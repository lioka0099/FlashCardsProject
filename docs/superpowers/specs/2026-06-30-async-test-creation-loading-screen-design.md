# Async test creation + loading screen — design

**Date:** 2026-06-30
**Status:** Draft for review

## Problem

Pressing **Create Test** currently fires a single blocking `POST /exams/from-upload`
that runs the entire pipeline (ingest → topics → cards, ~100s) before responding.
The user stares at an inline fake progress bar with no insight into what's happening,
and the request must stay open the whole time.

We want a dedicated loading screen (per the provided mockup) that shows the real
pipeline steps advancing, with green checkmarks as each completes — and the work
must keep running if the user closes the tab. The deployment target is the cloud
(multiple instances, instances can restart/scale down), so the design must not
depend on work living in a single web process.

## Decisions (locked)

1. **Real backend progress**, not a simulated timeline.
2. **DB-backed job queue** for execution (no Redis/Celery; reuses the existing DB).
   All progress + state is persisted in the DB, so polling is correct across
   instances, and a worker that dies mid-job gets its job re-claimed.
3. **Embedded worker thread now**: the web app starts a worker thread on boot that
   polls the jobs table. The job logic lives in a standalone module so moving to
   dedicated worker containers in the cloud is a deploy/config change, not a rewrite.

## Architecture

```
Create Test (frontend)
   │  POST /exams/from-upload  (cheap validation only)
   ▼
endpoint: persist files → create exam(state="processing") → enqueue Job(queued)
   │  returns { exam_id, state:"processing" }  immediately
   ▼
frontend: router.push(/exams/{id}/creating)
   │  polls GET /exams/{id} every ~1.5s
   ▼
worker thread (embedded): claim queued/stale job → run pipeline →
   write exam.info.progress per phase + heartbeat → state="diagnostic" | "failed"
   ▼
frontend sees state=="diagnostic" → router.replace(/exams/{id})
              state=="failed"     → friendly error + back / retry
```

### New exam states

`processing` (job queued/running), `failed` (job gave up), plus today's
`diagnostic` (ready). `state` is a free-text column — no schema migration needed.

### Progress shape (stored in `exam.info.progress`)

```jsonc
{
  "steps": [
    { "key": "uploading",     "label": "Uploading document",      "detail": "Document uploaded successfully", "status": "done" },
    { "key": "reading",       "label": "Reading pages",           "detail": "Reading 120 pages",              "status": "active" },
    { "key": "understanding", "label": "Understanding concepts",  "detail": "AI is identifying key concepts", "status": "pending" },
    { "key": "topics",        "label": "Finding important topics","detail": "Extracting relevant topics",     "status": "pending" },
    { "key": "questions",     "label": "Creating questions",      "detail": "Generating high-quality questions","status": "pending" },
    { "key": "finalizing",    "label": "Building your study set", "detail": "Organizing and optimizing your flashcards","status": "pending" }
  ],
  "updated_at": "2026-06-30T12:00:00Z"
}
```

`status ∈ done | active | pending | failed`. The frontend renders straight from
this array. The "Reading N pages" detail uses the real total page count (available
after load); a live "page 17 of 120" ticker is **out of scope** (pages load in one
batch — would need loader instrumentation). Noted as a future nice-to-have.

### Step → real phase mapping

| Step | Backend phase |
|---|---|
| Uploading document | files received (done as soon as the job starts) |
| Reading pages | `ingest_documents` |
| Understanding concepts | `classify_document_math_profile` |
| Finding important topics | `build_topics_for_exam` |
| Creating questions | `generate_starter_cards_v2` |
| Building your study set | finalize lifecycle → `diagnostic` |

## Backend components

### `Job` model (`app/data/models.py`)

| column | type | notes |
|---|---|---|
| job_id | str PK | uuid |
| exam_id | str | the exam being built |
| user_id | str | |
| type | str | `"bootstrap_exam"` |
| status | str | `queued` \| `running` \| `done` \| `failed` |
| payload | JSON | `{ paths, filenames, title, mode }` |
| attempts | int | retry counter |
| max_attempts | int | default 3 |
| heartbeat_at | datetime\|null | updated while running; stale ⇒ reclaimable |
| error | text\|null | last failure message |
| created_at / updated_at | datetime | |

New table → created by existing `Base.metadata.create_all` in `init_db()`.
No migration to existing tables.

### Repository (`app/data/db_repository.py`)

- `enqueue_job(...)` — insert a `queued` job.
- `claim_next_job()` — atomically pick one `queued` job, **or** a `running` job
  whose `heartbeat_at` is older than `STALE_SECONDS`; set `status=running`,
  `heartbeat_at=now`, bump `attempts`. SQLite: do this inside a single
  transaction with a guarded `UPDATE ... WHERE status=... ` and re-read.
- `heartbeat_job(job_id)` — bump `heartbeat_at`.
- `finish_job(job_id, ok, error=None)` — set `done` or, if `attempts >= max`,
  `failed`; otherwise back to `queued` for retry.
- `set_exam_progress(exam_id, steps)` — patch `info.progress`.

### Worker (`app/services/job_worker.py`, new — standalone)

- `run_once()` — claim a job, dispatch by `type`, heartbeat between phases,
  finish/fail. Pure DB in/out (no web-process coupling) so it runs unchanged
  in a cloud worker container.
- `worker_loop(stop_event, poll_interval=1.0)` — loop calling `run_once`.
- `start_embedded_worker()` — launch `worker_loop` in a daemon thread; called
  from a FastAPI startup hook. Uses its own DB session per `get_db()` (SQLite:
  separate connection per thread).

### Lifecycle split (`app/services/diagnostic_lifecycle.py`)

Split today's `bootstrap_exam_from_upload` (create + run) into:

- `create_processing_exam(user_id, title, mode, info)` → creates the exam with
  `state="processing"` and an initial `info.progress` (all steps pending), returns
  `exam_id`. Called by the endpoint.
- `run_bootstrap(exam_id, paths, user_id, *, reporter)` → does ingest → math
  profile → topics → cards → finalize on the existing exam, calling
  `reporter(step_key, status, detail)` at each phase boundary, and setting
  `state="diagnostic"` on success. On exception, records the message and re-raises
  so the worker can mark the job failed/retry; terminal failure sets
  `state="failed"` + `info.bootstrap_error`. Called by the worker.

`reporter` writes through `set_exam_progress`. The existing synchronous function
is kept as a thin `create_processing_exam` + `run_bootstrap(no-op reporter)` for
any sync callers/tests.

### Endpoint changes (`app/api/endpoints.py`)

- `POST /exams/from-upload`:
  1. Persist files (existing `_extract_uploaded_files` → `uploads/`).
  2. **Cheap sync validation only** — reject unsupported extensions with the same
     422 as today (good immediate UX). Deep failures (too few cards, etc.) happen
     in the worker and become `failed` state.
  3. `create_processing_exam(...)`, `enqueue_job(type="bootstrap_exam", payload=...)`.
  4. Return `{ exam_id, state: "processing" }`.
- FastAPI startup hook → `start_embedded_worker()`.
- `GET /exams/{id}` — unchanged; already returns `state` + `info` (incl. progress).
- `GET /exams` (list) — exclude `failed`; keep `processing` (sidebar shows them).
- Uploaded files cleaned up by the worker after the job finishes.

## Frontend components

- **`src/lib/api/client.ts`**: `createExamFromUpload` now returns
  `{ exam_id, state }`. Add `BootstrapProgress` / `ProgressStep` types on
  `ExamDetails.info`. `getExamById` already exists for polling.
- **`upload-exam-form.tsx`**: `onSubmit` calls `createExamFromUpload`, then
  `router.push('/exams/{exam_id}/creating')`. Inline `MagicUploadProgress` retired
  from this form (component file may stay, unused).
- **`src/app/exams/[examId]/creating/page.tsx`** (new): client page that polls
  `getExamById` via `useQuery({ refetchInterval: 1500 })` and renders
  `<CreatingTest>`. On `state==="diagnostic"` → `router.replace('/exams/{id}')`;
  on `state==="failed"` → error view (message via `mapHomeApiError` on
  `info.bootstrap_error`) with **Back home** / **Try again**.
- **`src/components/exam/creating-test.tsx`** (new) + **`creating-test.css`**:
  the mockup — sparkle header "Creating your test…", subtitle, file chip with green
  check, the 6-step stepper (done = green check, active = spinner, pending = hollow
  circle) from `info.progress.steps`, the document illustration + "AI in action"
  callout, a "Did you know?" tip rotator (3 static tips cycling), footer note
  "You can close this window — we'll keep working in the background." Reuses the
  `--p / --accent / --success` palette tokens from the home redesign.
- **`src/app/exams/[examId]/page.tsx`**: if `state==="processing"`, redirect to the
  creating screen (handles revisiting a still-building exam).

## Error handling

- Unsupported file type: rejected synchronously at upload (422), as today.
- Pipeline failure (e.g. too few cards): worker retries up to `max_attempts`, then
  job `failed` + exam `state="failed"` + `info.bootstrap_error`. The creating screen
  shows the friendly mapped message + retry.
- Worker/instance dies mid-job: stale `heartbeat_at` ⇒ another worker re-claims the
  job (idempotency: `run_bootstrap` overwrites topics/cards with `overwrite=True`
  as today, so a re-run is safe).
- Polling 404 (exam vanished): treat as generic failure in the UI.

## Testing

**Backend**
- enqueue → `claim_next_job` → run (mocked pipeline phases) → job `done`, exam
  `diagnostic`, all progress steps `done`.
- pipeline raises → after `max_attempts` → job `failed`, exam `failed`, error stored.
- stale running job (old heartbeat) → reclaimed by `claim_next_job`.

**Frontend**
- `CreatingTest` renders done/active/pending states from a progress fixture.
- creating page redirects when polled state becomes `diagnostic`.
- creating page shows the error view on `failed`.

## Out of scope

- Live per-page "17 of 120" counter (needs loader instrumentation).
- External broker (Redis/Celery) — DB queue is sufficient at this scale; the
  enqueue seam lets us swap later without touching callers.
- Cancelling an in-flight job from the UI.
