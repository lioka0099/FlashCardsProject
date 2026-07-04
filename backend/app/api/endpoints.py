import os

try:
    from fastapi import BackgroundTasks, Body, FastAPI, File, Form, Header, Query, Request, UploadFile  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    from fastapi.responses import FileResponse, JSONResponse  # type: ignore
except Exception as exc:
    raise ImportError(
        "FastAPI is required but not installed. Install it with 'pip install fastapi uvicorn'."
    ) from exc

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from app.api.schemas import (
    CardListResponse,
    CardResponse,
    ExamListResponse,
    ExamResponse,
    GenerateSingleCardRequest,
    GenerateSingleCardResponse,
    NextCardResponse,
    ProofSpan,
    ReviewCardRequest,
    SessionEventRequest,
    SessionEventResponse,
    TopicListResponse,
    TopicProgressResponse,
    TopicProficiencyResponse,
    TopicResponse,
)
from app.data.db_engine import get_db
from app.data.models import CardReview
from app.data.vector_store import VectorStore
from app.services.diagnostic_lifecycle import DiagnosticBootstrapError, DiagnosticLifecycleService
from app.services.exams import ImmutableExamError
from app.services.graph import generate_single_card
from app.services.ingestion import UnsupportedDocumentTypeError
from app.services.review_service import ReviewService
from app.services.session_card_generation import SessionCardGenerationService
from app.services.session_planner import SessionPlannerService
from app.services.context_packs import build_diverse_chunk_pack
from app.services.job_worker import start_embedded_worker

app = FastAPI(title="DocQA Proto")


@app.on_event("startup")
def _start_worker() -> None:
    start_embedded_worker()


def _latest_review_info_by_card(
    *,
    user_id: str,
    exam_id: str,
    card_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not card_ids:
        return {}
    with get_db() as db:
        rows = (
            db.query(CardReview)
            .filter(
                CardReview.user_id == user_id,
                CardReview.exam_id == exam_id,
                CardReview.card_id.in_(card_ids),
            )
            .order_by(CardReview.created_at.desc())
            .all()
        )
        latest: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if row.card_id in latest:
                continue
            latest[row.card_id] = {
                "rating": row.rating,
                "review_id": row.review_id,
                "reviewed_at": row.created_at.isoformat() if row.created_at else None,
            }
        return latest

_raw_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_cors_origins = [origin.strip() for origin in _raw_cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _immutable_exam_error_payload(*, exam_id: Optional[str] = None, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": "immutable_exam",
            "message": message,
            "exam_id": exam_id,
        }
    }


def _extract_uploaded_files(files: List[UploadFile]) -> tuple[List[Path], List[str]]:
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    temp_paths: List[Path] = []
    filenames: List[str] = []
    for file in files:
        temp_path = uploads / file.filename
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        temp_paths.append(temp_path)
        filenames.append(file.filename)
    return temp_paths, filenames


def _is_pdf_ref(ref: str) -> bool:
    """True when a document reference (a stored file path or an http doc_id URL)
    points at a PDF. Used so the frontend can render PDFs in the preview modal
    and fall back to a new tab for DOCX/TXT sources."""
    return (ref or "").lower().split("?")[0].split("#")[0].rstrip().endswith(".pdf")


def _card_to_response(store: VectorStore, card_id: str) -> Optional[CardResponse]:
    card = store.db.get_card(card_id=card_id)
    if card is None:
        return None
    topic_map = {t.topic_id: t.label for t in store.db.list_topics(exam_id=card.exam_id)}
    proofs = [
        ProofSpan(
            proof_id=p.proof_id,
            doc_id=p.doc_id,
            page=p.page,
            start=p.start,
            end=p.end,
            text=p.text,
            score=float(p.score or 0.0),
            is_pdf=_is_pdf_ref(store.db.get_document_path(doc_id=p.doc_id) or p.doc_id),
        )
        for p in store.db.list_card_proofs(card_id=card.card_id)
    ]
    return CardResponse(
        card_id=card.card_id,
        exam_id=card.exam_id,
        topic_id=card.topic_id,
        topic_label=topic_map.get(card.topic_id),
        question=card.question,
        answer=card.answer,
        difficulty=int(card.difficulty),
        created_at=card.created_at,
        status=card.status,
        proofs=proofs,
        info=card.info or {},
    )


def _exam_to_response(exam: Any) -> ExamResponse:
    info = dict(exam.info or {})
    info.update(
        {
            "state": exam.state,
            "diagnostic_total": exam.diagnostic_total,
            "diagnostic_answered": exam.diagnostic_answered,
            "diagnostic_started_at": exam.diagnostic_started_at,
            "diagnostic_completed_at": exam.diagnostic_completed_at,
        }
    )
    return ExamResponse(
        exam_id=exam.exam_id,
        user_id=exam.user_id,
        title=exam.title,
        mode=exam.mode,
        created_at=exam.created_at,
        updated_at=exam.updated_at,
        info=info,
    )


SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".docx", ".txt")


@app.post("/exams/from-upload")
async def create_exam_from_upload_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    title: str = Form(...),
    mode: str = Form(default="mastery"),
):
    for f in files:
        name = (f.filename or "").lower()
        if not name.endswith(SUPPORTED_UPLOAD_EXTENSIONS):
            return JSONResponse(
                {"error": "unsupported_document_type",
                 "message": f"Unsupported file type: {f.filename}. Use PDF, DOCX, or TXT."},
                status_code=422,
            )

    temp_paths, filenames = _extract_uploaded_files(files)
    store = VectorStore()
    svc = DiagnosticLifecycleService(store=store)
    exam_id = svc.create_processing_exam(
        user_id=user_id, title=title, mode=mode,
        info={"source": "from_upload", "filenames": filenames},
    )
    store.db.enqueue_job(
        exam_id=exam_id,
        payload={"paths": [str(p) for p in temp_paths], "filenames": filenames,
                 "title": title, "mode": mode},
    )
    return {"exam_id": exam_id, "state": "processing"}


@app.get("/exams", response_model=ExamListResponse)
async def list_exams_endpoint(
    user_id: str = Query(...),
    limit: int = Query(default=50, ge=1, le=500),
):
    store = VectorStore()
    exams = store.db.list_exams(user_id=user_id, limit=limit)
    exams = [x for x in exams if getattr(x, "state", None) != "failed"]
    return ExamListResponse(exams=[_exam_to_response(x) for x in exams])


@app.get("/exams/{exam_id}")
async def get_exam_endpoint(exam_id: str, user_id: str = Query(...)):
    store = VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        return JSONResponse({"error": f"Exam not found: {exam_id}"}, status_code=404)
    if exam.user_id != user_id:
        return JSONResponse({"error": "Exam does not belong to user"}, status_code=403)
    return {
        "exam_id": exam.exam_id,
        "user_id": exam.user_id,
        "title": exam.title,
        "mode": exam.mode,
        "state": exam.state,
        "diagnostic_total": exam.diagnostic_total,
        "diagnostic_answered": exam.diagnostic_answered,
        "diagnostic_started_at": exam.diagnostic_started_at,
        "diagnostic_completed_at": exam.diagnostic_completed_at,
        "created_at": exam.created_at,
        "updated_at": exam.updated_at,
        "info": exam.info,
    }


@app.get("/exams/{exam_id}/topics", response_model=TopicListResponse)
async def list_topics_endpoint(exam_id: str):
    store = VectorStore()
    rows = store.db.list_topics(exam_id=exam_id)
    topics: List[TopicResponse] = []
    for t in rows:
        topics.append(
            TopicResponse(
                topic_id=t.topic_id,
                exam_id=t.exam_id,
                label=t.label,
                created_at=t.created_at,
                n_chunks=int(t.info.get("n_chunks") or 0),
                info=t.info or {},
            )
        )
    return TopicListResponse(exam_id=exam_id, topics=topics)


@app.post("/exams/{exam_id}/topics/{topic_id}/cards/generate", response_model=GenerateSingleCardResponse)
async def generate_single_card_endpoint(exam_id: str, topic_id: str, req: GenerateSingleCardRequest):
    store = VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        return GenerateSingleCardResponse(card=None, error=f"Exam not found: {exam_id}")
    if store.vector_backend == "pinecone":
        namespace = f"u:{exam.user_id}|e:{exam_id}"
        store.set_namespace(namespace)
    topic_map = {t.topic_id: t for t in store.db.list_topics(exam_id=exam_id)}
    topic = topic_map.get(topic_id)
    if topic is None:
        return GenerateSingleCardResponse(card=None, error=f"Topic not found: {topic_id}")
    chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=topic_id)
    if not chunk_ids:
        return GenerateSingleCardResponse(card=None, error="Topic has no chunks")
    try:
        context_pack = build_diverse_chunk_pack(
            store=store,
            chunk_ids=chunk_ids,
            centroid=store.db.get_topic_vector(topic_id=topic_id),
        )
        card = generate_single_card(
            exam_id=exam_id,
            topic_id=topic_id,
            topic_label=topic.label,
            allowed_chunk_ids=chunk_ids,
            context_pack=context_pack,
            difficulty=req.difficulty,
            card_type="learning",
            user_id=req.user_id,
            store=store,
        )
    except Exception as exc:
        return GenerateSingleCardResponse(card=None, error=str(exc))
    if card is None:
        return GenerateSingleCardResponse(card=None, error="Could not generate card")
    payload = _card_to_response(store, card.card_id)
    return GenerateSingleCardResponse(card=payload, error=None)


@app.get("/exams/{exam_id}/cards", response_model=CardListResponse)
async def list_cards_endpoint(
    exam_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
):
    store = VectorStore()
    rows = store.db.list_cards_for_exam(exam_id=exam_id, limit=limit)
    cards: List[CardResponse] = []
    for row in rows:
        payload = _card_to_response(store, row.card_id)
        if payload is not None:
            cards.append(payload)
    return CardListResponse(cards=cards, total=len(cards))


@app.get("/exams/{exam_id}/session/next-card", response_model=NextCardResponse)
async def next_card_endpoint(
    exam_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Query(...),
):
    store = VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        return JSONResponse({"error": f"Exam not found: {exam_id}"}, status_code=404)
    if exam.user_id != user_id:
        return JSONResponse({"error": "Exam does not belong to user"}, status_code=403)

    # Resume guard: if the last served card hasn't been rated yet, return it again
    # instead of planning + recording a brand-new presentation. This makes the endpoint
    # idempotent across page mounts/reloads (the frontend fetches it from a useQuery that
    # refetches on every mount). Without it, each spurious fetch records a "presented" row
    # and permanently advances the progression queue for a card the user never saw/rated,
    # producing the "Not rated" ghost cards in history.
    session_state = store.db.get_exam_session_state(user_id=user_id, exam_id=exam_id)
    last_served_id = session_state.last_served_card_id if session_state else None
    if last_served_id:
        last_card = store.db.get_card(card_id=last_served_id)
        if last_card is not None and last_card.status == "active":
            reviewed = _latest_review_info_by_card(
                user_id=user_id, exam_id=exam_id, card_ids=[last_served_id]
            )
            if last_served_id not in reviewed:
                card = _card_to_response(store, last_served_id)
                return NextCardResponse(
                    card=card, reason="resumed", no_cards_available=False, message=None
                )

    planner = SessionPlannerService(repo=store.db)
    generator = SessionCardGenerationService(repo=store.db)
    generator.archive_invalid_prefetched_cards(user_id=user_id, exam_id=exam_id)
    planned = planner.plan_next_card(user_id=user_id, exam_id=exam_id)
    if planned is None:
        planned = generator.get_fresh_prefetched_card(user_id=user_id, exam_id=exam_id)
    if planned is None:
        planned = generator.generate_next_card(user_id=user_id, exam_id=exam_id, store=store)
    if planned is None:
        return NextCardResponse(card=None, reason=None, no_cards_available=True, message="No cards available")
    if planned.reason == "prefetched":
        generator.mark_prefetched_card_served(card_id=planned.card_id, user_id=user_id)

    now = datetime.now(timezone.utc)
    store.db.upsert_exam_session_state(
        user_id=user_id,
        exam_id=exam_id,
        last_served_card_id=planned.card_id,
        last_presented_at=now,
    )
    store.db.append_card_presentation(
        user_id=user_id,
        exam_id=exam_id,
        card_id=planned.card_id,
        presented_at=now,
        info={"source": "session_orchestrator", "reason": planned.reason},
    )
    background_tasks.add_task(
        SessionCardGenerationService().refill_prefetch_if_needed,
        user_id=user_id,
        exam_id=exam_id,
    )
    card = _card_to_response(store, planned.card_id)
    return NextCardResponse(card=card, reason=planned.reason, no_cards_available=False, message=None)


@app.get("/exams/{exam_id}/session/previous-card", response_model=NextCardResponse)
async def previous_card_endpoint(exam_id: str, user_id: str = Query(...)):
    store = VectorStore()
    latest = store.db.get_latest_presentation(user_id=user_id, exam_id=exam_id)
    if latest is None:
        return NextCardResponse(card=None, reason=None, no_cards_available=True, message="No previous card")
    prev = store.db.get_previous_presentation(
        user_id=user_id,
        exam_id=exam_id,
        current_sequence_no=latest.sequence_no,
    )
    if prev is None:
        return NextCardResponse(card=None, reason=None, no_cards_available=True, message="No previous card")
    card = _card_to_response(store, prev.card_id)
    return NextCardResponse(card=card, reason="previous", no_cards_available=False, message=None)


@app.get("/exams/{exam_id}/cards/presented-history", response_model=CardListResponse)
async def presented_history_endpoint(
    exam_id: str,
    user_id: str = Query(...),
    limit: int = Query(default=500, ge=1, le=1000),
):
    store = VectorStore()
    rows = store.db.list_presentations(user_id=user_id, exam_id=exam_id, ascending=False, limit=limit)
    learning_rows = []
    seen_card_ids = set()
    learning_reasons = {"diagnostic", "overdue", "remediation", "progression", "generated", "prefetched"}
    for row in rows:
        info = row.info or {}
        if info.get("source") != "session_orchestrator":
            continue
        if info.get("reason") not in learning_reasons:
            continue
        if row.card_id in seen_card_ids:
            continue
        seen_card_ids.add(row.card_id)
        learning_rows.append(row)
    latest_reviews = _latest_review_info_by_card(
        user_id=user_id,
        exam_id=exam_id,
        card_ids=[row.card_id for row in learning_rows],
    )
    cards: List[CardResponse] = []
    for row in learning_rows:
        payload = _card_to_response(store, row.card_id)
        if payload is not None:
            info = dict(payload.info or {})
            review_info = latest_reviews.get(row.card_id)
            info.update(
                {
                    "presented_at": row.presented_at,
                    "presentation_sequence_no": row.sequence_no,
                    "presentation_source": (row.info or {}).get("source"),
                    "presentation_reason": (row.info or {}).get("reason"),
                }
            )
            if review_info is not None:
                info.update(
                    {
                        "rating": review_info.get("rating"),
                        "last_rating": review_info.get("rating"),
                        "review_id": review_info.get("review_id"),
                        "reviewed_at": review_info.get("reviewed_at"),
                    }
                )
            payload.info = info
            cards.append(payload)
    return CardListResponse(cards=cards, total=len(cards))


@app.get("/documents/{doc_id}/source")
async def document_source_endpoint(doc_id: str, exam_id: str = Query(...), user_id: str = Query(...)):
    store = VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        return JSONResponse({"error": f"Exam not found: {exam_id}"}, status_code=404)
    if exam.user_id != user_id:
        return JSONResponse({"error": "Exam does not belong to user"}, status_code=403)
    exam_doc_ids = set(store.db.list_exam_documents(exam_id=exam_id))
    if doc_id not in exam_doc_ids:
        return JSONResponse({"error": f"Document not found in exam: {doc_id}"}, status_code=404)

    doc_path = store.db.get_document_path(doc_id=doc_id)
    if not doc_path:
        return JSONResponse({"error": f"Document path not found: {doc_id}"}, status_code=404)
    source_path = Path(doc_path)
    if not source_path.exists() or not source_path.is_file():
        return JSONResponse({"error": f"Document file is unavailable: {doc_id}"}, status_code=404)

    suffix = source_path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".txt":
        media_type = "text/plain; charset=utf-8"
    elif suffix == ".docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(path=source_path, media_type=media_type, filename=source_path.name)


@app.post("/exams/{exam_id}/cards/{card_id}/review")
async def review_card_endpoint(
    exam_id: str,
    card_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    body: Optional[ReviewCardRequest] = Body(default=None),
    user_id: Optional[str] = Form(default=None),
    rating: Optional[str] = Form(default=None),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    parsed_user_id = user_id
    parsed_rating = rating
    if body is not None:
        parsed_user_id = body.user_id
        parsed_rating = body.rating
    if not parsed_user_id or not parsed_rating:
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
            parsed_user_id = parsed_user_id or payload.get("user_id")
            parsed_rating = parsed_rating or payload.get("rating")
    if not parsed_user_id or not parsed_rating:
        return JSONResponse({"error": "Missing user_id or rating"}, status_code=422)

    key = (idempotency_key or "").strip() or (
        f"{parsed_user_id}:{exam_id}:{card_id}:{parsed_rating}:{datetime.now(timezone.utc).isoformat()}"
    )
    service = ReviewService()
    try:
        result = service.apply_review(
            user_id=parsed_user_id,
            exam_id=exam_id,
            card_id=card_id,
            rating=parsed_rating,
            idempotency_key=key,
        )
    except ValueError as exc:
        message = str(exc)
        status = 400
        if "not found" in message.lower():
            status = 404
        if "does not belong" in message.lower():
            status = 403
        return JSONResponse({"error": message}, status_code=status)

    if not result.idempotent_replay:
        background_tasks.add_task(
            SessionCardGenerationService().refill_prefetch_if_needed,
            user_id=parsed_user_id,
            exam_id=exam_id,
        )

    # Keep backward-compatible fields while exposing schema fields.
    response = {
        "review_id": result.review_id,
        "card_id": result.card_id,
        "rating": result.rating,
        "due_at": result.due_at,
        "interval_days": result.interval_days,
        "ease": result.ease,
        "topic_proficiency": result.topic_proficiency,
        "diagnostic_answered": result.diagnostic_answered,
        "diagnostic_total": result.diagnostic_total,
        "exam_state": result.exam_state,
        "idempotent_replay": result.idempotent_replay,
    }
    return response


@app.post("/exams/{exam_id}/session/event", response_model=SessionEventResponse)
async def session_event_endpoint(exam_id: str, req: SessionEventRequest):
    store = VectorStore()
    event_id = store.db.add_event(
        user_id=req.user_id,
        exam_id=exam_id,
        type=req.event_type,
        payload=req.payload or {},
    )
    return SessionEventResponse(event_id=event_id)


@app.get("/exams/{exam_id}/progress", response_model=TopicProgressResponse)
async def topic_progress_endpoint(exam_id: str, user_id: str = Query(...)):
    store = VectorStore()
    topics = {t.topic_id: t for t in store.db.list_topics(exam_id=exam_id)}
    prof_rows = store.db.list_topic_proficiencies(user_id=user_id, exam_id=exam_id)
    payload: List[TopicProficiencyResponse] = []
    for p in prof_rows:
        payload.append(
            TopicProficiencyResponse(
                topic_id=p.topic_id,
                topic_label=topics.get(p.topic_id).label if topics.get(p.topic_id) else p.topic_id,
                proficiency=float(p.proficiency),
                last_updated_at=p.last_updated_at,
                n_reviews=int(p.seen_count),
                current_difficulty=int(p.current_difficulty),
                by_route=(p.info or {}).get("route_proficiency") if isinstance(p.info, dict) else None,
            )
        )
    overall = (sum(x.proficiency for x in payload) / len(payload)) if payload else None
    return TopicProgressResponse(
        exam_id=exam_id,
        user_id=user_id,
        topics=payload,
        overall_proficiency=overall,
    )


