import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="job_repo_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"

from app.data.db_engine import drop_all_tables, init_db, get_db
from app.data.db_repository import DBRepository
from app.data.models import User, Exam


def _repo():
    drop_all_tables()
    init_db()
    return DBRepository(_TEST_ROOT / "meta.sqlite")


def _seed_exam(exam_id="ex-job-1", user_id="u-job-1"):
    with get_db() as db:
        db.merge(User(user_id=user_id, name="t"))
        db.merge(Exam(exam_id=exam_id, user_id=user_id, title="t", state="processing"))
    return exam_id, user_id


def test_enqueue_and_claim_roundtrip():
    repo = _repo()
    exam_id, user_id = _seed_exam()
    repo.enqueue_job(exam_id=exam_id, payload={"paths": ["/tmp/a.pdf"], "title": "t", "mode": "mastery"})

    claimed = repo.claim_next_job()
    assert claimed is not None
    assert claimed.exam_id == exam_id
    assert claimed.user_id == user_id
    assert claimed.payload["paths"] == ["/tmp/a.pdf"]
    assert claimed.attempts == 1
    # second claim returns nothing (job now running)
    assert repo.claim_next_job() is None


def test_finish_failure_retries_then_fails():
    repo = _repo()
    exam_id, _ = _seed_exam(exam_id="ex-job-2")
    repo.enqueue_job(exam_id=exam_id, payload={})

    # attempt 1
    repo.claim_next_job()
    assert repo.finish_job(exam_id=exam_id, ok=False, error="boom") == "queued"
    # attempt 2
    repo.claim_next_job()
    assert repo.finish_job(exam_id=exam_id, ok=False, error="boom") == "queued"
    # attempt 3 -> terminal
    repo.claim_next_job()
    assert repo.finish_job(exam_id=exam_id, ok=False, error="boom") == "failed"


def test_stale_running_job_is_reclaimed():
    repo = _repo()
    exam_id, _ = _seed_exam(exam_id="ex-job-3")
    repo.enqueue_job(exam_id=exam_id, payload={})
    j = repo.claim_next_job()
    assert j.exam_id == exam_id
    # not reclaimable while fresh
    assert repo.claim_next_job(stale_seconds=300) is None
    # reclaimable once heartbeat is considered stale
    reclaimed = repo.claim_next_job(stale_seconds=0)
    assert reclaimed.exam_id == exam_id
