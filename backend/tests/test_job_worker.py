import os
import tempfile
from pathlib import Path
from unittest.mock import patch

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="job_worker_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"

from app.data.db_engine import drop_all_tables, init_db, get_db
from app.data.models import User, Exam
from app.data.db_repository import DBRepository
from app.services import job_worker


def _repo():
    drop_all_tables()
    init_db()
    return DBRepository(_TEST_ROOT / "meta.sqlite")


def _seed(exam_id, uid="u-w"):
    with get_db() as db:
        db.merge(User(user_id=uid, name="t"))
        db.merge(Exam(exam_id=exam_id, user_id=uid, title="t", state="processing"))
    return uid


def test_run_once_dispatches_bootstrap_and_marks_done():
    repo = _repo()
    uid = _seed("ex-w1")
    repo.enqueue_job(exam_id="ex-w1", payload={"paths": [], "title": "t", "mode": "mastery"})

    with patch("app.services.job_worker.DiagnosticLifecycleService") as Svc:
        Svc.return_value.run_bootstrap.return_value = None
        processed = job_worker.run_once(repo)

    assert processed is True
    Svc.return_value.run_bootstrap.assert_called_once()
    # empty queue now
    assert job_worker.run_once(repo) is False


def test_run_once_marks_failed_on_exception():
    repo = _repo()
    _seed("ex-w2")
    repo.enqueue_job(exam_id="ex-w2", payload={"paths": []})
    with patch("app.services.job_worker.DiagnosticLifecycleService") as Svc:
        Svc.return_value.run_bootstrap.side_effect = RuntimeError("kaboom")
        # exhaust retries
        for _ in range(3):
            job_worker.run_once(repo)
    # job ended failed
    with get_db() as db:
        assert db.query(Exam).filter(Exam.exam_id == "ex-w2").first().job_status == "failed"
