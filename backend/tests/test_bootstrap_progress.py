from unittest.mock import patch
import pytest
from app.data.db_engine import init_db, get_db
from app.data.models import User
from app.services.diagnostic_lifecycle import DiagnosticLifecycleService


def _seed_user(uid="u-prog"):
    with get_db() as db:
        db.merge(User(user_id=uid, name="t"))
    return uid


def test_create_processing_exam_initializes_progress():
    init_db()
    uid = _seed_user()
    svc = DiagnosticLifecycleService()
    exam_id = svc.create_processing_exam(user_id=uid, title="My Test", mode="mastery",
                                         info={"filenames": ["notes.pdf"]})
    exam = svc.repo.get_exam(exam_id)
    assert exam.state == "processing"
    steps = exam.info["progress"]["steps"]
    assert [s["key"] for s in steps] == ["uploading", "reading", "understanding",
                                         "topics", "questions", "finalizing"]
    assert all(s["status"] == "pending" for s in steps)
    assert exam.info["filenames"] == ["notes.pdf"]


def test_run_bootstrap_reports_steps_and_marks_ready():
    init_db()
    uid = _seed_user("u-prog2")
    svc = DiagnosticLifecycleService()
    exam_id = svc.create_processing_exam(user_id=uid, title="T", mode="mastery",
                                         info={"filenames": ["a.pdf"]})
    seen = []

    class _Doc:  # minimal ingest result
        doc_id = "d1"

    with patch("app.services.diagnostic_lifecycle.ingest_documents", return_value=[_Doc()]), \
         patch.object(svc.repo, "attach_documents_to_exam"), \
         patch.object(svc.repo, "list_chunks_by_doc", return_value=[]), \
         patch("app.services.diagnostic_lifecycle.classify_document_math_profile") as mp, \
         patch("app.services.diagnostic_lifecycle.build_topics_for_exam", return_value=[1, 2, 3]), \
         patch("app.services.diagnostic_lifecycle.generate_starter_cards_v2", return_value=[1, 2, 3]):
        mp.return_value.to_info.return_value = {}
        svc.run_bootstrap(exam_id=exam_id, user_id=uid, paths=["/tmp/a.pdf"],
                          reporter=lambda k, s, d: seen.append((k, s)))

    exam = svc.repo.get_exam(exam_id)
    assert exam.state == "diagnostic"
    # every step ended done
    assert all(st["status"] == "done" for st in exam.info["progress"]["steps"])
    # reporter saw an active->done arc for at least topics + questions
    assert ("topics", "active") in seen and ("questions", "done") in seen


def test_run_bootstrap_failure_marks_failed():
    init_db()
    uid = _seed_user("u-prog3")
    svc = DiagnosticLifecycleService()
    exam_id = svc.create_processing_exam(user_id=uid, title="T", mode="mastery", info={})
    with patch("app.services.diagnostic_lifecycle.ingest_documents", side_effect=RuntimeError("nope")):
        with pytest.raises(RuntimeError):
            svc.run_bootstrap(exam_id=exam_id, user_id=uid, paths=["/tmp/a.pdf"])
    exam = svc.repo.get_exam(exam_id)
    assert exam.state == "failed"
    assert "nope" in (exam.info.get("bootstrap_error") or "")
