import os
import tempfile
import unittest
from pathlib import Path

# Configure a dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase0_immutable_exam_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from fastapi.testclient import TestClient

from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.vector_store import VectorStore
from app.deps import get_repo
from app.services.exams import ImmutableExamError, attach_documents, create_exam
from app.services.ingestion import ingest_documents


def _write_text_doc(name: str, body: str) -> str:
    doc_path = _TEST_ROOT / name
    doc_path.write_text(body, encoding="utf-8")
    return str(doc_path)


class ImmutableExamGuardrailsTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.store = VectorStore(basepath=str(_TEST_ROOT / "store"))

    def test_first_exam_ingest_is_allowed_but_second_is_blocked(self) -> None:
        exam_id = create_exam(repo=get_repo(), user_id="user-a", title="Phase 0 test")
        initial_doc = _write_text_doc("doc_initial.txt", "intro text for first ingest")

        first_results = ingest_documents(
            [initial_doc],
            store=self.store,
            user_id="user-a",
            exam_id=exam_id,
        )
        attach_documents(
            repo=get_repo(),
            exam_id=exam_id,
            doc_ids=[result.doc_id for result in first_results],
        )

        second_doc = _write_text_doc("doc_second.txt", "new text that should be rejected")
        with self.assertRaises(ImmutableExamError):
            ingest_documents(
                [second_doc],
                store=self.store,
                user_id="user-a",
                exam_id=exam_id,
            )

    def test_second_attach_documents_call_is_rejected(self) -> None:
        exam_id = create_exam(repo=get_repo(), user_id="user-b", title="Attach guard test")
        first_doc = _write_text_doc("attach_first.txt", "first attach content")
        second_doc = _write_text_doc("attach_second.txt", "second attach content")

        first_results = ingest_documents([first_doc], store=self.store)
        attach_documents(
            repo=get_repo(),
            exam_id=exam_id,
            doc_ids=[first_results[0].doc_id],
        )

        second_results = ingest_documents([second_doc], store=self.store)
        with self.assertRaises(ImmutableExamError):
            attach_documents(
                repo=get_repo(),
                exam_id=exam_id,
                doc_ids=[second_results[0].doc_id],
            )


if __name__ == "__main__":
    unittest.main()
