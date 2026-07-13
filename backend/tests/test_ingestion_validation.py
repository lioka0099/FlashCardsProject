import os
import tempfile
import unittest
from pathlib import Path

# Configure a dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase_ingestion_validation_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.vector_store import VectorStore
from app.services.corpus.ingestion import UnsupportedDocumentTypeError, ingest_documents


def _write_file(name: str, body: str) -> str:
    doc_path = _TEST_ROOT / name
    doc_path.write_text(body, encoding="utf-8")
    return str(doc_path)


class IngestionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.store = VectorStore(basepath=str(_TEST_ROOT / "store"))
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: "u-invalid"

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_current_user, None)

    def test_ingest_documents_rejects_unsupported_extension(self) -> None:
        bad_doc = _write_file("unsupported.csv", "a,b,c\n1,2,3")
        with self.assertRaises(UnsupportedDocumentTypeError):
            ingest_documents([bad_doc], store=self.store)

    def test_upload_endpoint_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/exams/from-upload",
            data={"user_id": "u-invalid", "title": "Invalid upload", "mode": "mastery"},
            files=[("files", ("notes.csv", "a,b,c\n1,2,3", "text/csv"))],
        )
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload.get("error"), "unsupported_document_type")
        self.assertIn("Unsupported file type", payload.get("message", ""))


if __name__ == "__main__":
    unittest.main()
