import io
import pytest
from fastapi.testclient import TestClient
from app.api.auth import get_current_user
from app.api.endpoints import app
from app.data.db_engine import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def _override_auth():
    app.dependency_overrides[get_current_user] = lambda: "u-ep"
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_from_upload_returns_processing_immediately():
    init_db()
    files = {"files": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}
    data = {"user_id": "u-ep", "title": "Net", "mode": "mastery"}
    r = client.post("/exams/from-upload", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "processing"
    assert body["exam_id"]
    # exam is immediately readable and processing
    g = client.get(f"/exams/{body['exam_id']}", params={"user_id": "u-ep"})
    assert g.json()["state"] == "processing"
    assert g.json()["info"]["progress"]["steps"][0]["key"] == "uploading"


def test_from_upload_rejects_unsupported_type_synchronously():
    init_db()
    files = {"files": ("bad.exe", io.BytesIO(b"x"), "application/octet-stream")}
    data = {"user_id": "u-ep2", "title": "X", "mode": "mastery"}
    r = client.post("/exams/from-upload", files=files, data=data)
    assert r.status_code == 422
