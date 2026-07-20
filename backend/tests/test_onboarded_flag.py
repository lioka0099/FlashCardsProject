import uuid
from fastapi.testclient import TestClient
from app.api.endpoints import app
from app.data.db_engine import init_db

client = TestClient(app)


def _register_and_token():
    init_db()
    email = f"{uuid.uuid4().hex}@example.com"
    r = client.post("/auth/register", json={"email": email, "password": "pw123456", "name": "Ada"})
    return r.json()["token"]


def test_new_user_is_not_onboarded():
    token = _register_and_token()
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["onboarded"] is False


def test_mark_onboarded_sets_flag_and_is_idempotent():
    token = _register_and_token()
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post("/auth/me/onboarded", headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["onboarded"] is True

    # Idempotent: calling again still succeeds and stays onboarded.
    second = client.post("/auth/me/onboarded", headers=headers)
    assert second.status_code == 200
    assert second.json()["onboarded"] is True

    me = client.get("/auth/me", headers=headers)
    assert me.json()["onboarded"] is True


def test_mark_onboarded_without_token_401():
    assert client.post("/auth/me/onboarded").status_code == 401
