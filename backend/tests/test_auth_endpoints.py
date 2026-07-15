import uuid
from fastapi.testclient import TestClient
from app.api.endpoints import app
from app.data.db_engine import init_db

client = TestClient(app)


def _email():
    return f"{uuid.uuid4().hex}@example.com"


def test_register_returns_token_and_user_id():
    init_db()
    r = client.post("/auth/register", json={"email": _email(), "password": "pw123456", "name": "Ada"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token"]
    assert body["user_id"]


def test_register_rejects_duplicate_email():
    init_db()
    email = _email()
    assert client.post("/auth/register", json={"email": email, "password": "pw123456"}).status_code == 200
    dup = client.post("/auth/register", json={"email": email, "password": "pw123456"})
    assert dup.status_code == 409


def test_login_succeeds_then_me_returns_identity():
    init_db()
    email = _email()
    client.post("/auth/register", json={"email": email, "password": "pw123456", "name": "Ada"})
    login = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    assert login.status_code == 200, login.text
    token = login.json()["token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["email"] == email
    assert me.json()["name"] == "Ada"


def test_login_wrong_password_401():
    init_db()
    email = _email()
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    bad = client.post("/auth/login", json={"email": email, "password": "nope"})
    assert bad.status_code == 401


def test_me_without_token_401():
    assert client.get("/auth/me").status_code == 401


def test_login_rate_limited_after_10_per_minute():
    init_db()
    app.state.limiter.reset()  # isolate from other tests' calls to /auth/login
    email = _email()
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    for _ in range(10):
        client.post("/auth/login", json={"email": email, "password": "wrong"})
    limited = client.post("/auth/login", json={"email": email, "password": "wrong"})
    assert limited.status_code == 429
