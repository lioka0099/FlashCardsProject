from fastapi.testclient import TestClient
from app.api.endpoints import app
from app.data.db_engine import init_db

client = TestClient(app)


def test_list_exams_requires_auth():
    init_db()
    # No Authorization header → 401 (previously trusted ?user_id=)
    r = client.get("/exams", params={"user_id": "anyone"})
    assert r.status_code == 401


def test_next_card_requires_auth():
    init_db()
    r = client.get("/exams/some-exam/session/next-card", params={"user_id": "anyone"})
    assert r.status_code == 401


def test_list_topics_requires_auth():
    init_db()
    r = client.get("/exams/some-exam/topics")
    assert r.status_code == 401


def test_list_cards_requires_auth():
    init_db()
    r = client.get("/exams/some-exam/cards")
    assert r.status_code == 401
