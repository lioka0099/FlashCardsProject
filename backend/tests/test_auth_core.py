import time
import pytest
from app.api import auth


def test_password_hash_roundtrip():
    stored = auth.hash_password("hunter2")
    assert stored != "hunter2"
    assert auth.verify_password("hunter2", stored) is True
    assert auth.verify_password("wrong", stored) is False


def test_verify_password_rejects_garbage_stored_value():
    assert auth.verify_password("x", "not-a-valid-hash") is False
    assert auth.verify_password("x", "") is False


def test_token_roundtrip():
    token = auth.issue_token("user-123")
    assert auth.verify_token(token) == "user-123"


def test_token_rejects_tampered_signature():
    token = auth.issue_token("user-123")
    payload, _sig = token.split(".")
    forged = f"{payload}.{'A' * 43}"
    assert auth.verify_token(forged) is None


def test_token_rejects_garbage():
    assert auth.verify_token("garbage") is None
    assert auth.verify_token("") is None


def test_token_rejects_expired(monkeypatch):
    token = auth.issue_token("user-123")
    original_time = time.time
    monkeypatch.setattr(time, "time", lambda: original_time() + auth._TOKEN_TTL_SECONDS + 10)
    assert auth.verify_token(token) is None
