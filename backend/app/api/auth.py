"""Self-hosted auth: stdlib password hashing (pbkdf2) + HMAC-signed bearer tokens.

No external crypto dependencies. Tokens are issued and verified only here with a
single fixed algorithm (HMAC-SHA256), so JWT alg-confusion footguns don't apply.
"""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from fastapi import Header, HTTPException

_PBKDF2_ITERATIONS = 600_000
_TOKEN_TTL_SECONDS = 30 * 24 * 3600  # 30 days
# ponytail: dev default so local runs work with no setup; MUST set JWT_SECRET in prod.
_DEV_SECRET = "dev-insecure-secret-change-me"


def _secret() -> bytes:
    return os.getenv("JWT_SECRET", _DEV_SECRET).encode()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"{_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iters_s, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters_s)
        )
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(user_id: str) -> str:
    payload = _b64(
        json.dumps({"sub": user_id, "exp": int(time.time()) + _TOKEN_TTL_SECONDS}).encode()
    )
    sig = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[str]:
    try:
        payload, sig = token.split(".")
    except (ValueError, AttributeError):
        return None
    expected = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_b64d(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("exp", 0) < time.time():
        return None
    return data.get("sub")


def get_current_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    user_id = verify_token(authorization[len("Bearer "):])
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id
