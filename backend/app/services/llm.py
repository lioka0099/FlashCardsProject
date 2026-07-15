"""OpenAI client wrapper: chat/embedding calls with retry+backoff, model config, and JSON-response parsing."""

from typing import Any, Callable, List, TypeVar
import json
import logging
import random
import re
import time
import threading
import numpy as np
from app.utils.common import getenv

_thread_local = threading.local()
logger = logging.getLogger(__name__)

T = TypeVar("T")

def client():
    # Lazy import to avoid hard dependency during unrelated code paths.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise ImportError(
            "OpenAI SDK is required but not installed. Install it with 'pip install openai'."
        ) from exc
    api_key = getenv("OPENAI_API_KEY")
    if api_key.lower().startswith("sk-your"):
        raise RuntimeError(
            "OPENAI_API_KEY is still set to the placeholder value. "
            "Update /Users/lioka/Desktop/flashcards/docqa-proto/.env with your real key."
        )
    c = getattr(_thread_local, "client", None)
    if c is None:
        logger.debug("Initializing OpenAI client (key prefix: %s****)", api_key[:8])
        c = OpenAI(api_key=api_key)
        _thread_local.client = c
    return c

EMBED_MODEL = getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = getenv("CHAT_MODEL", "gpt-4o")
CHAT_MODEL_FAST = getenv("CHAT_MODEL_FAST", "gpt-4o-mini")

def _is_retryable_openai_exception(exc: Exception) -> bool:
    """
    Check if the exception is retryable.
    """
    # Prefer typed exceptions when available.
    try:
        from openai import (  # type: ignore
            APIConnectionError,
            APIError,
            APITimeoutError,
            RateLimitError,
        )
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
                return True
    except Exception:
        # If imports fail, fall back to heuristics below.
        pass

    # Heuristic fallback: look for status code fields / common phrases.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "timeout" in msg or "timed out" in msg or "connection" in msg:
        return True
    return False


def openai_call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
) -> T:
    """
    Retry wrapper for transient OpenAI errors (429/5xx/timeouts/connection issues).
    Uses exponential backoff with jitter.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _is_retryable_openai_exception(exc):
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            # Add small jitter to avoid synchronized retries across threads.
            delay = delay * (1.0 + random.uniform(0.0, 0.2))
            logger.warning(
                "OpenAI call failed (attempt %d/%d). Retrying in %.2fs. Error=%s",
                attempt,
                max_retries,
                delay,
                repr(exc),
            )
            time.sleep(delay)


def chat_completions_create(**kwargs: Any):
    return openai_call_with_retries(lambda: client().chat.completions.create(**kwargs))


def embeddings_create(**kwargs: Any):
    return openai_call_with_retries(lambda: client().embeddings.create(**kwargs))


def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI returns list of vectors
    # Create embeddings for the texts
    resp = embeddings_create(model=EMBED_MODEL, input=texts)
    # Extract the embeddings from the response to an array of arrays
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])


_JSON_VALID_ESCAPES = set('"\\/bfnrtu')
_BACKSLASH_TOKEN = re.compile(r'\\(.)', re.DOTALL)


def _escape_stray_backslashes(raw: str) -> str:
    """
    Double any backslash that isn't already part of a valid JSON escape pair.

    Prompts that ask an LLM for JSON containing LaTeX (e.g. \\( ... \\), \\begin{..})
    require the model to double every backslash so the JSON stays valid. Models
    occasionally forget for one command, which breaks parsing for the whole
    response. This repairs that specific slip without touching backslash pairs
    that are already correctly escaped (including the ones LaTeX itself uses,
    e.g. the matrix row separator \\\\).
    """
    def repair(match: "re.Match[str]") -> str:
        ch = match.group(1)
        return match.group(0) if ch in _JSON_VALID_ESCAPES else "\\\\" + ch

    return _BACKSLASH_TOKEN.sub(repair, raw)


def safe_json_load(raw: str) -> dict:
    """
    Parse an LLM JSON response, returning {} on failure or non-object result.

    Tries a strict parse first, then relaxes in two ways an LLM's near-JSON can
    fail: literal control characters inside strings (strict=False), and a stray
    unescaped backslash from embedded LaTeX (_escape_stray_backslashes).
    """
    for candidate in (raw, _escape_stray_backslashes(raw)):
        for strict in (True, False):
            try:
                data = json.loads(candidate, strict=strict)
            except Exception:
                continue
            return data if isinstance(data, dict) else {}
    return {}
