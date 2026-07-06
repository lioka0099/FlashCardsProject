"""Wiring seam: the one place that builds the persistence repo and the vector store.

Both are stateless (the repo is a method-bag over the global SQLAlchemy engine; the
store takes its namespace per call), so a single shared instance of each is safe across
requests and the worker thread. Services default their `repo=`/`store=` params to these
singletons; tests inject fakes instead. FastAPI endpoints receive them via `Depends`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.data.db_repository import DBRepository
from app.data.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_repo() -> DBRepository:
    return DBRepository(Path("store/meta.sqlite"))


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    # Share the one repo so the store's internal chunk/cache work and the injected
    # persistence repo are the same instance.
    return VectorStore(repo=get_repo())
