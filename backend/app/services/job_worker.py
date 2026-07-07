"""Standalone DB-backed job worker. No FastAPI/request coupling: this module
runs unchanged as an embedded thread now and as a separate cloud process later."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from app.data.db_repository import DBRepository
from app.deps import get_repo
from app.services.diagnostic.lifecycle import DiagnosticLifecycleService

logger = logging.getLogger(__name__)


def _cleanup_paths(paths) -> None:
    for p in paths or []:
        try:
            os.remove(p)
        except OSError:
            pass


def run_once(repo: Optional[DBRepository] = None) -> bool:
    repo = repo or get_repo()
    job = repo.claim_next_job()
    if job is None:
        return False
    try:
        svc = DiagnosticLifecycleService(repo=repo)
        repo.heartbeat_job(job.exam_id)
        svc.run_bootstrap(
            exam_id=job.exam_id,
            user_id=job.user_id,
            paths=job.payload.get("paths", []),
            reporter=lambda k, s, d: repo.heartbeat_job(job.exam_id),
        )
        _cleanup_paths(job.payload.get("paths"))
        repo.finish_job(exam_id=job.exam_id, ok=True)
    except Exception as exc:  # noqa: BLE001 - worker must not die on a bad job
        logger.exception("Job for exam %s failed", job.exam_id)
        repo.finish_job(exam_id=job.exam_id, ok=False, error=str(exc))
    return True


def worker_loop(stop_event: threading.Event, *, poll_interval: float = 1.0) -> None:
    repo = get_repo()
    while not stop_event.is_set():
        try:
            did_work = run_once(repo)
        except Exception:  # noqa: BLE001
            logger.exception("worker_loop iteration error")
            did_work = False
        if not did_work:
            stop_event.wait(poll_interval)


def start_embedded_worker() -> threading.Event:
    stop_event = threading.Event()
    t = threading.Thread(target=worker_loop, args=(stop_event,), name="job-worker", daemon=True)
    t.start()
    logger.info("Embedded job worker started")
    return stop_event
