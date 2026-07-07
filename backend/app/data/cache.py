"""In-process cache for per-document context summaries."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class DocContextSummaryCache:
    """
    Tiny JSON-backed cache that persists sampled context excerpts and their summaries
    per document. Avoids re-summarizing the exact same content during every QA run.
    """

    def __init__(self, basepath: str, filename: str = "doc_context_cache.json"):
        self.path = Path(basepath) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt caches - fall back to empty to avoid blocking the run.
            self._data = {}

    # Flush the cache to disk
    def _flush(self) -> None: 
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.path)

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(doc_id)

    def set(self, doc_id: str, context: str, summary: str) -> None:
        with self._lock:
            self._data[doc_id] = {
                "context": context,
                "summary": summary,
                "updated_at": time.time(),
            }
            self._flush()

