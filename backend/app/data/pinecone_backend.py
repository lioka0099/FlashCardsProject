"""Pinecone vector-DB client and per-exam namespace helpers (the cloud vector backend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.utils.common import getenv


def pinecone_namespace(*, user_id: str, exam_id: str) -> str:
    #namespaces per exam
    return f"u:{user_id}|e:{exam_id}"


def _normalize_vecs_L2(x: np.ndarray) -> np.ndarray:
    """
    Return a float32 L2-normalized copy/view of x (supports [dim] or [n, dim]).
    """
    arr = x.astype("float32", copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str
    index_chunks: str
    index_questions: str

    @staticmethod
    def from_env() -> "PineconeConfig":
        return PineconeConfig(
            api_key=getenv("PINECONE_API_KEY").strip(),
            index_chunks=getenv("PINECONE_INDEX_CHUNKS").strip(),
            index_questions=getenv("PINECONE_INDEX_QUESTIONS").strip(),
        )


class PineconeClient:
    """
    Minimal Pinecone wrapper for:
    - chunks index: vector_id = chunk_id
    - questions index: vector_id = question_id
    Both scoped by namespace = u:{user_id}|e:{exam_id}
    """

    def __init__(self, cfg: Optional[PineconeConfig] = None):
        self.cfg = cfg or PineconeConfig.from_env()

        try:
            from pinecone import Pinecone 
        except Exception as exc:
            raise ImportError(
                "Pinecone SDK not installed. Install it with 'pip install pinecone'."
            ) from exc

        pc = Pinecone(api_key=self.cfg.api_key)
        self.chunks = pc.Index(self.cfg.index_chunks)
        self.questions = pc.Index(self.cfg.index_questions)

    @staticmethod
    def _vec_to_list(vec: np.ndarray) -> List[float]:
        v = vec.astype("float32", copy=False).reshape(-1)
        return v.tolist()

    def upsert(
        self,
        *,
        index: Any,
        namespace: str,
        vectors: Sequence[Tuple[str, np.ndarray]],
        metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        if not vectors:
            return
        md_map = metadata_by_id or {}
        batch: List[Tuple[str, List[float], Dict[str, Any]]] = []
        for vid, vec in vectors:
            vec_n = _normalize_vecs_L2(vec)[0]
            batch.append((vid, self._vec_to_list(vec_n), md_map.get(vid, {})))
            if len(batch) >= batch_size:
                index.upsert(vectors=batch, namespace=namespace)
                batch = []
        if batch:
            index.upsert(vectors=batch, namespace=namespace)

    def query(
        self,
        *,
        index: Any,
        namespace: str,
        query_vec: np.ndarray,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        q = _normalize_vecs_L2(query_vec)[0]
        res = index.query(
            namespace=namespace,
            vector=self._vec_to_list(q),
            top_k=int(top_k),
            include_metadata=False,
            filter=filter or None,
        )
        matches = getattr(res, "matches", None) or []
        out: List[Tuple[str, float]] = []
        for m in matches:
            out.append((str(getattr(m, "id", "")), float(getattr(m, "score", 0.0))))
        return out

    def fetch_vectors(
        self,
        *,
        index: Any,
        namespace: str,
        ids: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        id_list = [i for i in ids if i]
        if not id_list:
            return {}
        res = index.fetch(ids=id_list, namespace=namespace)
        vectors = getattr(res, "vectors", None) or {}
        out: Dict[str, np.ndarray] = {}
        for vid, item in vectors.items():
            values = getattr(item, "values", None)
            if values is None:
                continue
            arr = np.array(values, dtype="float32")
            out[str(vid)] = arr
        return out

