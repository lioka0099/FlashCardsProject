from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence, Optional
import copy
import os
import numpy as np
from app.data.db_repository import DBRepository, StoredChunk
from app.data.pinecone_backend import PineconeClient


def _require_namespace(namespace: Optional[str], op: str) -> str:
    if not namespace:
        raise RuntimeError(
            f"VectorStore.{op} requires a namespace. For the Pinecone backend, pass "
            "namespace=pinecone_namespace(user_id=..., exam_id=...); retrieval is exam-scoped."
        )
    return namespace

VEC_DIM = 3072  # OpenAI text-embedding-3-large

def _normalize_L2(x: np.ndarray) -> None:
    # In-place L2 normalization along rows, similar to faiss.normalize_L2
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x[:] = x / norms

class _NumpyIPIndex:
    def __init__(self, dim: int, path: Path):
        self.dim = dim
        self.path = path
        if self.path.exists():
            arr = np.load(self.path)
            self.vectors = arr.astype("float32", copy=False)
        else:
            self.vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        _normalize_L2(vectors)
        if self.vectors.size == 0:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.save()

    def search(self, query_vec: np.ndarray, k: int):
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        _normalize_L2(query_vec)
        n = self.vectors.shape[0]
        if n == 0:
            D = np.zeros((1, k), dtype="float32")
            I = -np.ones((1, k), dtype="int64")
            return D, I
        # query_vec is (1, dim); compute dot-product scores
        scores = self.vectors @ query_vec[0]
        k_eff = min(k, n)
        idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
        idx_sorted = idx[np.argsort(-scores[idx])]
        D = scores[idx_sorted].astype("float32")
        I = idx_sorted.astype("int64")
        # pad to k
        if k_eff < k:
            pad_d = np.zeros(k - k_eff, dtype="float32")
            pad_i = -np.ones(k - k_eff, dtype="int64")
            D = np.concatenate([D, pad_d], axis=0)
            I = np.concatenate([I, pad_i], axis=0)
        return D[None, :k], I[None, :k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, self.vectors)


class VectorStore:
    def __init__(self, basepath: str = "store", *, repo: Optional[DBRepository] = None):
        self.base = Path(basepath)
        self.base.mkdir(parents=True, exist_ok=True)

        # Metadata store (SQLite). Injected by the wiring seam (app/deps.py) so the
        # store and the persistence repo share one instance; kept private so callers
        # depend on VectorStore for vectors and on the repo for persistence, never
        # reaching through the store into the database.
        self._repo = repo or DBRepository(self.base / "meta.sqlite")

        # Vector backend selection (default keeps current behavior)
        self.vector_backend: str = os.getenv("VECTOR_BACKEND", "pinecone").strip().lower()
        self._pinecone: Optional[PineconeClient] = None

        # Exam namespace. None on the shared instance (which holds no request state, so it's
        # safe to share across requests); bound only on the immutable copies returned by
        # for_namespace(). Vector methods resolve namespace from their arg or this field.
        self._namespace: Optional[str] = None

        # Local fallback index (numpy-only). Pinecone is the primary backend when configured.
        self.vectors_path = self.base / "vectors.npy"
        self.index = _NumpyIPIndex(VEC_DIM, self.vectors_path)
        self.vector_dimension = VEC_DIM

    def for_namespace(self, namespace: str) -> "VectorStore":
        """Return an exam-scoped view. It's a shallow copy that shares the repo, index and
        Pinecone client with this instance and only binds the namespace, so the shared store
        is never mutated and the namespace rides on the object down any retrieval chain."""
        scoped = copy.copy(self)
        scoped._namespace = namespace
        return scoped

    def _pinecone_client(self) -> PineconeClient:
        if self._pinecone is None:
            self._pinecone = PineconeClient()
        return self._pinecone

    def _index_size(self) -> int:
        return int(getattr(self.index, "vectors", np.zeros((0, VEC_DIM), dtype="float32")).shape[0])

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self._repo.add_document(doc_id, path, title, info)

    def add_chunks(
        self,
        chunk_rows: Iterable[StoredChunk],
        vectors: np.ndarray,
        *,
        namespace: Optional[str] = None,
    ):
        # Materialize to preserve insertion order for vector_index_map.
        chunk_list = list(chunk_rows)

        # 1) Save metadata to SQL
        self._repo.add_chunks(chunk_list)

        # 2a) Pinecone backend: upsert vectors by chunk_id into the exam-scoped namespace.
        if self.vector_backend == "pinecone":
            ns = _require_namespace(namespace or self._namespace, "add_chunks")
            if vectors.dtype != np.float32:
                vectors = vectors.astype("float32")
            pc = self._pinecone_client()
            # Attach minimal metadata for debugging/filters.
            meta_by_id: Dict[str, Dict[str, object]] = {}
            for ch in chunk_list:
                meta_by_id[ch.chunk_id] = {
                    "doc_id": ch.doc_id,
                    "page": int(ch.page),
                }
            pc.upsert(
                index=pc.chunks,
                namespace=ns,
                vectors=[(ch.chunk_id, vec) for ch, vec in zip(chunk_list, vectors)],
                metadata_by_id=meta_by_id,
                batch_size=100,
            )
            return

        # 2) Local fallback: add vectors to numpy index + persist stable mapping.
        # ponytail: shared in-memory numpy index — concurrent add_chunks in local-backend
        # mode race on self.index/vectors.npy. Pinecone (the default) sidesteps this. Add a
        # global index lock only if you ever run VECTOR_BACKEND=local under concurrency.
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        start_index = self._index_size()
        self.index.add(vectors)

        # Persist mapping from vector positions -> chunk_id for stable retrieval
        self._repo.add_vector_index_mapping(
            start_index=start_index,
            chunk_ids=[c.chunk_id for c in chunk_list],
        )

    # ------ read/search ------
    def topk(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        *,
        namespace: Optional[str] = None,
    ) -> List[Tuple[StoredChunk, float]]:
        # Pinecone path: return (chunk, score) by querying chunk_id vectors.
        if self.vector_backend == "pinecone":
            ns = _require_namespace(namespace or self._namespace, "topk")
            pc = self._pinecone_client()
            matches = pc.query(
                index=pc.chunks,
                namespace=ns,
                query_vec=query_vec,
                top_k=int(k),
                filter=None,
            )
            if not matches:
                return []
            chunk_ids = [cid for cid, _ in matches]
            by_id = self._repo.get_chunks_by_ids(chunk_ids)
            out: List[Tuple[StoredChunk, float]] = []
            for cid, score in matches:
                ch = by_id.get(cid)
                if ch is None:
                    continue
                out.append((ch, float(score)))
            return out

        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
            
        D, I = self.index.search(query_vec, k)
            
        # Reconstruct chunks from SQL using the index from FAISS
        out: List[Tuple[StoredChunk, float]] = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0: continue
            
            chunk = self._repo.get_chunk_by_vector_index(idx)
            if not chunk: continue

            out.append((chunk, float(score)))
        return out

    def get_vectors_for_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Fetch vectors (in embedding space) for the given chunk_ids.
        Returns (resolved_chunk_ids_in_order, vectors[N, dim]).

        Notes:
        - Requires vector_index_map entries; for older stores, missing chunk_ids are skipped.
        """
        # Pinecone path: fetch vectors by chunk_id from the exam-scoped namespace.
        if self.vector_backend == "pinecone":
            ns = _require_namespace(namespace or self._namespace, "get_vectors_for_chunk_ids")
            ids = [c for c in chunk_ids if c]
            if not ids:
                return [], np.zeros((0, self.vector_dimension), dtype="float32")
            pc = self._pinecone_client()
            fetched = pc.fetch_vectors(index=pc.chunks, namespace=ns, ids=ids)
            resolved_ids = [cid for cid in ids if cid in fetched]
            if not resolved_ids:
                return [], np.zeros((0, self.vector_dimension), dtype="float32")
            X = np.stack([fetched[cid] for cid in resolved_ids]).astype("float32", copy=False)
            return resolved_ids, X

        ids = [c for c in chunk_ids if c]
        if not ids:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        mapping = self._repo.list_vector_indices_by_chunk_ids(ids)
        resolved: List[Tuple[str, int]] = [(cid, mapping[cid]) for cid in ids if cid in mapping]
        if not resolved:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        resolved_chunk_ids = [cid for cid, _ in resolved]
        indices = [idx for _, idx in resolved]

        # Numpy fallback index stores vectors directly
        vec_mat = getattr(self.index, "vectors", None)
        if vec_mat is None:
            return resolved_chunk_ids, np.zeros((0, self.vector_dimension), dtype="float32")
        # Guard against out-of-range indices (shouldn't happen if mapping is correct).
        max_n = int(vec_mat.shape[0])
        safe_pairs: List[Tuple[str, int]] = [(cid, idx) for cid, idx in resolved if 0 <= idx < max_n]
        if not safe_pairs:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        resolved_chunk_ids = [cid for cid, _ in safe_pairs]
        safe_indices = [idx for _, idx in safe_pairs]
        return resolved_chunk_ids, np.array(vec_mat[safe_indices], dtype="float32", copy=False)

    # ------ doc helpers ------
    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        return self._repo.list_chunks_by_doc(doc_id)

    def sample_chunks_by_doc(self, doc_id: str, n: int = 20) -> List[StoredChunk]:
        import random
        chunks = self.list_chunks_by_doc(doc_id)
        if len(chunks) <= n:
            return chunks
        idxs = list(range(len(chunks)))
        random.shuffle(idxs)
        return [chunks[i] for i in idxs[:n]]

    # ------ cache helpers ------
    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        return self._repo.get_cached_embeddings(hashes)

    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        return self._repo.add_cached_embeddings(mapping)
