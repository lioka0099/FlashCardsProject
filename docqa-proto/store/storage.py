from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import json, sqlite3, numpy as np
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

VEC_DIM = 1536  # OpenAI text-embedding-3-small

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

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str

class VectorStore:
    def __init__(self, basepath: str = "store"):
        self.base = Path(basepath)
        self.base.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base / "meta.sqlite"
        self.index_path = self.base / "faiss.index"
        self.vectors_path = self.base / "vectors.npy"

        self.conn = sqlite3.connect(self.db_path)
        self._ensure_schema()

        if _FAISS_AVAILABLE:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))  # type: ignore
            else:
                self.index = faiss.IndexFlatIP(VEC_DIM)  # type: ignore  # cosine via normalized dot
        else:
            self.index = _NumpyIPIndex(VEC_DIM, self.vectors_path)

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY,
            path TEXT,
            title TEXT,
            info TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            start INTEGER,
            end INTEGER,
            text TEXT
        );
        """)
        self.conn.commit()

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id, path, title, info) VALUES(?,?,?,?)",
            (doc_id, path, title, json.dumps(info, ensure_ascii=False)))
        self.conn.commit()

    def add_chunks(self, chunk_rows: Iterable[StoredChunk], vectors: np.ndarray):
        rows = [(c.chunk_id, c.doc_id, c.page, c.start, c.end, c.text) for c in chunk_rows]
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page, start, end, text) VALUES(?,?,?,?,?,?)", rows)
        self.conn.commit()

        # normalize for cosine similarity
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(vectors)  # type: ignore
            self.index.add(vectors)  # type: ignore
            faiss.write_index(self.index, str(self.index_path))  # type: ignore
        else:
            # fallback index keeps its own storage
            self.index.add(vectors)

    # ------ read/search ------
    def topk(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[StoredChunk, float]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(query_vec)  # type: ignore
            D, I = self.index.search(query_vec, k)  # type: ignore
        else:
            D, I = self.index.search(query_vec, k)
        # FAISS doesn't store metadata; we align by row order using rowid from chunks table.
        # Easiest is to store an external table mapping row order; here we rely on implicit order.
        # So instead, we’ll rebuild I -> chunk by rowid using LIMIT/OFFSET (works for proto).
        out: List[Tuple[StoredChunk, float]] = []
        cur = self.conn.cursor()
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0: continue
            cur.execute("SELECT chunk_id, doc_id, page, start, end, text FROM chunks LIMIT 1 OFFSET ?", (idx,))
            row = cur.fetchone()
            if not row: continue
            c = StoredChunk(*row)
            out.append((c, float(score)))
        return out
