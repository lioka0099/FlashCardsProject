"""Documents, chunks, vector-index map, embedding cache."""
import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence

from app.data.db_engine import get_db
from app.data.models import (
    Document, Chunk, EmbeddingCache, VectorIndexMap,
)
from app.data.db_repository.dtos import (
    StoredChunk,
)


class CorpusRepo:
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        """Add or update a document."""
        with get_db() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.path = path
                doc.title = title
                doc.info = info
            else:
                doc = Document(doc_id=doc_id, path=path, title=title, info=info)
                db.add(doc)

    def get_document_path(self, *, doc_id: str) -> Optional[str]:
        """Get absolute document path by document ID."""
        with get_db() as db:
            row = db.query(Document).filter(Document.doc_id == doc_id).first()
            if row is None:
                return None
            return row.path
    
    def add_chunks(self, chunk_rows: Iterable[StoredChunk]):
        """Add or update chunks."""
        with get_db() as db:
            for c in chunk_rows:
                chunk = db.query(Chunk).filter(Chunk.chunk_id == c.chunk_id).first()
                if chunk:
                    chunk.doc_id = c.doc_id
                    chunk.page = c.page
                    chunk.start = c.start
                    chunk.end = c.end
                    chunk.text = c.text
                else:
                    chunk = Chunk(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        page=c.page,
                        start=c.start,
                        end=c.end,
                        text=c.text,
                    )
                    db.add(chunk)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Get a chunk by its ID."""
        with get_db() as db:
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            return StoredChunk.from_orm(chunk)

    def get_chunk_by_vector_index(self, vector_index: int) -> Optional[StoredChunk]:
        """Get chunk by vector index using the mapping table."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.vector_index == vector_index
            ).first()
            if not mapping:
                return None
            # Extract chunk_id while session is still open
            chunk_id = mapping.chunk_id
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            # Extract all values while session is still open
            return StoredChunk.from_orm(chunk)
    
    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        """List all chunks for a document."""
        with get_db() as db:
            chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(
                Chunk.page, Chunk.start
            ).all()
            return [
                StoredChunk.from_orm(c)
                for c in chunks
            ]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> Dict[str, StoredChunk]:
        """Fetch chunks by IDs in one query (used by Pinecone-backed retrieval)."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            rows = db.query(Chunk).filter(Chunk.chunk_id.in_(ids)).all()
            out: Dict[str, StoredChunk] = {}
            for chunk in rows:
                out[chunk.chunk_id] = StoredChunk.from_orm(chunk)
            return out
    def add_vector_index_mapping(self, *, start_index: int, chunk_ids: Sequence[str]) -> None:
        """Add mapping from vector indices to chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return
        with get_db() as db:
            for i, cid in enumerate(ids):
                idx = start_index + i
                existing = db.query(VectorIndexMap).filter(
                    VectorIndexMap.vector_index == idx
                ).first()
                if existing:
                    existing.chunk_id = cid
                else:
                    db.add(VectorIndexMap(vector_index=idx, chunk_id=cid))
    
    def get_vector_index_by_chunk_id(self, chunk_id: str) -> Optional[int]:
        """Get vector index for a chunk ID."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id == chunk_id
            ).first()
            return mapping.vector_index if mapping else None
    
    def list_vector_indices_by_chunk_ids(self, chunk_ids: Sequence[str]) -> Dict[str, int]:
        """Get vector indices for multiple chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            mappings = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id.in_(ids)
            ).all()
            # Extract values while session is still open
            return {m.chunk_id: m.vector_index for m in mappings}
    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        """Get cached embeddings by hash."""
        if not hashes:
            return {}
        with get_db() as db:
            cached = db.query(EmbeddingCache).filter(
                EmbeddingCache.hash.in_(hashes)
            ).all()
            result: Dict[str, np.ndarray] = {}
            for c in cached:
                try:
                    arr = np.frombuffer(c.vector, dtype="float32")
                    if arr.size == c.dim:
                        result[c.hash] = arr
                except Exception:
                    continue
            return result
    
    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        """Cache embeddings."""
        if not mapping:
            return
        with get_db() as db:
            for key, vec in mapping.items():
                arr = vec.astype("float32", copy=False)
                existing = db.query(EmbeddingCache).filter(EmbeddingCache.hash == key).first()
                if existing:
                    existing.dim = arr.size
                    existing.vector = arr.tobytes()
                else:
                    db.add(EmbeddingCache(
                        hash=key,
                        dim=arr.size,
                        vector=arr.tobytes(),
                    ))
