"""Document ingestion: load, chunk, embed and persist source files into the corpus."""

import hashlib
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from app.data.vector_store import StoredChunk, VectorStore
from app.data.pinecone_backend import pinecone_namespace
from app.deps import get_repo, get_store
from app.services.exams import ensure_exam_ingest_allowed
from app.services.llm import EMBED_MODEL, embed_texts
from app.utils.chunking import Chunk, make_chunks
from app.utils.loaders.docx_loader import load_docx
from app.utils.loaders.pdf_loader import load_pdf
from app.utils.loaders.txt_loader import load_txt

logger = logging.getLogger(__name__)

_CHUNK_TARGET_CHARS = 1400
_CHUNK_OVERLAP = 200
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedDocumentTypeError(ValueError):
    """Raised when attempting to ingest a file with an unsupported extension."""


@dataclass
class IngestResult:
    """Result of ingesting a single document into the store (service-layer type)."""

    doc_id: str
    num_chunks: int


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_loader(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf
    if ext == ".docx":
        return load_docx
    if ext == ".txt":
        return load_txt
    supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
    raise UnsupportedDocumentTypeError(
        f"Unsupported document type for '{Path(path).name}': '{ext or '(no extension)'}'. "
        f"Supported types: {supported}."
    )


def _persist_source_file(
    src_path: str,
    doc_id: str,
    dest_dir: Optional[Path] = None,
) -> str:
    """Copy the uploaded file to a doc_id-keyed path so it survives the
    worker's temp-file cleanup. Returns the absolute destination path."""
    if dest_dir is None:
        dest_dir = Path(os.getenv("UPLOAD_DIR", "uploads")) / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_id}{Path(src_path).suffix.lower()}"
    shutil.copyfile(src_path, dest)
    return str(dest.resolve())


def _ingest_single(
    path: str,
    store: VectorStore,
    *,
    user_id: Optional[str] = None,
    exam_id: Optional[str] = None,
) -> IngestResult:
    logger.info("Starting ingestion for %s", path)
    loader = _detect_loader(path)
    logger.debug("Selected loader %s", loader.__name__)

    pages = loader(path)
    logger.info("Loaded %s with %d pages", Path(path).name, len(pages))

    doc_id = uuid.uuid4().hex[:12]
    persisted_path = _persist_source_file(path, doc_id)
    store.add_document(
        doc_id,
        path=persisted_path,
        title=Path(path).name,
        info={"pages": len(pages)},
    )
    logger.info("Registered document %s in metadata store", doc_id)

    # Chunk (slightly larger chunks + overlap improve coherence)
    chunks: List[Chunk] = make_chunks(
        doc_id,
        pages,
        target_chars=_CHUNK_TARGET_CHARS,
        overlap=_CHUNK_OVERLAP,
    )
    logger.info(
        "Created %d chunks (target_chars=%d, overlap=%d)",
        len(chunks),
        _CHUNK_TARGET_CHARS,
        _CHUNK_OVERLAP,
    )

    # Embeddings (with cache)
    texts = [c["text"] for c in chunks]
    hashes = [_hash_text(text) for text in texts]
    cached = store.get_cached_embeddings(hashes)
    logger.info(
        "Embedding %d chunks with model %s (cache hits=%d, misses=%d)",
        len(chunks),
        EMBED_MODEL,
        len(cached),
        len(hashes) - len(cached),
    )

    missing_indices: List[int] = [i for i, h in enumerate(hashes) if h not in cached]
    missing_texts = [texts[i] for i in missing_indices]

    if missing_texts:
        new_vectors = embed_texts(missing_texts)
        for idx, vec in zip(missing_indices, new_vectors):
            cached[hashes[idx]] = vec.astype("float32", copy=False)
        store.add_cached_embeddings({hashes[idx]: cached[hashes[idx]] for idx in missing_indices})
        logger.info("Cached %d new embeddings", len(missing_indices))
    elif not hashes:
        logger.info("No chunks to embed for %s", path)

    if hashes:
        vectors = np.stack([cached[h] for h in hashes]).astype("float32", copy=False)
    else:
        vectors = np.zeros((0, store.vector_dimension), dtype="float32")

    # Persist chunks + vectors
    stored = [
        StoredChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            page=c["page"],
            start=c["start"],
            end=c["end"],
            text=c["text"],
        )
        for c in chunks
    ]
    # If using Pinecone backend, ingestion must be exam-scoped so vectors go into the exam namespace.
    if store.vector_backend == "pinecone":
        if not user_id or not exam_id:
            raise RuntimeError(
                "Pinecone backend requires exam-scoped ingestion: provide user_id and exam_id."
            )
        store = store.for_namespace(pinecone_namespace(user_id=user_id, exam_id=exam_id))
    store.add_chunks(stored, vectors)
    logger.info("Finished ingestion for %s (doc_id=%s)", path, doc_id)

    return IngestResult(doc_id=doc_id, num_chunks=len(chunks))


def ingest_documents(
    paths: Sequence[str],
    store: Optional[VectorStore] = None,
    *,
    user_id: Optional[str] = None,
    exam_id: Optional[str] = None,
) -> List[IngestResult]:
    if not paths:
        logger.info("No paths provided for ingestion; skipping")
        return []
    store = store or get_store()
    if exam_id:
        ensure_exam_ingest_allowed(repo=get_repo(), exam_id=exam_id)
    results: List[IngestResult] = []
    for path in paths:
        results.append(_ingest_single(path, store, user_id=user_id, exam_id=exam_id))
    logger.info("Ingested %d document(s)", len(results))
    return results

