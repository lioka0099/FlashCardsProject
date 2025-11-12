import uuid
from pathlib import Path
from typing import List, Optional
from ingest.pdf_loader import load_pdf
from ingest.docx_loader import load_docx
from ingest.txt_loader import load_txt
from ingest.chunker import make_chunks, Chunk
from app.llm import embed_texts
from store.storage import VectorStore, StoredChunk
from app.models import IngestResult, ProofSpan

def _detect_loader(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf": return load_pdf
    if ext in [".docx"]: return load_docx
    return load_txt

def ingest_document(path: str, store: Optional[VectorStore] = None) -> IngestResult:
    store = store or VectorStore()
    pages = _detect_loader(path)(path)
    doc_id = uuid.uuid4().hex[:12]
    store.add_document(doc_id, path=str(Path(path).resolve()),
                       title=Path(path).name, info={"pages": len(pages)})

    # chunk
    # Slightly larger chunks and overlap improve coherence and recall
    chunks: List[Chunk] = make_chunks(doc_id, pages, target_chars=900, overlap=200)

    # embed
    vectors = embed_texts([c["text"] for c in chunks])

    # persist
    stored = [
        StoredChunk(
            chunk_id=c["chunk_id"], doc_id=c["doc_id"], page=c["page"],
            start=c["start"], end=c["end"], text=c["text"]
        )
        for c in chunks
    ]
    store.add_chunks(stored, vectors)
    return IngestResult(doc_id=doc_id, num_chunks=len(chunks))

def retrieve_with_proofs(question: str, k: int = 5, store: Optional[VectorStore] = None) -> List[ProofSpan]:
    from app.retriever import Retriever
    store = store or VectorStore()
    # Use the smarter pipeline by default
    hits = Retriever(store, k=k).search_smart(question, k=k)
    return [ProofSpan(
        doc_id=h.chunk.doc_id, page=h.chunk.page, start=h.chunk.start,
        end=h.chunk.end, text=h.chunk.text[:500], score=h.score
    ) for h in hits]

def retrieve_with_proofs_for_doc(question: str, doc_id: str, k: int = 5, store: Optional[VectorStore] = None) -> List[ProofSpan]:
    from app.retriever import Retriever
    store = store or VectorStore()
    hits = Retriever(store, k=k).search_smart(question, k=k)
    hits = [h for h in hits if h.chunk.doc_id == doc_id]
    return [ProofSpan(
        doc_id=h.chunk.doc_id, page=h.chunk.page, start=h.chunk.start,
        end=h.chunk.end, text=h.chunk.text[:500], score=h.score
    ) for h in hits]
