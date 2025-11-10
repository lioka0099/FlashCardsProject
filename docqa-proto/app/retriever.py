from dataclasses import dataclass
from typing import List
from app.llm import embed_query
from store.storage import VectorStore, StoredChunk

@dataclass
class Retrieval:
    chunk: StoredChunk
    score: float

class Retriever:
    def __init__(self, store: VectorStore, k: int = 5):
        self.store = store
        self.k = k

    def search(self, query: str) -> List[Retrieval]:
        qvec = embed_query(query)
        hits = self.store.topk(qvec, self.k)
        return [Retrieval(chunk=h[0], score=h[1]) for h in hits]
