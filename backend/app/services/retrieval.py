from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create, embed_query, embed_texts
from app.data.vector_store import VectorStore, StoredChunk
from app.deps import get_store
from app.api.schemas import ProofSpan

@dataclass
class Retrieval:
    chunk: StoredChunk
    score: float

def generate_alternate_queries(
    question: str,
    num_variations: int = 3,
    *,
    model: Optional[str] = None,
) -> List[str]:
    """
    Use the chat model to produce semantically diverse reformulations of the query
    to improve recall (multi-query retrieval).
    """
    prompt = (
        "Rewrite the user's question into diverse, concise search queries that capture different phrasings.\n"
        f"Original question: {question}\n"
        f"Return exactly {num_variations} queries, one per line, no numbering."
    )
    resp = chat_completions_create(
        model=model or CHAT_MODEL_FAST,
        messages=[
            {"role": "system", "content": "You are an expert search assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=120,
    )
    text = resp.choices[0].message.content or ""
    queries = [q.strip() for q in text.splitlines() if q.strip()]
    if len(queries) < num_variations:
        # pad with original question if the model returned fewer
        queries += [question] * (num_variations - len(queries))
    return queries[:num_variations]

class Retriever:
    def __init__(self, store: VectorStore, k: int = 5):
        self.store = store
        self.k = k

    def search(self, query: str, *, query_vec: Optional[np.ndarray] = None) -> List[Retrieval]:
        qvec = query_vec if query_vec is not None else embed_query(query)
        hits = self.store.topk(qvec, self.k)
        return [Retrieval(chunk=h[0], score=h[1]) for h in hits]

    def search_smart(self, query: str, k: Optional[int] = None, *, query_vec: Optional[np.ndarray] = None) -> List[Retrieval]:
        """
        Multi-query + diversification + LLM reranking pipeline:
        1) Generate alternate queries to improve recall
        2) Retrieve a larger candidate pool per query and merge/dedupe
        3) Diversify with a simple MMR over embeddings
        4) Rerank with LLM and return top-k
        """
        k_final = k if k is not None else self.k
        # Larger pool to recall more context for reranking
        k_pool = max(k_final * 5, 25)

        # 1) alternate queries
        alternates = [query]
        try:
            alternates += generate_alternate_queries(query, num_variations=2)
        except Exception:
            # If chat model unavailable, proceed with single query
            pass

        # 2) retrieve per alternate and merge
        per_chunk: Dict[str, Tuple[StoredChunk, float]] = {}
        for q in alternates:
            if query_vec is not None and q == query:
                qvec = query_vec
            else:
                qvec = embed_query(q)
            hits = self.store.topk(qvec, k_pool)
            for ch, score in hits:
                # keep the best vector score per chunk_id
                prev = per_chunk.get(ch.chunk_id)
                if (prev is None) or (score > prev[1]):
                    per_chunk[ch.chunk_id] = (ch, float(score))

        if not per_chunk:
            return []

        # make arrays
        candidates: List[StoredChunk] = [t[0] for t in per_chunk.values()]
        vec_scores: np.ndarray = np.array([t[1] for t in per_chunk.values()], dtype="float32")
        texts = [c.text[:1000] for c in candidates]

        # 3) diversify with MMR on embeddings
        try:
            cand_emb = embed_texts(texts)
            # L2 normalize
            norms = np.linalg.norm(cand_emb, axis=1, keepdims=True).clip(1e-12)
            cand_emb = cand_emb / norms
            q_vec = (query_vec if query_vec is not None else embed_query(query)).astype("float32")
            q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True).clip(1e-12)
            sim_to_q = (cand_emb @ q_vec[0])

            # MMR selection
            lambda_param = 0.5
            mmr_size = min(len(candidates), max(k_final * 3, 15)) #Max number of candidates to select
            selected: List[int] = []
            remaining = set(range(len(candidates)))
            if remaining:
                # seed with best sim_to_q
                first = int(np.argmax(sim_to_q)) #Index of the candidate with the highest cosine similarity to the query
                selected.append(first)
                remaining.remove(first)
            while remaining and len(selected) < mmr_size:
                best_idx, best_score = None, -1e9
                for i in list(remaining):
                    # relevance
                    relevance = sim_to_q[i] #Cosine similarity between the candidate and the query 
                    # diversity: max similarity to already selected
                    if selected:
                        div = np.max(cand_emb[i] @ cand_emb[selected].T) #Cosine similarity between the candidate and the already selected candidates
                    else:
                        div = 0.0
                    score = lambda_param * relevance - (1.0 - lambda_param) * div #Score - still relevant and not too similar to what we already have.
                    if score > best_score:
                        best_idx, best_score = i, score
                selected.append(best_idx)  # type: ignore
                remaining.remove(best_idx)  # type: ignore
            mmr_indices = selected #Indices of the candidates selected by MMR
        except Exception:
            # If embeddings fail, just take top by vector score
            order = np.argsort(-vec_scores)
            mmr_indices = order[: max(k_final * 3, 15)].tolist()

        # 4) Score diversified pool without extra LLM calls
        diversified = []
        for i in mmr_indices:
            chunk = candidates[i]
            vec_s = float(np.clip(vec_scores[i], 0.0, 1.0)) #Vector score of the candidate
            rel = float(sim_to_q[i]) if "sim_to_q" in locals() else vec_s #Cosine similarity between the candidate and the query
            score = 0.7 * rel + 0.3 * vec_s #Score - ordering the selected candidates before returning them
            diversified.append((chunk, score))

        diversified.sort(key=lambda x: x[1], reverse=True)
        diversified = diversified[:k_final]
        return [Retrieval(chunk=ch, score=sc) for ch, sc in diversified]

def retrieve_with_proofs(
    question: str,
    k: int = 8,
    store: Optional[VectorStore] = None,
    *,
    allowed_doc_ids: Optional[Sequence[str]] = None,
    allowed_chunk_ids: Optional[Sequence[str]] = None,
    query_vec: Optional[np.ndarray] = None,
) -> List[ProofSpan]:
    store = store or get_store()
    allowed_chunks = {c for c in (allowed_chunk_ids or []) if c} if allowed_chunk_ids else None
    # If we need to filter by chunk_id, ask for a bigger pool first to avoid starving after filtering.
    pool_k = k
    if allowed_chunks is not None:
        pool_k = min(max(k * 10, 50), 300)

    hits = Retriever(store, k=pool_k).search_smart(question, k=pool_k, query_vec=query_vec)
    allowed = {d for d in allowed_doc_ids if d} if allowed_doc_ids else None
    if allowed:
        hits = [h for h in hits if h.chunk.doc_id in allowed]
    if allowed_chunks is not None:
        hits = [h for h in hits if h.chunk.chunk_id in allowed_chunks]
        # If still short, try one more time with a bigger pool (still capped).
        if len(hits) < k and pool_k < 600:
            pool_k2 = min(max(pool_k * 2, 100), 600)
            hits2 = Retriever(store, k=pool_k2).search_smart(question, k=pool_k2, query_vec=query_vec)
            if allowed:
                hits2 = [h for h in hits2 if h.chunk.doc_id in allowed]
            hits2 = [h for h in hits2 if h.chunk.chunk_id in allowed_chunks]
            hits = hits2

    hits = hits[:k]
    return [ProofSpan(
        doc_id=h.chunk.doc_id, page=h.chunk.page, start=h.chunk.start,
        end=h.chunk.end, text=h.chunk.text, score=h.score
    ) for h in hits]
