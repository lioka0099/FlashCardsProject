from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from app.llm import embed_query, embed_texts, generate_alternate_queries, rerank_chunks
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

    def search_smart(self, query: str, k: int | None = None) -> List[Retrieval]:
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
            alternates += generate_alternate_queries(query, num_variations=3)
        except Exception:
            # If chat model unavailable, proceed with single query
            pass

        # 2) retrieve per alternate and merge
        per_chunk: Dict[str, Tuple[StoredChunk, float]] = {}
        for q in alternates:
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
            q_vec = embed_query(query).astype("float32")
            q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True).clip(1e-12)
            sim_to_q = (cand_emb @ q_vec[0])

            # MMR selection
            lambda_param = 0.5
            mmr_size = min(len(candidates), max(k_final * 3, 15))
            selected: List[int] = []
            remaining = set(range(len(candidates)))
            if remaining:
                # seed with best sim_to_q
                first = int(np.argmax(sim_to_q))
                selected.append(first)
                remaining.remove(first)
            while remaining and len(selected) < mmr_size:
                best_idx, best_score = None, -1e9
                for i in list(remaining):
                    # relevance
                    relevance = sim_to_q[i]
                    # diversity: max similarity to already selected
                    if selected:
                        div = np.max(cand_emb[i] @ cand_emb[selected].T)
                    else:
                        div = 0.0
                    score = lambda_param * relevance - (1.0 - lambda_param) * div
                    if score > best_score:
                        best_idx, best_score = i, score
                selected.append(best_idx)  # type: ignore
                remaining.remove(best_idx)  # type: ignore
            mmr_indices = selected
        except Exception:
            # If embeddings fail, just take top by vector score
            order = np.argsort(-vec_scores)
            mmr_indices = order[: max(k_final * 3, 15)].tolist()

        # 4) LLM rerank the diversified pool
        diversified = [(candidates[i], vec_scores[i]) for i in mmr_indices]
        cand_pairs = [(c.chunk_id, c.text) for c, _ in diversified]
        try:
            reranked = rerank_chunks(query, cand_pairs, top_n=max(k_final * 2, k_final))
            # Map rerank result to final list
            final: List[Tuple[StoredChunk, float]] = []
            for idx, llm_score in reranked:
                if 0 <= idx < len(diversified):
                    ch, vec_s = diversified[idx]
                    # blend scores
                    # vec_s expected ~ [-1,1] or [0,1]; clip and map to [0,1]
                    vs = float(np.clip(vec_s, 0.0, 1.0))
                    final_score = 0.6 * float(llm_score) + 0.4 * vs
                    final.append((ch, final_score))
            # If reranker returns fewer than needed, backfill
            if len(final) < k_final:
                remaining_idxs = [i for i in range(len(diversified)) if i not in {r[0] for r in reranked}]
                for i in remaining_idxs:
                    ch, vec_s = diversified[i]
                    vs = float(np.clip(vec_s, 0.0, 1.0))
                    final.append((ch, vs * 0.5))  # lower confidence
        except Exception:
            # If reranker unavailable, fallback to vector score of diversified pool
            final = [(ch, float(np.clip(score, 0.0, 1.0))) for ch, score in diversified]

        # sort and return top-k
        final.sort(key=lambda x: x[1], reverse=True)
        final = final[:k_final]
        return [Retrieval(chunk=ch, score=sc) for ch, sc in final]
