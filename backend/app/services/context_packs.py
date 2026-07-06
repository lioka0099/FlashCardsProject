from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.data.db_repository import DBRepository, StoredChunk
from app.data.vector_store import VectorStore
from app.deps import get_repo
from app.utils.vectors import l2_normalize


def build_representative_chunk_pack(
    *,
    store: VectorStore,
    repo: Optional[DBRepository] = None,
    chunk_ids: Sequence[str],
    centroid: Optional[np.ndarray] = None,
    # Optional fast-path: reuse precomputed normalized embeddings from caller.
    Xn: Optional[np.ndarray] = None,
    id_to_row: Optional[Dict[str, int]] = None,
    chunk_by_id: Optional[Dict[str, StoredChunk]] = None,
    max_chunks: int = 8,
    max_chars_per_chunk: int = 650,
    max_total_chars: int = 6000,
) -> str:
    """
    Build a compact context pack by selecting the most representative chunks.

    Representative = highest cosine similarity to the provided centroid, or to the
    centroid computed from the chunk vectors if none provided.

    This function supports two modes:
    - **Fast-path**: caller provides Xn (normalized vectors) and id_to_row mapping.
    - **Store-path**: reconstruct vectors for chunk_ids from the store.
    """
    repo = repo or get_repo()
    ids = [c for c in chunk_ids if c]
    if not ids:
        return ""

    # 1) Get vectors aligned with ids
    resolved_ids: List[str]
    V: np.ndarray

    if Xn is not None and id_to_row is not None:
        rows: List[int] = []
        kept: List[str] = []
        for cid in ids:
            r = id_to_row.get(cid)
            if r is None:
                continue
            rows.append(int(r))
            kept.append(cid)
        if not rows:
            return ""
        resolved_ids = kept
        V = Xn[rows]
    else:
        resolved_ids, X = store.get_vectors_for_chunk_ids(ids)
        if X.size == 0 or not resolved_ids:
            return ""
        V = l2_normalize(X)

    # 2) Determine centroid vector
    if centroid is None:
        centroid_vec = l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
    else:
        centroid_vec = l2_normalize(centroid.astype("float32", copy=False))[0]

    # 3) Rank chunks by similarity to centroid
    sims = (V @ centroid_vec).tolist()
    ranked = sorted(zip(resolved_ids, sims), key=lambda t: t[1], reverse=True)

    # 4) Build text pack
    buf: List[str] = []
    used = 0
    picked = 0
    for cid, sim in ranked:
        ch = chunk_by_id.get(cid) if chunk_by_id is not None else repo.get_chunk_by_id(cid)
        if not ch or not ch.text:
            continue
        text = ch.text.strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3].rstrip() + "..."
        block = f"[doc={ch.doc_id} page={ch.page} sim={sim:.2f}] {text}\n"
        if used + len(block) > max_total_chars:
            break
        buf.append(block)
        used += len(block)
        picked += 1
        if picked >= max_chunks:
            break
    return "".join(buf).strip()


def build_diverse_chunk_pack(
    *,
    store: VectorStore,
    repo: Optional[DBRepository] = None,
    chunk_ids: Sequence[str],
    centroid: Optional[np.ndarray] = None,
    max_chunks: int = 8,
    max_chars_per_chunk: int = 650,
    max_total_chars: int = 6000,
    lambda_param: float = 0.5,
) -> str:
    """
    Build context pack using MMR-style selection for diversity.

    Selects chunks that are:
    - Relevant (similar to topic centroid)
    - Diverse (dissimilar to already-selected chunks)

    Used for question generation to ensure broad topic coverage.

    Args:
        store: VectorStore instance
        chunk_ids: Pool of chunk IDs to select from
        centroid: Topic centroid vector (computed from chunks if None)
        max_chunks: Maximum number of chunks to select
        max_chars_per_chunk: Truncate chunks longer than this
        max_total_chars: Stop when total chars exceeds this
        lambda_param: Balance relevance vs diversity (0.5 = equal weight)

    Returns:
        Formatted context string with diverse excerpts.
    """
    repo = repo or get_repo()
    ids = [c for c in chunk_ids if c]
    if not ids:
        return ""

    # 1) Get vectors for all candidate chunks
    resolved_ids, X = store.get_vectors_for_chunk_ids(ids)
    if X.size == 0 or not resolved_ids:
        return ""
    V = l2_normalize(X)

    # 2) Determine centroid vector
    if centroid is None:
        centroid_vec = l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
    else:
        centroid_vec = l2_normalize(centroid.astype("float32", copy=False))[0]

    # 3) Compute relevance scores (similarity to centroid)
    relevance = V @ centroid_vec  # shape: (n_chunks,)

    # 4) MMR selection: balance relevance and diversity
    n = len(resolved_ids)
    selected_indices: List[int] = []
    remaining = set(range(n))

    # Seed with most relevant chunk
    if remaining:
        first = int(np.argmax(relevance))
        selected_indices.append(first)
        remaining.remove(first)

    # Greedily select remaining chunks
    while remaining and len(selected_indices) < max_chunks:
        best_idx = -1
        best_score = -1e9

        for i in remaining:
            rel = relevance[i]
            # Diversity: max similarity to already-selected chunks
            if selected_indices:
                sel_vecs = V[selected_indices]
                div = float(np.max(V[i] @ sel_vecs.T))
            else:
                div = 0.0
            # MMR score: relevance - penalty for similarity to selected
            score = lambda_param * rel - (1.0 - lambda_param) * div
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    # 5) Build text pack from selected chunks (in selection order)
    buf: List[str] = []
    used = 0
    for idx in selected_indices:
        cid = resolved_ids[idx]
        ch = repo.get_chunk_by_id(cid)
        if not ch or not ch.text:
            continue
        text = ch.text.strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3].rstrip() + "..."
        sim = float(relevance[idx])
        block = f"[doc={ch.doc_id} page={ch.page} sim={sim:.2f}] {text}\n"
        if used + len(block) > max_total_chars:
            break
        buf.append(block)
        used += len(block)

    return "".join(buf).strip()
