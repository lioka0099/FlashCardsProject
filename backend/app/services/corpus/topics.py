"""Topic building: cluster a corpus into grounded topics with evidence and centroid vectors."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from app.data.db_repository import DBRepository, StoredChunk, StoredTopic
from app.data.vector_store import VectorStore
from app.deps import get_repo, get_store
from app.services.exams import load_exam
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create
from app.utils.vectors import l2_normalize
from app.services.generation.context_packs import build_representative_chunk_pack
from app.services.generation.card_routing import classify_card_route

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those",
    "it", "its", "they", "their", "we", "our", "you", "your",
    "from", "at", "into", "over", "under", "between", "within",
    "not", "no", "yes",
}


@dataclass
class BuiltTopic:
    topic_id: str
    label: str
    chunk_ids: List[str]
    evidence: List[Dict[str, Any]]
    info: Dict[str, Any]


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer: keep alphanumerics, greek letters commonly appear as unicode words too.
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) >= 3]


def _ngram_phrases(tokens: List[str], n: int) -> Iterable[str]:
    if n <= 0:
        return []
    for i in range(0, len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _choose_grounded_label(texts: Sequence[str], max_words: int = 3) -> str:
    """
    Pick a topic label that appears in the cluster text (grounded) by scoring frequent n-grams.
    This is intentionally non-LLM for determinism and to avoid hallucinated labels.
    """
    joined = "\n".join(t for t in texts if t)
    tokens = _tokenize(joined)
    if not tokens:
        return "Topic"

    scores: Dict[str, float] = {}
    # Prefer longer phrases slightly.
    for n in range(min(max_words, 3), 0, -1):
        for phrase in _ngram_phrases(tokens, n):
            # Downweight overly generic endings.
            if phrase.split()[-1] in _STOPWORDS:
                continue
            scores[phrase] = scores.get(phrase, 0.0) + (1.0 + 0.2 * (n - 1))

    if not scores:
        return "Topic"

    # Choose best scoring phrase that is actually present as a substring in cluster text (case-insensitive).
    joined_lower = joined.lower()
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for phrase, _ in ranked[:200]:
        if phrase in joined_lower:
            # Title-case lightly for presentation, but keep original words.
            return " ".join(w.capitalize() if len(w) > 3 else w for w in phrase.split())

    # Fallback: top token
    return tokens[0].capitalize()


def _extract_evidence_for_label(chunks: Sequence[StoredChunk], label: str, max_items: int = 3) -> List[Dict[str, Any]]:
    """
    Find up to max_items evidence spans where the label appears in the chunk text.
    Evidence is grounded because it's literally taken from document text.
    """
    if not label:
        return []
    label_l = label.lower()
    evidence: List[Dict[str, Any]] = []
    for ch in chunks:
        hay = (ch.text or "")
        idx = hay.lower().find(label_l)
        if idx < 0:
            continue
        # Capture a small window around the match for inspection.
        window = 220
        s = max(0, idx - window // 2)
        e = min(len(hay), idx + len(label) + window // 2)
        snippet = hay[s:e].strip()
        evidence.append(
            {
                "evidence_id": uuid.uuid4().hex[:20],
                "doc_id": ch.doc_id,
                "page": ch.page,
                "start": int(ch.start + idx),
                "end": int(ch.start + idx + len(label)),
                "text": snippet,
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def _extract_evidence_for_phrases(
    chunks: Sequence[StoredChunk],
    phrases: Sequence[str],
    *,
    max_items: int = 3,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Try multiple candidate phrases and return (evidence, matched_phrase).
    """
    cleaned = [p.strip() for p in phrases if isinstance(p, str) and p.strip()]
    if not cleaned:
        return [], None
    for phrase in cleaned:
        ev = _extract_evidence_for_label(chunks, phrase, max_items=max_items)
        if ev:
            return ev, phrase
    return [], None


def _llm_label_cluster(
    *,
    context_pack: str,
    model: str = CHAT_MODEL_FAST,
) -> Dict[str, Any]:
    """
    Ask a fast LLM to propose a conceptual topic label and evidence phrases.
    Returns dict with keys: label (str), evidence_phrases (list[str]), key_terms (list[str]).
    """
    sys_prompt = (
        "You extract study topics from document excerpts.\n"
        "Propose conceptual, specific topic labels.\n"
        "Return JSON only."
    )
    user_prompt = (
        "Given the following excerpts that belong to ONE semantic cluster, propose a single topic.\n\n"
        "EXCERPTS:\n"
        f"{context_pack}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  \"label\": \"<short topic label>\",\n'
        '  \"key_terms\": [\"<term>\", ...],\n'
        '  \"evidence_phrases\": [\"<short phrase that should appear verbatim in the excerpts>\", ...]\n'
        "}\n\n"
        "Rules:\n"
        "- label should be 3–10 words, specific (not generic like 'Students' or 'Questions').\n"
        "- Prefer academic phrasing if appropriate (e.g., 'Hypothesis-based evaluation (H1–H3)').\n"
        "- evidence_phrases must be phrases you believe appear verbatim in the excerpts; include abbreviations too.\n"
        "- Keep evidence_phrases short (2–6 words) and include 3–8 items.\n"
        "- Return valid JSON only."
    )
    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    label = str(data.get("label", "")).strip()
    key_terms = data.get("key_terms") if isinstance(data.get("key_terms"), list) else []
    evidence_phrases = data.get("evidence_phrases") if isinstance(data.get("evidence_phrases"), list) else []
    return {
        "label": label,
        "key_terms": [str(x).strip() for x in key_terms if str(x).strip()],
        "evidence_phrases": [str(x).strip() for x in evidence_phrases if str(x).strip()],
    }

# def _kmeans_cluster(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
#     """
#     Returns assignments array of shape (n,) with values in [0..k-1].
#     Uses a small numpy kmeans (cosine-ish via normalized vectors).
#     """
#     n, d = X.shape
#     if n == 0:
#         return np.zeros((0,), dtype="int64")
#     k = max(1, min(int(k), n))

#     # Numpy fallback (small datasets): classic kmeans on normalized vectors.
#     rng = np.random.RandomState(seed)
#     Xn = l2_normalize(X.astype("float32", copy=False))
#     # init: pick k random points
#     centroids = Xn[rng.choice(n, size=k, replace=False)].copy()
#     for _ in range(20):
#         sims = Xn @ centroids.T
#         assign = np.argmax(sims, axis=1)
#         new_centroids = centroids.copy()
#         for j in range(k):
#             mask = assign == j
#             if not np.any(mask):
#                 new_centroids[j] = Xn[rng.randint(0, n)]
#                 continue
#             new_centroids[j] = l2_normalize(np.mean(Xn[mask], axis=0, keepdims=True))[0]
#         if np.allclose(new_centroids, centroids, atol=1e-4):
#             break
#         centroids = new_centroids
#     sims = Xn @ centroids.T
#     return np.argmax(sims, axis=1).astype("int64")

def _kmeans_cluster(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Returns assignments array of shape (n,) with values in [0..k-1].

    Cosine-kmeans on L2-normalized vectors with:
    - kmeans++ initialization (cosine distance = 1 - cosine_sim)
    - multiple restarts (n_init) and best objective selection
    - deterministic via seed

    This is intended to get much closer to FAISS KMeans quality than random init.
    """
    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype="int64")
    k = max(1, min(int(k), n))

    Xn = l2_normalize(X.astype("float32", copy=False))
    rng = np.random.RandomState(seed)

    n_init = 8
    max_iter = 35
    best_assign = None
    best_score = -1e18  # maximize mean cosine similarity to assigned centroid

    def _kmeanspp_init() -> np.ndarray:
        # centroids: (k, d)
        centroids = np.zeros((k, d), dtype="float32")

        # pick first centroid uniformly
        first = rng.randint(0, n)
        centroids[0] = Xn[first]

        # track best similarity to any chosen centroid so far
        best_sim = (Xn @ centroids[0].reshape(-1, 1)).reshape(-1)

        for ci in range(1, k):
            # cosine distance in [0..2] (since cosine sim in [-1..1])
            dist = 1.0 - best_sim
            dist = np.clip(dist, 0.0, None)
            probs = dist * dist
            s = float(np.sum(probs))
            if not np.isfinite(s) or s <= 1e-12:
                idx = rng.randint(0, n)
            else:
                probs = probs / s
                idx = int(rng.choice(n, p=probs))
            centroids[ci] = Xn[idx]

            sim_new = (Xn @ centroids[ci].reshape(-1, 1)).reshape(-1)
            best_sim = np.maximum(best_sim, sim_new)

        return centroids

    for _run in range(n_init):
        centroids = _kmeanspp_init()

        for _ in range(max_iter):
            sims = Xn @ centroids.T                      # (n, k)
            assign = np.argmax(sims, axis=1).astype("int64")

            new_centroids = centroids.copy()
            for j in range(k):
                mask = assign == j
                if not np.any(mask):
                    # re-seed empty cluster
                    new_centroids[j] = Xn[rng.randint(0, n)]
                else:
                    new_centroids[j] = l2_normalize(np.mean(Xn[mask], axis=0, keepdims=True))[0]

            if np.allclose(new_centroids, centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        # objective: average similarity to assigned centroid
        final_sims = Xn @ centroids.T
        score = float(np.mean(final_sims[np.arange(n), assign]))
        if score > best_score:
            best_score = score
            best_assign = assign

    return best_assign if best_assign is not None else np.zeros((n,), dtype="int64")


def _pick_k(num_chunks: int) -> int:
    """
    Heuristic for number of topics. Conservative to avoid over-fragmentation.
    """
    if num_chunks <= 8:
        return 2
    k = int(round(np.sqrt(num_chunks)))
    return max(2, min(12, k))


def _pick_hdbscan_min_cluster_size(num_chunks: int) -> int:
    if num_chunks <= 10:
        return 2
    return max(4, min(24, int(round(0.04 * num_chunks))))


def _pick_hdbscan_min_samples(min_cluster_size: int) -> int:
    return max(2, int(min_cluster_size // 2))


def _assign_noise_to_nearest_cluster(Xn: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Ensure each sample has a valid non-negative label by assigning noise (-1)
    to the nearest existing cluster centroid in cosine space.
    """
    assigned = labels.astype("int64", copy=True)
    cluster_ids = sorted({int(x) for x in assigned.tolist() if int(x) >= 0})
    if not cluster_ids:
        assigned[:] = 0
        return assigned

    centroids: List[np.ndarray] = []
    for cid in cluster_ids:
        rows = np.where(assigned == cid)[0]
        if rows.size == 0:
            continue
        centroid = l2_normalize(np.mean(Xn[rows], axis=0, keepdims=True))[0]
        centroids.append(centroid)
    if not centroids:
        assigned[:] = 0
        return assigned

    C = np.stack(centroids, axis=0)  # (k, d)
    noise_rows = np.where(assigned < 0)[0]
    if noise_rows.size == 0:
        return assigned
    sims = Xn[noise_rows] @ C.T
    best = np.argmax(sims, axis=1)
    for ridx, b in zip(noise_rows.tolist(), best.tolist()):
        assigned[ridx] = cluster_ids[int(b)]
    return assigned


def _agglomerative_single_link_threshold(Xn: np.ndarray, threshold: float = 0.82) -> np.ndarray:
    """
    Single-link agglomerative clustering approximation:
    build graph edges where cosine similarity >= threshold and return connected components.
    """
    n = Xn.shape[0]
    if n == 0:
        return np.zeros((0,), dtype="int64")
    if n == 1:
        return np.zeros((1,), dtype="int64")

    parent = np.arange(n, dtype="int64")

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sims = Xn @ Xn.T
    for i in range(n):
        for j in range(i + 1, n):
            if float(sims[i, j]) >= float(threshold):
                union(i, j)

    root_to_label: Dict[int, int] = {}
    labels = np.zeros((n,), dtype="int64")
    next_label = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]
    return labels


def _cluster_topic_embeddings(
    *,
    X: np.ndarray,
    seed: int,
    algorithm: str = "hdbscan",
    use_umap: bool = False,
    umap_n_components: int = 15,
    umap_min_chunk_count: int = 300,
    hdbscan_min_cluster_size: Optional[int] = None,
    hdbscan_min_samples: Optional[int] = None,
    agglomerative_threshold: float = 0.82,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster chunk embeddings for topics using adaptive methods.
    Returns (assignments, metadata).
    """
    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype="int64"), {"method": "none"}
    if n == 1:
        return np.zeros((1,), dtype="int64"), {"method": "singleton"}

    Xn = l2_normalize(X.astype("float32", copy=False))
    algo = (algorithm or "hdbscan").strip().lower()

    # Final safety net: preserve legacy behavior if explicitly requested.
    if algo == "kmeans":
        k = min(_pick_k(n), n)
        assign = _kmeans_cluster(Xn, k=k, seed=seed)
        return assign, {"method": "kmeans_embeddings", "k": int(k), "seed": int(seed)}

    mcs = int(hdbscan_min_cluster_size or _pick_hdbscan_min_cluster_size(n))
    mcs = max(2, min(mcs, n))
    ms = int(hdbscan_min_samples or _pick_hdbscan_min_samples(mcs))
    ms = max(1, min(ms, n - 1))
    hdbscan_reject_reason: Optional[str] = None

    if hdbscan is not None and algo in {"hdbscan", "auto"}:
        try:
            fit_X = Xn
            umap_used = False
            if use_umap and umap is not None and n >= int(umap_min_chunk_count):
                n_components = max(2, min(int(umap_n_components), d))
                n_neighbors = max(5, min(30, int(round(np.sqrt(n)))))
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=seed,
                )
                fit_X = reducer.fit_transform(Xn)
                umap_used = True

            # On L2-normalized vectors, euclidean distance preserves cosine ordering.
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(fit_X).astype("int64")
            if np.all(labels < 0):
                raise RuntimeError("HDBSCAN labeled all points as noise")

            labels = _assign_noise_to_nearest_cluster(Xn, labels)
            n_clusters = int(len(set(labels.tolist())))
            min_reasonable_clusters = max(2, int(math.ceil(_pick_k(n) * 0.5)))
            if n >= 16:
                min_reasonable_clusters = max(3, min_reasonable_clusters)
            if n_clusters < min_reasonable_clusters:
                raise RuntimeError(
                    f"HDBSCAN under-segmented chunks into {n_clusters} clusters; "
                    f"minimum reasonable clusters is {min_reasonable_clusters}"
                )
            return labels, {
                "method": "hdbscan_embeddings",
                "seed": int(seed),
                "hdbscan_min_cluster_size": int(mcs),
                "hdbscan_min_samples": int(ms),
                "use_umap": bool(umap_used),
                "umap_n_components": int(umap_n_components) if umap_used else None,
                "n_clusters": n_clusters,
            }
        except Exception as exc:
            hdbscan_reject_reason = str(exc)

    if hdbscan_reject_reason and "under-segmented" in hdbscan_reject_reason:
        k = min(_pick_k(n), n)
        labels = _kmeans_cluster(Xn, k=k, seed=seed)
        return labels, {
            "method": "kmeans_embeddings",
            "k": int(k),
            "seed": int(seed),
            "fallback_reason": hdbscan_reject_reason,
        }

    # First fallback: non-fixed-k thresholded single-link clustering.
    try:
        labels = _agglomerative_single_link_threshold(Xn, threshold=float(agglomerative_threshold))
        n_clusters = int(len(set(labels.tolist())))
        max_reasonable_clusters = max(3, int(math.ceil(_pick_k(n) * 1.5)))
        singleton_count = sum(1 for cid in set(labels.tolist()) if int(np.sum(labels == cid)) == 1)
        singleton_ratio = float(singleton_count / max(1, n_clusters))
        if n_clusters >= 2 and n_clusters <= max_reasonable_clusters and singleton_ratio <= 0.4:
            return labels, {
                "method": "agglomerative_cosine_threshold",
                "seed": int(seed),
                "agglomerative_threshold": float(agglomerative_threshold),
                "n_clusters": n_clusters,
                "singleton_ratio": singleton_ratio,
            }
    except Exception as exc:
        _ = exc

    # Final fallback: legacy cosine-kmeans.
    k = min(_pick_k(n), n)
    labels = _kmeans_cluster(Xn, k=k, seed=seed)
    return labels, {
        "method": "kmeans_embeddings",
        "k": int(k),
        "seed": int(seed),
        "fallback_reason": hdbscan_reject_reason,
    }


def _merge_clusters_by_centroid(
    *,
    clusters: List[tuple[int, List[str]]],
    Xn: np.ndarray,
    id_to_row: Dict[str, int],
    threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    """
    Post-process clusters to reduce redundancy:
    - compute centroid per cluster (mean of normalized chunk vectors)
    - greedily merge clusters whose centroids have cosine similarity >= threshold

    Returns list of merged cluster dicts:
    { "cluster_ids": [..], "chunk_ids": [..], "centroid": np.ndarray, "merged": bool }

    This is deliberately LLM-free and cheap (matrix ops only).
    """
    if not clusters:
        return []

    # Build centroid per cluster (on Xn).
    items: List[Dict[str, Any]] = []
    for cluster_id, cids in clusters:
        rows = [id_to_row[cid] for cid in cids if cid in id_to_row]
        if not rows:
            continue
        V = Xn[rows]
        centroid = l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
        items.append(
            {
                "cluster_ids": [int(cluster_id)],
                "chunk_ids": list(cids),
                "centroid": centroid,
                "merged": False,
                "merge_sims": [],  # list of floats for audit
            }
        )

    # Sort by size desc for stable greedy merges (keep biggest themes).
    items.sort(key=lambda it: len(it["chunk_ids"]), reverse=True)

    kept: List[Dict[str, Any]] = []
    for it in items:
        c = it["centroid"]
        best_j = None
        best_sim = -1.0
        for j, k_it in enumerate(kept):
            sim = float(np.dot(c, k_it["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None and best_sim >= float(threshold):
            # Merge into best existing cluster
            target = kept[best_j]
            target["cluster_ids"].extend(it["cluster_ids"])
            target["chunk_ids"].extend(it["chunk_ids"])
            target["merged"] = True
            target["merge_sims"].append(best_sim)

            # Recompute centroid using mean of member vectors (approx by averaging centroids weighted by sizes)
            # For better accuracy without heavy computation, recompute from underlying rows:
            rows = [id_to_row[cid] for cid in target["chunk_ids"] if cid in id_to_row]
            if rows:
                V = Xn[rows]
                target["centroid"] = l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
        else:
            kept.append(it)

    # De-duplicate chunk_ids within each merged cluster, preserve first-seen order.
    for it in kept:
        seen = set()
        deduped = []
        for cid in it["chunk_ids"]:
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(cid)
        it["chunk_ids"] = deduped
    return kept


def _attach_small_clusters_to_nearest(
    *,
    clusters: List[Dict[str, Any]],
    Xn: np.ndarray,
    id_to_row: Dict[str, int],
    min_topic_chunks: int = 2,
) -> List[Dict[str, Any]]:
    if not clusters:
        return []

    threshold = max(1, int(min_topic_chunks))
    large = [it for it in clusters if len(it.get("chunk_ids", [])) >= threshold]
    small = [it for it in clusters if len(it.get("chunk_ids", [])) < threshold]
    if not small or not large:
        return clusters

    kept = list(large)
    for it in small:
        c = it.get("centroid")
        if c is None:
            continue
        best_j = 0
        best_sim = -1.0
        for j, target in enumerate(kept):
            sim = float(np.dot(c, target["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_j = j

        target = kept[best_j]
        target["cluster_ids"].extend(it.get("cluster_ids", []))
        target["chunk_ids"].extend(it.get("chunk_ids", []))
        target["merged"] = True

        rows = [id_to_row[cid] for cid in target["chunk_ids"] if cid in id_to_row]
        if rows:
            V = Xn[rows]
            target["centroid"] = l2_normalize(np.mean(V, axis=0, keepdims=True))[0]

    for it in kept:
        seen = set()
        deduped = []
        for cid in it["chunk_ids"]:
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(cid)
        it["chunk_ids"] = deduped

    kept.sort(key=lambda it: len(it["chunk_ids"]), reverse=True)
    return kept


def build_topics_for_exam(
    *,
    exam_id: str,
    store: Optional[VectorStore] = None,
    repo: Optional[DBRepository] = None,
    overwrite: bool = True,
    seed: int = 0,
    use_llm_labels: bool = True,
    llm_model: str = CHAT_MODEL_FAST,
    merge_threshold: float = 0.88,
    topic_cluster_algorithm: str = "hdbscan",
    use_umap: bool = False,
    umap_n_components: int = 15,
    umap_min_chunk_count: int = 300,
    hdbscan_min_cluster_size: Optional[int] = None,
    hdbscan_min_samples: Optional[int] = None,
    agglomerative_threshold: float = 0.82,
) -> List[BuiltTopic]:
    """
    Build semantic topics for an exam by clustering chunk embeddings and producing grounded labels + evidence.
    Persists topics + topic_chunks + topic_evidence into SQLite.
    """
    store = store or get_store()
    repo = repo or get_repo()
    # For Pinecone backend, topic building needs the exam namespace for vector fetches.
    if store.vector_backend == "pinecone":
        exam_row = repo.get_exam(exam_id)
        if exam_row is None:
            raise ValueError(f"Exam not found: {exam_id}")
        from app.data.pinecone_backend import pinecone_namespace
        store = store.for_namespace(pinecone_namespace(user_id=exam_row.user_id, exam_id=exam_id))
    ws = load_exam(repo=repo, exam_id=exam_id)
    doc_ids = ws.doc_ids
    if not doc_ids:
        return []
    document_math_profile = (ws.exam.info or {}).get("math_profile") if isinstance(ws.exam.info, dict) else None

    # 1) collect chunks
    chunks: List[StoredChunk] = []
    for doc_id in doc_ids:
        chunks.extend(store.list_chunks_by_doc(doc_id))
    if not chunks:
        return []

    chunk_ids = [c.chunk_id for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in chunks}

    # 2) fetch vectors aligned to chunk_ids (skips any missing)
    resolved_ids, X = store.get_vectors_for_chunk_ids(chunk_ids)
    if X.size == 0 or not resolved_ids:
        return []
    id_to_row = {cid: i for i, cid in enumerate(resolved_ids)}
    # Normalize once and reuse everywhere (merging, context pack selection, centroid persistence).
    Xn = l2_normalize(X.astype("float32", copy=False))

    # 3) cluster
    assign, cluster_meta = _cluster_topic_embeddings(
        X=X,
        seed=seed,
        algorithm=topic_cluster_algorithm,
        use_umap=use_umap,
        umap_n_components=umap_n_components,
        umap_min_chunk_count=umap_min_chunk_count,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        agglomerative_threshold=agglomerative_threshold,
    )
    # Drop empty clusters and re-index to contiguous ids
    groups: Dict[int, List[str]] = {}
    for cid, a in zip(resolved_ids, assign.tolist()):
        groups.setdefault(int(a), []).append(cid)
    # Sort clusters by size descending for stable output.
    clusters = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    # 3.5) merge near-duplicate clusters by centroid similarity (cheap redundancy reduction)
    merged_clusters = _merge_clusters_by_centroid(
        clusters=clusters,
        Xn=Xn,
        id_to_row=id_to_row,
        threshold=merge_threshold,
    )
    merged_clusters = _attach_small_clusters_to_nearest(
        clusters=merged_clusters,
        Xn=Xn,
        id_to_row=id_to_row,
        min_topic_chunks=2,
    )

    built: List[BuiltTopic] = []
    for merged in merged_clusters:
        cids = list(merged["chunk_ids"])
        c_chunks = [chunk_by_id[cid] for cid in cids if cid in chunk_by_id]
        texts = [c.text for c in c_chunks if c and c.text]
        label = ""
        evidence: List[Dict[str, Any]] = []
        label_source = "ngram"

        if use_llm_labels:
            context_pack = build_representative_chunk_pack(
                store=store,
                chunk_ids=cids,
                centroid=merged.get("centroid"),
                Xn=Xn,
                id_to_row=id_to_row,
                chunk_by_id=chunk_by_id,
            )
            if context_pack:
                llm_out = _llm_label_cluster(context_pack=context_pack, model=llm_model)
                proposed_label = str(llm_out.get("label") or "").strip()
                phrases = list(llm_out.get("evidence_phrases") or [])
                ev, _matched = _extract_evidence_for_phrases(c_chunks, [proposed_label] + phrases)
                if ev and proposed_label:
                    label = proposed_label
                    evidence = ev
                    label_source = "llm"

        if not label:
            # Fallback: deterministic grounded n-gram label.
            label = _choose_grounded_label(texts)
            evidence = _extract_evidence_for_label(c_chunks, label)
            if not evidence:
                # Pick a label from the most frequent token (still grounded because it appears in text)
                label = _choose_grounded_label(texts, max_words=1)
                evidence = _extract_evidence_for_label(c_chunks, label)
        topic_id = uuid.uuid4().hex[:16]
        # Persist a centroid vector for routing: mean of normalized chunk vectors in this topic.
        rows = [id_to_row[cid] for cid in cids if cid in id_to_row]
        if rows:
            centroid = l2_normalize(np.mean(Xn[rows], axis=0, keepdims=True))[0]
            try:
                repo.upsert_topic_vector(topic_id=topic_id, vector=centroid)
            except Exception:
                pass
        info = {
            "n_chunks": len(cids),
            "method": str(cluster_meta.get("method") or "unknown"),
            "k": cluster_meta.get("k"),
            "seed": int(seed),
            "cluster_params": {
                "algorithm": topic_cluster_algorithm,
                "use_umap": bool(use_umap),
                "umap_n_components": int(umap_n_components),
                "umap_min_chunk_count": int(umap_min_chunk_count),
                "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                "hdbscan_min_samples": hdbscan_min_samples,
                "agglomerative_threshold": float(agglomerative_threshold),
            },
            "cluster_stats": cluster_meta,
            "label_source": label_source,
            "merged_from_clusters": merged.get("cluster_ids", []),
            "merge_sims": merged.get("merge_sims", []),
            "merge_threshold": float(merge_threshold),
        }
        try:
            route_context = build_representative_chunk_pack(
                store=store,
                chunk_ids=cids,
                centroid=merged.get("centroid"),
                Xn=Xn,
                id_to_row=id_to_row,
                chunk_by_id=chunk_by_id,
            )
            # Topic-build time: route from a representative chunk pack for the topic.
            route_decision = classify_card_route(
                topic_label=label,
                context_pack=route_context or "\n".join(texts[:4]),
                document_math_profile=document_math_profile if isinstance(document_math_profile, dict) else None,
                store=store,
            )
            info["route_candidate"] = route_decision.to_info()
        except Exception:
            pass
        built.append(
            BuiltTopic(
                topic_id=topic_id,
                label=label,
                chunk_ids=cids,
                evidence=evidence,
                info=info,
            )
        )

    # 4) persist
    if overwrite:
        repo.delete_topics_for_exam(exam_id=exam_id)
    for t in built:
        repo.upsert_topic(topic_id=t.topic_id, exam_id=exam_id, label=t.label, info=t.info)
        repo.replace_topic_chunks(topic_id=t.topic_id, chunk_ids=t.chunk_ids)
        repo.replace_topic_evidence(topic_id=t.topic_id, evidence=t.evidence)

    return built


def list_topics_for_exam(*, exam_id: str, repo: Optional[DBRepository] = None) -> List[StoredTopic]:
    repo = repo or get_repo()
    return repo.list_topics(exam_id=exam_id)


