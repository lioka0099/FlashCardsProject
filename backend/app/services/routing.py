from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.data.vector_store import VectorStore
from app.data.db_repository import DBRepository
from app.data.pinecone_backend import pinecone_namespace
from app.deps import get_repo, get_store
from app.services.llm import embed_query
from app.services.qa import generate_answer, AnswerWithCitations
from app.utils.vectors import l2_normalize


@dataclass
class TopicRoute:
    topic_id: str
    label: str
    score: float


def _topic_centroid_for_chunk_ids(store: VectorStore, chunk_ids: List[str]) -> Optional[np.ndarray]:
    # Reconstruct vectors for the topic's chunks and compute a centroid.
    resolved, X = store.get_vectors_for_chunk_ids(chunk_ids)
    if X.size == 0 or not resolved:
        return None
    Xn = l2_normalize(X)
    centroid = np.mean(Xn, axis=0, keepdims=True)
    return l2_normalize(centroid)[0]


def route_question_to_topic(
    *,
    exam_id: str,
    question: str,
    store: Optional[VectorStore] = None,
    repo: Optional[DBRepository] = None,
    top_n: int = 2,
    question_vec: Optional[np.ndarray] = None,
) -> List[TopicRoute]:
    """
    Deterministic topic router:
    - compute centroid embedding per topic (mean of topic chunk embeddings)
    - embed question once
    - return top-N topics by cosine similarity

    Pass an exam-scoped store (store.for_namespace(...)) when topic vectors may need
    recomputation from chunk embeddings under the Pinecone backend.
    """
    store = store or get_store()
    repo = repo or get_repo()
    topics = repo.list_topics(exam_id=exam_id)
    if not topics:
        return []
    # Prefer persisted topic vectors; fall back to recomputing if missing.
    centroids: Dict[str, np.ndarray] = repo.list_topic_vectors_for_exam(exam_id=exam_id)
    labels: Dict[str, str] = {}
    for t in topics:
        labels[t.topic_id] = t.label
        if t.topic_id not in centroids:
            cids = repo.list_chunk_ids_for_topic(topic_id=t.topic_id)
            c = _topic_centroid_for_chunk_ids(store, cids)
            if c is not None:
                centroids[t.topic_id] = c

    if not centroids:
        return []

    if question_vec is None:
        q = embed_query(question).astype("float32", copy=False)
    else:
        q = question_vec.astype("float32", copy=False)
        if q.ndim == 1:
            q = q[None, :]
    q = l2_normalize(q)[0]

    scored: List[TopicRoute] = []
    for topic_id, c in centroids.items():
        score = float(np.dot(q, c))
        scored.append(TopicRoute(topic_id=topic_id, label=labels.get(topic_id, ""), score=score))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[: max(1, int(top_n))]


def answer_in_exam(
    *,
    exam_id: str,
    question: str,
    store: Optional[VectorStore] = None,
    repo: Optional[DBRepository] = None,
    k: int = 8,
    min_score: float = 0.4,
    top_n_topics: int = 1,
) -> Tuple[AnswerWithCitations, List[TopicRoute]]:
    """
    Automatic end-to-end:
    route question -> select top topic(s) -> topic-scoped retrieve -> answer with proofs.
    """
    store = store or get_store()
    repo = repo or get_repo()
    # For Pinecone backend, scope all retrieval to this exam namespace.
    if store.vector_backend == "pinecone":
        exam_row = repo.get_exam(exam_id)
        if exam_row is None:
            raise ValueError(f"Exam not found: {exam_id}")
        store = store.for_namespace(pinecone_namespace(user_id=exam_row.user_id, exam_id=exam_id))
    q_vec = embed_query(question).astype("float32", copy=False)
    routes = route_question_to_topic(
        exam_id=exam_id,
        question=question,
        store=store,
        repo=repo,
        top_n=top_n_topics,
        question_vec=q_vec,
    )
    if not routes:
        # Fallback: unrestricted within store
        return generate_answer(question=question, k=k, min_score=min_score, store=store), []
    allowed_chunk_ids: List[str] = []
    for r in routes:
        allowed_chunk_ids.extend(repo.list_chunk_ids_for_topic(topic_id=r.topic_id))
    ans = generate_answer(
        question=question,
        k=k,
        min_score=min_score,
        store=store,
        allowed_chunk_ids=allowed_chunk_ids,
        query_vec=q_vec,
    )
    return ans, routes


