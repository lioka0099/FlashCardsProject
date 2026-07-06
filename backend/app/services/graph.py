"""
Card Generation LangGraph Flow

Topic-scoped, difficulty-aware card generation with:
- FAISS-based question deduplication (exam-scoped)
- Bloom's taxonomy difficulty levels
- Robust retry logic (full restart on persistent failures)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import logging
import time
import uuid
import numpy as np
from langgraph.graph import StateGraph, END

from app.data.vector_store import VectorStore
from app.data.pinecone_backend import PineconeClient, pinecone_namespace
from app.deps import get_repo, get_store
from app.services.student_memory import StudentMemoryService
from app.services.student_model import StudentModelService
from app.services.teacher_model import TeacherModelService
from app.services.card_routing import classify_card_route, default_route_decision
from app.services.difficulty_frameworks import (
    card_info_for_difficulty,
    clamp_difficulty,
    framework_for_route,
    get_level,
)
from app.services.math_student_model import MathStudentModelService
from app.services.math_teacher_model import MathTeacherModelService
from app.services.math_verification import verify_math_solution, verify_compound_solution
from app.services.math_concept_inventory import (
    INVENTORY_INFO_KEY,
    MathConceptInventoryService,
)
from app.services.llm import (
    chat_completions_create,
    CHAT_MODEL,
    embed_texts,
)
from app.services.cards import (
    GeneratedCard,
    pick_starter_topics,
)
from app.services.context_packs import build_diverse_chunk_pack
from app.api.schemas import ProofSpan

logger = logging.getLogger(__name__)

# ------------- Configuration -------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_question_attempts": 3,
    "max_answer_attempts": 3,
    "max_full_restarts": 5,
    "math_max_question_attempts": 3,
    "math_max_answer_attempts": 2,
    "math_max_full_restarts": 3,
    "uniqueness_threshold": 0.85,
    "math_uniqueness_threshold": 0.97,
    "uniqueness_top_k": 50,
    "validation_threshold": 0.7,
    "math_validation_threshold": 0.7,
    "initial_k": 8,
    "initial_min_score": 0.4,
    "strengthen_k_delta": 2,
    "strengthen_min_score_delta": 0.05,
    "commit_retry_attempts": 3,
    "commit_retry_sleep_s": 0.25,
    "graph_recursion_limit": 140,
}


@dataclass(frozen=True)
class BatchSeenQuestion:
    question_id: str
    topic_id: str
    question_text: str
    difficulty: int
    embedding: np.ndarray
    created_seq: int
    fingerprint: str = ""
    archetype: str = ""

# ------------- Helpers (kept from old implementation) -------------


def _hash_text(text: str) -> str:
    """Hash text for embedding cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_with_store_cache(texts: List[str], store: VectorStore) -> np.ndarray:
    """Embed texts using the store's embedding cache."""
    if not texts:
        return np.zeros((0, store.vector_dimension), dtype="float32")
    hashes = [_hash_text(t) for t in texts]
    cached = store.get_cached_embeddings(hashes)
    missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_vectors = embed_texts(missing_texts)
        for idx, vec in zip(missing_idx, new_vectors):
            cached[hashes[idx]] = vec.astype("float32", copy=False)
        store.add_cached_embeddings({hashes[idx]: cached[hashes[idx]] for idx in missing_idx})
    vectors = np.stack([cached[h] for h in hashes]).astype("float32", copy=False)
    return vectors


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = a.astype("float32", copy=False).reshape(-1)
    vb = b.astype("float32", copy=False).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _validate_answer(question: str, answer: str, proofs: List[ProofSpan]) -> Dict[str, Any]:
    """
    Validate answer groundedness against source proofs.
    Returns dict with 'score' (0.0-1.0) and 'critique' (string).
    """
    sources = []
    for i, p in enumerate(proofs, 1):
        sources.append(f"S{i}: doc={p.doc_id} page={p.page} score={p.score:.2f}\n{p.text}")
    src = "\n\n".join(sources)
    prompt = (
        "Evaluate the answer strictly for grounding and completeness using the sources.\n"
        "Return JSON with fields: score (0.0-1.0), critique (string). "
        "Penalize missing or incorrect citations.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{src}"
    )
    try:
        resp = chat_completions_create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict fact-checker."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        resp = chat_completions_create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict fact-checker."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {"score": 0.5, "critique": "Could not parse evaluation."}
    if "score" not in data:
        data["score"] = 0.5
    if "critique" not in data:
        data["critique"] = "No critique provided."
    return data


def _validate_math_answer(question: str, answer: str, proofs: List[ProofSpan]) -> Dict[str, Any]:
    """
    Validate worked math answers for source grounding and completeness.
    Mathematical correctness is checked separately by SymPy.
    """
    sources = []
    for i, p in enumerate(proofs, 1):
        sources.append(f"S{i}: doc={p.doc_id} page={p.page} score={p.score:.2f}\n{p.text}")
    src = "\n\n".join(sources)
    prompt = (
        "Evaluate the worked math flashcard answer for grounding and completeness only.\n"
        "Do not judge algebraic correctness here; a deterministic verifier handles that.\n"
        "Return JSON with fields: score (0.0-1.0), critique (string).\n"
        "Give high scores only when the problem givens/rules are supported by sources and "
        "the answer has enough steps plus a final answer.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{src}"
    )
    try:
        resp = chat_completions_create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict math-flashcard grounding evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        raw = "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {"score": 0.5, "critique": "Could not parse math grounding evaluation."}
    data.setdefault("score", 0.5)
    data.setdefault("critique", "No critique provided.")
    return data


# ------------- State -------------


class CardGenState(TypedDict):
    """State for single card generation flow."""
    
    # Input
    exam_id: str
    user_id: str
    store_basepath: str
    batch_id: str
    
    # Current topic
    topic_id: str
    topic_label: str
    difficulty: int
    card_route: str
    card_route_override: Optional[str]
    route_metadata: Dict[str, Any]
    difficulty_framework: str
    difficulty_level_name: str
    card_type: str
    allowed_chunk_ids: List[str]
    context_pack: str
    batch_seen_questions: Tuple[BatchSeenQuestion, ...]
    student_memory: Dict[str, Any]
    
    # Question state
    question: Optional[str]
    question_generation_failed: bool
    question_failure_reason: Optional[str]
    math_question_payload: Dict[str, Any]
    math_solver_payload: Dict[str, Any]
    question_embedding: Optional[np.ndarray]
    question_id: Optional[str]
    is_unique: bool
    
    # Answer state
    answer: Optional[str]
    math_answer_payload: Dict[str, Any]
    proofs: Optional[List[ProofSpan]]
    validation_score: Optional[float]
    validation_critique: Optional[str]
    grounding_validation_score: Optional[float]
    verification_result: Dict[str, Any]
    math_validation_passed: bool
    
    # Retry tracking
    question_attempts: int
    answer_attempts: int
    full_restart_count: int
    math_validation_fail_cycles: int
    
    # Limits
    max_question_attempts: int
    max_answer_attempts: int
    max_full_restarts: int
    uniqueness_threshold: float
    uniqueness_top_k: int
    validation_threshold: float
    commit_retry_attempts: int
    commit_retry_sleep_s: float
    
    # Retrieval params (for strengthening)
    initial_k: int
    initial_min_score: float
    strengthen_k_delta: int
    strengthen_min_score_delta: float
    k: int
    min_score: float
    
    # Flow control
    stop_after_embedding: bool  # For parallel starter pack: stop after Phase 1
    
    # Output
    card: Optional[GeneratedCard]


# ------------- Nodes -------------


def _get_or_build_concept_inventory(state: CardGenState, store: VectorStore):
    """Load the cached math concept inventory for a topic, building+caching if absent."""
    topic_info: Dict[str, Any] = {}
    topic_label = state["topic_label"]
    try:
        topics = get_repo().list_topics(exam_id=state["exam_id"])
        stored = next((t for t in topics if t.topic_id == state["topic_id"]), None)
        topic_info = stored.info or {} if stored is not None else {}
        topic_label = stored.label if stored is not None else topic_label
    except Exception:
        topic_info = {}
    cached = topic_info.get(INVENTORY_INFO_KEY) if isinstance(topic_info, dict) else None
    inventory = MathConceptInventoryService().get_or_build(
        topic_label=topic_label,
        context_pack=state["context_pack"],
        cached=cached if isinstance(cached, dict) else None,
    )
    # Persist the inventory once so later cards on this topic reuse it.
    if not (isinstance(cached, dict) and cached) and not inventory.is_empty():
        try:
            merged = dict(topic_info)
            merged[INVENTORY_INFO_KEY] = inventory.to_info()
            get_repo().upsert_topic(
                topic_id=state["topic_id"],
                exam_id=state["exam_id"],
                label=topic_label,
                info=merged,
            )
        except Exception:
            logger.debug("Could not persist concept inventory for topic %s", state.get("topic_id"))
    return inventory


def node_generate_question(state: CardGenState) -> CardGenState:
    """Generate a question at the specified difficulty level."""
    route = state.get("card_route") or "default"
    question = ""
    math_payload: Dict[str, Any] = {}
    math_solver_payload: Dict[str, Any] = {}
    failed = False
    failure_reason = None
    try:
        if route == "math_calculation":
            store = VectorStore(basepath=state["store_basepath"])
            blocked_questions = [
                str(seen.question_text or "").strip()
                for seen in state.get("batch_seen_questions", ())
                if str(seen.question_text or "").strip()
            ]
            blocked_questions.extend(
                get_repo().list_question_texts_for_exam(
                    exam_id=state["exam_id"],
                    limit=100,
                )
            )
            # Structural-diversity signal: fingerprints already used this batch + in the exam.
            blocked_fingerprints = [
                str(seen.fingerprint or "").strip()
                for seen in state.get("batch_seen_questions", ())
                if str(seen.fingerprint or "").strip()
            ]
            blocked_fingerprints.extend(
                get_repo().list_math_fingerprints_for_exam(exam_id=state["exam_id"], limit=100)
            )
            inventory = _get_or_build_concept_inventory(state, store)
            result = MathStudentModelService().generate_question(
                topic_label=state["topic_label"],
                context_pack=state["context_pack"],
                difficulty=state["difficulty"],
                memory=state.get("student_memory") or {},
                route_metadata=state.get("route_metadata") or {},
                avoid_questions=blocked_questions,
                avoid_fingerprints=blocked_fingerprints,
                concept_inventory=inventory,
                attempt_no=int(state.get("question_attempts") or 0) + 1,
            )
            question = result.question
            math_payload = result.payload
            # The compound spec was already verified deterministically inside the
            # student model (every step solvable; final answer = CAS canonical).
            math_solver_payload = {
                "method": "sympy-compound",
                "status": "solved",
                "final_answer": math_payload.get("expected_final_answer"),
                "solution_steps": [
                    str(step.get("description") or "")
                    for step in (math_payload.get("steps") or [])
                    if str(step.get("description") or "").strip()
                ],
                "verification_target": math_payload.get("verification_target") or {},
            }
        else:
            question = StudentModelService().generate_question(
                topic_label=state["topic_label"],
                context_pack=state["context_pack"],
                difficulty=state["difficulty"],
                memory=state.get("student_memory") or {},
            )
    except Exception as exc:
        failed = True
        failure_reason = str(exc)
    return {
        **state,
        "question": question,
        "question_generation_failed": failed or not bool(question),
        "question_failure_reason": failure_reason,
        "math_question_payload": math_payload,
        "math_solver_payload": math_solver_payload,
        "question_attempts": state["question_attempts"] + 1,
        "math_validation_fail_cycles": 0,
    }


def node_route_card(state: CardGenState) -> CardGenState:
    """Choose default vs math-calculation generation for this topic/context."""
    store = VectorStore(basepath=state["store_basepath"])
    override = state.get("card_route_override")
    if override in {"default", "math_calculation", "math_conceptual"}:
        # "math_conceptual" is an intent token only: it resolves to the default
        # generation path, distinguished solely by the conceptual tag below.
        resolved_route = "default" if override == "math_conceptual" else override
        framework = framework_for_route(resolved_route)
        difficulty = clamp_difficulty(framework, state["difficulty"])
        level = get_level(framework, difficulty)
        route_metadata = {
            "card_route": resolved_route,
            "subject_type": "math" if override.startswith("math_") else "general",
            "math_kind": "calculation" if override == "math_calculation" else "conceptual" if override == "math_conceptual" else "none",
            "confidence": 1.0,
            "reason": "Session generation forced this route as a fallback.",
        }
        return {
            **state,
            "card_route": resolved_route,
            "route_metadata": route_metadata,
            "difficulty_framework": framework,
            "difficulty": difficulty,
            "difficulty_level_name": level.name,
            "validation_threshold": (
                DEFAULT_CONFIG["math_validation_threshold"]
                if override == "math_calculation"
                else DEFAULT_CONFIG["validation_threshold"]
            ),
            "uniqueness_threshold": (
                DEFAULT_CONFIG["math_uniqueness_threshold"]
                if override == "math_calculation"
                else DEFAULT_CONFIG["uniqueness_threshold"]
            ),
            "max_answer_attempts": (
                DEFAULT_CONFIG["math_max_answer_attempts"]
                if override == "math_calculation"
                else DEFAULT_CONFIG["max_answer_attempts"]
            ),
            "max_question_attempts": (
                DEFAULT_CONFIG["math_max_question_attempts"]
                if override == "math_calculation"
                else DEFAULT_CONFIG["max_question_attempts"]
            ),
            "max_full_restarts": (
                DEFAULT_CONFIG["math_max_full_restarts"]
                if override == "math_calculation"
                else DEFAULT_CONFIG["max_full_restarts"]
            ),
        }
    topic_info: Dict[str, Any] = {}
    document_math_profile: Optional[Dict[str, Any]] = None
    try:
        exam = get_repo().get_exam(state["exam_id"])
        if exam is not None and isinstance(exam.info, dict):
            raw_profile = exam.info.get("math_profile")
            document_math_profile = raw_profile if isinstance(raw_profile, dict) else {"kind": "non_math"}
        topics = get_repo().list_topics(exam_id=state["exam_id"])
        topic_info = next((t.info or {} for t in topics if t.topic_id == state["topic_id"]), {})
    except Exception:
        topic_info = {}
    cached_route = topic_info.get("route_candidate") if isinstance(topic_info, dict) else None
    decision = classify_card_route(
        topic_label=state["topic_label"],
        context_pack=state["context_pack"],
        cached_topic_route=cached_route if isinstance(cached_route, dict) else None,
        document_math_profile=document_math_profile,
        store=store,
    )
    framework = framework_for_route(decision.card_route)
    difficulty = clamp_difficulty(framework, state["difficulty"])
    level = get_level(framework, difficulty)
    return {
        **state,
        "card_route": decision.card_route,
        "route_metadata": decision.to_info(),
        "difficulty_framework": framework,
        "difficulty": difficulty,
        "difficulty_level_name": level.name,
        "validation_threshold": (
            DEFAULT_CONFIG["math_validation_threshold"]
            if decision.card_route == "math_calculation"
            else state.get("validation_threshold", DEFAULT_CONFIG["validation_threshold"])
        ),
        "uniqueness_threshold": (
            DEFAULT_CONFIG["math_uniqueness_threshold"]
            if decision.card_route == "math_calculation"
            else state.get("uniqueness_threshold", DEFAULT_CONFIG["uniqueness_threshold"])
        ),
        "max_answer_attempts": (
            DEFAULT_CONFIG["math_max_answer_attempts"]
            if decision.card_route == "math_calculation"
            else state.get("max_answer_attempts", DEFAULT_CONFIG["max_answer_attempts"])
        ),
        "max_question_attempts": (
            DEFAULT_CONFIG["math_max_question_attempts"]
            if decision.card_route == "math_calculation"
            else state.get("max_question_attempts", DEFAULT_CONFIG["max_question_attempts"])
        ),
        "max_full_restarts": (
            DEFAULT_CONFIG["math_max_full_restarts"]
            if decision.card_route == "math_calculation"
            else state.get("max_full_restarts", DEFAULT_CONFIG["max_full_restarts"])
        ),
    }


def node_embed_question(state: CardGenState) -> CardGenState:
    """Embed the generated question."""
    store = VectorStore(basepath=state["store_basepath"])
    embedding = _embed_with_store_cache([state.get("question") or ""], store)[0]
    return {
        **state,
        "question_embedding": embedding,
    }


def node_check_uniqueness(state: CardGenState) -> CardGenState:
    """
    Check if question is semantically unique within this exam.
    Uses Pinecone question index search for exam-scoped similarity lookup.
    """
    threshold = float(state["uniqueness_threshold"])
    query_vec = state["question_embedding"]
    route = state.get("card_route") or "default"
    question_norm = " ".join(str(state.get("question") or "").strip().lower().split())

    if route == "math_calculation":
        candidate_fp = str((state.get("math_question_payload") or {}).get("fingerprint") or "").strip()
        # Structural-diversity gate: reject same problem *type* (archetype + concepts +
        # operation family), not just identical text — this is what stops "same template,
        # different numbers" repetition.
        for seen in state.get("batch_seen_questions", ()):
            seen_norm = " ".join(str(seen.question_text or "").strip().lower().split())
            if seen_norm and seen_norm == question_norm:
                logger.warning(
                    "Math uniqueness rejected exact in-batch duplicate: topic_id=%s seen_topic_id=%s",
                    state.get("topic_id"),
                    seen.topic_id,
                )
                return {**state, "is_unique": False}
            if candidate_fp and str(seen.fingerprint or "").strip() == candidate_fp:
                logger.info(
                    "Math uniqueness rejected in-batch structural duplicate: topic_id=%s fingerprint=%s",
                    state.get("topic_id"),
                    candidate_fp,
                )
                return {**state, "is_unique": False}
        if get_repo().has_equivalent_question_text(
            exam_id=state["exam_id"],
            question_text=state.get("question") or "",
        ):
            logger.warning(
                "Math uniqueness rejected exact indexed duplicate: topic_id=%s",
                state.get("topic_id"),
            )
            return {**state, "is_unique": False}
        if candidate_fp and candidate_fp in set(
            get_repo().list_math_fingerprints_for_exam(exam_id=state["exam_id"], limit=200)
        ):
            logger.info(
                "Math uniqueness rejected indexed structural duplicate: topic_id=%s fingerprint=%s",
                state.get("topic_id"),
                candidate_fp,
            )
            return {**state, "is_unique": False}
        return {**state, "is_unique": True}

    if query_vec is None:
        return {**state, "is_unique": False}

    # In-batch visibility check (deterministic sequential snapshot).
    for seen in state.get("batch_seen_questions", ()):
        sim = _cosine_similarity(query_vec, seen.embedding)
        if sim >= threshold:
            logger.info(
                "Question uniqueness rejected in-batch candidate: route=%s topic_id=%s "
                "seen_topic_id=%s similarity=%.3f threshold=%.3f question=%s",
                state.get("card_route") or "default",
                state.get("topic_id"),
                seen.topic_id,
                sim,
                threshold,
                (state.get("question") or "")[:120],
            )
            return {**state, "is_unique": False}

    store = VectorStore(basepath=state["store_basepath"])
    if store.vector_backend != "pinecone":
        # Phase 5 contract: uniqueness gate is mandatory.
        raise RuntimeError(
            "Mandatory uniqueness gate requires VECTOR_BACKEND=pinecone for card generation."
        )

    ns = pinecone_namespace(user_id=state["user_id"], exam_id=state["exam_id"])
    pc = PineconeClient()
    matches = pc.query(
        index=pc.questions,
        namespace=ns,
        query_vec=query_vec,
        top_k=int(state.get("uniqueness_top_k") or DEFAULT_CONFIG["uniqueness_top_k"]),
    )
    for _qid, sim in matches:
        if sim >= threshold:
            logger.info(
                "Question uniqueness rejected indexed candidate: route=%s topic_id=%s "
                "matched_question_id=%s similarity=%.3f threshold=%.3f question=%s",
                state.get("card_route") or "default",
                state.get("topic_id"),
                _qid,
                sim,
                threshold,
                (state.get("question") or "")[:120],
            )
            return {**state, "is_unique": False}
    return {**state, "is_unique": True}


def node_save_question_id(state: CardGenState) -> CardGenState:
    """
    Reservation-only step: assign a stable question_id.
    Permanent SQL/Pinecone commit happens only after validated card storage.
    """
    question_id = state.get("question_id") or uuid.uuid4().hex[:16]
    return {**state, "question_id": question_id}


def node_commit_question_index(state: CardGenState) -> CardGenState:
    """Persist question embedding/metadata after successful card generation."""
    question_id = state.get("question_id")
    embedding = state.get("question_embedding")
    if not question_id or embedding is None:
        raise RuntimeError("Cannot commit question index without question_id and embedding.")

    store = VectorStore(basepath=state["store_basepath"])
    get_repo().add_question_index_entry(
        question_id=question_id,
        exam_id=state["exam_id"],
        topic_id=state["topic_id"],
        question_text=state["question"] or "",
        difficulty=int(state["difficulty"]) if state.get("difficulty") is not None else None,
        embedding=embedding,
    )

    if store.vector_backend != "pinecone":
        raise RuntimeError("Question index commit requires VECTOR_BACKEND=pinecone.")

    ns = pinecone_namespace(user_id=state["user_id"], exam_id=state["exam_id"])
    pc = PineconeClient()
    max_retries = max(1, int(state["commit_retry_attempts"]))
    sleep_s = float(state["commit_retry_sleep_s"])
    for attempt in range(1, max_retries + 1):
        try:
            pc.upsert(
                index=pc.questions,
                namespace=ns,
                vectors=[(question_id, embedding)],
                metadata_by_id={
                    question_id: {
                        "topic_id": state["topic_id"],
                        "difficulty": int(state["difficulty"]),
                        "card_route": state.get("card_route") or "default",
                        "difficulty_framework": state.get("difficulty_framework") or "bloom",
                    }
                },
                batch_size=100,
            )
            break
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_s * attempt)

    get_repo().add_event(
        user_id=state["user_id"],
        exam_id=state["exam_id"],
        type="question_index_committed",
        payload={
            "question_id": question_id,
            "topic_id": state["topic_id"],
            "difficulty": state["difficulty"],
            "card_route": state.get("card_route") or "default",
            "difficulty_framework": state.get("difficulty_framework") or "bloom",
            "batch_id": state.get("batch_id"),
        },
    )
    return {**state, "question_id": question_id}


def node_generate_answer(state: CardGenState) -> CardGenState:
    """Generate answer using topic-scoped retrieval."""
    store = VectorStore(basepath=state["store_basepath"])
    if store.vector_backend == "pinecone":
        store = store.for_namespace(pinecone_namespace(user_id=state["user_id"], exam_id=state["exam_id"]))
    math_answer_payload: Dict[str, Any] = {}
    if state.get("card_route") == "math_calculation":
        math_result = MathTeacherModelService().generate_answer(
            question=state["question"] or "",
            question_payload=state.get("math_question_payload") or {},
            k=state["k"],
            min_score=state["min_score"],
            store=store,
            allowed_chunk_ids=state["allowed_chunk_ids"],
            solver_payload=state.get("math_solver_payload") or {},
            query_vec=state.get("question_embedding"),
        )
        answer = math_result.answer
        proofs = math_result.proofs
        math_answer_payload = math_result.payload
    else:
        result = TeacherModelService().generate_answer(
            question=state["question"],
            k=state["k"],
            min_score=state["min_score"],
            store=store,
            allowed_chunk_ids=state["allowed_chunk_ids"],
            query_vec=state.get("question_embedding"),
        )
        answer = result.answer
        proofs = result.proofs
    
    return {
        **state,
        "answer": answer,
        "proofs": proofs,
        "math_answer_payload": math_answer_payload,
        "answer_attempts": state["answer_attempts"] + 1,
    }


def node_validate(state: CardGenState) -> CardGenState:
    """Validate the answer for groundedness."""
    if state.get("card_route") == "math_calculation":
        validation = _validate_math_answer(
            question=state["question"] or "",
            answer=state["answer"] or "",
            proofs=state["proofs"] or [],
        )
    else:
        validation = _validate_answer(
            question=state["question"],
            answer=state["answer"],
            proofs=state["proofs"] or [],
        )
    score = float(validation.get("score", 0.5))
    return {
        **state,
        "validation_score": score,
        "grounding_validation_score": score,
        "validation_critique": str(validation.get("critique", "")),
    }


def node_verify_math(state: CardGenState) -> CardGenState:
    """Run deterministic math verification for math calculation cards."""
    if state.get("card_route") != "math_calculation":
        return {
            **state,
            "verification_result": {
                "status": "not_applicable",
                "method": "none",
                "confidence": 1.0,
                "checked_final_answer": False,
                "checked_steps": False,
                "details": {},
            },
            "math_validation_passed": True,
        }
    question_payload = dict(state.get("math_question_payload") or {})
    answer_payload = dict(state.get("math_answer_payload") or {})
    solver_payload = state.get("math_solver_payload") or {}
    if question_payload.get("compound"):
        # Compound path: confirm every step is solvable and the teacher's stated final
        # answer matches the CAS-canonical answer of the designated final step.
        declared = (
            answer_payload.get("final_answer")
            or question_payload.get("expected_final_answer")
        )
        result = verify_compound_solution(question_payload, declared_final_answer=declared)
    else:
        solver_target = solver_payload.get("verification_target") if isinstance(solver_payload, dict) else None
        if isinstance(solver_target, dict) and solver_target:
            question_payload["verification_target"] = solver_target
            answer_payload["verification_target"] = solver_target
        result = verify_math_solution(question_payload, answer_payload)
    threshold = float(state.get("validation_threshold", DEFAULT_CONFIG["math_validation_threshold"]))
    validation_ok = (state.get("validation_score") or 0.0) >= threshold
    math_ok = result.status == "verified"
    return {
        **state,
        "verification_result": result.to_info(),
        "math_validation_passed": math_ok,
        "math_validation_fail_cycles": (
            0
            if validation_ok and math_ok
            else int(state.get("math_validation_fail_cycles") or 0) + 1
        ),
    }


def node_strengthen(state: CardGenState) -> CardGenState:
    """Strengthen retrieval parameters for retry."""
    next_k = state["k"] + state["strengthen_k_delta"]
    next_min_score = max(0.2, state["min_score"] - state["strengthen_min_score_delta"])
    logger.info(
        "Strengthening retrieval after validation failure: route=%s topic_id=%s "
        "answer_attempts=%s/%s validation_score=%.2f threshold=%.2f "
        "math_status=%s next_k=%s next_min_score=%.2f",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("answer_attempts"),
        state.get("max_answer_attempts"),
        float(state.get("validation_score") or 0.0),
        float(state.get("validation_threshold") or 0.0),
        (state.get("verification_result") or {}).get("status"),
        next_k,
        next_min_score,
    )
    return {
        **state,
        "k": next_k,
        "min_score": next_min_score,
    }


def _is_math_question_failure(state: CardGenState) -> bool:
    if state.get("card_route") != "math_calculation":
        return False
    if not state.get("question_generation_failed"):
        return False
    reason = str(state.get("question_failure_reason") or "").lower()
    return any(
        token in reason
        for token in (
            "insufficient evidence",
            "invalid_spec",
            "unsupported",
            "parse_error",
            "could not solve generated question",
            "math problem spec",
            "math question generation returned no question",
        )
    )


def node_adapt_math_question_failure(state: CardGenState) -> CardGenState:
    """Lower math difficulty, then fall back to conceptual math if calculation still starves."""
    if state.get("card_route") != "math_calculation":
        return state

    if int(state.get("difficulty") or 1) > 1:
        difficulty = clamp_difficulty("tag", int(state["difficulty"]) - 1)
        level = get_level("tag", difficulty)
        logger.info(
            "Adapting math calculation card after repeated failure: topic_id=%s "
            "difficulty=%s next_difficulty=%s validation_score=%.2f math_status=%s",
            state.get("topic_id"),
            state.get("difficulty"),
            difficulty,
            float(state.get("validation_score") or 0.0),
            (state.get("verification_result") or {}).get("status"),
        )
        return {
            **state,
            "difficulty": difficulty,
            "difficulty_level_name": level.name,
            "question": None,
            "question_embedding": None,
            "question_id": None,
            "is_unique": False,
            "answer": None,
            "proofs": None,
            "validation_score": None,
            "validation_critique": None,
            "grounding_validation_score": None,
            "verification_result": {},
            "math_validation_passed": False,
            "question_generation_failed": False,
            "question_failure_reason": None,
            "math_question_payload": {},
            "math_solver_payload": {},
            "math_answer_payload": {},
            "question_attempts": 0,
            "answer_attempts": 0,
            "math_validation_fail_cycles": 0,
            "k": state.get("initial_k", DEFAULT_CONFIG["initial_k"]),
            "min_score": state.get("initial_min_score", DEFAULT_CONFIG["initial_min_score"]),
        }

    framework = framework_for_route("default")
    difficulty = clamp_difficulty(framework, state.get("difficulty") or 1)
    level = get_level(framework, difficulty)
    route_metadata = dict(state.get("route_metadata") or {})
    route_metadata.update(
        {
            "card_route": "default",
            "subject_type": "math",
            "math_kind": "conceptual",
            "reason": "Calculation generation repeatedly lacked solvable evidence; falling back to conceptual math.",
            "adapted_from": "math_calculation",
        }
    )
    logger.info(
        "Falling back from math calculation to conceptual math: topic_id=%s "
        "validation_score=%.2f math_status=%s",
        state.get("topic_id"),
        float(state.get("validation_score") or 0.0),
        (state.get("verification_result") or {}).get("status"),
    )
    return {
        **state,
        "card_route": "default",
        "route_metadata": route_metadata,
        "difficulty_framework": framework,
        "difficulty": difficulty,
        "difficulty_level_name": level.name,
        "validation_threshold": DEFAULT_CONFIG["validation_threshold"],
        "max_answer_attempts": DEFAULT_CONFIG["max_answer_attempts"],
        "max_full_restarts": DEFAULT_CONFIG["max_full_restarts"],
        "question": None,
        "question_embedding": None,
        "question_id": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "grounding_validation_score": None,
        "verification_result": {},
        "math_validation_passed": False,
        "question_generation_failed": False,
        "question_failure_reason": None,
        "math_question_payload": {},
        "math_solver_payload": {},
        "math_answer_payload": {},
        "question_attempts": 0,
        "answer_attempts": 0,
        "math_validation_fail_cycles": 0,
        "k": state.get("initial_k", DEFAULT_CONFIG["initial_k"]),
        "min_score": state.get("initial_min_score", DEFAULT_CONFIG["initial_min_score"]),
    }


def node_store_card(state: CardGenState) -> CardGenState:
    """Store the validated card in the database."""
    card_id = uuid.uuid4().hex[:16]
    
    framework = state.get("difficulty_framework") or framework_for_route(state.get("card_route"))
    difficulty_info = card_info_for_difficulty(framework, state["difficulty"])
    info: Dict[str, Any] = {
        "question_id": state.get("question_id"),
        "topic_label": state["topic_label"],
        "user_id": state["user_id"],
        "validation_score": state["validation_score"],
        "validation_critique": state.get("validation_critique"),
        "grounding_validation_score": state.get("grounding_validation_score"),
        "card_type": state["card_type"],
        "card_route": state.get("card_route") or "default",
        "subject_type": (state.get("route_metadata") or {}).get("subject_type", "general"),
        "math_kind": (state.get("route_metadata") or {}).get("math_kind", "none"),
        "routing": state.get("route_metadata") or default_route_decision().to_info(),
        "verification": state.get("verification_result") or {},
    }
    info.update(difficulty_info)
    if state.get("card_route") == "math_calculation":
        math_question = state.get("math_question_payload") or {}
        math_answer = state.get("math_answer_payload") or {}
        info.update(
            {
                "problem_type": math_question.get("problem_type"),
                "expected_final_answer": math_question.get("expected_final_answer") or math_answer.get("final_answer"),
                "math_question": math_question,
                "math_answer": math_answer,
                "math_solver": state.get("math_solver_payload") or {},
                "math_fingerprint": math_question.get("fingerprint"),
                "math_archetype": math_question.get("archetype"),
                "math_concepts": math_question.get("concepts_used") or [],
            }
        )

    get_repo().upsert_card(
        card_id=card_id,
        exam_id=state["exam_id"],
        topic_id=state["topic_id"],
        question=state["question"],
        answer=state["answer"],
        difficulty=state["difficulty"],
        card_type=state["card_type"],
        status="active",
        info=info,
    )
    
    # Store proofs. ponytail: reuse the `text` column for the clean LLM quote,
    # overwriting the raw chunk copy (raw still lives in the vector store); add a
    # dedicated `quote` column + migration only if raw display text is ever needed.
    from app.services.qa import summarize_evidence

    source_proofs = list(state["proofs"] or [])
    quotes = summarize_evidence(
        answer=state["answer"] or "",
        proof_texts=[p.text or "" for p in source_proofs],
        question=state["question"] or "",
    )
    proofs_data = [
        {
            "doc_id": p.doc_id,
            "page": p.page,
            "start": p.start,
            "end": p.end,
            "text": quotes[i] if i < len(quotes) else (p.text or ""),
            "score": float(p.score or 0.0),
        }
        for i, p in enumerate(source_proofs)
    ]
    get_repo().replace_card_proofs(card_id=card_id, proofs=proofs_data)
    
    # Create output card
    card = GeneratedCard(
        card_id=card_id,
        exam_id=state["exam_id"],
        topic_id=state["topic_id"],
        topic_label=state["topic_label"],
        question=state["question"],
        answer=state["answer"],
        difficulty=state["difficulty"],
        proofs=proofs_data,
    )
    
    return {**state, "card": card}


def node_full_restart(state: CardGenState) -> CardGenState:
    """Reset state for full restart with new question."""
    next_restart_count = state["full_restart_count"] + 1
    logger.warning(
        "Full card generation restart: route=%s topic_id=%s restart=%s/%s "
        "validation_score=%s threshold=%.2f math_status=%s question_failure=%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        next_restart_count,
        state.get("max_full_restarts"),
        state.get("validation_score"),
        float(state.get("validation_threshold") or 0.0),
        (state.get("verification_result") or {}).get("status"),
        state.get("question_failure_reason"),
    )
    return {
        **state,
        "question": None,
        "question_embedding": None,
        "question_id": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "grounding_validation_score": None,
        "verification_result": {},
        "math_validation_passed": False,
        "math_question_payload": {},
        "math_solver_payload": {},
        "math_answer_payload": {},
        "question_generation_failed": False,
        "question_failure_reason": None,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": next_restart_count,
        "math_validation_fail_cycles": 0,
        "k": state["initial_k"],
        "min_score": state["initial_min_score"],
    }


# ------------- Conditional Edges -------------


def decide_after_question_generation(state: CardGenState) -> str:
    """Retry or restart when route-specific question generation cannot produce a question."""
    if state.get("question") and not state.get("question_generation_failed"):
        return "embed_question"
    if state["question_attempts"] < state["max_question_attempts"]:
        return "generate_question"
    if _is_math_question_failure(state):
        logger.info(
            "Adapting math calculation after question/spec failure: topic_id=%s "
            "question_attempts=%s/%s reason=%s",
            state.get("topic_id"),
            state.get("question_attempts"),
            state.get("max_question_attempts"),
            state.get("question_failure_reason"),
        )
        return "adapt_question"
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "full_restart"
    logger.warning(
        "Ending card generation after question attempts exhausted: route=%s topic_id=%s "
        "question_attempts=%s max_question_attempts=%s restarts=%s/%s reason=%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("question_attempts"),
        state.get("max_question_attempts"),
        state.get("full_restart_count"),
        state.get("max_full_restarts"),
        state.get("question_failure_reason"),
    )
    return "end"


def decide_after_uniqueness(state: CardGenState) -> str:
    """Decide next step after uniqueness check."""
    if state["is_unique"]:
        return "save_question_id"
    
    if state["question_attempts"] < state["max_question_attempts"]:
        return "regenerate_question"
    
    # Max question retries reached, do full restart
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "full_restart"
    
    # Absolute max reached - give up
    logger.warning(
        "Ending card generation after uniqueness attempts exhausted: route=%s topic_id=%s "
        "question_attempts=%s restarts=%s/%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("question_attempts"),
        state.get("full_restart_count"),
        state.get("max_full_restarts"),
    )
    return "end"


def decide_after_validation(state: CardGenState) -> str:
    """Decide next step after answer validation."""
    threshold = state["validation_threshold"]
    validation_ok = (state["validation_score"] or 0.0) >= threshold
    math_ok = state.get("math_validation_passed", True)

    if validation_ok and math_ok:
        return "store_card"
    
    logger.info(
        "Answer validation failed: route=%s topic_id=%s answer_attempts=%s/%s "
        "validation_score=%.2f threshold=%.2f math_passed=%s math_status=%s "
        "critique=%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("answer_attempts"),
        state.get("max_answer_attempts"),
        float(state.get("validation_score") or 0.0),
        float(threshold),
        math_ok,
        (state.get("verification_result") or {}).get("status"),
        state.get("validation_critique"),
    )

    if state["answer_attempts"] < state["max_answer_attempts"]:
        return "strengthen"

    if (
        state.get("card_route") == "math_calculation"
        and int(state.get("math_validation_fail_cycles") or 0) >= 1
    ):
        return "adapt_question"
    
    # Max answer retries reached, do full restart
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "full_restart"
    
    # Absolute max reached - give up
    logger.warning(
        "Ending card generation after validation attempts exhausted: route=%s topic_id=%s "
        "answer_attempts=%s restarts=%s/%s validation_score=%.2f threshold=%.2f "
        "math_status=%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("answer_attempts"),
        state.get("full_restart_count"),
        state.get("max_full_restarts"),
        float(state.get("validation_score") or 0.0),
        float(threshold),
        (state.get("verification_result") or {}).get("status"),
    )
    return "end"


def decide_after_restart(state: CardGenState) -> str:
    """Decide next step after full restart."""
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "generate_question"
    logger.warning(
        "Ending card generation after full restart cap: route=%s topic_id=%s restarts=%s/%s",
        state.get("card_route") or "default",
        state.get("topic_id"),
        state.get("full_restart_count"),
        state.get("max_full_restarts"),
    )
    return "end"


def decide_after_save_question_id(state: CardGenState) -> str:
    """Decide whether to continue to answering or stop (for parallel starter pack)."""
    if state.get("stop_after_embedding", False):
        return "end"
    return "generate_answer"


# ------------- Graph Builder -------------


def build_card_graph():
    """Build the LangGraph for single card generation."""
    g = StateGraph(CardGenState)
    
    # Add nodes
    g.add_node("route_card", node_route_card)
    g.add_node("generate_question", node_generate_question)
    g.add_node("embed_question", node_embed_question)
    g.add_node("check_uniqueness", node_check_uniqueness)
    g.add_node("save_question_id", node_save_question_id)
    g.add_node("generate_answer", node_generate_answer)
    g.add_node("validate", node_validate)
    g.add_node("verify_math", node_verify_math)
    g.add_node("strengthen", node_strengthen)
    g.add_node("adapt_question", node_adapt_math_question_failure)
    g.add_node("store_card", node_store_card)
    g.add_node("commit_question_index", node_commit_question_index)
    g.add_node("full_restart", node_full_restart)
    
    # Set entry point
    g.set_entry_point("route_card")
    
    # Linear edges
    g.add_edge("route_card", "generate_question")
    g.add_edge("embed_question", "check_uniqueness")
    g.add_edge("generate_answer", "validate")
    g.add_edge("validate", "verify_math")
    g.add_edge("strengthen", "generate_answer")
    g.add_conditional_edges(
        "generate_question",
        decide_after_question_generation,
        {
            "embed_question": "embed_question",
            "generate_question": "generate_question",
            "adapt_question": "adapt_question",
            "full_restart": "full_restart",
            "end": END,
        },
    )
    g.add_edge("adapt_question", "generate_question")

    g.add_edge("store_card", "commit_question_index")
    g.add_edge("commit_question_index", END)
    
    # Conditional edge after save_question_id (for parallel starter pack)
    g.add_conditional_edges(
        "save_question_id",
        decide_after_save_question_id,
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )
    
    # Conditional edges after uniqueness check
    g.add_conditional_edges(
        "check_uniqueness",
        decide_after_uniqueness,
        {
            "save_question_id": "save_question_id",
            "regenerate_question": "generate_question",
            "full_restart": "full_restart",
            "end": END,
        },
    )
    
    # Conditional edges after validation
    g.add_conditional_edges(
        "verify_math",
        decide_after_validation,
        {
            "store_card": "store_card",
            "strengthen": "strengthen",
            "adapt_question": "adapt_question",
            "full_restart": "full_restart",
            "end": END,
        },
    )
    
    # Conditional edges after full restart
    g.add_conditional_edges(
        "full_restart",
        decide_after_restart,
        {
            "generate_question": "generate_question",
            "end": END,
        },
    )
    
    return g.compile()


# ------------- Entry Points -------------


def _run_question_phase(
    *,
    exam_id: str,
    batch_id: str,
    topic_id: str,
    topic_label: str,
    allowed_chunk_ids: List[str],
    context_pack: str,
    difficulty: int,
    card_type: str,
    user_id: str,
    store: VectorStore,
    batch_seen_questions: Tuple[BatchSeenQuestion, ...],
) -> Optional[CardGenState]:
    """
    Phase 1: Generate and dedupe a unique question.
    
    Returns the state with question + embedding stored, or None if failed.
    """
    student_memory = StudentMemoryService(repo=get_repo()).get_topic_memory(
        user_id=user_id,
        exam_id=exam_id,
        topic_id=topic_id,
    )
    initial_state: CardGenState = {
        "exam_id": exam_id,
        "user_id": user_id,
        "store_basepath": str(store.base),
        "batch_id": batch_id,
        "topic_id": topic_id,
        "topic_label": topic_label,
        "difficulty": difficulty,
        "card_route": "default",
        "card_route_override": None,
        "route_metadata": {},
        "difficulty_framework": "bloom",
        "difficulty_level_name": "",
        "card_type": card_type,
        "allowed_chunk_ids": allowed_chunk_ids,
        "context_pack": context_pack,
        "batch_seen_questions": batch_seen_questions,
        "student_memory": student_memory,
        "question": None,
        "question_generation_failed": False,
        "question_failure_reason": None,
        "math_question_payload": {},
        "math_solver_payload": {},
        "question_embedding": None,
        "question_id": None,
        "is_unique": False,
        "answer": None,
        "math_answer_payload": {},
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "grounding_validation_score": None,
        "verification_result": {},
        "math_validation_passed": False,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": 0,
        "math_validation_fail_cycles": 0,
        "max_question_attempts": DEFAULT_CONFIG["max_question_attempts"],
        "max_answer_attempts": DEFAULT_CONFIG["max_answer_attempts"],
        "max_full_restarts": DEFAULT_CONFIG["max_full_restarts"],
        "uniqueness_threshold": DEFAULT_CONFIG["uniqueness_threshold"],
        "uniqueness_top_k": DEFAULT_CONFIG["uniqueness_top_k"],
        "validation_threshold": DEFAULT_CONFIG["validation_threshold"],
        "commit_retry_attempts": DEFAULT_CONFIG["commit_retry_attempts"],
        "commit_retry_sleep_s": DEFAULT_CONFIG["commit_retry_sleep_s"],
        "initial_k": DEFAULT_CONFIG["initial_k"],
        "initial_min_score": DEFAULT_CONFIG["initial_min_score"],
        "strengthen_k_delta": DEFAULT_CONFIG["strengthen_k_delta"],
        "strengthen_min_score_delta": DEFAULT_CONFIG["strengthen_min_score_delta"],
        "k": DEFAULT_CONFIG["initial_k"],
        "min_score": DEFAULT_CONFIG["initial_min_score"],
        "stop_after_embedding": True,
        "card": None,
    }
    
    graph = build_card_graph()
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": int(DEFAULT_CONFIG["graph_recursion_limit"])},
    )
    
    # Check if we got a valid question + embedding
    if final_state.get("question") and final_state.get("question_id"):
        return final_state
    return None


def _run_answer_phase(state: CardGenState) -> Optional[GeneratedCard]:
    """
    Phase 2: Generate answer, validate, and store card.
    
    Includes retry logic for validation failures.
    Returns GeneratedCard on success, None on failure.
    """
    # Continue from where Phase 1 left off
    state = {**state, "stop_after_embedding": False}
    
    max_answer_attempts = state["max_answer_attempts"]
    max_full_restarts = state["max_full_restarts"]
    
    for _ in range(max_full_restarts):
        for attempt in range(max_answer_attempts):
            # Generate answer
            state = node_generate_answer(state)
            
            # Validate
            state = node_validate(state)
            state = node_verify_math(state)
            
            # Check validation
            if (
                state["validation_score"] >= state["validation_threshold"]
                and state.get("math_validation_passed", True)
            ):
                # Success! Store the card
                state = node_store_card(state)
                state = node_commit_question_index(state)
                return state.get("card")
            
            # Failed validation - strengthen and retry
            if attempt < max_answer_attempts - 1:
                state = node_strengthen(state)
                logger.debug(
                    "Answer validation failed (%.2f), strengthening retrieval (attempt %d)",
                    state["validation_score"], attempt + 1
                )
        
        # All answer attempts exhausted for this question
        # For starter pack, we don't do full restart (question is already unique)
        # Just give up on this card
        logger.warning(
            "Answer validation failed after %d attempts for question: %s",
            max_answer_attempts, state["question"][:50]
        )
        break
    
    return None


def generate_single_card(
    *,
    exam_id: str,
    topic_id: str,
    topic_label: str,
    allowed_chunk_ids: List[str],
    context_pack: str,
    difficulty: int = 1,
    card_type: str = "learning",
    user_id: str = "system",
    store: Optional[VectorStore] = None,
    stop_after_embedding: bool = False,
    card_route_override: Optional[str] = None,
) -> Optional[GeneratedCard]:
    """
    Generate a single card using the LangGraph flow.
    
    Args:
        stop_after_embedding: If True, stop after storing embedding (Phase 1 only).
                              Used for parallel starter pack generation.
    
    Retries until success or max restarts reached.
    Returns None only if all retries exhausted (rare edge case).
    """
    store = store or get_store()
    if store.vector_backend == "pinecone":
        store = store.for_namespace(pinecone_namespace(user_id=user_id, exam_id=exam_id))
    student_memory = StudentMemoryService(repo=get_repo()).get_topic_memory(
        user_id=user_id,
        exam_id=exam_id,
        topic_id=topic_id,
    )
    
    initial_state: CardGenState = {
        "exam_id": exam_id,
        "user_id": user_id,
        "store_basepath": str(store.base),
        "batch_id": uuid.uuid4().hex[:12],
        "topic_id": topic_id,
        "topic_label": topic_label,
        "difficulty": difficulty,
        "card_route": "default",
        "card_route_override": card_route_override,
        "route_metadata": {},
        "difficulty_framework": "bloom",
        "difficulty_level_name": "",
        "card_type": card_type,
        "allowed_chunk_ids": allowed_chunk_ids,
        "context_pack": context_pack,
        "batch_seen_questions": tuple(),
        "student_memory": student_memory,
        "question": None,
        "question_generation_failed": False,
        "question_failure_reason": None,
        "math_question_payload": {},
        "math_solver_payload": {},
        "question_embedding": None,
        "question_id": None,
        "is_unique": False,
        "answer": None,
        "math_answer_payload": {},
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "grounding_validation_score": None,
        "verification_result": {},
        "math_validation_passed": False,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": 0,
        "math_validation_fail_cycles": 0,
        "max_question_attempts": DEFAULT_CONFIG["max_question_attempts"],
        "max_answer_attempts": DEFAULT_CONFIG["max_answer_attempts"],
        "max_full_restarts": DEFAULT_CONFIG["max_full_restarts"],
        "uniqueness_threshold": DEFAULT_CONFIG["uniqueness_threshold"],
        "uniqueness_top_k": DEFAULT_CONFIG["uniqueness_top_k"],
        "validation_threshold": DEFAULT_CONFIG["validation_threshold"],
        "commit_retry_attempts": DEFAULT_CONFIG["commit_retry_attempts"],
        "commit_retry_sleep_s": DEFAULT_CONFIG["commit_retry_sleep_s"],
        "initial_k": DEFAULT_CONFIG["initial_k"],
        "initial_min_score": DEFAULT_CONFIG["initial_min_score"],
        "strengthen_k_delta": DEFAULT_CONFIG["strengthen_k_delta"],
        "strengthen_min_score_delta": DEFAULT_CONFIG["strengthen_min_score_delta"],
        "k": DEFAULT_CONFIG["initial_k"],
        "min_score": DEFAULT_CONFIG["initial_min_score"],
        "stop_after_embedding": stop_after_embedding,
        "card": None,
    }
    
    graph = build_card_graph()
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": int(DEFAULT_CONFIG["graph_recursion_limit"])},
    )
    
    return final_state.get("card")


def generate_starter_cards_v2(
    *,
    exam_id: str,
    user_id: str,
    n: int = 5,
    difficulty: int = 1,
    card_type: str = "learning",
    store: Optional[VectorStore] = None,
    max_workers: int = 5,
) -> List[GeneratedCard]:
    """
    Generate N starter cards across distinct topics.
    
    Phase 1 (Sequential): Generate + dedupe unique questions for N topics
    Phase 2 (Parallel): Answer + validate all questions simultaneously
    
    Returns list of generated cards (may be < N if topics insufficient).
    """
    store = store or get_store()
    if store.vector_backend == "pinecone":
        store = store.for_namespace(pinecone_namespace(user_id=user_id, exam_id=exam_id))

    # Pick top N topics
    picked = pick_starter_topics(exam_id=exam_id, repo=get_repo(), n=n)
    if not picked:
        logger.warning("No topics found for exam %s", exam_id)
        return []
    
    # Prepare context packs for each topic
    topic_contexts: Dict[str, Dict[str, Any]] = {}
    for topic_id, topic_label in picked:
        allowed_chunk_ids = get_repo().list_chunk_ids_for_topic(topic_id=topic_id)
        if not allowed_chunk_ids:
            continue
        centroid = get_repo().get_topic_vector(topic_id=topic_id)
        context_pack = build_diverse_chunk_pack(
            store=store,
            chunk_ids=allowed_chunk_ids,
            centroid=centroid,
        )
        if not context_pack:
            continue
        topic_contexts[topic_id] = {
            "topic_label": topic_label,
            "allowed_chunk_ids": allowed_chunk_ids,
            "context_pack": context_pack,
        }
    
    if not topic_contexts:
        logger.warning("No valid topic contexts for exam %s", exam_id)
        return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Sequential question generation + deduplication
    # Must be sequential so each uniqueness check sees all previous questions
    # ═══════════════════════════════════════════════════════════════════════
    
    question_states: List[CardGenState] = []
    batch_id = uuid.uuid4().hex[:12]
    batch_seen_master: List[BatchSeenQuestion] = []
    next_created_seq = 1
    
    for topic_id, ctx in topic_contexts.items():
        if len(question_states) >= n:
            break
        
        state = _run_question_phase(
            exam_id=exam_id,
            batch_id=batch_id,
            topic_id=topic_id,
            topic_label=ctx["topic_label"],
            allowed_chunk_ids=ctx["allowed_chunk_ids"],
            context_pack=ctx["context_pack"],
            difficulty=difficulty,
            card_type=card_type,
            user_id=user_id,
            store=store,
            batch_seen_questions=tuple(batch_seen_master),
        )
        
        if state:
            question_states.append(state)
            emb = state.get("question_embedding")
            qid = state.get("question_id")
            if emb is not None and qid:
                math_payload = state.get("math_question_payload") or {}
                batch_seen_master.append(
                    BatchSeenQuestion(
                        question_id=qid,
                        topic_id=state["topic_id"],
                        question_text=state["question"] or "",
                        difficulty=int(state["difficulty"]),
                        embedding=emb,
                        created_seq=next_created_seq,
                        fingerprint=str(math_payload.get("fingerprint") or ""),
                        archetype=str(math_payload.get("archetype") or ""),
                    )
                )
                next_created_seq += 1
            logger.info(
                "Phase 1: Generated unique question for topic '%s': %s",
                ctx["topic_label"], state["question"][:50]
            )
    
    if not question_states:
        logger.warning("Phase 1: No unique questions generated for exam %s", exam_id)
        return []
    
    logger.info(
        "Phase 1 complete: %d unique questions ready for answering",
        len(question_states)
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Parallel answer generation + validation
    # Each answer is independent, so we can parallelize
    # ═══════════════════════════════════════════════════════════════════════
    
    cards: List[GeneratedCard] = []
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(question_states))) as executor:
        future_to_state = {
            executor.submit(_run_answer_phase, state): state
            for state in question_states
        }
        
        for future in as_completed(future_to_state):
            state = future_to_state[future]
            try:
                card = future.result()
                if card:
                    cards.append(card)
                    logger.info(
                        "Phase 2: Generated card %s for topic '%s'",
                        card.card_id, card.topic_label
                    )
                else:
                    logger.warning(
                        "Phase 2: Failed to generate answer for question: %s",
                        state["question"][:50]
                    )
            except Exception as e:
                logger.error(
                    "Phase 2: Exception generating answer for '%s': %s",
                    state["topic_label"], str(e)
                )
    
    logger.info(
        "Phase 2 complete: %d cards generated out of %d questions",
        len(cards), len(question_states)
    )
    
    return cards


# For backward compatibility - alias to old name pattern
def node_propose_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility stub.
    The old doc-level question proposer is deprecated.
    Use generate_starter_cards_v2() instead.
    """
    logger.warning(
        "node_propose_questions is deprecated. Use generate_starter_cards_v2() instead."
    )
    return state


