from dataclasses import dataclass
from typing import List, Sequence, Optional
import json
import re
import numpy as np
from app.services.llm import CHAT_MODEL, CHAT_MODEL_FAST, chat_completions_create, embed_texts
from app.services.math_formatting import MATH_DISPLAY_INSTRUCTION
from app.services.retrieval import retrieve_with_proofs
from app.data.vector_store import VectorStore
from app.deps import get_store
from app.api.schemas import ProofSpan

@dataclass
class AnswerWithCitations:
    answer: str
    proofs: List[ProofSpan]
    score: Optional[float] = None
    critique: Optional[str] = None

@dataclass
class CondenseArtifacts:
    sentences_by_proof: List[List[str]]
    sentence_meta: List[tuple[int, int]]
    sentence_vectors: np.ndarray

def _build_context(proofs: List[ProofSpan], max_chars: int = 10000) -> str:
    """
    Concatenate top proof snippets into a single context block while keeping within a char budget.
    """
    buf, used = [], 0
    for i, p in enumerate(proofs, start=1):
        header = f"Source S{i} — doc={p.doc_id} page={p.page} score={p.score:.2f}\n"
        block = (header + p.text.strip() + "\n\n")
        if used + len(block) > max_chars:
            break
        buf.append(block)
        used += len(block)
    return "".join(buf).strip()

def _clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

_BULLET_RE = re.compile(r"[▪•●■◦‣·]+")

_VERBATIM_SYS_PROMPT = (
    "You extract a citation for a flashcard app. From PASSAGE, copy the exact "
    "sentence or contiguous span that best supports the ANSWER. Copy it "
    "word-for-word in the original order — do NOT paraphrase, summarize, reorder, "
    "or add words. You may drop bullet characters. Keep it to one sentence if "
    "possible.\n"
    'Respond as JSON: {"quote": "<verbatim text from the passage>"}.'
)


def _match_key(text: str) -> str:
    """Lowercase word sequence, punctuation/bullets stripped, for substring checks."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _display_clean(text: str) -> str:
    """Verbatim words, bullet glyphs dropped and whitespace collapsed."""
    return _clean_text(_BULLET_RE.sub(" ", text or ""))


def _fallback_verbatim(proof_text: str, query: str) -> str:
    """Deterministically pick the passage segment with the most query-word overlap."""
    segments = [
        _display_clean(seg)
        for seg in re.split(r"(?<=[.!?])\s+|[▪•●■◦‣]", proof_text or "")
    ]
    segments = [s for s in segments if s]
    if not segments:
        return _display_clean(proof_text)
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(segment: str) -> int:
        tokens = re.findall(r"[a-z0-9]+", segment.lower())
        return sum(1 for token in tokens if token in query_tokens)

    best = max(range(len(segments)), key=lambda i: (score(segments[i]), -i))
    return segments[best]


def _verbatim_quote_one(
    answer: str,
    question: str,
    proof_text: str,
    model: str,
) -> str:
    """
    A verbatim citation for one proof: the LLM copies the supporting span, then we
    verify its words actually appear (in order) in the passage before trusting it.
    On paraphrase/hallucination/failure, fall back to a deterministic verbatim
    segment so the shown text is always findable in the source.
    """
    if not _display_clean(proof_text):
        return ""
    query = f"{answer} {question}"
    user_prompt = (
        f"ANSWER:\n{answer}\n\nQUESTION:\n{question}\n\nPASSAGE:\n{proof_text}"
    )
    try:
        resp = chat_completions_create(
            model=model,
            messages=[
                {"role": "system", "content": _VERBATIM_SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        candidate = str(json.loads(resp.choices[0].message.content or "").get("quote", "")).strip()
        candidate_key = _match_key(candidate)
        if candidate_key and candidate_key in _match_key(proof_text):
            return _display_clean(candidate)
    except Exception:
        pass
    return _fallback_verbatim(proof_text, query)


def summarize_evidence(
    answer: str,
    proof_texts: List[str],
    question: str = "",
    model: str = CHAT_MODEL_FAST,
) -> List[str]:
    """
    Extract a short verbatim citation from each proof chunk — text that actually
    appears in the source (so the user can find it), with bullet glyphs and stray
    whitespace tidied. One LLM call per proof, each verified to be a real substring
    of its passage; deterministic fallback keeps the result verbatim on any failure.
    """
    return [
        _verbatim_quote_one(answer, question, text, model)
        for text in proof_texts
    ]


def _condense_proof_text(
    proof: ProofSpan,
    max_sentences: int = 2,
    max_chars: int = 550,
    min_similarity: float = 0.1,
    ranked_sentences: Optional[List[tuple[int, float]]] = None,
    pretokenized_sentences: Optional[List[str]] = None,
) -> ProofSpan:
    original = _clean_text(proof.text or "")
    if not original:
        return proof
    sentences = pretokenized_sentences or _split_sentences(original)
    if not sentences:
        return ProofSpan(
            doc_id=proof.doc_id,
            page=proof.page,
            start=proof.start,
            end=proof.end,
            text=original,
            score=proof.score,
        )
    selected_idx: List[int] = []
    if ranked_sentences:
        for idx, score in ranked_sentences:
            if score < min_similarity and selected_idx:
                break
            selected_idx.append(idx)
            if len(selected_idx) >= max_sentences:
                break
    if not selected_idx:
        if ranked_sentences:
            selected_idx = [idx for idx, _ in ranked_sentences[:max_sentences]]
        else:
            selected_idx = list(range(min(len(sentences), max_sentences)))
    selected_idx = sorted(set(idx for idx in selected_idx if 0 <= idx < len(sentences)))
    if not selected_idx:
        selected_idx = list[int](range(min(len(sentences), max_sentences)))
    selected = [sentences[i] for i in selected_idx]
    filtered: List[str] = []
    for sent in selected:
        sentence = _clean_text(sent)
        if not sentence:
            continue
        if re.match(r'^"?[A-Z]{1,3}:', sentence):
            # skip role prefixes like "S:" or "T:"
            continue
        if len(sentence) < 40 and len(selected) > 1:
            continue
        filtered.append(sentence)
    if filtered:
        selected = filtered
    condensed_parts: List[str] = []
    total_len = 0
    for sentence in selected:
        prospective = total_len + len(sentence) + (1 if condensed_parts else 0)
        if condensed_parts and prospective > max_chars:
            break
        condensed_parts.append(sentence)
        total_len = prospective
    if not condensed_parts:
        condensed_parts = [selected[0].strip()]
    condensed = " ".join(condensed_parts).strip()
    if not condensed:
        condensed = original if len(original) <= max_chars else original[:max_chars]
    return ProofSpan(
        doc_id=proof.doc_id, page=proof.page, start=proof.start,
        end=proof.end, text=condensed, score=proof.score
    )

def _condense_proofs(
    question: str,
    answer: str,
    proofs: List[ProofSpan],
    *,
    artifacts: Optional[CondenseArtifacts] = None,
) -> tuple[List[ProofSpan], Optional[CondenseArtifacts]]:
    if not proofs:
        return [], artifacts
    use_artifacts = (
        artifacts is not None
        and len(artifacts.sentences_by_proof) == len(proofs)
        and len(artifacts.sentence_meta) == len(artifacts.sentence_vectors)
    )
    if use_artifacts:
        sentences_by_proof = artifacts.sentences_by_proof
        flat_sentences: List[tuple[int, int, str]] = []
        for (proof_idx, sent_idx) in artifacts.sentence_meta:
            if (
                0 <= proof_idx < len(sentences_by_proof)
                and 0 <= sent_idx < len(sentences_by_proof[proof_idx])
            ):
                flat_sentences.append(
                    (proof_idx, sent_idx, sentences_by_proof[proof_idx][sent_idx])
                )
        sent_vecs = artifacts.sentence_vectors
        artifact_out = artifacts
    else:
        sentences_by_proof = []
        flat_sentences = []
        for proof_idx, proof in enumerate(proofs):
            cleaned = _clean_text(proof.text or "")
            sentences = _split_sentences(cleaned) if cleaned else []
            sentences_by_proof.append(sentences)
            for sent_idx, sentence in enumerate(sentences):
                if sentence:
                    flat_sentences.append((proof_idx, sent_idx, sentence))
        if flat_sentences:
            try:
                sent_vecs = embed_texts([entry[2] for entry in flat_sentences])
                sent_norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
                sent_vecs = sent_vecs / np.clip(sent_norms, 1e-12, None)
            except Exception:
                sent_vecs = np.zeros((0, 1), dtype="float32")
        else:
            sent_vecs = np.zeros((0, 1), dtype="float32")
        artifact_out = (
            CondenseArtifacts(
                sentences_by_proof=sentences_by_proof,
                sentence_meta=[(proof_idx, sent_idx) for proof_idx, sent_idx, _ in flat_sentences],
                sentence_vectors=sent_vecs,
            )
            if flat_sentences and sent_vecs.size
            else None
        )
    ranked_map: dict[int, List[tuple[int, float]]] = {}
    if flat_sentences and sent_vecs.size:
        try:
            query = f"{question}\n{answer}".strip() or question
            qa_vec = embed_texts([query])[0]
            qa_norm = np.linalg.norm(qa_vec)
            qa_vec = qa_vec / np.clip(qa_norm, 1e-12, None)
            sims = (sent_vecs @ qa_vec).tolist()
            for (proof_idx, sent_idx, _), score in zip(flat_sentences, sims):
                ranked_map.setdefault(proof_idx, []).append((sent_idx, score))
            for proof_idx in ranked_map:
                ranked_map[proof_idx].sort(key=lambda x: x[1], reverse=True)
        except Exception:
            ranked_map = {}
    condensed = []
    for proof_idx, proof in enumerate(proofs):
        ranked = ranked_map.get(proof_idx)
        sentences = sentences_by_proof[proof_idx]
        condensed.append(
            _condense_proof_text(
                proof,
                ranked_sentences=ranked,
                pretokenized_sentences=sentences,
            )
        )
    deduped: List[ProofSpan] = []
    seen = set()
    for p in condensed:
        key = (p.doc_id, p.page, p.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped, artifact_out

def _select_display_proofs(
    question: str,
    answer: str,
    proofs: List[ProofSpan],
    max_items: int = 3,
) -> List[ProofSpan]:
    """
    Rank condensed proofs and keep only the most relevant snippets for presentation.
    """
    if not proofs or len(proofs) <= max_items:
        return proofs
    entries: List[tuple[ProofSpan, str]] = []
    for proof in proofs:
        cleaned = _clean_text(proof.text or "")
        if cleaned:
            entries.append((proof, cleaned))
    if not entries:
        return proofs[:max_items]
    try:
        key = (answer or question).strip() or question
        key_vec = embed_texts([key])[0]
        key_norm = np.linalg.norm(key_vec)
        key_vec = key_vec / np.clip(key_norm, 1e-12, None)
        text_vecs = embed_texts([text for _, text in entries])
        text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
        text_vecs = text_vecs / np.clip(text_norms, 1e-12, None)
        sims = (text_vecs @ key_vec).tolist()
    except Exception:
        sims = [0.0] * len(entries)
    scored: List[tuple[float, int, ProofSpan]] = []
    for idx, ((proof, _), sim) in enumerate(zip(entries, sims)):
        base_score = proof.score if proof.score is not None else 0.0
        base_score = max(min(base_score, 1.0), -1.0)
        combined = 0.85 * sim + 0.15 * base_score
        scored.append((combined, idx, proof))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = [proof for _, _, proof in scored[:max_items]]
    return top or proofs[:max_items]

def generate_answer(
    question: str,
    k: int = 8,
    min_score: float = 0.4,
    model: str = CHAT_MODEL,
    allowed_doc_ids: Optional[Sequence[str]] = None,
    allowed_chunk_ids: Optional[Sequence[str]] = None,
    store: Optional[VectorStore] = None,
    *,
    prefetched_pool: Optional[Sequence[ProofSpan]] = None,
    pool_k: Optional[int] = None,
    query_vec: Optional[np.ndarray] = None,
) -> AnswerWithCitations:
    """
    Retrieve top-k proofs, filter by score, then ask the LLM to answer using only those proofs.
    Returns the final answer text + the proofs used (for UI / logging).
    """
    # 1) retrieve
    allowed_set = {doc for doc in (allowed_doc_ids or []) if doc}
    store_obj = store or get_store()
    target_pool = max(pool_k or k, k)
    if prefetched_pool is not None:
        proofs_all = list(prefetched_pool)
        if len(proofs_all) < target_pool:
            prefetched_pool = None  # force fresh retrieval below
    if prefetched_pool is None:
        allowed_list = list(allowed_set) if allowed_set else None
        proofs_all = retrieve_with_proofs(
            question,
            k=target_pool,
            store=store_obj,
            allowed_doc_ids=allowed_list,
            allowed_chunk_ids=allowed_chunk_ids,
            query_vec=query_vec,
        )
    proofs_all = sorted(
        proofs_all,
        key=lambda p: (p.score if p.score is not None else 0.0),
        reverse=True,
    )
    proofs = [p for p in proofs_all if p.score is None or p.score >= min_score]
    if allowed_set:
        proofs = [p for p in proofs if p.doc_id in allowed_set]
    if not proofs:
        fallback_pool = [p for p in proofs_all if not allowed_set or p.doc_id in allowed_set]
        proofs = fallback_pool[: max(4, len(fallback_pool))]
    proofs = proofs[: max(6, min(len(proofs), k + 2))]

    # 2) prepare concise snippets for prompting
    prompt_proofs, proof_artifacts = _condense_proofs(question, "", proofs)

    # 3) context
    context = _build_context(prompt_proofs)

    # 3) ask the model, but force ground-truth behavior
    sys_prompt = (
        "You are a careful assistant that writes flashcard-friendly answers.\n"
        "Answer ONLY using the provided sources.\n"
        "If the answer is not contained in them, say you don't have enough information.\n"
        "Do NOT include inline citations or source identifiers in the answer; evidence is handled separately.\n"
        "Never cite a source that wasn't provided.\n"
        "Respond as JSON with keys: answer (string), score (0-1 float), critique (string). "
        "Use the critique to note any coverage gaps or uncertainties."
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES (S# refers to source order):\n{context}\n\n"
        "RULES:\n"
        "- Only use information found in SOURCES; do not invent.\n"
        "- Do not include citations, source identifiers, or brackets like [S1, p.3] in the answer text.\n"
        "- If you lack evidence, state that explicitly and mention that the sources do not cover it.\n"
        "- Flashcard style: respond in 1–2 short sentences totaling ≤35 words while staying precise.\n"
        f"- {MATH_DISPLAY_INSTRUCTION}\n"
        "- Output valid JSON only."
    )

    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"answer": raw}
    answer_text = str(payload.get("answer", "")).strip() or raw.strip()
    raw_score = payload.get("score")
    try:
        score_val = float(raw_score) if raw_score is not None else None
    except Exception:
        score_val = None
    critique = str(payload.get("critique", "")).strip()

    final_proofs, _ = _condense_proofs(
        question, answer_text, proofs, artifacts=proof_artifacts
    )
    display_proofs = _select_display_proofs(question, answer_text, final_proofs)

    # 4) return both the answer and the proof objects for UI - UI/logging
    return AnswerWithCitations(
        answer=answer_text,
        proofs=display_proofs,
        score=score_val,
        critique=critique or None,
    )
