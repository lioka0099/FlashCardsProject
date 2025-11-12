from dataclasses import dataclass
from typing import List
import re
import numpy as np
from openai import OpenAI
from app.utils import getenv
from app.llm import CHAT_MODEL, embed_texts
from app.api import retrieve_with_proofs, retrieve_with_proofs_for_doc
from app.models import ProofSpan

@dataclass
class AnswerWithCitations:
    answer: str
    proofs: List[ProofSpan]  # already includes page/start/end/text/score

def _format_citation(i: int, p: ProofSpan) -> str:
    # Simple inline citation format: [S{i}, p.{page}]
    return f"[S{i}, p.{p.page}]"

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

def _condense_proof_text(question: str, answer: str, proof: ProofSpan,
                         max_sentences: int = 2, max_chars: int = 550,
                         min_similarity: float = 0.1) -> ProofSpan:
    original = _clean_text(proof.text or "")
    if not original:
        return proof
    sentences = _split_sentences(original)
    if not sentences:
        return ProofSpan(
            doc_id=proof.doc_id, page=proof.page, start=proof.start,
            end=proof.end, text=original, score=proof.score
        )
    try:
        query = f"{question}\n{answer}".strip() or question
        qa_vec = embed_texts([query])[0]
        sent_vecs = embed_texts(sentences)
        qa_norm = np.linalg.norm(qa_vec)
        if qa_norm > 0:
            qa_vec = qa_vec / qa_norm
        sent_norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
        sent_vecs = sent_vecs / np.clip(sent_norms, 1e-12, None)
        sims = sent_vecs @ qa_vec
        order = np.argsort(-sims)
        selected_idx: List[int] = []
        for idx in order.tolist():
            if sims[idx] < min_similarity:
                break
            selected_idx.append(idx)
            if len(selected_idx) >= max_sentences:
                break
        if not selected_idx:
            selected_idx = order[:max_sentences].tolist()
        selected_idx = sorted(set(selected_idx))
        selected = [sentences[i] for i in selected_idx]
    except Exception:
        selected = sentences[:max_sentences]
    if not selected:
        selected = sentences[:max_sentences]
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
        if not sentence:
            continue
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

def _condense_proofs(question: str, answer: str, proofs: List[ProofSpan]) -> List[ProofSpan]:
    condensed = [_condense_proof_text(question, answer, p) for p in proofs]
    deduped: List[ProofSpan] = []
    seen = set()
    for p in condensed:
        key = (p.doc_id, p.page, p.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped

def generate_answer(question: str, k: int = 5, min_score: float = 0.35,
                    model: str = CHAT_MODEL, doc_id: str | None = None) -> AnswerWithCitations:
    """
    Retrieve top-k proofs, filter by score, then ask the LLM to answer using only those proofs.
    Returns the final answer text + the proofs used (for UI / logging).
    """
    # 1) retrieve
    if doc_id:
        proofs_all = retrieve_with_proofs_for_doc(question, doc_id=doc_id, k=k)
    else:
        proofs_all = retrieve_with_proofs(question, k=k)
    proofs = [p for p in proofs_all if p.score is None or p.score >= min_score]
    if not proofs:
        proofs = proofs_all[:2]  # fall back to something

    # 2) prepare concise snippets for prompting
    prompt_proofs = _condense_proofs(question, "", proofs)

    # 3) context
    context = _build_context(prompt_proofs)

    # 3) ask the model, but force ground-truth behavior
    client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
    sys_prompt = (
        "You are a careful assistant. Answer ONLY using the provided sources.\n"
        "If the answer is not contained in them, say you don't have enough information.\n"
        "Add an inline citation like [S1, p.3] AFTER every sentence that uses a source.\n"
        "Prefer precise wording and avoid speculation."
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES (S# refers to source order):\n{context}\n\n"
        "RULES:\n"
        "- Only use information found in SOURCES; do not invent.\n"
        "- Add inline citations [S#, p.#] after each sentence that uses a source.\n"
        "- If sources conflict or are unclear, say so and cite the conflicting sources.\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    answer_text = resp.choices[0].message.content.strip()

    final_proofs = _condense_proofs(question, answer_text, proofs)

    # 4) return both the answer and the proof objects for UI
    return AnswerWithCitations(answer=answer_text, proofs=final_proofs)
