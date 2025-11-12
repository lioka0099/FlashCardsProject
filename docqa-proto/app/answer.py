from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI
from app.utils import getenv
from app.llm import CHAT_MODEL
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

    # 2) context
    context = _build_context(proofs)

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

    # 4) return both the answer and the proof objects for UI
    return AnswerWithCitations(answer=answer_text, proofs=proofs)
