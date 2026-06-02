from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from app.api.schemas import ProofSpan
from app.data.vector_store import VectorStore
from app.services.llm import CHAT_MODEL, chat_completions_create
from app.services.qa import _build_context, _condense_proofs, _select_display_proofs
from app.services.retrieval import retrieve_with_proofs


@dataclass
class MathTeacherModelResult:
    answer: str
    proofs: List[ProofSpan]
    payload: Dict[str, Any]


def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"answer": raw}


@dataclass
class MathTeacherModelService:
    """Grounded teacher-side worked-solution generation for math calculation cards."""

    model: str = CHAT_MODEL

    def generate_answer(
        self,
        *,
        question: str,
        question_payload: Dict[str, Any],
        k: int,
        min_score: float,
        store: VectorStore,
        allowed_chunk_ids: list[str],
    ) -> MathTeacherModelResult:
        proofs_all = retrieve_with_proofs(
            question,
            k=max(k, 8),
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        proofs_all = sorted(
            proofs_all,
            key=lambda p: (p.score if p.score is not None else 0.0),
            reverse=True,
        )
        proofs = [p for p in proofs_all if p.score is None or p.score >= min_score]
        if not proofs:
            proofs = proofs_all[: max(4, min(len(proofs_all), k))]
        proofs = proofs[: max(6, min(len(proofs), k + 2))]
        prompt_proofs, proof_artifacts = _condense_proofs(question, "", proofs)
        context = _build_context(prompt_proofs, max_chars=12000)

        sys_prompt = (
            "You write worked solutions for math calculation flashcards.\n"
            "Use ONLY the provided sources for the rule, formula, values, and procedure.\n"
            "Return JSON only."
        )
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"QUESTION STRUCTURE:\n{json.dumps(question_payload, ensure_ascii=True)}\n\n"
            f"SOURCES:\n{context}\n\n"
            "Write a concise worked solution.\n"
            "Rules:\n"
            "- Use only the provided sources and the givens in the question.\n"
            "- Include step-by-step work.\n"
            "- Include a clear final answer.\n"
            "- Keep the solution flashcard-friendly, not a long lecture.\n"
            "- Use SymPy-compatible syntax for final_answer and verification_target fields where possible.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "answer": "Step 1: ...\\nStep 2: ...\\n\\nFinal answer: ...",\n'
            '  "final_answer": "...",\n'
            '  "solution_steps": ["...", "..."],\n'
            '  "verification_target": {\n'
            '    "kind": "...",\n'
            '    "expression": "...",\n'
            '    "equation": "...",\n'
            '    "equations": ["..."],\n'
            '    "variable": "x",\n'
            '    "variables": ["x", "y"],\n'
            '    "expected": "..."\n'
            "  }\n"
            "}\n"
        )

        resp = chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        payload = _safe_json_load(resp.choices[0].message.content or "{}")
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            final = str(payload.get("final_answer") or "").strip()
            answer = f"Final answer: {final}" if final else ""
        if not answer:
            raise ValueError("Math answer generation returned no answer.")

        final_proofs, _ = _condense_proofs(question, answer, proofs, artifacts=proof_artifacts)
        display_proofs = _select_display_proofs(question, answer, final_proofs)
        payload["answer"] = answer
        payload.setdefault("verification_target", question_payload.get("verification_target") or {})
        return MathTeacherModelResult(answer=answer, proofs=display_proofs, payload=payload)

