from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.services.difficulty_frameworks import get_level
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create
from app.services.math_problem_spec import normalize_math_problem_spec, render_math_question


@dataclass
class MathQuestionResult:
    question: str
    payload: Dict[str, Any]


def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class MathStudentModelService:
    """Generates grounded math calculation flashcard questions."""

    model: str = CHAT_MODEL_FAST

    def generate_question(
        self,
        *,
        topic_label: str,
        context_pack: str,
        difficulty: int,
        memory: Optional[Dict[str, Any]] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
        avoid_questions: Optional[list[str]] = None,
        attempt_no: int = 1,
    ) -> MathQuestionResult:
        level = get_level("tag", difficulty)
        mem = memory or {}
        route = route_metadata or {}
        problem_hints = route.get("problem_types") or []
        known = [str(x) for x in mem.get("known_facts", [])[:8]]
        misconceptions = [str(x) for x in mem.get("misconceptions", [])[:6]]
        blocked = [str(x).strip() for x in (avoid_questions or []) if str(x or "").strip()][:20]
        retry_hint = (
            "This is a retry attempt. You MUST produce a different problem expression than blocked examples."
            if int(attempt_no) > 1
            else "Produce a fresh problem."
        )

        sys_prompt = (
            "You write strict JSON specs for math calculation flashcards.\n"
            "Use the provided excerpts for formulas, relationships, and procedures.\n"
            "You may introduce simple numeric givens only when a source-backed rule supports them.\n"
            "Return JSON only."
        )
        user_prompt = (
            f"TOPIC:\n{topic_label}\n\n"
            f"TAG LEVEL: {level.level} - {level.name}\n"
            f"TAG INSTRUCTION: {level.instruction}\n"
            f"PROMPT HINT: {level.prompt_hint}\n\n"
            f"ROUTE PROBLEM TYPE HINTS: {problem_hints or '(none)'}\n\n"
            f"ATTEMPT: {int(attempt_no)}\n"
            f"RETRY GUIDANCE: {retry_hint}\n\n"
            "STUDENT KNOWN FACTS:\n- " + ("\n- ".join(known) if known else "(none)") + "\n\n"
            "STUDENT MISCONCEPTIONS:\n- " + ("\n- ".join(misconceptions) if misconceptions else "(none)") + "\n\n"
            "BLOCKED PREVIOUS QUESTIONS (DO NOT REUSE OR PARAPHRASE THESE):\n- "
            + ("\n- ".join(blocked) if blocked else "(none)")
            + "\n\n"
            f"EVIDENCE EXCERPTS:\n{context_pack}\n\n"
            "Create ONE SymPy-solvable calculation problem spec.\n"
            "Rules:\n"
            "- Return status='ready' only if the spec is clean and solvable.\n"
            "- Return status='insufficient_evidence' with a reason if no relevant source-backed rule/procedure is present.\n"
            "- Do not write a natural-language question; the backend renders it from the spec.\n"
            "- Ground formulas, relationships, and procedures in the excerpts.\n"
            "- Do not invent arithmetic solely from incidental numbers in the excerpts.\n"
            "- You may invent small integer or simple fractional givens only when the excerpt teaches a math rule, formula, relationship, or procedure.\n"
            "- Include one source_rule copied or tightly paraphrased from the excerpts.\n"
            "- Keep generated givens simple enough for a flashcard and do not introduce topics outside the excerpts.\n"
            "- Use only these kind values: arithmetic, substitution, simplify, equivalence, expand, factor, equation, system, derivative, integral.\n"
            "- Use Python/SymPy syntax only: ** for powers, * for multiplication, sqrt(x) for roots. Do not use LaTeX.\n"
            "- For derivative/integral specs include expression and variable.\n"
            "- For equation specs include exactly one equation string with exactly one '=' and a variable.\n"
            "- For system specs include equations and variables arrays.\n"
            "- For arithmetic/substitution specs include expression and optional substitutions object.\n"
            "- For simplify/equivalence/expand/factor specs include expression and optional expected.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "status": "ready|insufficient_evidence",\n'
            '  "reason": "short reason when insufficient_evidence",\n'
            '  "kind": "derivative|integral|equation|system|simplify|equivalence|expand|factor|arithmetic|substitution",\n'
            '  "source_rule": "source-backed rule or procedure",\n'
            '  "expression": "x**2 + 3*x",\n'
            '  "equation": "2*x + 3 = 7",\n'
            '  "equations": ["x + y = 5", "x - y = 1"],\n'
            '  "variable": "x",\n'
            '  "variables": ["x", "y"],\n'
            '  "substitutions": {"x": "3"},\n'
            '  "expected": "optional expected result"\n'
            "}"
        )

        temperature = 0.25 if int(attempt_no) <= 1 else min(0.55, 0.25 + 0.1 * (int(attempt_no) - 1))
        resp = chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        payload = _safe_json_load(resp.choices[0].message.content or "{}")
        normalized = normalize_math_problem_spec(payload, require_source_rule=True)
        if not normalized.ok:
            reason = normalized.reason or "Math problem spec was not usable."
            if normalized.status == "insufficient_evidence":
                raise ValueError(f"Insufficient evidence: {reason}")
            raise ValueError(f"{normalized.status}: {reason}")
        spec = normalized.spec
        question = render_math_question(spec)
        spec["question"] = question
        spec.setdefault("tag_level", level.level)
        spec.setdefault("tag_level_name", level.name)
        return MathQuestionResult(question=question, payload=spec)

