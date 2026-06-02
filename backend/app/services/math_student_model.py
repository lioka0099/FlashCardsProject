from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.services.difficulty_frameworks import get_level
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create


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
    ) -> MathQuestionResult:
        level = get_level("tag", difficulty)
        mem = memory or {}
        route = route_metadata or {}
        problem_hints = route.get("problem_types") or []
        known = [str(x) for x in mem.get("known_facts", [])[:8]]
        misconceptions = [str(x) for x in mem.get("misconceptions", [])[:6]]

        sys_prompt = (
            "You write math calculation flashcard questions only.\n"
            "Use ONLY the provided excerpts for formulas, values, variables, and procedures.\n"
            "Reject conceptual-only questions. Return JSON only."
        )
        user_prompt = (
            f"TOPIC:\n{topic_label}\n\n"
            f"TAG LEVEL: {level.level} - {level.name}\n"
            f"TAG INSTRUCTION: {level.instruction}\n"
            f"PROMPT HINT: {level.prompt_hint}\n\n"
            f"ROUTE PROBLEM TYPE HINTS: {problem_hints or '(none)'}\n\n"
            "STUDENT KNOWN FACTS:\n- " + ("\n- ".join(known) if known else "(none)") + "\n\n"
            "STUDENT MISCONCEPTIONS:\n- " + ("\n- ".join(misconceptions) if misconceptions else "(none)") + "\n\n"
            f"EVIDENCE EXCERPTS:\n{context_pack}\n\n"
            "Create ONE calculation-based flashcard question.\n"
            "Rules:\n"
            "- The question must require actual mathematical work.\n"
            "- Include all givens needed to solve it.\n"
            "- Use only formulas, values, variables, and procedures grounded in the excerpts.\n"
            "- Do not ask conceptual questions like 'What is a derivative?' or 'Explain slope'.\n"
            "- Prefer problem types SymPy can verify: arithmetic, simplify, equivalence, equation, system, derivative, integral.\n"
            "- If the excerpts do not support a grounded calculation problem, return {\"question\": \"\", \"reason\": \"insufficient evidence\"}.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "question": "...",\n'
            '  "problem_type": "derivative|integral|equation|system|simplify|equivalence|arithmetic|substitution",\n'
            '  "givens": ["..."],\n'
            '  "source_rules": ["..."],\n'
            '  "expected_operation": "...",\n'
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
            "Use Python/SymPy-compatible syntax in verification_target where possible."
        )

        resp = chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        payload = _safe_json_load(resp.choices[0].message.content or "{}")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError(str(payload.get("reason") or "Math question generation returned no question."))
        payload["question"] = question
        payload.setdefault("tag_level", level.level)
        payload.setdefault("tag_level_name", level.name)
        return MathQuestionResult(question=question, payload=payload)

