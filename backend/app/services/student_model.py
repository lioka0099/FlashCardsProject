from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.services.cards import DIFFICULTY_LEVELS, generate_question_at_difficulty
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create
from app.services.math_formatting import MATH_DISPLAY_INSTRUCTION


@dataclass
class StudentModelService:
    """
    Generates learner-perspective questions from bounded student memory.
    """

    model: str = CHAT_MODEL_FAST

    def generate_question(
        self,
        *,
        topic_label: str,
        context_pack: str,
        difficulty: int,
        memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        mem = memory or {}
        level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS[1])
        known = [str(x) for x in mem.get("known_facts", [])[:12]]
        misconceptions = [str(x) for x in mem.get("misconceptions", [])[:8]]
        confidence = float(mem.get("confidence", 0.5))

        sys_prompt = (
            "You are a curious student asking a flashcard question.\n"
            "Ask one useful question that helps learn the next concept.\n"
            "Respect the target difficulty and the student's current knowledge limits.\n"
            "Return JSON only."
        )
        user_prompt = (
            f"TOPIC:\n{topic_label}\n\n"
            f"TARGET DIFFICULTY: {level['name']}\n"
            f"DIFFICULTY INSTRUCTION: {level['instruction']}\n\n"
            f"STUDENT CONFIDENCE: {confidence:.2f}\n"
            f"STUDENT KNOWN FACTS:\n- " + ("\n- ".join(known) if known else "(none)") + "\n\n"
            f"STUDENT MISCONCEPTIONS:\n- " + ("\n- ".join(misconceptions) if misconceptions else "(none)") + "\n\n"
            f"EVIDENCE EXCERPTS:\n{context_pack}\n\n"
            "Rules:\n"
            "- Ask exactly one question.\n"
            "- Keep it answerable from excerpts.\n"
            "- If confidence is low, prefer concrete and focused question.\n"
            "- If confidence is high, prefer slightly more challenging but still grounded question.\n"
            "- Do not mention memory, confidence, or excerpts directly.\n"
            f"- {MATH_DISPLAY_INSTRUCTION}\n"
            "- Return JSON as {\"question\": \"...\"}."
        )

        try:
            resp = chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=220,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            question = str(data.get("question") or "").strip()
            if question:
                return question
        except Exception:
            pass

        return generate_question_at_difficulty(
            topic_label=topic_label,
            context_pack=context_pack,
            difficulty=difficulty,
            model=self.model,
        )
