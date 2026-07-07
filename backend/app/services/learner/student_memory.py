"""Maintain per-topic student memory state (what the learner knows and struggles with)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.data.db_repository import DBRepository


DEFAULT_MEMORY_VERSION = 1
DEFAULT_MAX_KNOWN_FACTS = 24
DEFAULT_MAX_MISCONCEPTIONS = 24
DEFAULT_MAX_RECENT_EVENTS = 32


@dataclass
class StudentMemoryService:
    """Read/update bounded per-topic student memory."""

    repo: DBRepository
    max_known_facts: int = DEFAULT_MAX_KNOWN_FACTS
    max_misconceptions: int = DEFAULT_MAX_MISCONCEPTIONS
    max_recent_events: int = DEFAULT_MAX_RECENT_EVENTS

    def get_topic_memory(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
    ) -> Dict[str, Any]:
        row = self.repo.get_student_knowledge_state(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
        )
        if row is None:
            return self._empty_memory()
        return self._normalize_memory(row.memory)

    def update_from_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        rating: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        difficulty: Optional[int] = None,
        source: str = "review",
    ) -> Dict[str, Any]:
        memory = self.get_topic_memory(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
        )

        event = {
            "rating": rating,
            "difficulty": int(difficulty) if difficulty is not None else None,
            "question": (question or "").strip(),
            "answer": (answer or "").strip(),
            "source": source,
        }
        memory["recent_events"].append(event)
        if len(memory["recent_events"]) > self.max_recent_events:
            memory["recent_events"] = memory["recent_events"][-self.max_recent_events :]

        if rating in {"i_knew_it", "almost_knew", "learned_now"}:
            if question:
                memory["known_facts"].append(question.strip())
            if answer:
                memory["known_facts"].append(answer.strip())
            memory["confidence"] = min(1.0, float(memory["confidence"]) + 0.05)
        else:
            if question:
                memory["misconceptions"].append(question.strip())
            memory["confidence"] = max(0.0, float(memory["confidence"]) - 0.08)

        memory["known_facts"] = self._dedupe_keep_last(
            memory["known_facts"], self.max_known_facts
        )
        memory["misconceptions"] = self._dedupe_keep_last(
            memory["misconceptions"], self.max_misconceptions
        )

        self.repo.upsert_student_knowledge_state(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
            memory=memory,
        )
        return memory

    @staticmethod
    def _dedupe_keep_last(items: list[str], limit: int) -> list[str]:
        seen = set()
        out_reversed: list[str] = []
        for value in reversed(items):
            key = value.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out_reversed.append(key)
            if len(out_reversed) >= limit:
                break
        return list(reversed(out_reversed))

    @staticmethod
    def _empty_memory() -> Dict[str, Any]:
        return {
            "version": DEFAULT_MEMORY_VERSION,
            "known_facts": [],
            "misconceptions": [],
            "recent_events": [],
            "confidence": 0.5,
        }

    def _normalize_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._empty_memory()
        merged.update(memory or {})
        merged["known_facts"] = [str(x).strip() for x in merged.get("known_facts", []) if str(x).strip()]
        merged["misconceptions"] = [str(x).strip() for x in merged.get("misconceptions", []) if str(x).strip()]
        merged["recent_events"] = list(merged.get("recent_events", []))
        merged["confidence"] = float(merged.get("confidence", 0.5))
        return merged
