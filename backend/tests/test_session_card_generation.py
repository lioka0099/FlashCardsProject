import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.data.db_repository import StoredCard, StoredCardTopic, StoredExam, StoredTopic, StoredTopicProficiency
from app.services.card_routing import RouteDecision
from app.services.cards import GeneratedCard
from app.services import session_card_generation as generation_mod
from app.services.session_card_generation import SessionCardGenerationService


class FakeStore:
    vector_backend = "numpy"


class FallbackRepo:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.exam = StoredExam(
            exam_id="e1",
            user_id="u1",
            title="Exam",
            mode="mastery",
            created_at=now,
            updated_at=now,
            info={"math_profile": {"kind": "calculation"}},
            state="active_learning",
        )
        self.topic = StoredTopic(topic_id="t1", exam_id="e1", label="Algebra", created_at=now, info={})
        self.proficiency = StoredTopicProficiency(
            user_id="u1",
            exam_id="e1",
            topic_id="t1",
            proficiency=0.5,
            current_difficulty=1,
            streak_up=0,
            streak_down=0,
            seen_count=1,
            correctish_count=0,
            last_updated_at=now,
            info={},
        )
        self.cards = {}
        self.card_topics = {}

    def get_exam(self, exam_id: str) -> StoredExam:
        return self.exam

    def list_topic_proficiencies(self, *, user_id: str, exam_id: str):
        return [self.proficiency]

    def list_topics(self, *, exam_id: str):
        return [self.topic]

    def list_presentations(self, *, user_id: str, exam_id: str, ascending: bool = False, limit: int = 100):
        return []

    def list_cards_for_exam(self, *, exam_id: str, limit: int = 5000):
        return list(self.cards.values())

    def list_card_topics(self, *, card_id: str):
        return self.card_topics.get(card_id, [])

    def list_chunk_ids_for_topic(self, *, topic_id: str):
        return ["chunk-1"]

    def get_topic_vector(self, *, topic_id: str):
        return None

    def get_topic_proficiency(self, *, user_id: str, exam_id: str, topic_id: str):
        return self.proficiency

    def replace_card_topics(self, *, card_id: str, topics):
        self.card_topics[card_id] = [
            StoredCardTopic(
                card_id=card_id,
                topic_id=topic["topic_id"],
                role=topic["role"],
                weight=topic["weight"],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            for topic in topics
        ]


class SessionCardGenerationTests(unittest.TestCase):
    def test_overlapping_prefetch_generation_returns_none(self) -> None:
        key = ("u1", "e1")
        lock = generation_mod._GENERATION_LOCKS.setdefault(key, generation_mod.threading.Lock())  # type: ignore[attr-defined]
        lock.acquire()
        try:
            svc = SessionCardGenerationService(repo=object())  # type: ignore[arg-type]
            self.assertIsNone(svc.generate_next_card(user_id="u1", exam_id="e1", prefetch=True))
        finally:
            lock.release()
            generation_mod._GENERATION_LOCKS.pop(key, None)  # type: ignore[attr-defined]

    def test_math_generation_falls_back_to_conceptual_card(self) -> None:
        repo = FallbackRepo()
        calls = []

        def fake_generate_single_card(**kwargs):
            calls.append(kwargs)
            if kwargs.get("card_route_override") != "math_conceptual":
                return None
            now = datetime.now(timezone.utc).isoformat()
            repo.cards["conceptual-1"] = StoredCard(
                card_id="conceptual-1",
                exam_id="e1",
                topic_id="t1",
                question="What is a linear equation?",
                answer="An equation whose graph is a line.",
                difficulty=1,
                card_type="learning",
                status="active",
                created_at=now,
                info={"card_route": "math_conceptual"},
            )
            return GeneratedCard(
                card_id="conceptual-1",
                exam_id="e1",
                topic_id="t1",
                topic_label="Algebra",
                question="What is a linear equation?",
                answer="An equation whose graph is a line.",
                difficulty=1,
                proofs=[],
            )

        with (
            patch("app.services.session_card_generation.build_diverse_chunk_pack", return_value="math context"),
            patch(
                "app.services.session_card_generation.classify_card_route",
                return_value=RouteDecision(card_route="math_calculation", subject_type="math", math_kind="calculation"),
            ),
            patch("app.services.session_card_generation.generate_single_card", side_effect=fake_generate_single_card),
        ):
            result = SessionCardGenerationService(repo=repo)._generate_next_card_unlocked(
                user_id="u1",
                exam_id="e1",
                store=FakeStore(),  # type: ignore[arg-type]
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.card_id, "conceptual-1")
        self.assertEqual(calls[1].get("card_route_override"), "math_conceptual")


if __name__ == "__main__":
    unittest.main()
