import unittest
from datetime import datetime, timezone

from app.data.db_repository import StoredCard, StoredCardPresentation, StoredCardScheduling, StoredTopicProficiency
from app.services.session.planner import SessionPlannerService


class FakeRepo:
    def list_topic_proficiencies(self, *, user_id, exam_id):
        return [
            StoredTopicProficiency(
                user_id=user_id,
                exam_id=exam_id,
                topic_id="t1",
                proficiency=0.6,
                current_difficulty=1,
                streak_up=0,
                streak_down=0,
                seen_count=1,
                correctish_count=1,
                last_updated_at="",
                info={},
            )
        ]

    def list_cards_for_exam(self, *, exam_id, limit):
        return [
            StoredCard(
                card_id="already-seen",
                exam_id=exam_id,
                topic_id="t1",
                question="Q1",
                answer="A1",
                difficulty=1,
                created_at="1",
                status="active",
                info={},
                card_type="learning",
            ),
            StoredCard(
                card_id="new-card",
                exam_id=exam_id,
                topic_id="t1",
                question="Q2",
                answer="A2",
                difficulty=1,
                created_at="2",
                status="active",
                info={},
                card_type="learning",
            ),
        ]

    def get_card_scheduling(self, *, card_id):
        return None

    def list_due_cards(self, *, user_id, exam_id, at_or_before, limit):
        return [
            StoredCardScheduling(
                card_id="diagnostic-due",
                due_at="",
                state="review",
                interval_days=1.0,
                ease=2.5,
                reps=1,
                lapses=0,
                last_reviewed_at="",
            )
        ]

    def get_card(self, *, card_id):
        if card_id == "diagnostic-due":
            return StoredCard(
                card_id="diagnostic-due",
                exam_id="e1",
                topic_id="t1",
                question="Diagnostic?",
                answer="Answer",
                difficulty=1,
                created_at="0",
                status="active",
                info={},
                card_type="diagnostic",
            )
        return None

    def list_presentations(self, *, user_id, exam_id, ascending, limit):
        return [
            StoredCardPresentation(
                user_id=user_id,
                exam_id=exam_id,
                sequence_no=1,
                card_id="already-seen",
                presented_at="",
                info={},
            )
        ]

    def list_card_topics(self, *, card_id):
        return []


class SessionPlannerTests(unittest.TestCase):
    def test_progression_queue_skips_presented_unscheduled_cards(self) -> None:
        planner = SessionPlannerService(repo=FakeRepo())  # type: ignore[arg-type]
        cards = planner._build_progression_queue(
            user_id="u1",
            exam_id="e1",
            now=datetime.now(timezone.utc),
        )

        self.assertEqual([card.card_id for card in cards], ["new-card"])

    def test_overdue_queue_skips_diagnostic_cards_after_transition(self) -> None:
        planner = SessionPlannerService(repo=FakeRepo())  # type: ignore[arg-type]
        cards = planner._build_overdue_queue(
            user_id="u1",
            exam_id="e1",
            now=datetime.now(timezone.utc),
        )

        self.assertEqual(cards, [])


if __name__ == "__main__":
    unittest.main()
