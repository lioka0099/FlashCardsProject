import unittest

from app.data.db_repository import StoredCardTopic, StoredTopicProficiency
from app.services.topic_proficiency_state import TopicProficiencyStateService


def _link() -> StoredCardTopic:
    return StoredCardTopic(
        card_id="c1",
        topic_id="t1",
        role="primary",
        weight=1.0,
        created_at="",
    )


def _prof() -> StoredTopicProficiency:
    return StoredTopicProficiency(
        user_id="u1",
        exam_id="e1",
        topic_id="t1",
        proficiency=0.8,
        current_difficulty=4,
        streak_up=2,
        streak_down=0,
        seen_count=10,
        correctish_count=9,
        last_updated_at="",
        info={},
    )


class RouteProficiencyTests(unittest.TestCase):
    def test_math_route_does_not_change_default_columns(self) -> None:
        service = TopicProficiencyStateService()
        current = _prof()
        transition = service.apply_rating(
            user_id="u1",
            exam_id="e1",
            topic_link=_link(),
            rating="i_knew_it",
            current=current,
            card_route="math_calculation",
            difficulty_framework="tag",
        )
        self.assertEqual(transition.current_difficulty, 4)
        self.assertAlmostEqual(transition.proficiency, 0.8)
        math_state = transition.info["route_proficiency"]["math_calculation"]
        self.assertEqual(math_state["difficulty_framework"], "tag")
        self.assertEqual(math_state["current_difficulty"], 1)
        self.assertEqual(math_state["seen_count"], 1)

    def test_default_route_preserves_existing_column_behavior(self) -> None:
        service = TopicProficiencyStateService()
        current = _prof()
        transition = service.apply_rating(
            user_id="u1",
            exam_id="e1",
            topic_link=_link(),
            rating="i_knew_it",
            current=current,
            card_route="default",
            difficulty_framework="bloom",
        )
        self.assertEqual(transition.current_difficulty, 5)
        self.assertEqual(transition.info["route_proficiency"]["default"]["current_difficulty"], 5)


if __name__ == "__main__":
    unittest.main()

