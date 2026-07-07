import unittest

from app.data.db_repository import StoredCardTopic, StoredTopicProficiency
from app.services.review.topic_proficiency_state import TopicProficiencyStateService


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
    def test_math_route_projects_visible_columns_from_route_state(self) -> None:
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
        self.assertAlmostEqual(transition.proficiency, 0.92)
        self.assertEqual(transition.seen_count, 11)
        self.assertEqual(transition.correctish_count, 10)
        math_state = transition.info["route_proficiency"]["math_calculation"]
        self.assertEqual(math_state["difficulty_framework"], "tag")
        self.assertEqual(math_state["current_difficulty"], 4)
        self.assertEqual(math_state["seen_count"], 11)

    def test_math_route_difficulty_progresses_from_canonical_route_state(self) -> None:
        service = TopicProficiencyStateService()
        current = None
        transition = None
        for _ in range(3):
            transition = service.apply_rating(
                user_id="u1",
                exam_id="e1",
                topic_link=_link(),
                rating="i_knew_it",
                current=current,
                card_route="math_calculation",
                difficulty_framework="tag",
            )
            current = StoredTopicProficiency(
                user_id=transition.user_id,
                exam_id=transition.exam_id,
                topic_id=transition.topic_id,
                proficiency=transition.proficiency,
                current_difficulty=transition.current_difficulty,
                streak_up=transition.streak_up,
                streak_down=transition.streak_down,
                seen_count=transition.seen_count,
                correctish_count=transition.correctish_count,
                last_updated_at="",
                info=transition.info,
            )

        assert transition is not None
        self.assertEqual(transition.current_difficulty, 2)
        self.assertEqual(transition.info["route_proficiency"]["math_calculation"]["current_difficulty"], 2)
        self.assertEqual(transition.seen_count, 3)

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

    def test_first_math_route_review_keeps_existing_seen_count(self) -> None:
        service = TopicProficiencyStateService()
        current = StoredTopicProficiency(
            user_id="u1",
            exam_id="e1",
            topic_id="t1",
            proficiency=0.75,
            current_difficulty=2,
            streak_up=1,
            streak_down=0,
            seen_count=1,
            correctish_count=1,
            last_updated_at="",
            info={"source": "diagnostic_state_service"},
        )

        transition = service.apply_rating(
            user_id="u1",
            exam_id="e1",
            topic_link=_link(),
            rating="almost_knew",
            current=current,
            card_route="math_conceptual",
            difficulty_framework="solo",
        )

        self.assertEqual(transition.seen_count, 2)
        self.assertEqual(transition.correctish_count, 2)
        self.assertGreaterEqual(transition.proficiency, 0.75)


    def test_second_route_does_not_double_count_reviews(self) -> None:
        # Regression: a topic reviewed under one route, then under a second route,
        # must report total reviews = number of ratings (not 2*N+1). The new route
        # used to inherit seen_count from the summed top-level column, so the
        # projection counted the prior reviews twice the moment the route switched.
        service = TopicProficiencyStateService()
        current = None
        # 3 reviews on the default (bloom) route.
        for _ in range(3):
            transition = service.apply_rating(
                user_id="u1",
                exam_id="e1",
                topic_link=_link(),
                rating="i_knew_it",
                current=current,
                card_route="default",
                difficulty_framework="bloom",
            )
            current = StoredTopicProficiency(
                user_id=transition.user_id,
                exam_id=transition.exam_id,
                topic_id=transition.topic_id,
                proficiency=transition.proficiency,
                current_difficulty=transition.current_difficulty,
                streak_up=transition.streak_up,
                streak_down=transition.streak_down,
                seen_count=transition.seen_count,
                correctish_count=transition.correctish_count,
                last_updated_at="",
                info=transition.info,
            )
        self.assertEqual(current.seen_count, 3)

        # 4th review switches to the math_calculation route.
        transition = service.apply_rating(
            user_id="u1",
            exam_id="e1",
            topic_link=_link(),
            rating="i_knew_it",
            current=current,
            card_route="math_calculation",
            difficulty_framework="tag",
        )
        self.assertEqual(transition.seen_count, 4)  # not 7
        self.assertEqual(transition.correctish_count, 4)


if __name__ == "__main__":
    unittest.main()

