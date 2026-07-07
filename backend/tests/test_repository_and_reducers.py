import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase2_repo_reducers_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.data.db_engine import drop_all_tables, init_db
from app.data.db_repository import DBRepository
from app.services.review.card_scheduling_state import CardSchedulingStateService
from app.services.review.state_reducer import ReviewStateReducer
from app.services.review.topic_proficiency_state import TopicProficiencyStateService


class RepositoryAndReducerTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.repo = DBRepository(_TEST_ROOT / "meta.sqlite")
        self.repo.ensure_user("phase2-user")
        self.exam_id = self.repo.create_exam(user_id="phase2-user", title="Phase 2")
        self.repo.upsert_topic(topic_id="t1", exam_id=self.exam_id, label="Topic 1")
        self.repo.upsert_topic(topic_id="t2", exam_id=self.exam_id, label="Topic 2")
        self.repo.upsert_card(
            card_id="c1",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Q1?",
            answer="A1",
            difficulty=2,
            status="active",
            info={},
        )

    def test_repository_card_topics_and_primary_guard(self) -> None:
        self.repo.replace_card_topics(
            card_id="c1",
            topics=[
                {"topic_id": "t1", "role": "primary", "weight": 1.0},
                {"topic_id": "t2", "role": "secondary", "weight": 0.4},
            ],
        )
        links = self.repo.list_card_topics(card_id="c1")
        self.assertEqual(len(links), 2)
        self.assertEqual(len([l for l in links if l.role == "primary"]), 1)

        with self.assertRaises(ValueError):
            self.repo.replace_card_topics(
                card_id="c1",
                topics=[
                    {"topic_id": "t1", "role": "secondary", "weight": 0.4},
                    {"topic_id": "t2", "role": "secondary", "weight": 0.4},
                ],
            )

    def test_repository_scheduling_proficiency_session_presentation_helpers(self) -> None:
        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c1",
            due_at=now - timedelta(hours=1),
            state="learning",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )
        sched = self.repo.get_card_scheduling(card_id="c1")
        self.assertIsNotNone(sched)
        self.assertEqual(sched.state, "learning")

        due = self.repo.list_due_cards(
            user_id="phase2-user",
            exam_id=self.exam_id,
            at_or_before=now,
        )
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].card_id, "c1")

        self.repo.upsert_topic_proficiency(
            user_id="phase2-user",
            exam_id=self.exam_id,
            topic_id="t1",
            proficiency=0.6,
            current_difficulty=2,
            streak_up=1,
            streak_down=0,
            seen_count=1,
            correctish_count=1,
            info={},
        )
        prof = self.repo.get_topic_proficiency(
            user_id="phase2-user",
            exam_id=self.exam_id,
            topic_id="t1",
        )
        self.assertIsNotNone(prof)
        self.assertAlmostEqual(prof.proficiency, 0.6, places=6)

        self.repo.upsert_exam_session_state(
            user_id="phase2-user",
            exam_id=self.exam_id,
            last_served_card_id="c1",
            last_presented_at=now,
        )
        ss = self.repo.get_exam_session_state(user_id="phase2-user", exam_id=self.exam_id)
        self.assertIsNotNone(ss)
        self.assertEqual(ss.last_served_card_id, "c1")

        p1 = self.repo.append_card_presentation(
            user_id="phase2-user",
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now,
        )
        self.assertEqual(p1.sequence_no, 1)
        p2 = self.repo.append_card_presentation(
            user_id="phase2-user",
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now + timedelta(minutes=1),
        )
        self.assertEqual(p2.sequence_no, 2)

        latest = self.repo.get_latest_presentation(user_id="phase2-user", exam_id=self.exam_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.sequence_no, 2)
        prev = self.repo.get_previous_presentation(
            user_id="phase2-user",
            exam_id=self.exam_id,
            current_sequence_no=2,
        )
        self.assertIsNotNone(prev)
        self.assertEqual(prev.sequence_no, 1)

    def test_due_cards_are_user_scoped(self) -> None:
        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c1",
            due_at=now - timedelta(minutes=5),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )

        other_user = "phase2-user-other"
        self.repo.ensure_user(other_user)
        other_exam = self.repo.create_exam(user_id=other_user, title="Other exam")
        self.repo.upsert_topic(topic_id="other-t1", exam_id=other_exam, label="Other Topic")
        self.repo.upsert_card(
            card_id="other-c1",
            exam_id=other_exam,
            topic_id="other-t1",
            question="Other Q?",
            answer="Other A",
            difficulty=1,
            status="active",
            info={},
        )
        self.repo.upsert_card_scheduling(
            card_id="other-c1",
            due_at=now - timedelta(minutes=5),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )

        due_self = self.repo.list_due_cards(
            user_id="phase2-user",
            exam_id=self.exam_id,
            at_or_before=now,
        )
        self.assertEqual([x.card_id for x in due_self], ["c1"])

    def test_scheduling_transition_matrix(self) -> None:
        service = CardSchedulingStateService()
        now = datetime.now(timezone.utc)

        first = service.apply_rating(card_id="c1", rating="learned_now", current=None, now=now)
        self.assertIn(first.state, ("learning", "review"))
        self.assertGreaterEqual(first.interval_days, 1.0)

        almost = service.apply_rating(
            card_id="c1",
            rating="almost_knew",
            current=self.repo.get_card_scheduling(card_id="c1"),
            now=now,
        ) if self.repo.get_card_scheduling(card_id="c1") else service.apply_rating(
            card_id="c1", rating="almost_knew", current=None, now=now
        )
        self.assertEqual(almost.state, "review")

        knew = service.apply_rating(card_id="c1", rating="i_knew_it", current=None, now=now)
        self.assertEqual(knew.state, "retired")

        dont = service.apply_rating(card_id="c1", rating="dont_understand", current=None, now=now)
        self.assertEqual(dont.state, "relearning")
        self.assertGreater(dont.lapses, 0)

    def test_topic_weighted_impact_primary_vs_secondary(self) -> None:
        service = TopicProficiencyStateService()
        self.repo.replace_card_topics(
            card_id="c1",
            topics=[
                {"topic_id": "t1", "role": "primary", "weight": 1.0},
                {"topic_id": "t2", "role": "secondary", "weight": 0.3},
            ],
        )
        links = {f"{x.topic_id}:{x.role}": x for x in self.repo.list_card_topics(card_id="c1")}
        p = service.apply_rating(
            user_id="phase2-user",
            exam_id=self.exam_id,
            topic_link=links["t1:primary"],
            rating="almost_knew",
            current=None,
        )
        s = service.apply_rating(
            user_id="phase2-user",
            exam_id=self.exam_id,
            topic_link=links["t2:secondary"],
            rating="almost_knew",
            current=None,
        )
        self.assertGreater(p.proficiency, s.proficiency)

    def test_orchestrator_idempotency_and_rollback(self) -> None:
        self.repo.replace_card_topics(
            card_id="c1",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        reducer = ReviewStateReducer(repo=self.repo)
        key = "same-review-key"
        first = reducer.apply_review(
            user_id="phase2-user",
            exam_id=self.exam_id,
            card_id="c1",
            rating="learned_now",
            idempotency_key=key,
        )
        second = reducer.apply_review(
            user_id="phase2-user",
            exam_id=self.exam_id,
            card_id="c1",
            rating="learned_now",
            idempotency_key=key,
        )
        self.assertEqual(first.review_id, second.review_id)
        self.assertTrue(second.idempotent_replay)

        before = self.repo.get_card_scheduling(card_id="c1")

        class BoomProficiencyService:
            def apply_rating(self, **kwargs):
                raise RuntimeError("boom")

        failing = ReviewStateReducer(
            repo=self.repo,
            proficiency_service=BoomProficiencyService(),  # type: ignore[arg-type]
        )
        with self.assertRaises(RuntimeError):
            failing.apply_review(
                user_id="phase2-user",
                exam_id=self.exam_id,
                card_id="c1",
                rating="almost_knew",
                idempotency_key="new-key-causes-write",
            )

        after = self.repo.get_card_scheduling(card_id="c1")
        self.assertEqual(before.due_at if before else "", after.due_at if after else "")
        self.assertEqual(before.state if before else "", after.state if after else "")


if __name__ == "__main__":
    unittest.main()
