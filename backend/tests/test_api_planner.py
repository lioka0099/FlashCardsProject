import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase4_api_planner_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.db_repository import DBRepository
from app.services.session.card_generation import GeneratedSessionCard, SessionCardGenerationService


class ApiPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.client = TestClient(app)
        self.repo = DBRepository(_TEST_ROOT / "meta.sqlite")
        self.refill_patcher = patch(
            "app.api.endpoints.SessionCardGenerationService.refill_prefetch_if_needed",
            autospec=True,
        )
        self.mock_refill = self.refill_patcher.start()
        self.addCleanup(self.refill_patcher.stop)
        self.user_id = "phase4-user"
        self.repo.ensure_user(self.user_id)
        self.exam_id = self.repo.create_exam(user_id=self.user_id, title="Phase 4 Exam")
        self.repo.update_exam_lifecycle(exam_id=self.exam_id, state="active_learning")
        self.doc_id = "doc-source-1"
        self.source_path = _TEST_ROOT / "source.txt"
        self.source_path.write_text("source body", encoding="utf-8")
        self.repo.add_document(
            doc_id=self.doc_id,
            path=self.source_path.as_posix(),
            title="source.txt",
            info={},
        )
        self.repo.attach_documents_to_exam(exam_id=self.exam_id, doc_ids=[self.doc_id])

        self.repo.upsert_topic(topic_id="t1", exam_id=self.exam_id, label="Topic 1")
        self.repo.upsert_topic(topic_id="t2", exam_id=self.exam_id, label="Topic 2")
        self.repo.upsert_card(
            card_id="c1",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Q1?",
            answer="A1",
            difficulty=1,
            card_type="learning",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="c1",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        self.repo.upsert_card(
            card_id="c2",
            exam_id=self.exam_id,
            topic_id="t2",
            question="Q2?",
            answer="A2",
            difficulty=2,
            card_type="learning",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="c2",
            topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
        )
        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c1",
            due_at=now - timedelta(minutes=10),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )
        self.repo.upsert_card_scheduling(
            card_id="c2",
            due_at=now + timedelta(days=2),
            state="new",
            interval_days=1.0,
            ease=2.5,
            reps=0,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )
        for topic_id in ("t1", "t2"):
            self.repo.upsert_topic_proficiency(
                user_id=self.user_id,
                exam_id=self.exam_id,
                topic_id=topic_id,
                proficiency=0.5,
                current_difficulty=1,
                streak_up=0,
                streak_down=0,
                seen_count=1,
                correctish_count=0,
                info={},
            )

    def _archive_seed_learning_cards(self) -> None:
        for card_id, topic_id, question, answer in [
            ("c1", "t1", "Q1?", "A1"),
            ("c2", "t2", "Q2?", "A2"),
        ]:
            self.repo.upsert_card(
                card_id=card_id,
                exam_id=self.exam_id,
                topic_id=topic_id,
                question=question,
                answer=answer,
                difficulty=1,
                card_type="learning",
                status="archived",
                info={},
            )

    def _seed_prefetched_card(
        self,
        *,
        card_id: str,
        topic_id: str,
        generated_difficulty: int,
    ) -> None:
        self.repo.upsert_card(
            card_id=card_id,
            exam_id=self.exam_id,
            topic_id=topic_id,
            question=f"Prefetched {card_id}?",
            answer=f"Prefetched {card_id} answer",
            difficulty=generated_difficulty,
            card_type="learning",
            status="active",
            info={
                "prefetch_status": "ready",
                "generated_for_user_id": self.user_id,
                "generated_difficulty": generated_difficulty,
            },
        )
        self.repo.replace_card_topics(
            card_id=card_id,
            topics=[{"topic_id": topic_id, "role": "primary", "weight": 1.0}],
        )

    def test_next_previous_and_presented_history(self) -> None:
        with patch(
            "app.api.endpoints.SessionCardGenerationService.generate_next_card",
            side_effect=AssertionError("due SRS card should be served before generation"),
        ):
            n1 = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )
        self.assertEqual(n1.status_code, 200)
        p1 = n1.json()
        self.assertFalse(p1["no_cards_available"])
        self.assertEqual(p1["reason"], "overdue")
        self.assertEqual(p1["card"]["card_id"], "c1")

        review = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "phase4-history-review-c1"},
        )
        self.assertEqual(review.status_code, 200)

        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c2",
            due_at=now - timedelta(minutes=1),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )

        n2 = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(n2.status_code, 200)

        prev = self.client.get(
            f"/exams/{self.exam_id}/session/previous-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(prev.status_code, 200)
        self.assertFalse(prev.json()["no_cards_available"])
        self.assertEqual(prev.json()["reason"], "previous")

        history = self.client.get(
            f"/exams/{self.exam_id}/cards/presented-history",
            params={"user_id": self.user_id},
        )
        self.assertEqual(history.status_code, 200)
        self.assertGreaterEqual(history.json()["total"], 2)

    def test_presented_history_dedupes_learning_flow_cards_and_includes_rating(self) -> None:
        now = datetime.now(timezone.utc)
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now,
            info={"source": "session_orchestrator", "reason": "generated"},
        )
        self.repo.add_card_review(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            topic_id="t1",
            rating="learned_now",
        )
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now + timedelta(minutes=1),
            info={"source": "review_state_reducer"},
        )
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now + timedelta(minutes=2),
            info={"source": "session_orchestrator", "reason": "progression"},
        )
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c2",
            presented_at=now + timedelta(minutes=3),
            info={"source": "session_orchestrator", "reason": "overdue"},
        )
        self.repo.add_card_review(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c2",
            topic_id="t2",
            rating="i_knew_it",
        )

        history = self.client.get(
            f"/exams/{self.exam_id}/cards/presented-history",
            params={"user_id": self.user_id},
        )

        self.assertEqual(history.status_code, 200)
        payload = history.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual([card["card_id"] for card in payload["cards"]], ["c2", "c1"])
        self.assertEqual(payload["cards"][0]["info"]["rating"], "i_knew_it")
        self.assertEqual(payload["cards"][1]["info"]["rating"], "learned_now")

    def test_next_card_resumes_unrated_card_without_recording_ghost_presentations(self) -> None:
        # First fetch serves the overdue card and records one presentation.
        first = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["card"]["card_id"], "c1")

        # Re-fetching without rating (mount/reload) must resume the same card and NOT
        # record additional presentations or advance the queue.
        for _ in range(3):
            again = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )
            self.assertEqual(again.status_code, 200)
            self.assertEqual(again.json()["card"]["card_id"], "c1")
            self.assertEqual(again.json()["reason"], "resumed")

        def orchestrator_presentations():
            rows = self.repo.list_presentations(
                user_id=self.user_id, exam_id=self.exam_id, ascending=True, limit=100
            )
            return [r for r in rows if (r.info or {}).get("source") == "session_orchestrator"]

        self.assertEqual([r.card_id for r in orchestrator_presentations()], ["c1"])

        # After rating, the next fetch advances to a new card and records it.
        review = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "resume-guard-review-c1"},
        )
        self.assertEqual(review.status_code, 200)

        # Make c2 due so the advance has a deterministic candidate (mirrors the SRS flow).
        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c2",
            due_at=now - timedelta(minutes=1),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )

        advanced = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(advanced.status_code, 200)
        self.assertEqual(advanced.json()["card"]["card_id"], "c2")
        self.assertEqual([r.card_id for r in orchestrator_presentations()], ["c1", "c2"])

    def test_active_learning_review_uses_unified_orchestration(self) -> None:
        res = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "learned_now"},
            headers={"Idempotency-Key": "phase4-active-review"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["card_id"], "c1")
        self.assertEqual(payload["exam_state"], "active_learning")
        self.assertIn("due_at", payload)
        self.assertIn("interval_days", payload)
        self.assertIn("ease", payload)

    def test_progress_contract_endpoint(self) -> None:
        # Seed one proficiency row through review.
        review = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "phase4-progress-review"},
        )
        self.assertEqual(review.status_code, 200)
        progress = self.client.get(
            f"/exams/{self.exam_id}/progress",
            params={"user_id": self.user_id},
        )
        self.assertEqual(progress.status_code, 200)
        payload = progress.json()
        self.assertEqual(payload["exam_id"], self.exam_id)
        self.assertEqual(payload["user_id"], self.user_id)
        self.assertIn("topics", payload)
        self.assertGreaterEqual(len(payload["topics"]), 1)

        # Reviewed topic should appear with updated proficiency metrics.
        t1_row = next((t for t in payload["topics"] if t["topic_id"] == "t1"), None)
        self.assertIsNotNone(t1_row)
        assert t1_row is not None
        self.assertEqual(t1_row["topic_label"], "Topic 1")
        self.assertGreaterEqual(float(t1_row["proficiency"]), 0.0)
        self.assertLessEqual(float(t1_row["proficiency"]), 1.0)
        self.assertGreaterEqual(int(t1_row["n_reviews"]), 1)

        # Overall proficiency should be computed when at least one topic row exists.
        self.assertIn("overall_proficiency", payload)
        self.assertIsNotNone(payload["overall_proficiency"])
        self.assertGreaterEqual(float(payload["overall_proficiency"]), 0.0)
        self.assertLessEqual(float(payload["overall_proficiency"]), 1.0)

    def test_document_source_endpoint_returns_exam_document(self) -> None:
        res = self.client.get(
            f"/documents/{self.doc_id}/source",
            params={"exam_id": self.exam_id, "user_id": self.user_id},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/plain", (res.headers.get("content-type") or ""))
        self.assertIn("source body", res.text)

    def test_diagnostic_exam_serves_unanswered_diagnostic_cards(self) -> None:
        self.repo.update_exam_lifecycle(exam_id=self.exam_id, state="diagnostic")
        for topic_id in ("t1", "t2"):
            self.repo.upsert_topic_proficiency(
                user_id=self.user_id,
                exam_id=self.exam_id,
                topic_id=topic_id,
                proficiency=0.5,
                current_difficulty=1,
                streak_up=0,
                streak_down=0,
                seen_count=0,
                correctish_count=0,
                info={},
            )
        self.repo.upsert_card(
            card_id="d1",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Diagnostic Q1?",
            answer="Diagnostic A1",
            difficulty=1,
            card_type="diagnostic",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="d1",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        self.repo.upsert_card(
            card_id="d2",
            exam_id=self.exam_id,
            topic_id="t2",
            question="Diagnostic Q2?",
            answer="Diagnostic A2",
            difficulty=1,
            card_type="diagnostic",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="d2",
            topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
        )

        first = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertFalse(first_payload["no_cards_available"])
        self.assertEqual(first_payload["reason"], "diagnostic")
        self.assertIn(first_payload["card"]["card_id"], {"d1", "d2"})

        first_card_id = first_payload["card"]["card_id"]
        first_review = self.client.post(
            f"/exams/{self.exam_id}/cards/{first_card_id}/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "phase4-diagnostic-first"},
        )
        self.assertEqual(first_review.status_code, 200)
        self.assertEqual(first_review.json()["exam_state"], "diagnostic")

        second = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()
        self.assertFalse(second_payload["no_cards_available"])
        self.assertEqual(second_payload["reason"], "diagnostic")
        self.assertNotEqual(second_payload["card"]["card_id"], first_card_id)

    def test_diagnostic_exhaustion_falls_through_to_active_learning(self) -> None:
        self.repo.update_exam_lifecycle(
            exam_id=self.exam_id,
            state="diagnostic",
            diagnostic_total=2,
            diagnostic_answered=1,
        )

        result = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertFalse(payload["no_cards_available"])
        self.assertEqual(payload["reason"], "overdue")
        self.assertEqual(payload["card"]["card_id"], "c1")
        exam = self.repo.get_exam(self.exam_id)
        self.assertIsNotNone(exam)
        assert exam is not None
        self.assertEqual(exam.state, "active_learning")

    def test_diagnostic_does_not_transition_while_topics_remain_undiagnosed(self) -> None:
        self.repo.upsert_topic(topic_id="t3", exam_id=self.exam_id, label="Topic 3")
        self.repo.upsert_topic_proficiency(
            user_id=self.user_id,
            exam_id=self.exam_id,
            topic_id="t3",
            proficiency=0.5,
            current_difficulty=1,
            streak_up=0,
            streak_down=0,
            seen_count=0,
            correctish_count=0,
            info={},
        )
        self.repo.update_exam_lifecycle(
            exam_id=self.exam_id,
            state="diagnostic",
            diagnostic_total=3,
            diagnostic_answered=2,
        )

        def fake_generate(service, *, user_id: str, exam_id: str, store, prefetch=False):
            self.assertEqual(user_id, self.user_id)
            self.assertEqual(exam_id, self.exam_id)
            self.assertFalse(prefetch)
            service.repo.upsert_card(
                card_id="diag-gap-1",
                exam_id=exam_id,
                topic_id="t3",
                question="Gap diagnostic?",
                answer="Gap answer",
                difficulty=1,
                card_type="diagnostic",
                status="active",
                info={},
            )
            service.repo.replace_card_topics(
                card_id="diag-gap-1",
                topics=[{"topic_id": "t3", "role": "primary", "weight": 1.0}],
            )
            return GeneratedSessionCard(card_id="diag-gap-1", reason="diagnostic", topic_id="t3")

        with patch("app.api.endpoints.SessionCardGenerationService.generate_next_card", new=fake_generate):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertFalse(payload["no_cards_available"])
        self.assertEqual(payload["reason"], "diagnostic")
        self.assertEqual(payload["card"]["card_id"], "diag-gap-1")
        exam = self.repo.get_exam(self.exam_id)
        self.assertIsNotNone(exam)
        assert exam is not None
        self.assertEqual(exam.state, "diagnostic")

    def test_learning_prefetch_for_undiagnosed_topic_is_not_served(self) -> None:
        self.repo.upsert_topic(topic_id="t3", exam_id=self.exam_id, label="Topic 3")
        self._seed_prefetched_card(card_id="prefetch-undiagnosed", topic_id="t3", generated_difficulty=1)

        result = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertFalse(payload["no_cards_available"])
        self.assertNotEqual(payload["card"]["card_id"], "prefetch-undiagnosed")
        archived = self.repo.get_card(card_id="prefetch-undiagnosed")
        self.assertIsNotNone(archived)
        assert archived is not None
        self.assertEqual(archived.info.get("prefetch_status"), "stale")

    def test_next_card_generates_learning_card_when_planner_has_no_candidate(self) -> None:
        self._archive_seed_learning_cards()

        def fake_generate(service, *, user_id: str, exam_id: str, store):
            self.assertEqual(user_id, self.user_id)
            self.assertEqual(exam_id, self.exam_id)
            service.repo.upsert_card(
                card_id="generated-1",
                exam_id=exam_id,
                topic_id="t2",
                question="Generated?",
                answer="Generated answer",
                difficulty=1,
                card_type="learning",
                status="active",
                info={"card_type": "learning"},
            )
            service.repo.replace_card_topics(
                card_id="generated-1",
                topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
            )
            return GeneratedSessionCard(card_id="generated-1", reason="generated", topic_id="t2")

        with patch("app.api.endpoints.SessionCardGenerationService.generate_next_card", new=fake_generate):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertFalse(payload["no_cards_available"])
        self.assertEqual(payload["reason"], "generated")
        self.assertEqual(payload["card"]["card_id"], "generated-1")
        latest = self.repo.get_latest_presentation(user_id=self.user_id, exam_id=self.exam_id)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.card_id, "generated-1")

    def test_fresh_prefetched_card_is_served_before_sync_generation(self) -> None:
        self._archive_seed_learning_cards()
        self._seed_prefetched_card(card_id="prefetch-1", topic_id="t1", generated_difficulty=1)

        with patch(
            "app.api.endpoints.SessionCardGenerationService.generate_next_card",
            side_effect=AssertionError("fresh prefetch should be served before sync generation"),
        ):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertFalse(payload["no_cards_available"])
        self.assertEqual(payload["reason"], "prefetched")
        self.assertEqual(payload["card"]["card_id"], "prefetch-1")
        served_card = self.repo.get_card(card_id="prefetch-1")
        self.assertIsNotNone(served_card)
        assert served_card is not None
        self.assertEqual(served_card.info.get("prefetch_status"), "served")

    def test_ready_prefetch_that_was_already_presented_is_not_served_again(self) -> None:
        self._archive_seed_learning_cards()
        self._seed_prefetched_card(card_id="prefetch-presented", topic_id="t1", generated_difficulty=1)
        now = datetime.now(timezone.utc)
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="prefetch-presented",
            presented_at=now,
            info={"source": "test"},
        )

        def fake_generate(service, *, user_id: str, exam_id: str, store):
            service.repo.upsert_card(
                card_id="generated-after-presented-prefetch",
                exam_id=exam_id,
                topic_id="t2",
                question="Generated after presented prefetch?",
                answer="Generated answer",
                difficulty=1,
                card_type="learning",
                status="active",
                info={"card_type": "learning"},
            )
            service.repo.replace_card_topics(
                card_id="generated-after-presented-prefetch",
                topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
            )
            return GeneratedSessionCard(
                card_id="generated-after-presented-prefetch",
                reason="generated",
                topic_id="t2",
            )

        with patch("app.api.endpoints.SessionCardGenerationService.generate_next_card", new=fake_generate):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["reason"], "generated")
        self.assertEqual(payload["card"]["card_id"], "generated-after-presented-prefetch")
        presented_prefetch = self.repo.get_card(card_id="prefetch-presented")
        self.assertIsNotNone(presented_prefetch)
        assert presented_prefetch is not None
        self.assertEqual(presented_prefetch.info.get("prefetch_status"), "served")

    def test_prefetch_refill_does_not_mark_already_presented_card_ready(self) -> None:
        service = SessionCardGenerationService(repo=self.repo)
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c2",
            presented_at=datetime.now(timezone.utc),
            info={"source": "test"},
        )

        service._mark_prefetched_card_ready(
            card_id="c2",
            user_id=self.user_id,
            choice=service.select_next_topic(user_id=self.user_id, exam_id=self.exam_id),
        )

        card = self.repo.get_card(card_id="c2")
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card.info.get("prefetch_status"), "served")
        self.assertEqual(card.info.get("prefetch_skipped_reason"), "already_presented")

    def test_stale_prefetched_card_is_skipped_for_sync_generation(self) -> None:
        self._archive_seed_learning_cards()
        self._seed_prefetched_card(card_id="prefetch-stale", topic_id="t1", generated_difficulty=1)
        self.repo.upsert_topic_proficiency(
            user_id=self.user_id,
            exam_id=self.exam_id,
            topic_id="t1",
            proficiency=0.9,
            current_difficulty=4,
            streak_up=0,
            streak_down=0,
            seen_count=1,
            correctish_count=1,
            info={},
        )

        def fake_generate(service, *, user_id: str, exam_id: str, store):
            service.repo.upsert_card(
                card_id="generated-after-stale",
                exam_id=exam_id,
                topic_id="t2",
                question="Generated after stale?",
                answer="Generated answer",
                difficulty=1,
                card_type="learning",
                status="active",
                info={"card_type": "learning"},
            )
            service.repo.replace_card_topics(
                card_id="generated-after-stale",
                topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
            )
            return GeneratedSessionCard(card_id="generated-after-stale", reason="generated", topic_id="t2")

        with patch("app.api.endpoints.SessionCardGenerationService.generate_next_card", new=fake_generate):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["reason"], "generated")
        self.assertEqual(payload["card"]["card_id"], "generated-after-stale")
        stale = self.repo.get_card(card_id="prefetch-stale")
        self.assertIsNotNone(stale)
        assert stale is not None
        self.assertEqual(stale.status, "archived")
        self.assertEqual(stale.info.get("prefetch_status"), "stale")

    def test_due_srs_card_beats_ready_prefetched_card(self) -> None:
        self._seed_prefetched_card(card_id="prefetch-1", topic_id="t2", generated_difficulty=1)

        with patch(
            "app.api.endpoints.SessionCardGenerationService.generate_next_card",
            side_effect=AssertionError("due SRS card should be served before generation"),
        ):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["reason"], "overdue")
        self.assertEqual(payload["card"]["card_id"], "c1")

    def test_retired_due_card_is_not_served_as_overdue(self) -> None:
        self._archive_seed_learning_cards()
        now = datetime.now(timezone.utc)
        self.repo.upsert_card(
            card_id="retired-due",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Retired?",
            answer="Retired answer",
            difficulty=1,
            card_type="learning",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="retired-due",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        self.repo.upsert_card_scheduling(
            card_id="retired-due",
            due_at=now,
            state="retired",
            interval_days=7.0,
            ease=2.7,
            reps=1,
            lapses=0,
            last_reviewed_at=now,
        )

        def fake_generate(service, *, user_id: str, exam_id: str, store):
            service.repo.upsert_card(
                card_id="generated-after-retired",
                exam_id=exam_id,
                topic_id="t2",
                question="Generated after retired?",
                answer="Generated answer",
                difficulty=1,
                card_type="learning",
                status="active",
                info={"card_type": "learning"},
            )
            service.repo.replace_card_topics(
                card_id="generated-after-retired",
                topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
            )
            return GeneratedSessionCard(card_id="generated-after-retired", reason="generated", topic_id="t2")

        with patch("app.api.endpoints.SessionCardGenerationService.generate_next_card", new=fake_generate):
            result = self.client.get(
                f"/exams/{self.exam_id}/session/next-card",
                params={"user_id": self.user_id},
            )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["reason"], "generated")
        self.assertEqual(payload["card"]["card_id"], "generated-after-retired")

    def test_background_refill_requested_after_review(self) -> None:
        res = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "learned_now"},
            headers={"Idempotency-Key": "phase4-refill-review"},
        )

        self.assertEqual(res.status_code, 200)
        self.assertTrue(self.mock_refill.called)

    def test_generation_topic_selection_balances_recent_topic_counts(self) -> None:
        now = datetime.now(timezone.utc)
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now - timedelta(minutes=2),
            info={"source": "test"},
        )
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now - timedelta(minutes=1),
            info={"source": "test"},
        )

        service = SessionCardGenerationService(repo=self.repo, max_consecutive_topic_cards=2)
        choice = service.select_next_topic(user_id=self.user_id, exam_id=self.exam_id)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.topic_id, "t2")

    def test_generation_keeps_blocked_topic_as_last_resort_choice(self) -> None:
        now = datetime.now(timezone.utc)
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now - timedelta(minutes=2),
            info={"source": "test"},
        )
        self.repo.append_card_presentation(
            user_id=self.user_id,
            exam_id=self.exam_id,
            card_id="c1",
            presented_at=now - timedelta(minutes=1),
            info={"source": "test"},
        )

        service = SessionCardGenerationService(repo=self.repo, max_consecutive_topic_cards=2)
        choices = service._ordered_topic_choices(user_id=self.user_id, exam_id=self.exam_id)

        self.assertEqual([choice.topic_id for choice in choices], ["t2", "t1"])


if __name__ == "__main__":
    unittest.main()
