import os
import tempfile
import unittest
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase5_student_memory_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.data.db_repository import DBRepository  # noqa: E402
from app.services.review.service import ReviewService  # noqa: E402
from app.services.learner.student_memory import StudentMemoryService  # noqa: E402
from app.services.learner import student_model as student_model_mod  # noqa: E402


class StudentMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = DBRepository(_TEST_ROOT / "meta.sqlite")

    def test_memory_bounded_and_versioned(self):
        exam_id = self.repo.create_exam(user_id="u1", title="Exam 1")
        self.repo.upsert_topic(topic_id="t1", exam_id=exam_id, label="Topic 1")
        svc = StudentMemoryService(
            repo=self.repo,
            max_known_facts=5,
            max_misconceptions=5,
            max_recent_events=6,
        )
        for i in range(10):
            svc.update_from_review(
                user_id="u1",
                exam_id=exam_id,
                topic_id="t1",
                rating="i_knew_it" if i % 2 == 0 else "dont_understand",
                question=f"Q{i}",
                answer=f"A{i}",
                difficulty=2,
            )
        mem = svc.get_topic_memory(user_id="u1", exam_id=exam_id, topic_id="t1")
        self.assertEqual(mem["version"], 1)
        self.assertLessEqual(len(mem["known_facts"]), 5)
        self.assertLessEqual(len(mem["misconceptions"]), 5)
        self.assertLessEqual(len(mem["recent_events"]), 6)

    def test_review_service_updates_student_memory(self):
        exam_id = self.repo.create_exam(user_id="u2", title="Exam")
        self.repo.upsert_topic(topic_id="tdiag", exam_id=exam_id, label="Topic D")
        self.repo.upsert_card(
            card_id="cdiag",
            exam_id=exam_id,
            topic_id="tdiag",
            question="What is X?",
            answer="X is Y",
            difficulty=1,
            card_type="diagnostic",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="cdiag",
            topics=[{"topic_id": "tdiag", "role": "primary", "weight": 1.0}],
        )
        self.repo.update_exam_lifecycle(
            exam_id=exam_id,
            state="diagnostic",
            diagnostic_total=1,
            diagnostic_answered=0,
        )

        service = ReviewService(repo=self.repo)
        result = service.apply_review(
            user_id="u2",
            exam_id=exam_id,
            card_id="cdiag",
            rating="learned_now",
            idempotency_key="k1",
        )
        self.assertFalse(result.idempotent_replay)
        mem_row = self.repo.get_student_knowledge_state(
            user_id="u2",
            exam_id=exam_id,
            topic_id="tdiag",
        )
        self.assertIsNotNone(mem_row)
        self.assertIn("known_facts", mem_row.memory)

    def test_student_model_uses_memory_in_prompt(self):
        captured = {"text": ""}
        original = student_model_mod.chat_completions_create
        try:
            def fake_chat_completions_create(**kwargs):
                captured["text"] = kwargs["messages"][1]["content"]
                class _Msg:
                    content = "{\"question\": \"What follows next?\"}"
                class _Choice:
                    message = _Msg()
                class _Resp:
                    choices = [_Choice()]
                return _Resp()

            student_model_mod.chat_completions_create = fake_chat_completions_create
            q = student_model_mod.StudentModelService().generate_question(
                topic_label="T",
                context_pack="proof text",
                difficulty=2,
                memory={"known_facts": ["Known one"], "misconceptions": ["Wrong one"], "confidence": 0.3},
            )
            self.assertEqual(q, "What follows next?")
            self.assertIn("Known one", captured["text"])
            self.assertIn("Wrong one", captured["text"])
        finally:
            student_model_mod.chat_completions_create = original


if __name__ == "__main__":
    unittest.main()
