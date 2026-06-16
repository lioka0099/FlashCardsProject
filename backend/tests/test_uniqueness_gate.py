import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase5_uniqueness_gate_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.services.graph import BatchSeenQuestion, node_check_uniqueness  # noqa: E402


def _state(*, batch_seen_questions):
    return {
        "exam_id": "exam1",
        "user_id": "u1",
        "store_basepath": str(_TEST_ROOT / "store"),
        "batch_id": "b1",
        "topic_id": "t1",
        "topic_label": "Topic 1",
        "difficulty": 2,
        "card_type": "learning",
        "allowed_chunk_ids": ["c1"],
        "context_pack": "ctx",
        "batch_seen_questions": batch_seen_questions,
        "student_memory": {},
        "question": "Q?",
        "question_embedding": np.array([1.0, 0.0], dtype="float32"),
        "question_id": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": 0,
        "max_question_attempts": 3,
        "max_answer_attempts": 3,
        "max_full_restarts": 3,
        "uniqueness_threshold": 0.85,
        "validation_threshold": 0.7,
        "commit_retry_attempts": 3,
        "commit_retry_sleep_s": 0.01,
        "initial_k": 8,
        "initial_min_score": 0.4,
        "strengthen_k_delta": 2,
        "strengthen_min_score_delta": 0.05,
        "k": 8,
        "min_score": 0.4,
        "stop_after_embedding": True,
        "card": None,
    }


class UniquenessGateTests(unittest.TestCase):
    def test_duplicate_from_same_batch_is_rejected(self):
        seen = BatchSeenQuestion(
            question_id="q1",
            topic_id="t1",
            question_text="same meaning",
            difficulty=2,
            embedding=np.array([1.0, 0.0], dtype="float32"),
            created_seq=1,
        )
        out = node_check_uniqueness(_state(batch_seen_questions=(seen,)))
        self.assertFalse(out["is_unique"])

    def test_duplicate_from_different_topic_same_batch_is_rejected(self):
        seen = BatchSeenQuestion(
            question_id="q1",
            topic_id="other-topic",
            question_text="same meaning",
            difficulty=2,
            embedding=np.array([1.0, 0.0], dtype="float32"),
            created_seq=1,
        )
        out = node_check_uniqueness(_state(batch_seen_questions=(seen,)))
        self.assertFalse(out["is_unique"])

    def test_non_pinecone_backend_raises_for_mandatory_gate(self):
        with self.assertRaises(RuntimeError):
            node_check_uniqueness(_state(batch_seen_questions=tuple()))

    def test_math_calculation_allows_semantic_similarity_when_text_differs(self):
        seen = BatchSeenQuestion(
            question_id="q1",
            topic_id="t1",
            question_text="Differentiate x**2 with respect to x.",
            difficulty=2,
            embedding=np.array([1.0, 0.0], dtype="float32"),
            created_seq=1,
        )
        state = _state(batch_seen_questions=(seen,))
        state["card_route"] = "math_calculation"
        state["question"] = "Differentiate x**3 with respect to x."
        out = node_check_uniqueness(state)
        self.assertTrue(out["is_unique"])

    def test_math_calculation_rejects_exact_text_duplicate(self):
        state = _state(batch_seen_questions=tuple())
        state["card_route"] = "math_calculation"
        state["question"] = "Differentiate x**2 with respect to x."

        class _FakeDB:
            @staticmethod
            def has_equivalent_question_text(*, exam_id, question_text):
                return True

        class _FakeStore:
            def __init__(self, **_kwargs):
                self.db = _FakeDB()

        with patch("app.services.graph.VectorStore", _FakeStore):
            out = node_check_uniqueness(state)
        self.assertFalse(out["is_unique"])


if __name__ == "__main__":
    unittest.main()
