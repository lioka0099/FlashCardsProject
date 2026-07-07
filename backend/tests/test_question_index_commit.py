import os
import tempfile
import unittest
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase5_qindex_commit_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.services.generation import graph as graph_mod  # noqa: E402


class QuestionIndexCommitTests(unittest.TestCase):
    def test_save_question_id_is_reservation_only(self):
        state = {
            "question_id": None,
        }
        out = graph_mod.node_save_question_id(state)  # type: ignore[arg-type]
        self.assertIsNotNone(out["question_id"])

    def test_commit_called_only_on_success(self):
        calls = {"commit": 0}

        original_generate = graph_mod.node_generate_answer
        original_validate = graph_mod.node_validate
        original_store_card = graph_mod.node_store_card
        original_commit = graph_mod.node_commit_question_index
        try:
            def fake_generate(state):
                return {**state, "answer": "A", "proofs": []}

            def fake_validate_ok(state):
                return {**state, "validation_score": 0.99}

            def fake_validate_fail(state):
                return {**state, "validation_score": 0.01}

            def fake_store_card(state):
                return {**state, "card": object()}

            def fake_commit(state):
                calls["commit"] += 1
                return state

            graph_mod.node_generate_answer = fake_generate
            graph_mod.node_store_card = fake_store_card
            graph_mod.node_commit_question_index = fake_commit

            ok_state = {
                "max_answer_attempts": 2,
                "max_full_restarts": 2,
                "validation_threshold": 0.7,
                "strengthen_k_delta": 2,
                "strengthen_min_score_delta": 0.05,
                "validation_score": None,
                "question": "Q",
            }
            graph_mod.node_validate = fake_validate_ok
            card = graph_mod._run_answer_phase(ok_state)  # type: ignore[arg-type]
            self.assertIsNotNone(card)
            self.assertEqual(calls["commit"], 1)

            calls["commit"] = 0
            graph_mod.node_validate = fake_validate_fail
            fail_state = {
                "max_answer_attempts": 1,
                "max_full_restarts": 1,
                "validation_threshold": 0.7,
                "strengthen_k_delta": 2,
                "strengthen_min_score_delta": 0.05,
                "validation_score": None,
                "question": "Q",
                "k": 8,
                "min_score": 0.4,
            }
            card2 = graph_mod._run_answer_phase(fail_state)  # type: ignore[arg-type]
            self.assertIsNone(card2)
            self.assertEqual(calls["commit"], 0)
        finally:
            graph_mod.node_generate_answer = original_generate
            graph_mod.node_validate = original_validate
            graph_mod.node_store_card = original_store_card
            graph_mod.node_commit_question_index = original_commit


if __name__ == "__main__":
    unittest.main()
