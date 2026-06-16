import unittest

from app.data.db_repository import StoredExam, StoredTopic
from app.services import graph as graph_mod


class MathGenerationGraphTests(unittest.TestCase):
    def test_math_verification_required_before_store(self) -> None:
        state = {
            "card_route": "math_calculation",
            "math_question_payload": {
                "verification_target": {
                    "kind": "derivative",
                    "expression": "x**2 + 3*x",
                    "variable": "x",
                }
            },
            "math_answer_payload": {"final_answer": "2*x"},
        }
        out = graph_mod.node_verify_math(state)  # type: ignore[arg-type]
        self.assertFalse(out["math_validation_passed"])
        self.assertEqual(out["verification_result"]["status"], "failed")

        decision_state = {
            **out,
            "validation_score": 0.99,
            "validation_threshold": 0.75,
            "answer_attempts": 1,
            "max_answer_attempts": 2,
            "full_restart_count": 0,
            "max_full_restarts": 1,
        }
        self.assertEqual(graph_mod.decide_after_validation(decision_state), "strengthen")  # type: ignore[arg-type]

    def test_default_route_skips_math_verification(self) -> None:
        out = graph_mod.node_verify_math({"card_route": "default"})  # type: ignore[arg-type]
        self.assertTrue(out["math_validation_passed"])
        self.assertEqual(out["verification_result"]["status"], "not_applicable")

    def test_route_node_clamps_tag_difficulty(self) -> None:
        original = graph_mod.classify_card_route
        try:
            class FakeDecision:
                card_route = "math_calculation"

                def to_info(self):
                    return {
                        "card_route": "math_calculation",
                        "subject_type": "math",
                        "math_kind": "calculation",
                        "confidence": 0.9,
                    }

            graph_mod.classify_card_route = lambda **_: FakeDecision()  # type: ignore[assignment]
            out = graph_mod.node_route_card(
                {
                    "store_basepath": "store",
                    "exam_id": "e1",
                    "topic_id": "t1",
                    "topic_label": "Derivatives",
                    "context_pack": "f(x)=x^2, differentiate.",
                    "difficulty": 5,
                    "validation_threshold": 0.7,
                }
            )  # type: ignore[arg-type]
            self.assertEqual(out["card_route"], "math_calculation")
            self.assertEqual(out["difficulty_framework"], "tag")
            self.assertEqual(out["difficulty"], 4)
            self.assertEqual(out["validation_threshold"], graph_mod.DEFAULT_CONFIG["math_validation_threshold"])
            self.assertEqual(out["uniqueness_threshold"], graph_mod.DEFAULT_CONFIG["math_uniqueness_threshold"])
            self.assertEqual(out["max_question_attempts"], graph_mod.DEFAULT_CONFIG["math_max_question_attempts"])
            self.assertEqual(out["max_answer_attempts"], graph_mod.DEFAULT_CONFIG["math_max_answer_attempts"])
            self.assertEqual(out["max_full_restarts"], graph_mod.DEFAULT_CONFIG["math_max_full_restarts"])
        finally:
            graph_mod.classify_card_route = original  # type: ignore[assignment]

    def test_route_node_passes_exam_math_profile(self) -> None:
        original_classifier = graph_mod.classify_card_route
        original_store = graph_mod.VectorStore
        captured = {}

        class FakeDB:
            def get_exam(self, _exam_id):
                return StoredExam(
                    exam_id="e1",
                    user_id="u1",
                    title="Exam",
                    mode="mastery",
                    created_at="",
                    updated_at="",
                    info={"math_profile": {"kind": "non_math", "confidence": 0.95}},
                )

            def list_topics(self, *, exam_id):
                return [
                    StoredTopic(
                        topic_id="t1",
                        exam_id=exam_id,
                        label="Topic",
                        created_at="",
                        info={"route_candidate": {"card_route": "math_calculation", "confidence": 0.9}},
                    )
                ]

        class FakeStore:
            def __init__(self, **_kwargs):
                self.db = FakeDB()

        class FakeDecision:
            card_route = "default"

            def to_info(self):
                return {
                    "card_route": "default",
                    "subject_type": "general",
                    "math_kind": "none",
                    "confidence": 0.95,
                }

        def fake_classifier(**kwargs):
            captured.update(kwargs)
            return FakeDecision()

        try:
            graph_mod.VectorStore = FakeStore  # type: ignore[assignment]
            graph_mod.classify_card_route = fake_classifier  # type: ignore[assignment]
            out = graph_mod.node_route_card(
                {
                    "store_basepath": "store",
                    "exam_id": "e1",
                    "topic_id": "t1",
                    "topic_label": "Derivatives",
                    "context_pack": "f(x)=x^2, differentiate.",
                    "difficulty": 3,
                    "validation_threshold": 0.7,
                }
            )  # type: ignore[arg-type]
        finally:
            graph_mod.classify_card_route = original_classifier  # type: ignore[assignment]
            graph_mod.VectorStore = original_store  # type: ignore[assignment]

        self.assertEqual(captured["document_math_profile"]["kind"], "non_math")
        self.assertEqual(out["card_route"], "default")
        self.assertEqual(out["validation_threshold"], graph_mod.DEFAULT_CONFIG["validation_threshold"])
        self.assertEqual(out["uniqueness_threshold"], graph_mod.DEFAULT_CONFIG["uniqueness_threshold"])
        self.assertEqual(out["max_question_attempts"], graph_mod.DEFAULT_CONFIG["max_question_attempts"])
        self.assertEqual(out["max_answer_attempts"], graph_mod.DEFAULT_CONFIG["max_answer_attempts"])
        self.assertEqual(out["max_full_restarts"], graph_mod.DEFAULT_CONFIG["max_full_restarts"])

    def test_verification_uses_solver_target_over_teacher_target(self) -> None:
        state = {
            "card_route": "math_calculation",
            "math_question_payload": {
                "verification_target": {
                    "kind": "derivative",
                    "expression": "x**2 + 3*x",
                    "variable": "x",
                }
            },
            "math_solver_payload": {
                "final_answer": "2*x + 3",
                "verification_target": {
                    "kind": "derivative",
                    "expression": "x**2 + 3*x",
                    "variable": "x",
                    "expected": "2*x + 3",
                },
            },
            "math_answer_payload": {
                "final_answer": "2*x",
                "verification_target": {"kind": "equivalence", "expected": "2*x"},
            },
        }
        out = graph_mod.node_verify_math(state)  # type: ignore[arg-type]
        self.assertFalse(out["math_validation_passed"])
        self.assertEqual(out["verification_result"]["status"], "failed")

    def test_math_failure_adapts_to_lower_difficulty_then_conceptual(self) -> None:
        lowered = graph_mod.node_adapt_math_question_failure(
            {
                "card_route": "math_calculation",
                "difficulty": 3,
                "route_metadata": {},
                "question_failure_reason": "insufficient evidence",
            }
        )  # type: ignore[arg-type]
        self.assertEqual(lowered["card_route"], "math_calculation")
        self.assertEqual(lowered["difficulty"], 2)
        self.assertEqual(lowered["question_attempts"], 0)

        conceptual = graph_mod.node_adapt_math_question_failure(
            {
                "card_route": "math_calculation",
                "difficulty": 1,
                "route_metadata": {"confidence": 0.7},
                "question_failure_reason": "insufficient evidence",
            }
        )  # type: ignore[arg-type]
        self.assertEqual(conceptual["card_route"], "math_conceptual")
        self.assertEqual(conceptual["difficulty_framework"], "bloom")
        self.assertEqual(conceptual["route_metadata"]["math_kind"], "conceptual")
        self.assertEqual(conceptual["answer_attempts"], 0)
        self.assertEqual(conceptual["math_validation_fail_cycles"], 0)

    def test_math_validation_failure_adapts_after_answer_attempts_exhausted(self) -> None:
        state = {
            "card_route": "math_calculation",
            "topic_id": "t1",
            "validation_score": 0.99,
            "validation_threshold": graph_mod.DEFAULT_CONFIG["math_validation_threshold"],
            "validation_critique": "ok grounding",
            "verification_result": {"status": "failed"},
            "math_validation_passed": False,
            "math_validation_fail_cycles": 2,
            "answer_attempts": graph_mod.DEFAULT_CONFIG["math_max_answer_attempts"],
            "max_answer_attempts": graph_mod.DEFAULT_CONFIG["math_max_answer_attempts"],
            "full_restart_count": 0,
            "max_full_restarts": graph_mod.DEFAULT_CONFIG["math_max_full_restarts"],
        }
        self.assertEqual(graph_mod.decide_after_validation(state), "adapt_question")  # type: ignore[arg-type]

    def test_math_question_spec_failure_adapts_without_full_restart(self) -> None:
        state = {
            "card_route": "math_calculation",
            "topic_id": "t1",
            "question": "",
            "question_generation_failed": True,
            "question_failure_reason": "invalid_spec: equation target missing equation",
            "question_attempts": graph_mod.DEFAULT_CONFIG["math_max_question_attempts"],
            "max_question_attempts": graph_mod.DEFAULT_CONFIG["math_max_question_attempts"],
            "full_restart_count": 0,
            "max_full_restarts": graph_mod.DEFAULT_CONFIG["math_max_full_restarts"],
        }

        self.assertEqual(graph_mod.decide_after_question_generation(state), "adapt_question")  # type: ignore[arg-type]

    def test_default_validation_failure_still_restarts_after_answer_attempts(self) -> None:
        state = {
            "card_route": "default",
            "topic_id": "t1",
            "validation_score": 0.1,
            "validation_threshold": graph_mod.DEFAULT_CONFIG["validation_threshold"],
            "validation_critique": "not grounded",
            "verification_result": {"status": "not_applicable"},
            "math_validation_passed": True,
            "answer_attempts": graph_mod.DEFAULT_CONFIG["max_answer_attempts"],
            "max_answer_attempts": graph_mod.DEFAULT_CONFIG["max_answer_attempts"],
            "full_restart_count": 0,
            "max_full_restarts": graph_mod.DEFAULT_CONFIG["max_full_restarts"],
        }
        self.assertEqual(graph_mod.decide_after_validation(state), "full_restart")  # type: ignore[arg-type]

    def test_math_threshold_relaxes_grounding_but_keeps_verifier_required(self) -> None:
        ok_state = {
            "card_route": "math_calculation",
            "topic_id": "t1",
            "validation_score": graph_mod.DEFAULT_CONFIG["math_validation_threshold"],
            "validation_threshold": graph_mod.DEFAULT_CONFIG["math_validation_threshold"],
            "verification_result": {"status": "verified"},
            "math_validation_passed": True,
        }
        self.assertEqual(graph_mod.decide_after_validation(ok_state), "store_card")  # type: ignore[arg-type]

        verifier_failed_state = {
            **ok_state,
            "verification_result": {"status": "failed"},
            "math_validation_passed": False,
            "math_validation_fail_cycles": 1,
            "answer_attempts": 1,
            "max_answer_attempts": graph_mod.DEFAULT_CONFIG["math_max_answer_attempts"],
            "full_restart_count": 0,
            "max_full_restarts": graph_mod.DEFAULT_CONFIG["math_max_full_restarts"],
        }
        self.assertEqual(graph_mod.decide_after_validation(verifier_failed_state), "strengthen")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

