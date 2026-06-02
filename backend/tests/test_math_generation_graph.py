import unittest

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
        finally:
            graph_mod.classify_card_route = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()

