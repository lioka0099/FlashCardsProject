import unittest

from app.services.card_routing import classify_card_route
from app.services.difficulty_frameworks import clamp_difficulty, framework_for_route, get_level


class MathRoutingTests(unittest.TestCase):
    def test_routes_grounded_calculation_context_to_math(self) -> None:
        decision = classify_card_route(
            topic_label="Polynomial Derivatives",
            context_pack="The derivative rule is d/dx x^n = n*x^(n-1). Example: f(x)=x^2+3x, differentiate term by term.",
        )
        self.assertEqual(decision.card_route, "math_calculation")
        self.assertEqual(decision.subject_type, "math")
        self.assertIn("derivative", decision.problem_types)

    def test_conceptual_math_falls_back_to_default(self) -> None:
        decision = classify_card_route(
            topic_label="Derivative Meaning",
            context_pack="Explain what a derivative represents and why slope is useful for rates of change.",
        )
        self.assertEqual(decision.card_route, "default")

    def test_numeric_mentions_without_calculation_fall_back(self) -> None:
        decision = classify_card_route(
            topic_label="Chapter 2 Overview",
            context_pack="There are 3 definitions and 7 examples in this section.",
        )
        self.assertEqual(decision.card_route, "default")

    def test_difficulty_registry_keeps_frameworks_separate(self) -> None:
        self.assertEqual(framework_for_route("default"), "bloom")
        self.assertEqual(framework_for_route("math_calculation"), "tag")
        self.assertEqual(get_level("bloom", 3).name, "Apply")
        self.assertEqual(get_level("tag", 3).name, "Procedures With Connections")
        self.assertEqual(clamp_difficulty("tag", 5), 4)
        self.assertEqual(clamp_difficulty("bloom", 5), 5)


if __name__ == "__main__":
    unittest.main()

