import unittest

from app.services import card_routing
from app.services.card_routing import classify_card_route
from app.services.difficulty_frameworks import clamp_difficulty, framework_for_route, get_level
from app.services.math_classification import MathClassificationEvidence


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
        self.assertEqual(decision.card_route, "math_conceptual")
        self.assertEqual(decision.math_kind, "conceptual")

    def test_numeric_mentions_without_calculation_fall_back(self) -> None:
        decision = classify_card_route(
            topic_label="Chapter 2 Overview",
            context_pack="There are 3 definitions and 7 examples in this section.",
        )
        self.assertEqual(decision.card_route, "default")

    def test_non_math_document_profile_blocks_math_routes(self) -> None:
        decision = classify_card_route(
            topic_label="Polynomial Derivatives",
            context_pack="The derivative rule is d/dx x^n = n*x^(n-1). Differentiate term by term.",
            cached_topic_route={"card_route": "math_calculation", "confidence": 0.95},
            document_math_profile={"kind": "non_math", "confidence": 0.95},
        )
        self.assertEqual(decision.card_route, "default")
        self.assertEqual(decision.subject_type, "general")

    def test_non_math_document_profile_overrides_cached_conceptual_math(self) -> None:
        decision = classify_card_route(
            topic_label="Function Meaning",
            context_pack="Inputs and outputs are related by a function.",
            cached_topic_route={"card_route": "math_conceptual", "confidence": 0.9},
            document_math_profile={"kind": "non_math", "confidence": 0.95},
        )
        self.assertEqual(decision.card_route, "default")

    def test_mixed_document_requires_local_math_evidence(self) -> None:
        original = card_routing.MathClassificationService

        class FakeClassifier:
            def __init__(self, **_kwargs):
                pass

            def score(self, **_kwargs):
                return MathClassificationEvidence(
                    semantic_math_score=0.8,
                    semantic_calculation_score=0.75,
                    semantic_conceptual_score=0.2,
                    source="test",
                )

        try:
            card_routing.MathClassificationService = FakeClassifier  # type: ignore[assignment]
            decision = classify_card_route(
                topic_label="Operations Overview",
                context_pack="This section has 5 examples and 100 practice items.",
                document_math_profile={"kind": "mixed", "confidence": 0.7},
                store=object(),  # type: ignore[arg-type]
            )
        finally:
            card_routing.MathClassificationService = original  # type: ignore[assignment]
        self.assertEqual(decision.card_route, "default")

    def test_mixed_document_allows_grounded_calculation_topic(self) -> None:
        decision = classify_card_route(
            topic_label="Polynomial Derivatives",
            context_pack="The derivative rule is d/dx x^n = n*x^(n-1). Example: f(x)=x^2+3x, differentiate term by term.",
            document_math_profile={"kind": "mixed", "confidence": 0.75},
        )
        self.assertEqual(decision.card_route, "math_calculation")

    def test_math_document_preserves_conceptual_math(self) -> None:
        decision = classify_card_route(
            topic_label="Derivative Meaning",
            context_pack="Explain what a derivative represents and why slope is useful for rates of change.",
            document_math_profile={"kind": "math", "confidence": 0.9},
        )
        self.assertEqual(decision.card_route, "math_conceptual")

    def test_difficulty_registry_keeps_frameworks_separate(self) -> None:
        self.assertEqual(framework_for_route("default"), "bloom")
        self.assertEqual(framework_for_route("math_calculation"), "tag")
        self.assertEqual(framework_for_route("math_conceptual"), "bloom")
        self.assertEqual(get_level("bloom", 3).name, "Apply")
        self.assertEqual(get_level("tag", 3).name, "Procedures With Connections")
        self.assertEqual(clamp_difficulty("tag", 5), 4)
        self.assertEqual(clamp_difficulty("bloom", 5), 5)

    def test_semantic_math_without_formula_routes_to_conceptual(self) -> None:
        original = card_routing.MathClassificationService

        class FakeClassifier:
            def __init__(self, **_kwargs):
                pass

            def score(self, **_kwargs):
                return MathClassificationEvidence(
                    semantic_math_score=0.72,
                    semantic_calculation_score=0.2,
                    semantic_conceptual_score=0.72,
                    source="test",
                )

        try:
            card_routing.MathClassificationService = FakeClassifier  # type: ignore[assignment]
            decision = classify_card_route(
                topic_label="Linear Functions",
                context_pack="A function maps each input to exactly one output.",
                store=object(),  # type: ignore[arg-type]
            )
        finally:
            card_routing.MathClassificationService = original  # type: ignore[assignment]
        self.assertEqual(decision.card_route, "math_conceptual")

    def test_cached_conceptual_route_is_preserved(self) -> None:
        decision = classify_card_route(
            topic_label="Function Meaning",
            context_pack="Inputs and outputs are related by a function.",
            cached_topic_route={"card_route": "math_conceptual", "confidence": 0.8},
        )
        self.assertEqual(decision.card_route, "math_conceptual")


if __name__ == "__main__":
    unittest.main()

