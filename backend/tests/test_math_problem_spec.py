import unittest

from app.services.math_problem_spec import normalize_math_problem_spec, render_math_question


class MathProblemSpecTests(unittest.TestCase):
    def test_normalizes_derivative_spec(self) -> None:
        result = normalize_math_problem_spec(
            {
                "status": "ready",
                "kind": "derivative",
                "source_rule": "Use the power rule.",
                "expression": "f(x)=x^2 + 3x",
                "variable": "x",
            },
            require_source_rule=True,
        )

        self.assertTrue(result.ok)
        target = result.spec["verification_target"]
        self.assertEqual(target["kind"], "derivative")
        self.assertEqual(target["expression"], "x**2 + 3*x")
        self.assertEqual(render_math_question(result.spec), "Differentiate x**2 + 3*x with respect to x.")

    def test_normalizes_equation_and_rejects_bad_equation(self) -> None:
        result = normalize_math_problem_spec(
            {
                "kind": "equation",
                "source_rule": "Solve linear equations by isolating the variable.",
                "equation": "2x + 3 = 7",
                "variable": "x",
            },
            require_source_rule=True,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.spec["verification_target"]["equation"], "2*x + 3 = 7")

        bad = normalize_math_problem_spec(
            {
                "kind": "equation",
                "source_rule": "Solve equations.",
                "equation": "2*x + 3",
                "variable": "x",
            },
            require_source_rule=True,
        )
        self.assertFalse(bad.ok)
        self.assertEqual(bad.status, "invalid_spec")

    def test_insufficient_evidence_is_structured_failure(self) -> None:
        result = normalize_math_problem_spec(
            {"status": "insufficient_evidence", "reason": "No formula or procedure is present."},
            require_source_rule=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "insufficient_evidence")

    def test_requires_source_rule_when_requested(self) -> None:
        result = normalize_math_problem_spec(
            {"kind": "simplify", "expression": "x^2 + 2x + 1"},
            require_source_rule=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "invalid_spec")


if __name__ == "__main__":
    unittest.main()
