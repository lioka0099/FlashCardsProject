import unittest

from app.services.math_verification import verify_math_solution


class MathVerificationTests(unittest.TestCase):
    def test_expression_equivalence(self) -> None:
        result = verify_math_solution(
            {"verification_target": {"kind": "equivalence", "expected": "x**2 + 2*x + 1"}},
            {"final_answer": "(x + 1)^2"},
        )
        self.assertEqual(result.status, "verified")

    def test_derivative(self) -> None:
        result = verify_math_solution(
            {"verification_target": {"kind": "derivative", "expression": "x**2 + 3*x", "variable": "x"}},
            {"final_answer": "2*x + 3"},
        )
        self.assertEqual(result.status, "verified")

    def test_equation_solving(self) -> None:
        result = verify_math_solution(
            {"verification_target": {"kind": "equation", "equation": "2*x + 3 = 7", "variable": "x"}},
            {"final_answer": "x = 2"},
        )
        self.assertEqual(result.status, "verified")

    def test_system_solving(self) -> None:
        result = verify_math_solution(
            {
                "verification_target": {
                    "kind": "system",
                    "equations": ["x + y = 5", "x - y = 1"],
                    "variables": ["x", "y"],
                }
            },
            {"final_answer": "x = 3, y = 2"},
        )
        self.assertEqual(result.status, "verified")

    def test_failed_answer(self) -> None:
        result = verify_math_solution(
            {"verification_target": {"kind": "derivative", "expression": "x**2 + 3*x", "variable": "x"}},
            {"final_answer": "2*x"},
        )
        self.assertEqual(result.status, "failed")

    def test_substitution_with_values(self) -> None:
        result = verify_math_solution(
            {
                "verification_target": {
                    "kind": "substitution",
                    "expression": "2*x + 1",
                    "substitutions": {"x": 3},
                }
            },
            {"final_answer": "7"},
        )
        self.assertEqual(result.status, "verified")

    def test_missing_final_answer_is_parse_error(self) -> None:
        result = verify_math_solution(
            {"verification_target": {"kind": "arithmetic", "expression": "2 + 2"}},
            {},
        )
        self.assertEqual(result.status, "parse_error")


if __name__ == "__main__":
    unittest.main()

