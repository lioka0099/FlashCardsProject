import unittest

from app.services.math_solver import solve_math_problem


class MathSolverTests(unittest.TestCase):
    def test_derivative(self) -> None:
        result = solve_math_problem(
            {"verification_target": {"kind": "derivative", "expression": "x**2 + 3*x", "variable": "x"}}
        )
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.final_answer, "2*x + 3")
        self.assertEqual(result.verification_target["expected"], "2*x + 3")

    def test_equation(self) -> None:
        result = solve_math_problem(
            {"verification_target": {"kind": "equation", "equation": "2*x + 3 = 7", "variable": "x"}}
        )
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.final_answer, "x = 2")

    def test_system(self) -> None:
        result = solve_math_problem(
            {
                "verification_target": {
                    "kind": "system",
                    "equations": ["x + y = 5", "x - y = 1"],
                    "variables": ["x", "y"],
                }
            }
        )
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.final_answer, "x = 3, y = 2")

    def test_substitution(self) -> None:
        result = solve_math_problem(
            {
                "verification_target": {
                    "kind": "substitution",
                    "expression": "2*x + 1",
                    "substitutions": {"x": 3},
                }
            }
        )
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.final_answer, "7")

    def test_unsupported(self) -> None:
        result = solve_math_problem({"verification_target": {"kind": "geometry_proof"}})
        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
