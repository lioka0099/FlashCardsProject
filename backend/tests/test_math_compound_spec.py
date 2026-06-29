import unittest

from app.services.difficulty_frameworks import BLOOM_LEVELS, TAG_LEVELS, get_level
from app.services.math_compound_spec import (
    math_structural_fingerprint,
    normalize_compound_spec,
)
from app.services.math_solver import solve_math_problem
from app.services.math_verification import verify_compound_solution, verify_math_solution


def _eigen_problem(final_answer="1, 3"):
    return {
        "status": "ready",
        "problem_statement": "Matrix A=[[2,1],[1,2]]. Find eigenvalues and decide positive definiteness.",
        "concepts_used": ["eigenvalues", "positive definiteness"],
        "archetype": "eigenvalues-and-definiteness",
        "source_rule": "A symmetric matrix is positive definite iff all eigenvalues are positive.",
        "final_answer": final_answer,
        "steps": [
            {
                "description": "Characteristic polynomial det(A - lambda I) = 0",
                "check": {"kind": "matrix", "operation": "eigenvals", "matrix": [[2, 1], [1, 2]]},
                "is_final": True,
            },
            {"description": "Both eigenvalues positive => positive definite", "check": None},
        ],
    }


class CompoundSpecTests(unittest.TestCase):
    def test_normalize_and_fingerprint(self) -> None:
        result = normalize_compound_spec(_eigen_problem())
        self.assertTrue(result.ok)
        self.assertEqual(len(result.spec["steps"]), 2)
        fp = math_structural_fingerprint(result.spec)
        self.assertIn("arch=eigenvalues-and-definiteness", fp)
        self.assertIn("matrix:eigenvals", fp)

    def test_fingerprint_distinguishes_archetypes(self) -> None:
        a = normalize_compound_spec(_eigen_problem()).spec
        other = _eigen_problem()
        other["archetype"] = "determinant-only"
        other["steps"][0]["check"] = {"kind": "matrix", "operation": "det", "matrix": [[2, 1], [1, 2]]}
        b = normalize_compound_spec(other).spec
        self.assertNotEqual(math_structural_fingerprint(a), math_structural_fingerprint(b))

    def test_verify_compound_passes_on_correct_chain(self) -> None:
        spec = normalize_compound_spec(_eigen_problem()).spec
        result = verify_compound_solution(spec)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.details["canonical_final_answer"], "1, 3")

    def test_verify_compound_fails_on_wrong_final_answer(self) -> None:
        spec = normalize_compound_spec(_eigen_problem(final_answer="2, 4")).spec
        result = verify_compound_solution(spec, declared_final_answer="2, 4")
        self.assertEqual(result.status, "failed")

    def test_verify_compound_fails_on_unsolvable_step(self) -> None:
        bad = _eigen_problem()
        bad["steps"][0]["check"] = {"kind": "equation", "equation": "x = ", "variable": "x"}
        # The malformed check is dropped during normalization, leaving no checkable step.
        result = normalize_compound_spec(bad)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "insufficient_evidence")

    def test_requires_checkable_step(self) -> None:
        narrative_only = _eigen_problem()
        for step in narrative_only["steps"]:
            step["check"] = None
        result = normalize_compound_spec(narrative_only)
        self.assertFalse(result.ok)

    def test_single_primitive_backward_compatible(self) -> None:
        single = {
            "kind": "derivative",
            "expression": "x**2 + 3*x",
            "variable": "x",
            "source_rule": "power rule",
            "problem_statement": "Differentiate x**2 + 3x.",
            "final_answer": "2*x + 3",
        }
        result = normalize_compound_spec(single)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.spec["steps"]), 1)
        self.assertEqual(verify_compound_solution(result.spec).status, "verified")

    def test_missing_problem_statement_rejected(self) -> None:
        spec = _eigen_problem()
        spec.pop("problem_statement")
        self.assertFalse(normalize_compound_spec(spec).ok)


class ExtendedPrimitiveTests(unittest.TestCase):
    def _solve_and_verify(self, target):
        payload = {"kind": target["kind"], "problem_type": target["kind"], "source_rule": "r", "verification_target": target}
        solved = solve_math_problem(payload)
        self.assertEqual(solved.status, "solved", msg=str(solved.details))
        verified = verify_math_solution(
            payload, {"final_answer": solved.final_answer, "verification_target": solved.verification_target}
        )
        self.assertEqual(verified.status, "verified", msg=str(verified.details))
        return solved.final_answer

    def test_matrix_determinant(self) -> None:
        self.assertEqual(self._solve_and_verify({"kind": "matrix", "operation": "det", "matrix": [[2, 1], [1, 2]]}), "3")

    def test_matrix_eigenvalues(self) -> None:
        self.assertEqual(self._solve_and_verify({"kind": "matrix", "operation": "eigenvals", "matrix": [[2, 1], [1, 2]]}), "1, 3")

    def test_limit(self) -> None:
        self.assertEqual(self._solve_and_verify({"kind": "limit", "expression": "sin(x)/x", "variable": "x", "point": "0"}), "1")

    def test_summation(self) -> None:
        self.assertEqual(self._solve_and_verify({"kind": "summation", "expression": "k", "variable": "k", "lower": "1", "upper": "n"}), "n*(n + 1)/2")

    def test_definite_integral(self) -> None:
        self.assertEqual(self._solve_and_verify({"kind": "integral", "expression": "x", "variable": "x", "lower": "0", "upper": "2"}), "2")


class DifficultyDepthTests(unittest.TestCase):
    def test_tag_levels_expose_increasing_depth(self) -> None:
        depths = [TAG_LEVELS[i].depth for i in range(1, 5)]
        self.assertTrue(all(d is not None for d in depths))
        self.assertEqual([d.min_steps for d in depths], [1, 2, 3, 4])
        self.assertTrue(depths[0].min_concepts <= depths[3].min_concepts)
        self.assertFalse(depths[0].requires_method_selection)
        self.assertTrue(depths[3].requires_method_selection)
        self.assertTrue(depths[3].requires_modeling)

    def test_bloom_levels_have_no_depth(self) -> None:
        self.assertTrue(all(level.depth is None for level in BLOOM_LEVELS.values()))

    def test_depth_prompt_mentions_concepts_not_numbers(self) -> None:
        prompt = get_level("tag", 4).depth.as_prompt()
        self.assertIn("concept", prompt.lower())
        self.assertIn("never through larger numbers", prompt.lower())


class AuthoringYieldTests(unittest.TestCase):
    """The authoring gate must trust the solver, not the LLM's stated answer."""

    def _die_variance(self, final_answer="35/12", final_expr="91/6 - (7/2)**2"):
        return {
            "status": "ready",
            "problem_statement": "For a fair six-sided die X, compute Var(X).",
            "archetype": "variance-of-distribution",
            "concepts_used": ["expectation", "variance"],
            "final_answer": final_answer,
            "steps": [
                {"description": "E[X]", "check": {"kind": "summation", "expression": "x*(1/6)", "variable": "x", "lower": 1, "upper": 6}, "is_final": False},
                {"description": "Var", "check": {"kind": "arithmetic", "expression": final_expr}, "is_final": True},
            ],
        }

    def test_solver_answer_adopted_despite_wrong_declared_answer(self) -> None:
        # The LLM's stated answer is wrong, but the spine solves: we keep the problem
        # and adopt the CAS-canonical answer instead of rejecting it (declared="" skips
        # the equality gate, mirroring the authoring path in math_student_model).
        spec = normalize_compound_spec(self._die_variance(final_answer="2.9"), require_source_rule=False)
        self.assertTrue(spec.ok)
        result = verify_compound_solution(spec.spec, declared_final_answer="")
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.details["canonical_final_answer"], "35/12")

    def test_missing_source_rule_does_not_block(self) -> None:
        spec = normalize_compound_spec(self._die_variance(), require_source_rule=False)
        self.assertTrue(spec.ok)


if __name__ == "__main__":
    unittest.main()
