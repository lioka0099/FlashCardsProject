from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import sympy as sp


@dataclass
class VerificationResult:
    status: str
    method: str = "sympy"
    confidence: float = 0.0
    checked_final_answer: bool = False
    checked_steps: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_expr_text(text: Any) -> str:
    raw = str(text or "").strip()
    raw = raw.replace("^", "**")
    raw = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", raw)
    raw = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", raw)
    return raw


def _parse_expr(text: Any) -> sp.Expr:
    return sp.sympify(_normalize_expr_text(text))


def _parse_symbols(names: Iterable[str] | None, fallback: str = "x") -> Dict[str, sp.Symbol]:
    cleaned = [str(n).strip() for n in (names or []) if str(n).strip()]
    if not cleaned:
        cleaned = [fallback]
    return {name: sp.Symbol(name) for name in cleaned}


def _equivalent(actual: Any, expected: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        lhs = _parse_expr(actual)
        rhs = _parse_expr(expected)
        simplified = sp.simplify(lhs - rhs)
        if simplified == 0:
            return True
        if simplified.free_symbols:
            return False
        return abs(float(simplified.evalf())) <= tolerance
    except Exception:
        try:
            return abs(float(actual) - float(expected)) <= tolerance
        except Exception:
            raise


def _verify_equivalence(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expected = target.get("expected") or target.get("equivalent_to") or target.get("answer")
    if expected is None:
        return _unsupported("equivalence target missing expected expression", target)
    ok = _equivalent(final_answer, expected)
    return _checked(ok, {"kind": "equivalence", "expected": str(expected), "actual": str(final_answer)})


def _verify_arithmetic(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expression = target.get("expression")
    expected = target.get("expected")
    if expected is None and expression is not None:
        expr = _parse_expr(expression)
        substitutions = target.get("substitutions") or target.get("values")
        if isinstance(substitutions, dict):
            sub_map = {sp.Symbol(str(k)): _parse_expr(v) for k, v in substitutions.items()}
            expr = expr.subs(sub_map)
        expected = sp.N(expr) if not expr.free_symbols else sp.simplify(expr)
    if expected is None:
        return _unsupported("arithmetic target missing expression or expected answer", target)
    tolerance = float(target.get("tolerance") or 1e-6)
    ok = _equivalent(final_answer, expected, tolerance=tolerance)
    return _checked(ok, {"kind": "arithmetic", "expected": str(expected), "actual": str(final_answer)})


def _verify_derivative(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "x")
    if not expression:
        return _unsupported("derivative target missing expression", target)
    symbol = sp.Symbol(variable)
    expected = target.get("expected") or sp.diff(_parse_expr(expression), symbol)
    ok = _equivalent(final_answer, expected)
    return _checked(
        ok,
        {
            "kind": "derivative",
            "expression": str(expression),
            "variable": variable,
            "expected": str(expected),
            "actual": str(final_answer),
        },
    )


def _verify_integral(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "x")
    if not expression:
        return _unsupported("integral target missing expression", target)
    symbol = sp.Symbol(variable)
    if target.get("expected") is not None:
        expected = target["expected"]
        ok = _equivalent(final_answer, expected)
    else:
        ok = _equivalent(sp.diff(_parse_expr(final_answer), symbol), _parse_expr(expression))
        expected = "derivative backcheck"
    return _checked(
        ok,
        {
            "kind": "integral",
            "expression": str(expression),
            "variable": variable,
            "expected": str(expected),
            "actual": str(final_answer),
        },
    )


def _verify_equation(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    equation = target.get("equation")
    variable = str(target.get("variable") or "x")
    expected = target.get("expected")
    if expected is not None:
        actual = _extract_assignment_value(final_answer, variable)
        ok = _equivalent(actual, expected)
        return _checked(ok, {"kind": "equation", "expected": str(expected), "actual": str(final_answer)})
    if not equation:
        return _unsupported("equation target missing equation or expected answer", target)

    lhs_text, rhs_text = _split_equation(str(equation))
    symbol = sp.Symbol(variable)
    solutions = sp.solve(sp.Eq(_parse_expr(lhs_text), _parse_expr(rhs_text)), symbol)
    actual = _extract_assignment_value(final_answer, variable)
    ok = any(_equivalent(actual, sol) for sol in solutions)
    return _checked(
        ok,
        {
            "kind": "equation",
            "equation": str(equation),
            "variable": variable,
            "expected_solutions": [str(s) for s in solutions],
            "actual": str(final_answer),
        },
    )


def _verify_system(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    equations = target.get("equations")
    variables = [str(v) for v in (target.get("variables") or []) if str(v).strip()]
    expected = target.get("expected")
    if expected is not None:
        return _checked(
            _compare_solution_mapping(final_answer, expected, variables),
            {"kind": "system", "expected": expected, "actual": final_answer},
        )
    if not equations or not variables:
        return _unsupported("system target missing equations or variables", target)

    symbols = [sp.Symbol(v) for v in variables]
    sympy_equations = []
    for equation in equations:
        lhs_text, rhs_text = _split_equation(str(equation))
        sympy_equations.append(sp.Eq(_parse_expr(lhs_text), _parse_expr(rhs_text)))
    solutions = sp.solve(sympy_equations, symbols, dict=True)
    actual_map = _parse_solution_mapping(final_answer, variables)
    ok = any(
        all(_equivalent(actual_map.get(str(sym)), value) for sym, value in sol.items())
        for sol in solutions
    )
    return _checked(
        ok,
        {
            "kind": "system",
            "expected_solutions": [{str(k): str(v) for k, v in sol.items()} for sol in solutions],
            "actual": actual_map,
        },
    )


def _parse_matrix(value: Any) -> sp.Matrix:
    if isinstance(value, sp.MatrixBase):
        return value
    if isinstance(value, (list, tuple)):
        rows = []
        for row in value:
            if isinstance(row, (list, tuple)):
                rows.append([_parse_expr(cell) for cell in row])
            else:
                rows.append([_parse_expr(row)])
        return sp.Matrix(rows)
    return sp.Matrix(sp.sympify(_normalize_expr_text(value)))


def _matrices_equal(actual: Any, expected: Any) -> bool:
    try:
        A = _parse_matrix(actual)
        B = _parse_matrix(expected)
        if A.shape != B.shape:
            return False
        return sp.simplify(A - B).is_zero_matrix is True
    except Exception:
        return False


def _value_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip().strip("[]")
    return [part.strip() for part in text.split(",") if part.strip()]


def _multiset_equivalent(actual: Any, expected: Any) -> bool:
    actual_items = _value_list(actual)
    expected_items = _value_list(expected)
    if len(actual_items) != len(expected_items):
        return False
    remaining = list(expected_items)
    for item in actual_items:
        match_index = None
        for idx, candidate in enumerate(remaining):
            try:
                if _equivalent(item, candidate):
                    match_index = idx
                    break
            except Exception:
                continue
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _verify_matrix(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    operation = str(target.get("operation") or "").strip().lower()
    matrix = target.get("matrix")
    if matrix is None:
        return _unsupported("matrix target missing matrix", target)
    A = _parse_matrix(matrix)
    expected = target.get("expected")
    if operation in {"determinant", "det"}:
        computed = expected if expected is not None else sp.simplify(A.det())
        ok = _equivalent(final_answer, computed)
    elif operation == "rank":
        computed = expected if expected is not None else A.rank()
        ok = _equivalent(final_answer, computed)
    elif operation in {"eigenvals", "eigenvalues", "eig"}:
        if expected is None:
            eig = A.eigenvals()
            computed = []
            for value, multiplicity in eig.items():
                computed.extend([value] * int(multiplicity))
        else:
            computed = expected
        ok = _multiset_equivalent(final_answer, computed)
    elif operation in {"inverse", "inv"}:
        computed = expected if expected is not None else sp.simplify(A.inv())
        ok = _matrices_equal(final_answer, computed)
    elif operation in {"transpose", "t"}:
        computed = expected if expected is not None else A.T
        ok = _matrices_equal(final_answer, computed)
    elif operation in {"multiply", "product", "matmul"}:
        matrix_b = target.get("matrix_b")
        if matrix_b is None:
            return _unsupported("matrix multiply target missing matrix_b", target)
        computed = expected if expected is not None else sp.simplify(A * _parse_matrix(matrix_b))
        ok = _matrices_equal(final_answer, computed)
    else:
        return _unsupported(f"unsupported matrix operation: {operation or '(missing)'}", target)
    return _checked(ok, {"kind": "matrix", "operation": operation, "expected": str(computed), "actual": str(final_answer)})


def _verify_limit(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "x")
    point = target.get("point")
    if not expression or point is None:
        return _unsupported("limit target missing expression or point", target)
    if target.get("expected") is not None:
        expected = target["expected"]
    else:
        direction = str(target.get("direction") or "+").strip()
        dir_arg = "-" if direction == "-" else "+"
        expected = sp.limit(_parse_expr(expression), sp.Symbol(variable), _parse_expr(point), dir_arg)
    ok = _equivalent(final_answer, expected)
    return _checked(ok, {"kind": "limit", "expression": str(expression), "expected": str(expected), "actual": str(final_answer)})


def _verify_summation(target: Dict[str, Any], final_answer: Any) -> VerificationResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "k")
    lower = target.get("lower")
    upper = target.get("upper")
    if not expression or lower is None or upper is None:
        return _unsupported("summation target missing expression, lower, or upper", target)
    if target.get("expected") is not None:
        expected = target["expected"]
    else:
        expected = sp.summation(_parse_expr(expression), (sp.Symbol(variable), _parse_expr(lower), _parse_expr(upper)))
    ok = _equivalent(final_answer, expected)
    return _checked(ok, {"kind": "summation", "expression": str(expression), "expected": str(expected), "actual": str(final_answer)})


def _split_equation(equation: str) -> tuple[str, str]:
    if "=" not in equation:
        raise ValueError("Equation must contain '='")
    lhs, rhs = equation.split("=", 1)
    return lhs.strip(), rhs.strip()


def _extract_assignment_value(value: Any, variable: str) -> Any:
    text = str(value)
    match = re.search(rf"\b{re.escape(variable)}\s*=\s*([^,;\n]+)", text)
    return match.group(1).strip() if match else value


def _parse_solution_mapping(value: Any, variables: List[str]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    text = str(value)
    out: Dict[str, Any] = {}
    for variable in variables:
        match = re.search(rf"\b{re.escape(variable)}\s*=\s*([^,;\n]+)", text)
        if match:
            out[variable] = match.group(1).strip()
    return out


def _compare_solution_mapping(actual: Any, expected: Any, variables: List[str]) -> bool:
    expected_map = expected if isinstance(expected, dict) else _parse_solution_mapping(expected, variables)
    actual_map = actual if isinstance(actual, dict) else _parse_solution_mapping(actual, variables)
    if not expected_map or not actual_map:
        return False
    keys = variables or [str(k) for k in expected_map.keys()]
    return all(_equivalent(actual_map.get(k), expected_map.get(k)) for k in keys)


def _unsupported(reason: str, target: Dict[str, Any]) -> VerificationResult:
    return VerificationResult(
        status="unsupported",
        confidence=0.0,
        checked_final_answer=False,
        checked_steps=False,
        details={"reason": reason, "target": target},
    )


def _checked(ok: bool, details: Dict[str, Any]) -> VerificationResult:
    return VerificationResult(
        status="verified" if ok else "failed",
        confidence=1.0 if ok else 0.0,
        checked_final_answer=True,
        checked_steps=False,
        details=details,
    )


def verify_compound_solution(
    spec: Optional[Dict[str, Any]],
    *,
    declared_final_answer: Any = None,
) -> VerificationResult:
    """
    Deterministically verify a compound (multi-step) calculation spec.

    Guarantees: every machine-checkable step is solvable by SymPy (the
    solver-of-record), and the declared final answer is CAS-equivalent to the
    result computed for the designated final step. Narrative steps (no ``check``)
    carry no guarantee and are ignored here. Returns ``verified`` only when the
    computational spine and the final answer all check out.
    """
    from app.services.math_solver import solve_math_problem  # local import avoids cycle

    spec = spec or {}
    steps = spec.get("steps") if isinstance(spec.get("steps"), list) else []
    checkable = [s for s in steps if isinstance(s, dict) and isinstance(s.get("check"), dict)]
    if not checkable:
        return _unsupported("compound spec has no machine-checkable step", spec)

    step_results: List[Dict[str, Any]] = []
    final_canonical: Any = None
    final_target: Dict[str, Any] = {}
    for index, step in enumerate(steps):
        check = step.get("check") if isinstance(step, dict) else None
        if not isinstance(check, dict):
            continue
        solver = solve_math_problem(check)
        ok = solver.status == "solved"
        step_results.append(
            {
                "index": index,
                "kind": str((check.get("verification_target") or check).get("kind") or ""),
                "status": solver.status,
                "final_answer": solver.final_answer,
            }
        )
        if not ok:
            return VerificationResult(
                status="failed",
                confidence=0.0,
                checked_final_answer=False,
                checked_steps=True,
                details={"reason": f"step {index} is not solvable", "steps": step_results, "solver": solver.details},
            )
        if step.get("is_final"):
            final_canonical = solver.final_answer
            final_target = solver.verification_target or (check.get("verification_target") or check)

    if final_canonical is None:  # no step flagged final: use the last checkable result
        last = step_results[-1]
        final_canonical = last["final_answer"]
        last_check = checkable[-1]["check"]
        final_target = last_check.get("verification_target") or last_check

    declared = declared_final_answer if declared_final_answer is not None else spec.get("final_answer")
    final_checked = False
    final_ok = True
    if declared is not None and str(declared).strip():
        sub = verify_math_solution(
            {"verification_target": final_target},
            {"final_answer": declared, "verification_target": final_target},
        )
        final_ok = sub.status == "verified"
        final_checked = sub.status in {"verified", "failed"}

    status = "verified" if final_ok else "failed"
    return VerificationResult(
        status=status,
        method="sympy",
        confidence=1.0 if status == "verified" else 0.0,
        checked_final_answer=final_checked,
        checked_steps=True,
        details={
            "steps": step_results,
            "checked_steps": len(step_results),
            "canonical_final_answer": str(final_canonical),
            "declared_final_answer": (str(declared) if declared is not None else None),
            "final_match": final_ok,
        },
    )


def verify_math_solution(
    question_payload: Optional[Dict[str, Any]],
    answer_payload: Optional[Dict[str, Any]],
) -> VerificationResult:
    question_payload = question_payload or {}
    answer_payload = answer_payload or {}
    target = answer_payload.get("verification_target") or question_payload.get("verification_target") or {}
    if not isinstance(target, dict):
        return _unsupported("verification target is not an object", {})

    final_answer = (
        answer_payload.get("final_answer")
        or target.get("actual")
        or answer_payload.get("expected_final_answer")
    )
    if final_answer is None:
        return VerificationResult(
            status="parse_error",
            confidence=0.0,
            checked_final_answer=False,
            checked_steps=False,
            details={"reason": "missing final_answer"},
        )

    kind = str(target.get("kind") or question_payload.get("problem_type") or "").strip().lower()
    try:
        if kind in {"arithmetic", "calculate", "numeric", "substitution"}:
            return _verify_arithmetic(target, final_answer)
        if kind in {"simplify", "equivalence", "expression_equivalence", "expand", "factor"}:
            return _verify_equivalence(target, final_answer)
        if kind in {"solve_equation", "equation", "linear_equation"}:
            return _verify_equation(target, final_answer)
        if kind in {"system", "system_of_equations", "solve_system"}:
            return _verify_system(target, final_answer)
        if kind in {"derivative", "differentiate"}:
            return _verify_derivative(target, final_answer)
        if kind in {"integral", "integrate"}:
            return _verify_integral(target, final_answer)
        if kind in {"matrix", "matrices"}:
            return _verify_matrix(target, final_answer)
        if kind in {"limit"}:
            return _verify_limit(target, final_answer)
        if kind in {"summation", "sum", "series"}:
            return _verify_summation(target, final_answer)
        if target.get("expected") is not None:
            return _verify_equivalence(target, final_answer)
        return _unsupported(f"unsupported verification kind: {kind or '(missing)'}", target)
    except Exception as exc:
        return VerificationResult(
            status="parse_error",
            confidence=0.0,
            checked_final_answer=False,
            checked_steps=False,
            details={"reason": str(exc), "target": target, "final_answer": str(final_answer)},
        )

