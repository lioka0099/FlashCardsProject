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

