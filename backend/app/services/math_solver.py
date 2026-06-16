from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import sympy as sp

from app.services.math_problem_spec import normalize_math_problem_spec


@dataclass
class MathSolverResult:
    status: str
    method: str = "sympy"
    final_answer: Optional[str] = None
    solution_steps: List[str] = field(default_factory=list)
    verification_target: Dict[str, Any] = field(default_factory=dict)
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


def _split_equation(equation: str) -> tuple[str, str]:
    if "=" not in equation:
        raise ValueError("Equation must contain '='")
    lhs, rhs = equation.split("=", 1)
    return lhs.strip(), rhs.strip()


def _sympify_mapping(raw: Any) -> Dict[sp.Symbol, sp.Expr]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[sp.Symbol, sp.Expr] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if name:
            out[sp.Symbol(name)] = _parse_expr(value)
    return out


def _format_expr(value: Any) -> str:
    return str(sp.simplify(value))


def _unsupported(reason: str, payload: Dict[str, Any]) -> MathSolverResult:
    return MathSolverResult(status="unsupported", details={"reason": reason, "payload": payload})


def _parse_error(exc: Exception, payload: Dict[str, Any]) -> MathSolverResult:
    return MathSolverResult(status="parse_error", details={"reason": str(exc), "payload": payload})


def _invalid_spec(reason: str, payload: Dict[str, Any]) -> MathSolverResult:
    return MathSolverResult(status="invalid_spec", details={"reason": reason, "payload": payload})


def _solved(final: Any, steps: List[str], target: Dict[str, Any]) -> MathSolverResult:
    final_answer = str(final)
    normalized_target = dict(target)
    normalized_target.setdefault("expected", final_answer)
    return MathSolverResult(
        status="solved",
        final_answer=final_answer,
        solution_steps=steps,
        verification_target=normalized_target,
        details={"checked_final_answer": True},
    )


def _target_from_payload(question_payload: Dict[str, Any]) -> Dict[str, Any]:
    target = question_payload.get("verification_target") or {}
    return target if isinstance(target, dict) else {}


def _kind(question_payload: Dict[str, Any], target: Dict[str, Any]) -> str:
    return str(target.get("kind") or question_payload.get("problem_type") or "").strip().lower()


def _solve_arithmetic(kind: str, target: Dict[str, Any]) -> MathSolverResult:
    expression = target.get("expression")
    expected = target.get("expected")
    substitutions = _sympify_mapping(target.get("substitutions") or target.get("values"))
    if expression is None and expected is None:
        return _unsupported("arithmetic target missing expression or expected answer", target)
    if expression is None:
        return _solved(str(expected), ["Use the provided expected result."], {**target, "kind": kind or "arithmetic"})
    expr = _parse_expr(expression)
    if substitutions:
        expr = expr.subs(substitutions)
    value = sp.simplify(expr)
    steps = [f"Start with {expression}.", f"Evaluate to get {_format_expr(value)}."]
    return _solved(_format_expr(value), steps, {**target, "kind": kind or "arithmetic"})


def _solve_expression(kind: str, target: Dict[str, Any]) -> MathSolverResult:
    expression = target.get("expression")
    expected = target.get("expected")
    if expression is None and expected is None:
        return _unsupported("expression target missing expression or expected answer", target)
    if expression is None:
        return _solved(str(expected), ["Use the provided equivalent expression."], {**target, "kind": kind})
    expr = _parse_expr(expression)
    if kind in {"factor"}:
        result = sp.factor(expr)
        action = "Factor"
    elif kind in {"expand"}:
        result = sp.expand(expr)
        action = "Expand"
    else:
        result = sp.simplify(expr)
        action = "Simplify"
    return _solved(
        _format_expr(result),
        [f"{action} {expression}.", f"The equivalent expression is {_format_expr(result)}."],
        {**target, "kind": kind or "simplify"},
    )


def _solve_equation(target: Dict[str, Any]) -> MathSolverResult:
    equation = target.get("equation")
    variable = str(target.get("variable") or "x")
    if not equation:
        return _unsupported("equation target missing equation", target)
    lhs_text, rhs_text = _split_equation(str(equation))
    symbol = sp.Symbol(variable)
    solutions = sp.solve(sp.Eq(_parse_expr(lhs_text), _parse_expr(rhs_text)), symbol)
    if not solutions:
        return _unsupported("equation has no symbolic solution", target)
    final = f"{variable} = {_format_expr(solutions[0])}" if len(solutions) == 1 else str([_format_expr(s) for s in solutions])
    return _solved(
        final,
        [f"Solve {equation} for {variable}.", f"The solution is {final}."],
        {**target, "kind": "equation", "expected": _format_expr(solutions[0]) if len(solutions) == 1 else final},
    )


def _solve_system(target: Dict[str, Any]) -> MathSolverResult:
    equations = target.get("equations")
    variables = [str(v).strip() for v in (target.get("variables") or []) if str(v).strip()]
    if not equations or not variables:
        return _unsupported("system target missing equations or variables", target)
    symbols = [sp.Symbol(v) for v in variables]
    sympy_equations = []
    for equation in equations:
        lhs_text, rhs_text = _split_equation(str(equation))
        sympy_equations.append(sp.Eq(_parse_expr(lhs_text), _parse_expr(rhs_text)))
    solutions = sp.solve(sympy_equations, symbols, dict=True)
    if not solutions:
        return _unsupported("system has no symbolic solution", target)
    solution = solutions[0]
    final_map = {str(sym): _format_expr(solution[sym]) for sym in symbols if sym in solution}
    final = ", ".join(f"{key} = {value}" for key, value in final_map.items())
    return _solved(
        final,
        ["Solve the system of equations.", f"The solution is {final}."],
        {**target, "kind": "system", "expected": final_map},
    )


def _solve_derivative(target: Dict[str, Any]) -> MathSolverResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "x")
    if not expression:
        return _unsupported("derivative target missing expression", target)
    symbol = sp.Symbol(variable)
    result = sp.diff(_parse_expr(expression), symbol)
    return _solved(
        _format_expr(result),
        [f"Differentiate {expression} with respect to {variable}.", f"The derivative is {_format_expr(result)}."],
        {**target, "kind": "derivative"},
    )


def _solve_integral(target: Dict[str, Any]) -> MathSolverResult:
    expression = target.get("expression")
    variable = str(target.get("variable") or "x")
    if not expression:
        return _unsupported("integral target missing expression", target)
    symbol = sp.Symbol(variable)
    result = sp.integrate(_parse_expr(expression), symbol)
    final = f"{_format_expr(result)} + C"
    return _solved(
        final,
        [f"Integrate {expression} with respect to {variable}.", f"An antiderivative is {final}."],
        {**target, "kind": "integral"},
    )


def solve_math_problem(question_payload: Optional[Dict[str, Any]]) -> MathSolverResult:
    payload = question_payload or {}
    if not isinstance(payload, dict):
        return _invalid_spec("math payload is not an object", {"payload": payload})

    if isinstance(payload.get("verification_target"), dict):
        target = _target_from_payload(payload)
    else:
        normalized = normalize_math_problem_spec(payload)
        if not normalized.ok:
            status = normalized.status if normalized.status in {"insufficient_evidence", "unsupported"} else "invalid_spec"
            return MathSolverResult(
                status=status,
                details={
                    "reason": normalized.reason,
                    "payload": payload,
                    **normalized.details,
                },
            )
        payload = normalized.spec
        target = _target_from_payload(payload)
    kind = _kind(payload, target)
    try:
        if kind in {"arithmetic", "calculate", "numeric", "substitution"}:
            return _solve_arithmetic(kind or "arithmetic", target)
        if kind in {"simplify", "equivalence", "expression_equivalence", "expand", "factor"}:
            return _solve_expression(kind or "simplify", target)
        if kind in {"solve_equation", "equation", "linear_equation"}:
            return _solve_equation(target)
        if kind in {"system", "system_of_equations", "solve_system"}:
            return _solve_system(target)
        if kind in {"derivative", "differentiate"}:
            return _solve_derivative(target)
        if kind in {"integral", "integrate"}:
            return _solve_integral(target)
        if target.get("expected") is not None:
            return _solved(str(target["expected"]), ["Use the provided expected result."], {**target, "kind": kind or "equivalence"})
        return _unsupported(f"unsupported solver kind: {kind or '(missing)'}", payload)
    except Exception as exc:
        return _parse_error(exc, payload)
