from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_KINDS = {
    "arithmetic",
    "substitution",
    "simplify",
    "equivalence",
    "expand",
    "factor",
    "equation",
    "system",
    "derivative",
    "integral",
}

KIND_ALIASES = {
    "calculate": "arithmetic",
    "numeric": "arithmetic",
    "expression_equivalence": "equivalence",
    "solve_equation": "equation",
    "linear_equation": "equation",
    "system_of_equations": "system",
    "solve_system": "system",
    "differentiate": "derivative",
    "integrate": "integral",
}


@dataclass
class MathProblemSpecResult:
    ok: bool
    status: str
    spec: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def normalize_math_problem_spec(
    payload: Optional[Dict[str, Any]],
    *,
    require_source_rule: bool = False,
) -> MathProblemSpecResult:
    """
    Normalize LLM-produced math problem specs into the exact shape SymPy solvers expect.

    The returned spec keeps a `verification_target` for compatibility with the rest of
    the card generation pipeline, while also exposing `kind` at the top level.
    """
    raw = payload or {}
    if not isinstance(raw, dict):
        return _invalid("spec payload is not an object", {"payload": raw})

    status = str(raw.get("status") or "ready").strip().lower()
    if status == "insufficient_evidence":
        return MathProblemSpecResult(
            ok=False,
            status="insufficient_evidence",
            reason=str(raw.get("reason") or "Insufficient evidence for a grounded calculation spec."),
            details={"payload": raw},
        )
    if status and status != "ready":
        return _invalid(f"unsupported spec status: {status}", {"payload": raw})

    target = raw.get("verification_target") if isinstance(raw.get("verification_target"), dict) else raw
    kind = _normalize_kind(target.get("kind") or raw.get("kind") or raw.get("problem_type"))
    if not kind:
        return _invalid("missing problem kind", {"payload": raw})
    if kind not in SUPPORTED_KINDS:
        return MathProblemSpecResult(
            ok=False,
            status="unsupported",
            reason=f"unsupported problem kind: {kind}",
            details={"payload": raw},
        )

    source_rule = _source_rule(raw, target)
    if require_source_rule and not source_rule:
        return _invalid("missing source_rule", {"payload": raw})

    normalized_target = _normalize_target(kind, target)
    if not normalized_target.ok:
        return normalized_target

    spec: Dict[str, Any] = {
        "status": "ready",
        "kind": kind,
        "problem_type": kind,
        "verification_target": normalized_target.spec,
    }
    if source_rule:
        spec["source_rule"] = source_rule
        spec["source_rules"] = [source_rule]
    if raw.get("reason"):
        spec["reason"] = str(raw["reason"])
    return MathProblemSpecResult(ok=True, status="ready", spec=spec)


def render_math_question(spec: Dict[str, Any]) -> str:
    """Render a deterministic user-facing question from a normalized math spec."""
    target = spec.get("verification_target") if isinstance(spec.get("verification_target"), dict) else spec
    kind = _normalize_kind(target.get("kind") or spec.get("kind"))
    if kind == "derivative":
        return f"Differentiate {target['expression']} with respect to {target['variable']}."
    if kind == "integral":
        return f"Integrate {target['expression']} with respect to {target['variable']}."
    if kind == "equation":
        return f"Solve {target['equation']} for {target['variable']}."
    if kind == "system":
        equations = "; ".join(str(eq) for eq in target["equations"])
        variables = ", ".join(str(v) for v in target["variables"])
        return f"Solve the system {equations} for {variables}."
    if kind in {"simplify", "factor", "expand"}:
        verb = {"simplify": "Simplify", "factor": "Factor", "expand": "Expand"}[kind]
        return f"{verb} {target['expression']}."
    if kind == "equivalence":
        if target.get("expression") and target.get("expected"):
            return f"Show that {target['expression']} is equivalent to {target['expected']}."
        return f"Find an equivalent expression for {target.get('expression') or target['expected']}."
    if kind in {"arithmetic", "substitution"}:
        substitutions = target.get("substitutions")
        if isinstance(substitutions, dict) and substitutions:
            values = ", ".join(f"{key}={value}" for key, value in substitutions.items())
            return f"Evaluate {target['expression']} when {values}."
        return f"Evaluate {target['expression']}."
    raise ValueError(f"Cannot render unsupported math problem kind: {kind}")


def _normalize_target(kind: str, target: Dict[str, Any]) -> MathProblemSpecResult:
    out: Dict[str, Any] = {"kind": kind}
    if kind in {"derivative", "integral"}:
        expression = _normalize_expression(target.get("expression"))
        variable = _normalize_variable(target.get("variable") or "x")
        if not expression:
            return _invalid(f"{kind} target missing expression", {"target": target})
        out.update({"expression": expression, "variable": variable})
        return MathProblemSpecResult(ok=True, status="ready", spec=out)

    if kind == "equation":
        equation = _normalize_equation(target.get("equation"))
        variable = _normalize_variable(target.get("variable") or "x")
        if not equation:
            return _invalid("equation target missing equation", {"target": target})
        out.update({"equation": equation, "variable": variable})
        if target.get("expected") is not None:
            out["expected"] = _normalize_expression(target["expected"])
        return MathProblemSpecResult(ok=True, status="ready", spec=out)

    if kind == "system":
        equations = [_normalize_equation(eq) for eq in (target.get("equations") or [])]
        equations = [eq for eq in equations if eq]
        variables = [_normalize_variable(v) for v in (target.get("variables") or []) if str(v).strip()]
        if not equations or not variables:
            return _invalid("system target missing equations or variables", {"target": target})
        out.update({"equations": equations, "variables": variables})
        if target.get("expected") is not None:
            out["expected"] = target["expected"]
        return MathProblemSpecResult(ok=True, status="ready", spec=out)

    if kind in {"arithmetic", "substitution"}:
        expression = _normalize_expression(target.get("expression"))
        expected = target.get("expected")
        if not expression and expected is None:
            return _invalid(f"{kind} target missing expression or expected", {"target": target})
        if expression:
            out["expression"] = expression
        if expected is not None:
            out["expected"] = _normalize_expression(expected)
        substitutions = _normalize_substitutions(target.get("substitutions") or target.get("values"))
        if substitutions:
            out["substitutions"] = substitutions
        return MathProblemSpecResult(ok=True, status="ready", spec=out)

    if kind in {"simplify", "equivalence", "expand", "factor"}:
        expression = _normalize_expression(target.get("expression"))
        expected = target.get("expected")
        if not expression and expected is None:
            return _invalid(f"{kind} target missing expression or expected", {"target": target})
        if expression:
            out["expression"] = expression
        if expected is not None:
            out["expected"] = _normalize_expression(expected)
        return MathProblemSpecResult(ok=True, status="ready", spec=out)

    return MathProblemSpecResult(
        ok=False,
        status="unsupported",
        reason=f"unsupported problem kind: {kind}",
        details={"target": target},
    )


def _normalize_kind(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return KIND_ALIASES.get(raw, raw)


def _source_rule(raw: Dict[str, Any], target: Dict[str, Any]) -> str:
    direct = str(raw.get("source_rule") or target.get("source_rule") or "").strip()
    if direct:
        return direct
    for candidate in (raw.get("source_rules"), target.get("source_rules")):
        if isinstance(candidate, list):
            for item in candidate:
                text = str(item or "").strip()
                if text:
                    return text
    return ""


def _normalize_expression(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = _strip_function_assignment(raw)
    raw = raw.replace("^", "**")
    raw = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", raw)
    raw = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", raw)
    return raw


def _strip_function_assignment(value: str) -> str:
    text = value.strip()
    if "=" not in text:
        return text
    left, right = text.split("=", 1)
    if re.fullmatch(r"[A-Za-z]\s*\([^)]*\)", left.strip()):
        return right.strip()
    return text


def _normalize_equation(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.count("=") != 1:
        return ""
    lhs, rhs = raw.split("=", 1)
    lhs_norm = _normalize_expression(lhs)
    rhs_norm = _normalize_expression(rhs)
    if not lhs_norm or not rhs_norm:
        return ""
    return f"{lhs_norm} = {rhs_norm}"


def _normalize_variable(value: Any) -> str:
    raw = str(value or "x").strip()
    return raw if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw) else "x"


def _normalize_substitutions(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for key, raw in value.items():
        variable = _normalize_variable(key)
        expr = _normalize_expression(raw)
        if variable and expr:
            out[variable] = expr
    return out


def _invalid(reason: str, details: Dict[str, Any]) -> MathProblemSpecResult:
    return MathProblemSpecResult(ok=False, status="invalid_spec", reason=reason, details=details)
