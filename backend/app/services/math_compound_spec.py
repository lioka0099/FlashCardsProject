"""
Compound math problem specs.

A compound spec models a calculation problem as an *ordered chain of steps*
instead of a single SymPy primitive. Each step may carry a machine-checkable
``check`` (a sub-spec reusing the existing primitive vocabulary in
``math_problem_spec``). Depth comes from chaining several concepts/steps; every
computational step is still individually verifiable by SymPy, and the final
answer is CAS-checked, so correctness stays deterministic.

A single-primitive problem is just a length-1 chain, so the legacy shape remains
valid (see :func:`normalize_compound_spec`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.math_problem_spec import (
    SUPPORTED_KINDS,
    normalize_math_problem_spec,
    _normalize_kind,
    _source_rule,
)


@dataclass
class CompoundSpecResult:
    ok: bool
    status: str
    spec: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def _clean_str_list(value: Any, *, limit: int = 12) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_step(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize one step; an unparseable check degrades the step to narrative."""
    if not isinstance(raw, dict):
        return None
    description = str(raw.get("description") or raw.get("text") or "").strip()
    is_final = bool(raw.get("is_final") or raw.get("final"))
    check_payload = raw.get("check")
    normalized_check: Optional[Dict[str, Any]] = None
    if isinstance(check_payload, dict) and check_payload:
        result = normalize_math_problem_spec(check_payload, require_source_rule=False)
        if result.ok:
            normalized_check = result.spec
    if not description and normalized_check is None:
        return None
    return {"description": description, "check": normalized_check, "is_final": is_final}


def normalize_compound_spec(
    payload: Optional[Dict[str, Any]],
    *,
    require_source_rule: bool = True,
) -> CompoundSpecResult:
    """
    Normalize an LLM-authored compound spec into a stable, verifiable structure.

    Accepts both the compound shape (``steps``) and a bare single-primitive spec
    (wrapped into a length-1 chain) for backward compatibility.
    """
    raw = payload or {}
    if not isinstance(raw, dict):
        return CompoundSpecResult(False, "invalid_spec", reason="spec payload is not an object")

    status = str(raw.get("status") or "ready").strip().lower()
    if status == "insufficient_evidence":
        return CompoundSpecResult(
            False,
            "insufficient_evidence",
            reason=str(raw.get("reason") or "Insufficient evidence for a grounded calculation problem."),
        )
    if status and status != "ready":
        return CompoundSpecResult(False, "invalid_spec", reason=f"unsupported spec status: {status}")

    # Backward compatibility: a bare single-primitive spec becomes a 1-step chain.
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        primitive = normalize_math_problem_spec(raw, require_source_rule=require_source_rule)
        if not primitive.ok:
            return CompoundSpecResult(False, primitive.status, reason=primitive.reason, details=primitive.details)
        raw_steps = [{"description": "", "check": raw, "is_final": True}]

    source_rule = _source_rule(raw, raw)
    if require_source_rule and not source_rule:
        return CompoundSpecResult(False, "invalid_spec", reason="missing source_rule")

    steps: List[Dict[str, Any]] = []
    for raw_step in raw_steps:
        step = _normalize_step(raw_step)
        if step is not None:
            steps.append(step)

    checkable = [s for s in steps if s["check"] is not None]
    if not checkable:
        return CompoundSpecResult(
            False,
            "insufficient_evidence",
            reason="No machine-checkable computational step was present; not a calculation problem.",
        )

    # The final/canonical step must be checkable so the final answer is CAS-verifiable.
    final_index = next((i for i, s in enumerate(steps) if s["is_final"] and s["check"] is not None), None)
    if final_index is None:
        # last checkable step
        final_index = max(i for i, s in enumerate(steps) if s["check"] is not None)
    for i, s in enumerate(steps):
        s["is_final"] = i == final_index

    problem_statement = str(raw.get("problem_statement") or raw.get("question") or "").strip()
    if not problem_statement:
        return CompoundSpecResult(False, "invalid_spec", reason="missing problem_statement")

    spec: Dict[str, Any] = {
        "status": "ready",
        "compound": True,
        "problem_statement": problem_statement,
        "concepts_used": _clean_str_list(raw.get("concepts_used") or raw.get("concepts")),
        "archetype": str(raw.get("archetype") or "").strip().lower(),
        "steps": steps,
        "final_answer": (str(raw.get("final_answer")).strip() if raw.get("final_answer") is not None else None),
    }
    if source_rule:
        spec["source_rule"] = source_rule
        spec["source_rules"] = _clean_str_list(raw.get("source_rules")) or [source_rule]
    return CompoundSpecResult(True, "ready", spec=spec)


def _step_check_kinds(spec: Dict[str, Any]) -> List[str]:
    kinds: List[str] = []
    for step in spec.get("steps") or []:
        check = step.get("check")
        if not isinstance(check, dict):
            continue
        target = check.get("verification_target") if isinstance(check.get("verification_target"), dict) else check
        kind = _normalize_kind(target.get("kind") or check.get("kind"))
        if kind == "matrix":
            operation = str(target.get("operation") or "").strip().lower()
            kind = f"matrix:{operation}" if operation else "matrix"
        if kind:
            kinds.append(kind)
    return kinds


def math_structural_fingerprint(spec: Dict[str, Any]) -> str:
    """
    Structural signature for diversity: archetype + concept set + step-kind set.

    Two problems that share the same archetype, concepts, and operation families
    are considered the same *type* even if their numbers/wording differ.
    """
    if not isinstance(spec, dict):
        return ""
    archetype = str(spec.get("archetype") or "").strip().lower()
    concepts = sorted({c.strip().lower() for c in (spec.get("concepts_used") or []) if str(c).strip()})
    kinds = sorted(set(_step_check_kinds(spec)))
    return "|".join(["arch=" + archetype, "concepts=" + ",".join(concepts), "kinds=" + ",".join(kinds)])
