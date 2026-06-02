from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from app.services.difficulty_frameworks import CardRoute


SubjectType = Literal["general", "math"]
MathKind = Literal["none", "calculation"]


@dataclass(frozen=True)
class RouteDecision:
    card_route: CardRoute
    subject_type: SubjectType = "general"
    math_kind: MathKind = "none"
    confidence: float = 0.0
    reason: str = ""
    evidence_chunk_ids: List[str] = field(default_factory=list)
    problem_types: List[str] = field(default_factory=list)

    def to_info(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["classified_at"] = datetime.now(timezone.utc).isoformat()
        return payload


_CALCULATION_TERMS = {
    "solve": "equation",
    "simplify": "simplify",
    "factor": "factor",
    "expand": "expand",
    "differentiate": "derivative",
    "derivative": "derivative",
    "integrate": "integral",
    "integral": "integral",
    "substitute": "substitution",
    "calculate": "arithmetic",
    "compute": "arithmetic",
    "evaluate": "substitution",
    "equivalent": "equivalence",
    "system of equations": "system",
}

_CONCEPTUAL_PATTERNS = [
    r"\bwhat does\b",
    r"\bwhat is\b",
    r"\bexplain\b",
    r"\bmeaning of\b",
    r"\brepresent\b",
    r"\bwhy\b",
    r"\bconcept\b",
    r"\bintuition\b",
]

_UNSUPPORTED_STEM_TERMS = {
    "physics",
    "chemistry",
    "statistics",
    "statistical",
    "economics",
    "algorithm",
    "algorithms",
}


def _has_formula_or_equation(text: str) -> bool:
    if re.search(r"[A-Za-z]\s*\([^)]*\)\s*=", text):
        return True
    if re.search(r"[A-Za-z0-9)\]]\s*=\s*[-+*/^A-Za-z0-9(]", text):
        return True
    if re.search(r"\b\d+\s*[A-Za-z]\s*[-+*/^=]", text):
        return True
    if re.search(r"[A-Za-z]\s*[\^*]\s*\d", text):
        return True
    if re.search(r"\\(?:frac|sqrt|int|sum|cdot|times)", text):
        return True
    return False


def _extract_chunk_ids(context: str) -> List[str]:
    ids: List[str] = []
    for pattern in (r"chunk[_ -]?id[:=]\s*([A-Za-z0-9_-]+)", r"chunk=([A-Za-z0-9_-]+)"):
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            ids.append(match.group(1))
    return list(dict.fromkeys(ids))


def _problem_types(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for term, problem_type in _CALCULATION_TERMS.items():
        if term in lowered and problem_type not in found:
            found.append(problem_type)
    return found


def _looks_conceptual_only(text: str) -> bool:
    lowered = text.lower()
    has_conceptual = any(re.search(pattern, lowered) for pattern in _CONCEPTUAL_PATTERNS)
    return has_conceptual and not _has_formula_or_equation(text)


def _fallback(reason: str, *, subject_type: SubjectType = "general", confidence: float = 0.2) -> RouteDecision:
    return RouteDecision(
        card_route="default",
        subject_type=subject_type,
        math_kind="none",
        confidence=confidence,
        reason=reason,
    )


def classify_card_route(
    *,
    topic_label: str,
    context_pack: str,
    cached_topic_route: Optional[Dict[str, Any]] = None,
) -> RouteDecision:
    text = f"{topic_label}\n{context_pack}".strip()
    lowered = text.lower()

    if any(term in lowered for term in _UNSUPPORTED_STEM_TERMS):
        return _fallback("The context appears to be an unsupported STEM subject for this phase.")

    problem_types = _problem_types(text)
    has_calculation_term = bool(problem_types)
    has_formula = _has_formula_or_equation(text)
    subject_type: SubjectType = "math" if has_formula or has_calculation_term else "general"

    if _looks_conceptual_only(text):
        return _fallback("The context is mathematical but conceptual rather than calculation-based.", subject_type="math")

    cached_confidence = 0.0
    if cached_topic_route and cached_topic_route.get("card_route") == "math_calculation":
        try:
            cached_confidence = float(cached_topic_route.get("confidence") or 0.0)
        except Exception:
            cached_confidence = 0.0

    if has_formula and has_calculation_term:
        confidence = max(0.82, min(0.95, 0.78 + 0.05 * len(problem_types), cached_confidence))
        return RouteDecision(
            card_route="math_calculation",
            subject_type="math",
            math_kind="calculation",
            confidence=confidence,
            reason="The context contains grounded formulas/equations and calculation procedures.",
            evidence_chunk_ids=_extract_chunk_ids(context_pack),
            problem_types=problem_types,
        )

    if has_formula and cached_confidence >= 0.75:
        return RouteDecision(
            card_route="math_calculation",
            subject_type="math",
            math_kind="calculation",
            confidence=min(0.9, cached_confidence),
            reason="The cached topic route and current context contain enough calculation evidence.",
            evidence_chunk_ids=_extract_chunk_ids(context_pack),
            problem_types=problem_types or ["calculation"],
        )

    if has_formula:
        return _fallback(
            "The context contains mathematical notation but not enough procedural evidence for a grounded calculation card.",
            subject_type="math",
            confidence=0.55,
        )

    return _fallback("No sufficient document-grounded calculation evidence was found.", subject_type=subject_type)


def classify_topic_route(
    *,
    topic_label: str,
    representative_context: str,
) -> RouteDecision:
    return classify_card_route(topic_label=topic_label, context_pack=representative_context)


def default_route_decision() -> RouteDecision:
    return RouteDecision(
        card_route="default",
        subject_type="general",
        math_kind="none",
        confidence=1.0,
        reason="Default route.",
    )

