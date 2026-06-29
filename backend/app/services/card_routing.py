from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from app.data.vector_store import VectorStore
from app.services.difficulty_frameworks import CardRoute
from app.services.document_math_profile import normalize_document_math_kind
from app.services.math_classification import MathClassificationEvidence, MathClassificationService


SubjectType = Literal["general", "math"]
MathKind = Literal["none", "calculation", "conceptual"]


@dataclass(frozen=True)
class RouteDecision:
    card_route: CardRoute
    subject_type: SubjectType = "general"
    math_kind: MathKind = "none"
    confidence: float = 0.0
    reason: str = ""
    evidence_chunk_ids: List[str] = field(default_factory=list)
    problem_types: List[str] = field(default_factory=list)
    classification: Dict[str, Any] = field(default_factory=dict)

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
    # Linear algebra
    "determinant": "matrix",
    "eigenvalue": "matrix",
    "eigenvalues": "matrix",
    "inverse": "matrix",
    "matrix": "matrix",
    # Discrete / sequences
    "summation": "summation",
    "series": "summation",
    "limit": "limit",
    # Probability / statistics / optimization (decompose into existing primitives)
    "probability": "arithmetic",
    "expectation": "arithmetic",
    "variance": "arithmetic",
    "maximize": "derivative",
    "minimize": "derivative",
    "optimize": "derivative",
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

# Genuinely non-mathematical lab sciences. Note: statistics, probability, linear
# algebra, discrete math, optimization, etc. are SUPPORTED math domains and must NOT
# be blocked here. Document-level math profiling already gates non-math documents.
_UNSUPPORTED_STEM_TERMS = {
    "physics",
    "chemistry",
}

_MATH_VOCABULARY_TERMS = {
    "algebra",
    "calculus",
    "derivative",
    "differentiate",
    "equation",
    "formula",
    "function",
    "integral",
    "linear",
    "polynomial",
    "quadratic",
    "slope",
    "theorem",
    "variable",
    # Linear algebra
    "matrix",
    "matrices",
    "vector",
    "eigenvalue",
    "determinant",
    # Probability / statistics
    "probability",
    "distribution",
    "expectation",
    "variance",
    "random",
    # Discrete / combinatorics
    "combinatorics",
    "permutation",
    "summation",
    "sequence",
    # Geometry / optimization
    "geometry",
    "optimization",
    "gradient",
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


def _has_math_vocabulary(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _MATH_VOCABULARY_TERMS)


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


def _conceptual_math(reason: str, *, confidence: float, evidence: MathClassificationEvidence) -> RouteDecision:
    # Conceptual math uses the default generation path; the only thing that
    # distinguishes it is the math_kind/subject_type tag (kept for logs/metadata).
    return RouteDecision(
        card_route="default",
        subject_type="math",
        math_kind="conceptual",
        confidence=confidence,
        reason=reason,
        classification=evidence.to_info(),
    )


def classify_card_route(
    *,
    topic_label: str,
    context_pack: str,
    cached_topic_route: Optional[Dict[str, Any]] = None,
    document_math_profile: Optional[Dict[str, Any]] = None,
    store: Optional[VectorStore] = None,
) -> RouteDecision:
    text = f"{topic_label}\n{context_pack}".strip()
    lowered = text.lower()
    document_math_kind = normalize_document_math_kind(document_math_profile)

    if document_math_kind == "non_math":
        return _fallback("The document profile is non-math, so math routing is disabled.", confidence=0.95)

    if any(term in lowered for term in _UNSUPPORTED_STEM_TERMS):
        return _fallback("The context appears to be an unsupported STEM subject for this phase.")

    evidence = (
        MathClassificationService(store=store).score(topic_label=topic_label, context_pack=context_pack)
        if store is not None
        else MathClassificationEvidence()
    )
    semantic_math = evidence.semantic_math_score
    semantic_calc = evidence.semantic_calculation_score
    semantic_conceptual = evidence.semantic_conceptual_score

    problem_types = _problem_types(text)
    has_calculation_term = bool(problem_types)
    has_formula = _has_formula_or_equation(text)
    local_math_evidence = has_formula or has_calculation_term or _has_math_vocabulary(text)
    mixed_requires_grounding = document_math_kind == "mixed"
    subject_type: SubjectType = "math" if has_formula or has_calculation_term or semantic_math >= 0.42 else "general"

    if _looks_conceptual_only(text) and (not mixed_requires_grounding or local_math_evidence):
        return _conceptual_math(
            "The context is mathematical but conceptual rather than calculation-based.",
            confidence=max(0.62, min(0.9, semantic_conceptual or semantic_math)),
            evidence=evidence,
        )

    cached_confidence = 0.0
    cached_route = cached_topic_route.get("card_route") if cached_topic_route else None
    if cached_route == "math_calculation":
        try:
            cached_confidence = float(cached_topic_route.get("confidence") or 0.0)
        except Exception:
            cached_confidence = 0.0
    # Legacy: older cached routes stored "math_conceptual" before it was folded
    # into "default"; re-emit them as the conceptual-default decision.
    if (
        cached_route == "math_conceptual"
        and not mixed_requires_grounding
        and not (has_formula and has_calculation_term)
    ):
        try:
            conceptual_confidence = float(cached_topic_route.get("confidence") or 0.0)
        except Exception:
            conceptual_confidence = 0.0
        if conceptual_confidence >= 0.65:
            return _conceptual_math(
                "The cached topic route indicates conceptual math and no calculation evidence overrides it.",
                confidence=min(0.9, conceptual_confidence),
                evidence=evidence,
            )

    calculation_semantic_hit = semantic_calc >= 0.58 and semantic_calc >= semantic_conceptual - 0.04
    if mixed_requires_grounding and not (has_formula or has_calculation_term):
        calculation_semantic_hit = False
    if (has_formula and has_calculation_term) or (
        calculation_semantic_hit and (has_formula or has_calculation_term or cached_confidence >= 0.75)
    ):
        confidence = max(
            0.82,
            min(0.95, 0.78 + 0.05 * len(problem_types), cached_confidence, semantic_calc),
        )
        return RouteDecision(
            card_route="math_calculation",
            subject_type="math",
            math_kind="calculation",
            confidence=confidence,
            reason="The context contains grounded formulas/equations and calculation procedures.",
            evidence_chunk_ids=_extract_chunk_ids(context_pack),
            problem_types=problem_types,
            classification=evidence.to_info(),
        )

    if has_formula and cached_confidence >= 0.75 and not mixed_requires_grounding:
        return RouteDecision(
            card_route="math_calculation",
            subject_type="math",
            math_kind="calculation",
            confidence=min(0.9, cached_confidence),
            reason="The cached topic route and current context contain enough calculation evidence.",
            evidence_chunk_ids=_extract_chunk_ids(context_pack),
            problem_types=problem_types or ["calculation"],
            classification=evidence.to_info(),
        )

    if (
        subject_type == "math"
        and (semantic_conceptual >= 0.48 or has_calculation_term or has_formula)
        and (not mixed_requires_grounding or local_math_evidence)
    ):
        return _conceptual_math(
            "The context appears mathematical but does not contain enough procedural evidence for a calculation card.",
            confidence=max(0.55, min(0.86, semantic_conceptual or semantic_math)),
            evidence=evidence,
        )

    if has_formula:
        return _fallback(
            "The context contains mathematical notation but not enough procedural evidence for a grounded calculation card.",
            subject_type="math",
            confidence=0.55,
        )

    return _fallback("No sufficient document-grounded calculation evidence was found.", subject_type=subject_type)


def default_route_decision() -> RouteDecision:
    return RouteDecision(
        card_route="default",
        subject_type="general",
        math_kind="none",
        confidence=1.0,
        reason="Default route.",
    )

