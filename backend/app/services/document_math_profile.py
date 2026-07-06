from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence

from app.data.db_repository import StoredChunk
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create, safe_json_load


DocumentMathKind = Literal["math", "non_math", "mixed"]

_LABEL_TO_KIND: Dict[str, DocumentMathKind] = {
    "MATHEMATICAL": "math",
    "MATH": "math",
    "CONCEPTUAL": "non_math",
    "REAL": "non_math",
    "NON_MATH": "non_math",
    "BOTH": "mixed",
    "MIXED": "mixed",
}

_MATH_TASK_TERMS = {
    "solve",
    "calculate",
    "compute",
    "evaluate",
    "derive",
    "derivation",
    "differentiate",
    "integrate",
    "simplify",
    "factor",
    "expand",
    "prove",
    "proof",
    "manipulate",
}

_MATH_DOMAIN_TERMS = {
    "algebra",
    "calculus",
    "equation",
    "formula",
    "function",
    "derivative",
    "integral",
    "polynomial",
    "quadratic",
    "theorem",
    "variable",
    "matrix",
    "probability",
}


@dataclass(frozen=True)
class DocumentMathProfile:
    kind: DocumentMathKind
    label: str
    confidence: float
    reason: str
    source: str
    evidence: List[str] = field(default_factory=list)
    classified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_document_math_kind(profile: Optional[Dict[str, Any]]) -> DocumentMathKind:
    if not isinstance(profile, dict):
        return "math"

    raw = profile.get("kind") or profile.get("label") or profile.get("classification")
    label = str(raw or "").strip().upper()
    if not label:
        return "math"
    normalized = label.replace(" ", "_")
    return _LABEL_TO_KIND.get(normalized, "math")


def classify_document_math_profile(
    *,
    chunks: Sequence[StoredChunk],
    model: str = CHAT_MODEL_FAST,
) -> DocumentMathProfile:
    sample = build_document_math_sample(chunks)
    if not sample:
        return DocumentMathProfile(
            kind="non_math",
            label="CONCEPTUAL",
            confidence=0.6,
            reason="No document text was available for math classification.",
            source="fallback_empty",
        )

    try:
        return _classify_with_llm(sample=sample, model=model)
    except Exception:
        return _fallback_profile(sample)


def build_document_math_sample(
    chunks: Sequence[StoredChunk],
    *,
    max_chunks: int = 12,
    max_chars_per_chunk: int = 700,
    max_total_chars: int = 8000,
) -> str:
    nonempty = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
    if not nonempty:
        return ""

    selected_indices = _spread_indices(len(nonempty), max_chunks)
    buf: List[str] = []
    used = 0
    for idx in selected_indices:
        chunk = nonempty[idx]
        text = re.sub(r"\s+", " ", chunk.text).strip()
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3].rstrip() + "..."
        block = f"[doc={chunk.doc_id} page={chunk.page}] {text}\n"
        if used + len(block) > max_total_chars:
            break
        buf.append(block)
        used += len(block)
    return "".join(buf).strip()


def _spread_indices(length: int, limit: int) -> List[int]:
    if length <= limit:
        return list(range(length))
    if limit <= 1:
        return [0]
    raw = [round(i * (length - 1) / (limit - 1)) for i in range(limit)]
    out: List[int] = []
    for idx in raw:
        if idx not in out:
            out.append(idx)
    return out


def _classify_with_llm(*, sample: str, model: str) -> DocumentMathProfile:
    system_prompt = (
        "You are an educational document classifier.\n"
        "Classify by the dominant learning task required from the student.\n"
        "Return JSON only."
    )
    user_prompt = (
        "Classify the uploaded document into exactly one label: MATHEMATICAL, CONCEPTUAL, or BOTH.\n\n"
        "MATHEMATICAL: choose only when the main learning activity requires mathematical reasoning, "
        "calculations, formulas, equations, proofs, derivations, numeric problem solving, or formal quantitative thinking.\n"
        "CONCEPTUAL: choose when the main learning activity is understanding a conceptual, theoretical, practical, "
        "educational, scientific, social, historical, technical, or real-world topic.\n"
        "BOTH: choose only when substantial conceptual explanation and substantial mathematical reasoning/calculation "
        "are both central to the learning content.\n\n"
        "Important: numbers, percentages, tables, charts, graphs, statistics, experiment results, averages, p-values, "
        "code snippets, and technical examples do not make a document mathematical unless students are expected to do "
        "mathematical calculations or mathematical reasoning as the main task.\n"
        "Do not choose BOTH only because the document has numbers, statistics, graphs, or examples.\n\n"
        "Ask yourself: What is the main subject? What should the student mainly do after reading it? "
        "Would mathematical calculations be suitable for generating study questions from this document?\n\n"
        "DOCUMENT SAMPLE:\n"
        f"{sample}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "label": "MATHEMATICAL|CONCEPTUAL|BOTH",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short reason",\n'
        '  "evidence": ["short phrase from the document", "..."]\n'
        "}"
    )
    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    payload = safe_json_load(resp.choices[0].message.content or "{}")
    label = str(payload.get("label") or "").strip().upper()
    kind = _LABEL_TO_KIND.get(label, "non_math")
    confidence = _coerce_confidence(payload.get("confidence"), default=0.7)
    reason = str(payload.get("reason") or "Classified by dominant learning task.").strip()
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    return DocumentMathProfile(
        kind=kind,
        label=label if label in {"MATHEMATICAL", "CONCEPTUAL", "BOTH"} else "CONCEPTUAL",
        confidence=confidence,
        reason=reason,
        source="llm",
        evidence=[str(item).strip() for item in evidence if str(item).strip()][:8],
    )


def _fallback_profile(sample: str) -> DocumentMathProfile:
    lowered = sample.lower()
    has_task = any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _MATH_TASK_TERMS)
    has_domain = any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _MATH_DOMAIN_TERMS)
    has_formula = _has_formula_or_equation(sample)

    if has_formula and has_task:
        return DocumentMathProfile(
            kind="math",
            label="MATHEMATICAL",
            confidence=0.72,
            reason="The document includes mathematical notation together with explicit math-task language.",
            source="heuristic_fallback",
            evidence=_evidence_phrases(sample),
        )
    if has_domain and has_task:
        return DocumentMathProfile(
            kind="mixed",
            label="BOTH",
            confidence=0.62,
            reason="The document has math-domain and math-task language, but limited formula evidence.",
            source="heuristic_fallback",
            evidence=_evidence_phrases(sample),
        )
    return DocumentMathProfile(
        kind="non_math",
        label="CONCEPTUAL",
        confidence=0.65,
        reason="No strong evidence that mathematical work is the dominant learning task.",
        source="heuristic_fallback",
        evidence=_evidence_phrases(sample),
    )


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


def _evidence_phrases(text: str, *, limit: int = 4) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return [sentence[:160].strip() for sentence in sentences if sentence.strip()][:limit]


def _coerce_confidence(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(0.0, min(1.0, numeric))
