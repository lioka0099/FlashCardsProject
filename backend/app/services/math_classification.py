from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.data.vector_store import VectorStore
from app.services.llm import embed_texts


_CALCULATION_EXEMPLARS = [
    "solve an equation or system of equations for unknown variables",
    "differentiate or integrate a function using calculus rules",
    "simplify factor expand or compute an algebraic expression",
    "compute a determinant inverse rank or eigenvalues of a matrix",
    "calculate a probability expectation variance or value from a distribution",
    "evaluate a summation series or limit",
    "find a maximum or minimum by optimization",
    "compute lengths areas angles or coordinates in geometry",
]

_CONCEPTUAL_EXEMPLARS = [
    "explain what a concept such as a derivative eigenvalue or distribution represents",
    "describe the meaning of a mathematical object relationship or property",
    "understand a mathematical definition theorem axiom or property",
    "interpret the idea behind a formula or method without calculating values",
]

_PROCESS_CACHE: Dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class MathClassificationEvidence:
    semantic_math_score: float = 0.0
    semantic_calculation_score: float = 0.0
    semantic_conceptual_score: float = 0.0
    source: str = "heuristic"
    matched_exemplars: List[str] = field(default_factory=list)

    def to_info(self) -> Dict[str, Any]:
        return {
            "semantic_math_score": self.semantic_math_score,
            "semantic_calculation_score": self.semantic_calculation_score,
            "semantic_conceptual_score": self.semantic_conceptual_score,
            "source": self.source,
            "matched_exemplars": self.matched_exemplars,
        }


def _hash_text(text: str) -> str:
    return "math-classification:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype("float32", copy=False)
    matrix = matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)
    return matrix / norms


class MathClassificationService:
    """Semantic scorer for distinguishing calculation and conceptual math topics."""

    def __init__(self, *, store: Optional[VectorStore] = None) -> None:
        self.store = store

    def score(self, *, topic_label: str, context_pack: str) -> MathClassificationEvidence:
        text = f"{topic_label}\n{context_pack}".strip()
        if not text:
            return MathClassificationEvidence()

        exemplars = _CALCULATION_EXEMPLARS + _CONCEPTUAL_EXEMPLARS
        try:
            vectors = self._embed([text] + exemplars)
        except Exception:
            return MathClassificationEvidence(source="heuristic_fallback")
        if vectors.shape[0] != len(exemplars) + 1:
            return MathClassificationEvidence(source="heuristic_fallback")

        vectors = _normalize_rows(vectors)
        query = vectors[0]
        exemplar_vectors = vectors[1:]
        scores = exemplar_vectors @ query
        calc_scores = scores[: len(_CALCULATION_EXEMPLARS)]
        conceptual_scores = scores[len(_CALCULATION_EXEMPLARS) :]
        calc_score = float(np.max(calc_scores)) if calc_scores.size else 0.0
        conceptual_score = float(np.max(conceptual_scores)) if conceptual_scores.size else 0.0
        semantic_math_score = max(calc_score, conceptual_score)

        ranked = sorted(
            zip(exemplars, [float(s) for s in scores]),
            key=lambda item: item[1],
            reverse=True,
        )
        return MathClassificationEvidence(
            semantic_math_score=semantic_math_score,
            semantic_calculation_score=calc_score,
            semantic_conceptual_score=conceptual_score,
            source="embedding",
            matched_exemplars=[text for text, _score in ranked[:3]],
        )

    def _embed(self, texts: List[str]) -> np.ndarray:
        hashes = [_hash_text(text) for text in texts]
        cached: Dict[str, np.ndarray] = {}
        if self.store is not None:
            cached.update(self.store.get_cached_embeddings(hashes))
        cached.update({h: v for h, v in _PROCESS_CACHE.items() if h in hashes})

        missing = [i for i, h in enumerate(hashes) if h not in cached]
        if missing:
            new_vectors = embed_texts([texts[i] for i in missing])
            for idx, vector in zip(missing, new_vectors):
                cached[hashes[idx]] = vector.astype("float32", copy=False)
                _PROCESS_CACHE[hashes[idx]] = cached[hashes[idx]]
            if self.store is not None:
                self.store.add_cached_embeddings({hashes[idx]: cached[hashes[idx]] for idx in missing})

        return np.stack([cached[h] for h in hashes]).astype("float32", copy=False)
