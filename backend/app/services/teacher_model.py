from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.api.schemas import ProofSpan
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer


@dataclass
class TeacherModelResult:
    answer: str
    proofs: List[ProofSpan]


@dataclass
class TeacherModelService:
    """Grounded teacher-side answer generation."""

    def generate_answer(
        self,
        *,
        question: str,
        k: int,
        min_score: float,
        store: VectorStore,
        allowed_chunk_ids: list[str],
        query_vec: Optional[np.ndarray] = None,
    ) -> TeacherModelResult:
        result = generate_answer(
            question=question,
            k=k,
            min_score=min_score,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
            query_vec=query_vec,
        )
        return TeacherModelResult(answer=result.answer, proofs=result.proofs)
