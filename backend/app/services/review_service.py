from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.data.db_repository import DBRepository
from app.deps import get_repo
from app.services.diagnostic_review_reducer import DiagnosticReviewReducer
from app.services.review_state_reducer import ReviewStateReducer
from app.services.student_memory import StudentMemoryService


@dataclass
class ReviewServiceResult:
    review_id: str
    card_id: str
    rating: str
    exam_state: str
    topic_proficiency: float
    due_at: Optional[str] = None
    interval_days: Optional[float] = None
    ease: Optional[float] = None
    diagnostic_answered: Optional[int] = None
    diagnostic_total: Optional[int] = None
    idempotent_replay: bool = False


class ReviewService:
    """
    Thin orchestration adapter that routes review handling by exam state.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        diagnostic_reducer: Optional[DiagnosticReviewReducer] = None,
        active_reducer: Optional[ReviewStateReducer] = None,
        student_memory: Optional[StudentMemoryService] = None,
    ) -> None:
        self.repo = repo or get_repo()
        self.diagnostic_reducer = diagnostic_reducer or DiagnosticReviewReducer(repo=self.repo)
        self.active_reducer = active_reducer or ReviewStateReducer(repo=self.repo)
        self.student_memory = student_memory or StudentMemoryService(repo=self.repo)

    def _apply_student_memory_update(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
    ) -> None:
        card = self.repo.get_card(card_id=card_id)
        if card is None:
            return
        links = self.repo.list_card_topics(card_id=card_id)
        if links:
            primary_links = [x for x in links if x.role == "primary"]
            topic_id = primary_links[0].topic_id if primary_links else links[0].topic_id
        else:
            topic_id = card.topic_id
        self.student_memory.update_from_review(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
            rating=rating,
            question=card.question,
            answer=card.answer,
            difficulty=card.difficulty,
            source="review_service",
        )

    def apply_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        idempotency_key: str,
    ) -> ReviewServiceResult:
        exam = self.repo.get_exam(exam_id)
        if exam is None:
            raise ValueError(f"Exam not found: {exam_id}")
        if exam.user_id != user_id:
            raise ValueError("Exam does not belong to user")

        if exam.state == "diagnostic":
            d = self.diagnostic_reducer.apply_review(
                user_id=user_id,
                exam_id=exam_id,
                card_id=card_id,
                rating=rating,
                idempotency_key=idempotency_key,
            )
            if not d.idempotent_replay:
                self._apply_student_memory_update(
                    user_id=user_id,
                    exam_id=exam_id,
                    card_id=card_id,
                    rating=rating,
                )
            return ReviewServiceResult(
                review_id=d.review_id,
                card_id=d.card_id,
                rating=d.rating,
                exam_state=d.exam_state,
                topic_proficiency=d.topic_proficiency,
                diagnostic_answered=d.diagnostic_answered,
                diagnostic_total=d.diagnostic_total,
                idempotent_replay=d.idempotent_replay,
            )

        a = self.active_reducer.apply_review(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            rating=rating,
            idempotency_key=idempotency_key,
        )
        if not a.idempotent_replay:
            self._apply_student_memory_update(
                user_id=user_id,
                exam_id=exam_id,
                card_id=card_id,
                rating=rating,
            )
        return ReviewServiceResult(
            review_id=a.review_id,
            card_id=a.card_id,
            rating=a.rating,
            exam_state=exam.state,
            topic_proficiency=a.topic_proficiency,
            due_at=a.due_at,
            interval_days=a.interval_days,
            ease=a.ease,
            idempotent_replay=a.idempotent_replay,
        )
