from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.data.db_engine import get_db
from app.data.db_repository import DBRepository, StoredCardTopic, StoredTopicProficiency
from app.services.diagnostic_coverage import all_topics_diagnosed
from app.services.diagnostic_state import DiagnosticStateService


@dataclass
class DiagnosticReviewApplyResult:
    review_id: str
    card_id: str
    rating: str
    diagnostic_answered: int
    diagnostic_total: int
    exam_state: str
    topic_proficiency: float
    idempotent_replay: bool = False


class DiagnosticReviewReducer:
    """
    Applies diagnostic-phase reviews transactionally.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        diagnostic_service: Optional[DiagnosticStateService] = None,
    ) -> None:
        self.repo = repo or DBRepository(Path("store/meta.sqlite"))
        self.diagnostic_service = diagnostic_service or DiagnosticStateService()

    def apply_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        idempotency_key: str,
        now: Optional[datetime] = None,
    ) -> DiagnosticReviewApplyResult:
        now_dt = now or datetime.now(timezone.utc)
        with get_db() as session:
            return self._apply_review_in_tx(
                session=session,
                user_id=user_id,
                exam_id=exam_id,
                card_id=card_id,
                rating=rating,
                idempotency_key=idempotency_key,
                now=now_dt,
            )

    @staticmethod
    def _ensure_primary_topic(links: list[StoredCardTopic]) -> StoredCardTopic:
        primaries = [x for x in links if x.role == "primary"]
        if len(primaries) != 1:
            raise ValueError("Card must have exactly one primary topic link")
        return primaries[0]

    def _idempotent_result(
        self,
        *,
        session: Session,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        review_id: str,
    ) -> DiagnosticReviewApplyResult:
        exam = self.repo.get_exam(exam_id, session=session)
        if exam is None:
            raise ValueError(f"Exam not found: {exam_id}")
        links = self.repo.list_card_topics(card_id=card_id, session=session)
        if not links:
            raise ValueError("Card topic links missing")
        primary = self._ensure_primary_topic(links)
        prof = self.repo.get_topic_proficiency(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=primary.topic_id,
            session=session,
        )
        return DiagnosticReviewApplyResult(
            review_id=review_id,
            card_id=card_id,
            rating=rating,
            diagnostic_answered=exam.diagnostic_answered,
            diagnostic_total=exam.diagnostic_total,
            exam_state=exam.state,
            topic_proficiency=float(prof.proficiency) if prof else 0.5,
            idempotent_replay=True,
        )

    def _apply_review_in_tx(
        self,
        *,
        session: Session,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        idempotency_key: str,
        now: datetime,
    ) -> DiagnosticReviewApplyResult:
        existing_review_id = self.repo.get_card_review_by_idempotency_key(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            idempotency_key=idempotency_key,
            session=session,
        )
        if existing_review_id:
            return self._idempotent_result(
                session=session,
                user_id=user_id,
                exam_id=exam_id,
                card_id=card_id,
                rating=rating,
                review_id=existing_review_id,
            )

        exam = self.repo.get_exam(exam_id, session=session)
        if exam is None:
            raise ValueError(f"Exam not found: {exam_id}")
        if exam.user_id != user_id:
            raise ValueError("Exam does not belong to user")
        if exam.state != "diagnostic":
            raise ValueError("Exam is not in diagnostic state")

        card = self.repo.get_card(card_id=card_id)
        if card is None:
            raise ValueError(f"Card not found: {card_id}")
        if card.exam_id != exam_id:
            raise ValueError("Card does not belong to exam")
        if card.card_type != "diagnostic":
            raise ValueError("Card is not a diagnostic card")

        links = self.repo.list_card_topics(card_id=card_id, session=session)
        primary_count = sum(1 for x in links if x.role == "primary")
        if not links or primary_count != 1:
            # Fallback/heal for legacy or malformed card-topic links.
            self.repo.replace_card_topics(
                card_id=card_id,
                topics=[{"topic_id": card.topic_id, "role": "primary", "weight": 1.0}],
                session=session,
            )
            links = self.repo.list_card_topics(card_id=card_id, session=session)
        primary = self._ensure_primary_topic(links)

        current_prof = self.repo.get_topic_proficiency(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=primary.topic_id,
            session=session,
        )
        transition = self.diagnostic_service.apply_rating(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=primary.topic_id,
            rating=rating,
            current=current_prof,
        )
        self.repo.upsert_topic_proficiency(
            user_id=transition.user_id,
            exam_id=transition.exam_id,
            topic_id=transition.topic_id,
            proficiency=transition.proficiency,
            current_difficulty=transition.current_difficulty,
            streak_up=transition.streak_up,
            streak_down=transition.streak_down,
            seen_count=transition.seen_count,
            correctish_count=transition.correctish_count,
            info=transition.info,
            session=session,
        )

        review_id = self.repo.add_card_review(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            topic_id=primary.topic_id,
            rating=rating,
            info={"idempotency_key": idempotency_key, "phase": "diagnostic"},
            session=session,
        )
        diagnostic_answered = self.repo.increment_exam_diagnostic_answered(
            exam_id=exam_id,
            by=1,
            session=session,
        )
        self.repo.upsert_exam_session_state(
            user_id=user_id,
            exam_id=exam_id,
            last_served_card_id=card_id,
            last_presented_at=now,
            session=session,
        )
        self.repo.append_card_presentation(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            presented_at=now,
            info={"source": "diagnostic_review_reducer"},
            session=session,
        )

        state = exam.state
        diagnostic_total = int(exam.diagnostic_total or 0)
        if all_topics_diagnosed(repo=self.repo, user_id=user_id, exam_id=exam_id):
            state = "active_learning"
            self.repo.update_exam_lifecycle(
                exam_id=exam_id,
                state=state,
                diagnostic_completed_at=now,
                session=session,
            )

        return DiagnosticReviewApplyResult(
            review_id=review_id,
            card_id=card_id,
            rating=rating,
            diagnostic_answered=diagnostic_answered,
            diagnostic_total=diagnostic_total,
            exam_state=state,
            topic_proficiency=transition.proficiency,
            idempotent_replay=False,
        )

