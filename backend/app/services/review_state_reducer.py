from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from app.data.db_engine import get_db
from app.data.db_repository import (
    DBRepository,
    StoredCardTopic,
    StoredTopicProficiency,
)
from app.deps import get_repo
from app.services.card_scheduling_state import CardSchedulingStateService
from app.services.topic_proficiency_state import TopicProficiencyStateService


@dataclass
class ReviewApplyResult:
    review_id: str
    card_id: str
    rating: str
    due_at: str
    interval_days: float
    ease: float
    topic_proficiency: float
    idempotent_replay: bool = False


class ReviewStateReducer:
    """
    Orchestrates scheduling and proficiency reducers in a single transaction.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        scheduling_service: Optional[CardSchedulingStateService] = None,
        proficiency_service: Optional[TopicProficiencyStateService] = None,
    ) -> None:
        self.repo = repo or get_repo()
        self.scheduling_service = scheduling_service or CardSchedulingStateService()
        self.proficiency_service = proficiency_service or TopicProficiencyStateService()

    def _ensure_primary_topic(self, links: Dict[str, StoredCardTopic]) -> StoredCardTopic:
        primary = [l for l in links.values() if l.role == "primary"]
        if len(primary) != 1:
            raise ValueError("Card must have exactly one primary topic link")
        return primary[0]

    def apply_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        idempotency_key: str,
        now: Optional[datetime] = None,
    ) -> ReviewApplyResult:
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
    ) -> ReviewApplyResult:
        existing_review_id = self.repo.get_card_review_by_idempotency_key(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            idempotency_key=idempotency_key,
            session=session,
        )
        if existing_review_id:
            sched = self.repo.get_card_scheduling(card_id=card_id, session=session)
            topic_links = self.repo.list_card_topics(card_id=card_id, session=session)
            links = {f"{l.card_id}:{l.topic_id}:{l.role}": l for l in topic_links}
            primary = self._ensure_primary_topic(links)
            primary_prof = self.repo.get_topic_proficiency(
                user_id=user_id,
                exam_id=exam_id,
                topic_id=primary.topic_id,
                session=session,
            )
            return ReviewApplyResult(
                review_id=existing_review_id,
                card_id=card_id,
                rating=rating,
                due_at=sched.due_at if sched else "",
                interval_days=float(sched.interval_days) if sched else 0.0,
                ease=float(sched.ease) if sched else 0.0,
                topic_proficiency=float(primary_prof.proficiency) if primary_prof else 0.5,
                idempotent_replay=True,
            )

        card = self.repo.get_card(card_id=card_id)
        if not card:
            raise ValueError(f"Card not found: {card_id}")
        if card.exam_id != exam_id:
            raise ValueError("Card does not belong to exam")
        card_info = card.info or {}
        card_route = str(card_info.get("card_route") or "default")
        difficulty_framework = str(card_info.get("difficulty_framework") or "bloom")

        topic_links = self.repo.list_card_topics(card_id=card_id, session=session)
        if not topic_links:
            # Backward-compatible fallback for older cards that only have card.topic_id.
            self.repo.replace_card_topics(
                card_id=card_id,
                topics=[{"topic_id": card.topic_id, "role": "primary", "weight": 1.0}],
                session=session,
            )
            topic_links = self.repo.list_card_topics(card_id=card_id, session=session)

        links = {f"{l.card_id}:{l.topic_id}:{l.role}": l for l in topic_links}
        primary = self._ensure_primary_topic(links)

        current_sched = self.repo.get_card_scheduling(card_id=card_id, session=session)
        sched_transition = self.scheduling_service.apply_rating(
            card_id=card_id,
            rating=rating,
            current=current_sched,
            now=now,
        )
        self.repo.upsert_card_scheduling(
            card_id=sched_transition.card_id,
            due_at=sched_transition.due_at,
            state=sched_transition.state,
            interval_days=sched_transition.interval_days,
            ease=sched_transition.ease,
            reps=sched_transition.reps,
            lapses=sched_transition.lapses,
            last_reviewed_at=sched_transition.last_reviewed_at,
            session=session,
        )

        primary_result: Optional[StoredTopicProficiency] = None
        for link in topic_links:
            current_prof = self.repo.get_topic_proficiency(
                user_id=user_id,
                exam_id=exam_id,
                topic_id=link.topic_id,
                session=session,
            )
            prof_transition = self.proficiency_service.apply_rating(
                user_id=user_id,
                exam_id=exam_id,
                topic_link=link,
                rating=rating,
                current=current_prof,
                card_route=card_route,
                difficulty_framework=difficulty_framework,
            )
            self.repo.upsert_topic_proficiency(
                user_id=prof_transition.user_id,
                exam_id=prof_transition.exam_id,
                topic_id=prof_transition.topic_id,
                proficiency=prof_transition.proficiency,
                current_difficulty=prof_transition.current_difficulty,
                streak_up=prof_transition.streak_up,
                streak_down=prof_transition.streak_down,
                seen_count=prof_transition.seen_count,
                correctish_count=prof_transition.correctish_count,
                info=prof_transition.info,
                session=session,
            )
            if link.role == "primary":
                primary_result = self.repo.get_topic_proficiency(
                    user_id=user_id,
                    exam_id=exam_id,
                    topic_id=link.topic_id,
                    session=session,
                )

        review_id = self.repo.add_card_review(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            topic_id=primary.topic_id,
            rating=rating,
            info={"idempotency_key": idempotency_key},
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
            info={"source": "review_state_reducer"},
            session=session,
        )

        return ReviewApplyResult(
            review_id=review_id,
            card_id=card_id,
            rating=rating,
            due_at=sched_transition.due_at.isoformat(),
            interval_days=sched_transition.interval_days,
            ease=sched_transition.ease,
            topic_proficiency=(
                float(primary_result.proficiency)
                if primary_result is not None
                else 0.5
            ),
            idempotent_replay=False,
        )
