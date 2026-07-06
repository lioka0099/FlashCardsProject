"""SRS memory: scheduling, proficiency, knowledge, reviews."""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.data.db_engine import get_db
from app.data.models import (
    Exam, Card, CardReview, CardScheduling, TopicProficiency, StudentKnowledgeState,
)
from app.data.db_repository.dtos import (
    StoredCardScheduling, StoredTopicProficiency, StoredStudentKnowledgeState, _datetime_to_str,
)


class ReviewRepo:
    def get_card_scheduling(
        self,
        *,
        card_id: str,
        session: Optional[Session] = None,
    ) -> Optional[StoredCardScheduling]:
        """Get scheduling row for a card."""
        if session is not None:
            row = session.query(CardScheduling).filter(CardScheduling.card_id == card_id).first()
            if not row:
                return None
            return StoredCardScheduling.from_orm(row)

        with get_db() as db:
            row = db.query(CardScheduling).filter(CardScheduling.card_id == card_id).first()
            if not row:
                return None
            return StoredCardScheduling.from_orm(row)

    def upsert_card_scheduling(
        self,
        *,
        card_id: str,
        due_at: datetime,
        state: str,
        interval_days: float,
        ease: float,
        reps: int,
        lapses: int,
        last_reviewed_at: Optional[datetime],
        session: Optional[Session] = None,
    ) -> None:
        """Insert/update card scheduling state."""
        if session is not None:
            row = session.query(CardScheduling).filter(CardScheduling.card_id == card_id).first()
            if row:
                row.due_at = due_at
                row.state = state
                row.interval_days = interval_days
                row.ease = ease
                row.reps = reps
                row.lapses = lapses
                row.last_reviewed_at = last_reviewed_at
            else:
                session.add(
                    CardScheduling(
                        card_id=card_id,
                        due_at=due_at,
                        state=state,
                        interval_days=interval_days,
                        ease=ease,
                        reps=reps,
                        lapses=lapses,
                        last_reviewed_at=last_reviewed_at,
                    )
                )
            return

        with get_db() as db:
            row = db.query(CardScheduling).filter(CardScheduling.card_id == card_id).first()
            if row:
                row.due_at = due_at
                row.state = state
                row.interval_days = interval_days
                row.ease = ease
                row.reps = reps
                row.lapses = lapses
                row.last_reviewed_at = last_reviewed_at
            else:
                db.add(
                    CardScheduling(
                        card_id=card_id,
                        due_at=due_at,
                        state=state,
                        interval_days=interval_days,
                        ease=ease,
                        reps=reps,
                        lapses=lapses,
                        last_reviewed_at=last_reviewed_at,
                    )
                )

    def list_due_cards(
        self,
        *,
        user_id: str,
        exam_id: str,
        at_or_before: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[StoredCardScheduling]:
        """
        List due cards for a user/exam with deterministic ordering.

        Order: due_at asc, lapses desc, card_id asc.
        """
        cutoff = at_or_before or datetime.now(timezone.utc)
        with get_db() as db:
            rows = (
                db.query(CardScheduling)
                .join(Card, Card.card_id == CardScheduling.card_id)
                .join(Exam, Exam.exam_id == Card.exam_id)
                .filter(
                    Exam.user_id == user_id,
                    Card.exam_id == exam_id,
                    CardScheduling.due_at <= cutoff,
                    Card.status == "active",
                )
                .order_by(
                    CardScheduling.due_at.asc(),
                    CardScheduling.lapses.desc(),
                    CardScheduling.card_id.asc(),
                )
                .limit(limit)
                .all()
            )
            return [
                StoredCardScheduling.from_orm(r)
                for r in rows
            ]

    def list_card_scheduling_by_state(
        self,
        *,
        exam_id: str,
        state: str,
        limit: int = 200,
    ) -> List[StoredCardScheduling]:
        """List scheduling rows for cards in exam by scheduling state."""
        with get_db() as db:
            rows = (
                db.query(CardScheduling)
                .join(Card, Card.card_id == CardScheduling.card_id)
                .filter(
                    Card.exam_id == exam_id,
                    CardScheduling.state == state,
                    Card.status == "active",
                )
                .order_by(
                    CardScheduling.due_at.asc(),
                    CardScheduling.lapses.desc(),
                    CardScheduling.card_id.asc(),
                )
                .limit(limit)
                .all()
            )
            return [
                StoredCardScheduling.from_orm(r)
                for r in rows
            ]
    def get_topic_proficiency(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        session: Optional[Session] = None,
    ) -> Optional[StoredTopicProficiency]:
        """Get proficiency row by user+exam+topic."""
        if session is not None:
            row = (
                session.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == user_id,
                    TopicProficiency.exam_id == exam_id,
                    TopicProficiency.topic_id == topic_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredTopicProficiency.from_orm(row)

        with get_db() as db:
            row = (
                db.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == user_id,
                    TopicProficiency.exam_id == exam_id,
                    TopicProficiency.topic_id == topic_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredTopicProficiency.from_orm(row)

    def upsert_topic_proficiency(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        proficiency: float,
        current_difficulty: int,
        streak_up: int,
        streak_down: int,
        seen_count: int,
        correctish_count: int,
        info: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> None:
        """Insert/update topic proficiency row."""
        if session is not None:
            row = (
                session.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == user_id,
                    TopicProficiency.exam_id == exam_id,
                    TopicProficiency.topic_id == topic_id,
                )
                .first()
            )
            if row:
                row.proficiency = proficiency
                row.current_difficulty = current_difficulty
                row.streak_up = streak_up
                row.streak_down = streak_down
                row.seen_count = seen_count
                row.correctish_count = correctish_count
                row.info = info or {}
            else:
                session.add(
                    TopicProficiency(
                        user_id=user_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        proficiency=proficiency,
                        current_difficulty=current_difficulty,
                        streak_up=streak_up,
                        streak_down=streak_down,
                        seen_count=seen_count,
                        correctish_count=correctish_count,
                        info=info or {},
                    )
                )
            return

        with get_db() as db:
            row = (
                db.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == user_id,
                    TopicProficiency.exam_id == exam_id,
                    TopicProficiency.topic_id == topic_id,
                )
                .first()
            )
            if row:
                row.proficiency = proficiency
                row.current_difficulty = current_difficulty
                row.streak_up = streak_up
                row.streak_down = streak_down
                row.seen_count = seen_count
                row.correctish_count = correctish_count
                row.info = info or {}
            else:
                db.add(
                    TopicProficiency(
                        user_id=user_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        proficiency=proficiency,
                        current_difficulty=current_difficulty,
                        streak_up=streak_up,
                        streak_down=streak_down,
                        seen_count=seen_count,
                        correctish_count=correctish_count,
                        info=info or {},
                    )
                )

    def list_topic_proficiencies(
        self,
        *,
        user_id: str,
        exam_id: str,
    ) -> List[StoredTopicProficiency]:
        """List all topic proficiencies for a user/exam."""
        with get_db() as db:
            rows = (
                db.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == user_id,
                    TopicProficiency.exam_id == exam_id,
                )
                .order_by(TopicProficiency.topic_id.asc())
                .all()
            )
            return [
                StoredTopicProficiency.from_orm(r)
                for r in rows
            ]
    def get_student_knowledge_state(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        session: Optional[Session] = None,
    ) -> Optional[StoredStudentKnowledgeState]:
        """Get student memory state for a user/exam/topic."""
        if session is not None:
            row = (
                session.query(StudentKnowledgeState)
                .filter(
                    StudentKnowledgeState.user_id == user_id,
                    StudentKnowledgeState.exam_id == exam_id,
                    StudentKnowledgeState.topic_id == topic_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredStudentKnowledgeState.from_orm(row)

        with get_db() as db:
            row = (
                db.query(StudentKnowledgeState)
                .filter(
                    StudentKnowledgeState.user_id == user_id,
                    StudentKnowledgeState.exam_id == exam_id,
                    StudentKnowledgeState.topic_id == topic_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredStudentKnowledgeState.from_orm(row)

    def upsert_student_knowledge_state(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        memory: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> None:
        """Insert/update student memory state for a user/exam/topic."""
        payload = dict(memory or {})
        if session is not None:
            row = (
                session.query(StudentKnowledgeState)
                .filter(
                    StudentKnowledgeState.user_id == user_id,
                    StudentKnowledgeState.exam_id == exam_id,
                    StudentKnowledgeState.topic_id == topic_id,
                )
                .first()
            )
            if row:
                row.memory = payload
            else:
                session.add(
                    StudentKnowledgeState(
                        user_id=user_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        memory=payload,
                    )
                )
            return

        with get_db() as db:
            row = (
                db.query(StudentKnowledgeState)
                .filter(
                    StudentKnowledgeState.user_id == user_id,
                    StudentKnowledgeState.exam_id == exam_id,
                    StudentKnowledgeState.topic_id == topic_id,
                )
                .first()
            )
            if row:
                row.memory = payload
            else:
                db.add(
                    StudentKnowledgeState(
                        user_id=user_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        memory=payload,
                    )
                )

    def list_student_knowledge_states(
        self,
        *,
        user_id: str,
        exam_id: str,
    ) -> List[StoredStudentKnowledgeState]:
        """List student memory states for all topics in a user/exam."""
        with get_db() as db:
            rows = (
                db.query(StudentKnowledgeState)
                .filter(
                    StudentKnowledgeState.user_id == user_id,
                    StudentKnowledgeState.exam_id == exam_id,
                )
                .order_by(StudentKnowledgeState.topic_id.asc())
                .all()
            )
            return [
                StoredStudentKnowledgeState.from_orm(r)
                for r in rows
            ]
    def add_card_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        topic_id: str,
        rating: str,
        info: Optional[Dict[str, Any]] = None,
        review_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> str:
        """Insert a card review row and return review_id."""
        rid = review_id or uuid.uuid4().hex[:20]
        if session is not None:
            session.add(
                CardReview(
                    review_id=rid,
                    user_id=user_id,
                    exam_id=exam_id,
                    card_id=card_id,
                    topic_id=topic_id,
                    rating=rating,
                    info=info or {},
                )
            )
            return rid
        with get_db() as db:
            db.add(
                CardReview(
                    review_id=rid,
                    user_id=user_id,
                    exam_id=exam_id,
                    card_id=card_id,
                    topic_id=topic_id,
                    rating=rating,
                    info=info or {},
                )
            )
        return rid

    def get_card_review_by_idempotency_key(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        idempotency_key: str,
        session: Optional[Session] = None,
    ) -> Optional[str]:
        """Return existing review_id for a matching idempotency key."""
        if not idempotency_key:
            return None
        if session is not None:
            row = (
                session.query(CardReview)
                .filter(
                    CardReview.user_id == user_id,
                    CardReview.exam_id == exam_id,
                    CardReview.card_id == card_id,
                )
                .all()
            )
            for r in row:
                info = r.info or {}
                if str(info.get("idempotency_key") or "") == idempotency_key:
                    return r.review_id
            return None

        with get_db() as db:
            row = (
                db.query(CardReview)
                .filter(
                    CardReview.user_id == user_id,
                    CardReview.exam_id == exam_id,
                    CardReview.card_id == card_id,
                )
                .all()
            )
            for r in row:
                info = r.info or {}
                if str(info.get("idempotency_key") or "") == idempotency_key:
                    return r.review_id
            return None
