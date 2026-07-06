"""Session progression: session-state + presentation log."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.data.db_engine import get_db, db_scope
from app.data.models import (
    ExamSessionState, CardPresentationLog,
)
from app.data.db_repository.dtos import (
    StoredExamSessionState, StoredCardPresentation, _datetime_to_str,
)


class SessionRepo:
    def get_exam_session_state(
        self,
        *,
        user_id: str,
        exam_id: str,
        session: Optional[Session] = None,
    ) -> Optional[StoredExamSessionState]:
        """Get user exam session state."""
        with db_scope(session) as db:
            row = (
                db.query(ExamSessionState)
                .filter(
                    ExamSessionState.user_id == user_id,
                    ExamSessionState.exam_id == exam_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredExamSessionState.from_orm(row)

    def upsert_exam_session_state(
        self,
        *,
        user_id: str,
        exam_id: str,
        last_served_card_id: Optional[str],
        last_presented_at: Optional[datetime],
        session: Optional[Session] = None,
    ) -> None:
        """Insert/update user exam session pointer state."""
        with db_scope(session) as db:
            row = (
                db.query(ExamSessionState)
                .filter(
                    ExamSessionState.user_id == user_id,
                    ExamSessionState.exam_id == exam_id,
                )
                .first()
            )
            if row:
                row.last_served_card_id = last_served_card_id
                row.last_presented_at = last_presented_at
            else:
                db.add(
                    ExamSessionState(
                        user_id=user_id,
                        exam_id=exam_id,
                        last_served_card_id=last_served_card_id,
                        last_presented_at=last_presented_at,
                    )
                )

    def append_card_presentation(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        presented_at: Optional[datetime] = None,
        info: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> StoredCardPresentation:
        """Append next card presentation entry atomically for user+exam."""
        presented_at = presented_at or datetime.now(timezone.utc)

        with db_scope(session) as db:
            max_seq = (
                db.query(func.max(CardPresentationLog.sequence_no))
                .filter(
                    CardPresentationLog.user_id == user_id,
                    CardPresentationLog.exam_id == exam_id,
                )
                .scalar()
            )
            next_seq = int(max_seq or 0) + 1
            db.add(
                CardPresentationLog(
                    user_id=user_id,
                    exam_id=exam_id,
                    sequence_no=next_seq,
                    card_id=card_id,
                    presented_at=presented_at,
                    info=info or {},
                )
            )
            return StoredCardPresentation(
                user_id=user_id,
                exam_id=exam_id,
                sequence_no=next_seq,
                card_id=card_id,
                presented_at=_datetime_to_str(presented_at),
                info=info or {},
            )

    def get_latest_presentation(
        self,
        *,
        user_id: str,
        exam_id: str,
    ) -> Optional[StoredCardPresentation]:
        """Get latest presented card entry."""
        with get_db() as db:
            row = (
                db.query(CardPresentationLog)
                .filter(
                    CardPresentationLog.user_id == user_id,
                    CardPresentationLog.exam_id == exam_id,
                )
                .order_by(CardPresentationLog.sequence_no.desc())
                .first()
            )
            if not row:
                return None
            return StoredCardPresentation.from_orm(row)

    def get_previous_presentation(
        self,
        *,
        user_id: str,
        exam_id: str,
        current_sequence_no: int,
    ) -> Optional[StoredCardPresentation]:
        """Get immediately previous presentation entry by sequence number."""
        with get_db() as db:
            row = (
                db.query(CardPresentationLog)
                .filter(
                    CardPresentationLog.user_id == user_id,
                    CardPresentationLog.exam_id == exam_id,
                    CardPresentationLog.sequence_no < current_sequence_no,
                )
                .order_by(CardPresentationLog.sequence_no.desc())
                .first()
            )
            if not row:
                return None
            return StoredCardPresentation.from_orm(row)

    def list_presentations(
        self,
        *,
        user_id: str,
        exam_id: str,
        ascending: bool = True,
        limit: int = 500,
    ) -> List[StoredCardPresentation]:
        """List card presentation history for a user/exam."""
        with get_db() as db:
            q = db.query(CardPresentationLog).filter(
                CardPresentationLog.user_id == user_id,
                CardPresentationLog.exam_id == exam_id,
            )
            if ascending:
                q = q.order_by(CardPresentationLog.sequence_no.asc())
            else:
                q = q.order_by(CardPresentationLog.sequence_no.desc())
            rows = q.limit(limit).all()
            return [
                StoredCardPresentation.from_orm(row)
                for row in rows
            ]
