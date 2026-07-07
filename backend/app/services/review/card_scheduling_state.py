"""SRS card scheduling: compute the next due date / interval / ease from a review rating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.data.db_repository import StoredCardScheduling


@dataclass
class SchedulingTransition:
    card_id: str
    due_at: datetime
    state: str
    interval_days: float
    ease: float
    reps: int
    lapses: int
    last_reviewed_at: datetime


class CardSchedulingStateService:
    """
    Deterministic card-level scheduling reducer.

    This service only computes card scheduling transitions and never updates
    topic proficiency.
    """

    @staticmethod
    def _ensure_now(now: Optional[datetime]) -> datetime:
        if now is None:
            return datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now

    @staticmethod
    def _base_state(card_id: str, now: datetime) -> SchedulingTransition:
        return SchedulingTransition(
            card_id=card_id,
            due_at=now,
            state="new",
            interval_days=1.0,
            ease=2.5,
            reps=0,
            lapses=0,
            last_reviewed_at=now,
        )

    @staticmethod
    def _from_stored(current: StoredCardScheduling, now: datetime) -> SchedulingTransition:
        due_at = datetime.fromisoformat(current.due_at) if current.due_at else now
        last_reviewed_at = (
            datetime.fromisoformat(current.last_reviewed_at)
            if current.last_reviewed_at
            else now
        )
        return SchedulingTransition(
            card_id=current.card_id,
            due_at=due_at,
            state=current.state,
            interval_days=float(current.interval_days),
            ease=float(current.ease),
            reps=int(current.reps),
            lapses=int(current.lapses),
            last_reviewed_at=last_reviewed_at,
        )

    def apply_rating(
        self,
        *,
        card_id: str,
        rating: str,
        current: Optional[StoredCardScheduling],
        now: Optional[datetime] = None,
    ) -> SchedulingTransition:
        now_dt = self._ensure_now(now)
        state = (
            self._from_stored(current, now_dt)
            if current is not None
            else self._base_state(card_id, now_dt)
        )

        if rating == "i_knew_it":
            new_interval = max(7.0, state.interval_days * 3.0)
            new_ease = min(3.0, state.ease + 0.15)
            return SchedulingTransition(
                card_id=card_id,
                due_at=now_dt,
                state="retired",
                interval_days=new_interval,
                ease=new_ease,
                reps=state.reps + 1,
                lapses=state.lapses,
                last_reviewed_at=now_dt,
            )

        if rating == "almost_knew":
            new_interval = max(2.0, state.interval_days * 1.6)
            new_ease = min(3.0, state.ease + 0.05)
            return SchedulingTransition(
                card_id=card_id,
                due_at=now_dt + timedelta(days=new_interval),
                state="review",
                interval_days=new_interval,
                ease=new_ease,
                reps=state.reps + 1,
                lapses=state.lapses,
                last_reviewed_at=now_dt,
            )

        if rating == "learned_now":
            new_interval = max(1.0, state.interval_days * 1.2)
            new_ease = max(1.3, state.ease - 0.05)
            next_state = "learning" if (state.reps + 1) < 2 else "review"
            return SchedulingTransition(
                card_id=card_id,
                due_at=now_dt + timedelta(days=new_interval),
                state=next_state,
                interval_days=new_interval,
                ease=new_ease,
                reps=state.reps + 1,
                lapses=state.lapses,
                last_reviewed_at=now_dt,
            )

        if rating == "dont_understand":
            new_interval = 0.25
            new_ease = max(1.3, state.ease - 0.2)
            return SchedulingTransition(
                card_id=card_id,
                due_at=now_dt + timedelta(days=new_interval),
                state="relearning",
                interval_days=new_interval,
                ease=new_ease,
                reps=max(0, state.reps),
                lapses=state.lapses + 1,
                last_reviewed_at=now_dt,
            )

        raise ValueError(f"Unsupported rating: {rating}")
