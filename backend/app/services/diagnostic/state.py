"""Diagnostic state transitions: advance a topic's diagnostic status as answers arrive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from app.data.db_repository import StoredTopicProficiency


_RATING_TO_OBSERVED = {
    "i_knew_it": 1.0,
    "almost_knew": 0.75,
    "learned_now": 0.45,
    "dont_understand": 0.15,
}


@dataclass
class DiagnosticTopicTransition:
    user_id: str
    exam_id: str
    topic_id: str
    proficiency: float
    current_difficulty: int
    streak_up: int
    streak_down: int
    seen_count: int
    correctish_count: int
    info: Dict[str, object]


class DiagnosticStateService:
    """
    Topic-level diagnostic reducer.

    Simplified for current product behavior:
    one diagnostic card per topic, answered once.

    So proficiency is directly mapped from the rating signal.
    """

    def apply_rating(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        rating: str,
        current: Optional[StoredTopicProficiency],
    ) -> DiagnosticTopicTransition:
        if rating not in _RATING_TO_OBSERVED:
            raise ValueError(f"Unsupported rating: {rating}")

        observed = float(_RATING_TO_OBSERVED[rating])
        proficiency = max(0.0, min(1.0, observed))
        seen_count = 1
        correctish_count = 0 if rating == "dont_understand" else 1

        if rating == "dont_understand":
            streak_up = 0
            streak_down = 1
        else:
            streak_up = 1
            streak_down = 0

        # Initial ladder from single diagnostic signal.
        if proficiency >= 0.82:
            difficulty = 3
        elif proficiency >= 0.62:
            difficulty = 2
        else:
            difficulty = 1
        difficulty = max(1, min(5, difficulty))

        return DiagnosticTopicTransition(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
            proficiency=proficiency,
            current_difficulty=difficulty,
            streak_up=streak_up,
            streak_down=streak_down,
            seen_count=seen_count,
            correctish_count=correctish_count,
            info={
                "source": "diagnostic_state_service",
                "last_rating": rating,
                "observed_score": observed,
                "mode": "single_shot_mapping",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

