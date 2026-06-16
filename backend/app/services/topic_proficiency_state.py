from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from app.data.db_repository import StoredCardTopic, StoredTopicProficiency
from app.services.difficulty_frameworks import clamp_difficulty, framework_for_route
from app.services.topic_route_proficiency import (
    default_route_state,
    merge_route_state,
    normalize_route,
    project_route_columns,
    route_state_from_info,
)


@dataclass
class TopicProficiencyTransition:
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


class TopicProficiencyStateService:
    """
    Deterministic topic-level proficiency reducer.

    This service only computes topic progression updates and never updates card
    scheduling.
    """

    _RATING_DELTA = {
        "i_knew_it": 0.12,
        "almost_knew": 0.07,
        "learned_now": 0.03,
        "dont_understand": -0.15,
    }

    def _base(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
    ) -> TopicProficiencyTransition:
        return TopicProficiencyTransition(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
            proficiency=0.5,
            current_difficulty=1,
            streak_up=0,
            streak_down=0,
            seen_count=0,
            correctish_count=0,
            info={},
        )

    def _from_stored(
        self,
        current: StoredTopicProficiency,
    ) -> TopicProficiencyTransition:
        return TopicProficiencyTransition(
            user_id=current.user_id,
            exam_id=current.exam_id,
            topic_id=current.topic_id,
            proficiency=float(current.proficiency),
            current_difficulty=int(current.current_difficulty),
            streak_up=int(current.streak_up),
            streak_down=int(current.streak_down),
            seen_count=int(current.seen_count),
            correctish_count=int(current.correctish_count),
            info=current.info or {},
        )

    def apply_rating(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_link: StoredCardTopic,
        rating: str,
        current: Optional[StoredTopicProficiency],
        card_route: str = "default",
        difficulty_framework: Optional[str] = None,
    ) -> TopicProficiencyTransition:
        if rating not in self._RATING_DELTA:
            raise ValueError(f"Unsupported rating: {rating}")

        base = (
            self._from_stored(current)
            if current is not None
            else self._base(user_id=user_id, exam_id=exam_id, topic_id=topic_link.topic_id)
        )

        weight = float(topic_link.weight)
        if topic_link.role == "primary":
            weight = max(0.0, weight)
        else:
            weight = max(0.0, min(0.6, weight))

        route = normalize_route(card_route)
        framework = difficulty_framework or framework_for_route(route)
        default_fallback = default_route_state(
            card_route="default",
            proficiency=base.proficiency,
            current_difficulty=base.current_difficulty,
            streak_up=base.streak_up,
            streak_down=base.streak_down,
            seen_count=base.seen_count,
            correctish_count=base.correctish_count,
        )
        # When a route is seen for the first time (e.g. first math learning card
        # right after diagnostic), bootstrap from the existing topic-level state
        # so review counters keep accumulating instead of resetting to route-zero.
        route_fallback = default_route_state(
            card_route=route,
            proficiency=base.proficiency,
            current_difficulty=base.current_difficulty,
            streak_up=base.streak_up,
            streak_down=base.streak_down,
            seen_count=base.seen_count,
            correctish_count=base.correctish_count,
        )
        route_base = route_state_from_info(
            info=base.info,
            card_route=route,
            fallback=route_fallback,
        )

        delta = self._RATING_DELTA[rating] * weight
        new_proficiency = max(0.0, min(1.0, route_base.proficiency + delta))

        seen_count = route_base.seen_count + 1
        correctish_count = (
            route_base.correctish_count + 1
            if rating in ("i_knew_it", "almost_knew", "learned_now")
            else route_base.correctish_count
        )

        if delta >= 0:
            streak_up = route_base.streak_up + 1
            streak_down = 0
        else:
            streak_up = 0
            streak_down = route_base.streak_down + 1

        difficulty = route_base.current_difficulty
        if streak_up >= 3 and new_proficiency >= 0.7:
            difficulty = clamp_difficulty(framework, difficulty + 1)
            streak_up = 0
        if streak_down >= 2:
            difficulty = max(1, difficulty - 1)
            streak_down = 0

        route_state = default_route_state(
            card_route=route,
            proficiency=new_proficiency,
            current_difficulty=difficulty,
            streak_up=streak_up,
            streak_down=streak_down,
            seen_count=seen_count,
            correctish_count=correctish_count,
        )
        route_state.difficulty_framework = framework
        updated_at = datetime.now(timezone.utc).isoformat()
        info = merge_route_state(
            info=base.info,
            card_route=route,
            state=route_state,
            last_rating=rating,
            applied_weight=weight,
            updated_at=updated_at,
        )

        projection_fallback = default_route_state(
            card_route=route,
            proficiency=base.proficiency,
            current_difficulty=base.current_difficulty,
            streak_up=base.streak_up,
            streak_down=base.streak_down,
            seen_count=base.seen_count,
            correctish_count=base.correctish_count,
        )
        projection = project_route_columns(info=info, fallback=projection_fallback)

        return TopicProficiencyTransition(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_link.topic_id,
            proficiency=projection.proficiency,
            current_difficulty=projection.current_difficulty,
            streak_up=projection.streak_up,
            streak_down=projection.streak_down,
            seen_count=projection.seen_count,
            correctish_count=projection.correctish_count,
            info=info,
        )
