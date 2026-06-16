from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from app.services.difficulty_frameworks import clamp_difficulty, framework_for_route


DEFAULT_ROUTE = "default"
MATH_ROUTE = "math_calculation"
MATH_CONCEPTUAL_ROUTE = "math_conceptual"
ROUTE_PROFICIENCY_KEY = "route_proficiency"


@dataclass
class RouteProficiencyState:
    difficulty_framework: str
    current_difficulty: int
    proficiency: float
    streak_up: int
    streak_down: int
    seen_count: int
    correctish_count: int

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_route(card_route: Optional[str]) -> str:
    if card_route == MATH_ROUTE:
        return MATH_ROUTE
    if card_route == MATH_CONCEPTUAL_ROUTE:
        return MATH_CONCEPTUAL_ROUTE
    return DEFAULT_ROUTE


def default_route_state(
    *,
    card_route: Optional[str],
    proficiency: float = 0.5,
    current_difficulty: int = 1,
    streak_up: int = 0,
    streak_down: int = 0,
    seen_count: int = 0,
    correctish_count: int = 0,
) -> RouteProficiencyState:
    route = normalize_route(card_route)
    framework = framework_for_route(route)
    return RouteProficiencyState(
        difficulty_framework=framework,
        current_difficulty=clamp_difficulty(framework, current_difficulty),
        proficiency=max(0.0, min(1.0, float(proficiency))),
        streak_up=max(0, int(streak_up)),
        streak_down=max(0, int(streak_down)),
        seen_count=max(0, int(seen_count)),
        correctish_count=max(0, int(correctish_count)),
    )


def route_state_from_info(
    *,
    info: Optional[Dict[str, Any]],
    card_route: Optional[str],
    fallback: RouteProficiencyState,
) -> RouteProficiencyState:
    route = normalize_route(card_route)
    route_map = (info or {}).get(ROUTE_PROFICIENCY_KEY)
    raw = route_map.get(route) if isinstance(route_map, dict) else None
    if not isinstance(raw, dict):
        return fallback
    framework = raw.get("difficulty_framework") or framework_for_route(route)
    return RouteProficiencyState(
        difficulty_framework=framework,
        current_difficulty=clamp_difficulty(framework, raw.get("current_difficulty", fallback.current_difficulty)),
        proficiency=max(0.0, min(1.0, float(raw.get("proficiency", fallback.proficiency)))),
        streak_up=max(0, int(raw.get("streak_up", fallback.streak_up))),
        streak_down=max(0, int(raw.get("streak_down", fallback.streak_down))),
        seen_count=max(0, int(raw.get("seen_count", fallback.seen_count))),
        correctish_count=max(0, int(raw.get("correctish_count", fallback.correctish_count))),
    )


def merge_route_state(
    *,
    info: Optional[Dict[str, Any]],
    card_route: Optional[str],
    state: RouteProficiencyState,
    last_rating: str,
    applied_weight: float,
    updated_at: str,
) -> Dict[str, Any]:
    route = normalize_route(card_route)
    out = dict(info or {})
    route_map = dict(out.get(ROUTE_PROFICIENCY_KEY) or {})
    route_map[route] = state.to_info()
    out[ROUTE_PROFICIENCY_KEY] = route_map
    out["last_card_route"] = route
    out["last_difficulty_framework"] = state.difficulty_framework
    out["last_rating"] = last_rating
    out["applied_weight"] = applied_weight
    out["updated_at"] = updated_at
    return out

