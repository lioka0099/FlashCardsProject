from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


CardRoute = Literal["default", "math_calculation"]
DifficultyFramework = Literal["bloom", "tag"]


@dataclass(frozen=True)
class DifficultyLevel:
    framework: DifficultyFramework
    level: int
    name: str
    instruction: str
    prompt_hint: str


BLOOM_LEVELS: Dict[int, DifficultyLevel] = {
    1: DifficultyLevel(
        framework="bloom",
        level=1,
        name="Recall",
        instruction="Test basic recall of definitions, facts, or terminology.",
        prompt_hint="Ask 'What is...?', 'Define...', or 'Name...' questions.",
    ),
    2: DifficultyLevel(
        framework="bloom",
        level=2,
        name="Understand",
        instruction="Ask to explain concepts in own words or describe relationships.",
        prompt_hint="Use 'Explain...', 'Describe...', 'Why does...' questions.",
    ),
    3: DifficultyLevel(
        framework="bloom",
        level=3,
        name="Apply",
        instruction="Present a scenario requiring application of knowledge.",
        prompt_hint="Use 'How would you use...?', 'Given X, what would...?' questions.",
    ),
    4: DifficultyLevel(
        framework="bloom",
        level=4,
        name="Analyze",
        instruction="Ask to compare, contrast, or analyze trade-offs. May span multiple topics.",
        prompt_hint="Use 'Compare...', 'What are the trade-offs...?' questions.",
    ),
    5: DifficultyLevel(
        framework="bloom",
        level=5,
        name="Evaluate",
        instruction="Ask to design solutions, justify choices, or critique approaches.",
        prompt_hint="Use 'Design...', 'Justify...', 'What would be the best...?' questions.",
    ),
}


TAG_LEVELS: Dict[int, DifficultyLevel] = {
    1: DifficultyLevel(
        framework="tag",
        level=1,
        name="Memorization",
        instruction=(
            "Use a directly stated formula, fact, or rule in a near-identical form, "
            "with a concrete calculational result."
        ),
        prompt_hint="Ask for a direct calculation from a stated rule or formula.",
    ),
    2: DifficultyLevel(
        framework="tag",
        level=2,
        name="Procedures Without Connections",
        instruction="Require a familiar routine procedure with all givens supplied.",
        prompt_hint="Ask for a straightforward solve, simplify, expand, factor, substitute, or compute task.",
    ),
    3: DifficultyLevel(
        framework="tag",
        level=3,
        name="Procedures With Connections",
        instruction="Require choosing or connecting a procedure to the given representation.",
        prompt_hint="Ask for a multi-step calculation that uses a grounded formula or relationship.",
    ),
    4: DifficultyLevel(
        framework="tag",
        level=4,
        name="Doing Mathematics",
        instruction="Require bounded planning or reasoning before carrying out the calculation.",
        prompt_hint="Ask for a less direct but still verifiable calculation grounded in the excerpts.",
    ),
}


_FRAMEWORK_LEVELS: Dict[DifficultyFramework, Dict[int, DifficultyLevel]] = {
    "bloom": BLOOM_LEVELS,
    "tag": TAG_LEVELS,
}


def framework_for_route(card_route: str | None) -> DifficultyFramework:
    return "tag" if card_route == "math_calculation" else "bloom"


def max_level(framework: str | None) -> int:
    fw = normalize_framework(framework)
    return max(_FRAMEWORK_LEVELS[fw])


def normalize_framework(framework: str | None) -> DifficultyFramework:
    return "tag" if framework == "tag" else "bloom"


def clamp_difficulty(framework: str | None, value: Any) -> int:
    fw = normalize_framework(framework)
    try:
        raw = int(value)
    except Exception:
        raw = 1
    return max(1, min(max_level(fw), raw))


def get_level(framework: str | None, difficulty: Any) -> DifficultyLevel:
    fw = normalize_framework(framework)
    level_num = clamp_difficulty(fw, difficulty)
    return _FRAMEWORK_LEVELS[fw][level_num]


def card_info_for_difficulty(framework: str | None, difficulty: Any) -> Dict[str, Any]:
    level = get_level(framework, difficulty)
    if level.framework == "tag":
        return {
            "difficulty_framework": "tag",
            "tag_level": level.level,
            "tag_level_name": level.name,
        }
    return {
        "difficulty_framework": "bloom",
        "bloom_level": level.level,
        "bloom_level_name": level.name,
    }

