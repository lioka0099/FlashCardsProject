from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.data.db_repository import DBRepository
from app.data.vector_store import VectorStore
from app.deps import get_repo, get_store
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create, safe_json_load
from app.services.math_formatting import MATH_DISPLAY_INSTRUCTION


# Bloom's Taxonomy difficulty levels for question generation
DIFFICULTY_LEVELS = {
    1: {
        "name": "Recall",
        "instruction": "Test basic recall of definitions, facts, or terminology.",
        "prompt_hint": "Ask 'What is...?', 'Define...', or 'Name...' questions.",
    },
    2: {
        "name": "Understand",
        "instruction": "Ask to explain concepts in own words or describe relationships.",
        "prompt_hint": "Use 'Explain...', 'Describe...', 'Why does...' questions.",
    },
    3: {
        "name": "Apply",
        "instruction": "Present a scenario requiring application of knowledge.",
        "prompt_hint": "Use 'How would you use...?', 'Given X, what would...?' questions.",
    },
    4: {
        "name": "Analyze",
        "instruction": "Ask to compare, contrast, or analyze trade-offs. May span multiple topics.",
        "prompt_hint": "Use 'Compare...', 'What are the trade-offs...?' questions.",
    },
    5: {
        "name": "Evaluate",
        "instruction": "Ask to design solutions, justify choices, or critique approaches.",
        "prompt_hint": "Use 'Design...', 'Justify...', 'What would be the best...?' questions.",
    },
}


@dataclass
class GeneratedCard:
    card_id: str
    exam_id: str
    topic_id: str
    topic_label: str
    question: str
    answer: str
    difficulty: int
    proofs: List[Dict[str, Any]]


def pick_starter_topics(
    *,
    exam_id: str,
    repo: Optional[DBRepository] = None,
    n: int = 5,
) -> List[Tuple[str, str]]:
    """
    Pick N topics for starter flashcards.
    Current heuristic: highest n_chunks (stored in topics.info) first.
    Returns [(topic_id, label), ...].
    """
    repo = repo or get_repo()
    topics = repo.list_topics(exam_id=exam_id)
    scored: List[Tuple[int, str, str]] = []
    for t in topics:
        n_chunks = int(t.info.get("n_chunks") or 0)
        scored.append((n_chunks, t.topic_id, t.label))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(topic_id, label) for _, topic_id, label in scored[: max(0, int(n))]]


def generate_question_at_difficulty(
    *,
    topic_label: str,
    context_pack: str,
    difficulty: int = 1,
    model: str = CHAT_MODEL_FAST,
) -> str:
    """
    Generate one flashcard question at the specified Bloom's taxonomy level.
    
    Difficulty levels:
        1 - Recall: Basic definitions and facts
        2 - Understand: Explain concepts, describe relationships
        3 - Apply: Use knowledge in scenarios
        4 - Analyze: Compare, contrast, analyze trade-offs
        5 - Evaluate: Design solutions, justify choices
    
    Returns the question text (grounded in context_pack).
    """
    # Get difficulty config, default to level 1 if invalid
    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS[1])
    level_name = level["name"]
    level_instruction = level["instruction"]
    level_hint = level["prompt_hint"]
    
    sys_prompt = (
        "You write high-quality flashcard questions at specific cognitive levels.\n"
        "You must ground the question in the provided excerpts and avoid generic questions.\n"
        "Return JSON only."
    )
    
    user_prompt = (
        f"TOPIC LABEL:\n{topic_label}\n\n"
        "EXCERPTS (from the document, same topic):\n"
        f"{context_pack}\n\n"
        f"DIFFICULTY LEVEL: {level_name}\n"
        f"INSTRUCTION: {level_instruction}\n"
        f"HINT: {level_hint}\n\n"
        f"Create ONE flashcard question at the {level_name} level.\n"
        "Rules:\n"
        "- The question must be answerable using ONLY the excerpts.\n"
        "- Follow the difficulty level instruction carefully.\n"
        "- Ask about core concepts, not trivia.\n"
        "- Avoid referencing 'the excerpt' or 'the paper' in the question.\n"
        f"- {MATH_DISPLAY_INSTRUCTION}\n"
        "- Return JSON: {\"question\": \"...\"}\n"
    )
    
    # Higher difficulty = slightly more creative (higher temperature)
    temp = 0.3 + (difficulty - 1) * 0.1  # 0.3 to 0.7
    
    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temp,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    data = safe_json_load(raw)
    q = str(data.get("question") or "").strip()
    return q


def generate_starter_cards(
    *,
    exam_id: str,
    user_id: str,
    store: Optional[VectorStore] = None,
    n: int = 5,
    difficulty: int = 1,
    **kwargs,  # For backward compatibility
) -> List[GeneratedCard]:
    """
    Generate N starter cards across distinct topics using LangGraph flow.
    
    Features:
    - Topic-scoped question generation at specified difficulty
    - FAISS-based deduplication (exam-scoped)
    - Mandatory answer validation
    - Automatic retry on failures
    
    Requires topics to exist (call build_topics_for_exam() first).
    """
    from app.services.graph import generate_starter_cards_v2

    store = store or get_store()

    return generate_starter_cards_v2(
        exam_id=exam_id,
        user_id=user_id,
        n=n,
        difficulty=difficulty,
        store=store,
    )


