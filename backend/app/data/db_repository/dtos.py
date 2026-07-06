"""Detached DTOs returned by the repository, plus shared mapping helpers.

Each DTO owns the ORM->DTO mapping once via `from_orm`, so the coercion lives in a
single place instead of being retyped in every get_/list_ method.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

MAX_JOB_ATTEMPTS = 3


def _datetime_to_str(dt: Optional[datetime]) -> str:
    """Convert datetime to ISO string."""
    if dt is None:
        return ""
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


@dataclass
class JobData:
    exam_id: str
    user_id: str
    payload: dict
    attempts: int


@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str

    @classmethod
    def from_orm(cls, row) -> "StoredChunk":
        return cls(
            chunk_id=row.chunk_id,
            doc_id=row.doc_id,
            page=row.page or 0,
            start=row.start or 0,
            end=row.end or 0,
            text=row.text or "",
        )


@dataclass
class StoredExam:
    exam_id: str
    user_id: str
    title: str
    mode: str
    created_at: str
    updated_at: str
    info: Dict[str, Any]
    state: str = "diagnostic"
    diagnostic_total: int = 0
    diagnostic_answered: int = 0
    diagnostic_started_at: str = ""
    diagnostic_completed_at: str = ""

    @classmethod
    def from_orm(cls, row) -> "StoredExam":
        return cls(
            exam_id=row.exam_id,
            user_id=row.user_id,
            title=row.title,
            mode=row.mode,
            created_at=_datetime_to_str(row.created_at),
            updated_at=_datetime_to_str(row.updated_at),
            info=row.info or {},
            state=row.state,
            diagnostic_total=int(row.diagnostic_total or 0),
            diagnostic_answered=int(row.diagnostic_answered or 0),
            diagnostic_started_at=_datetime_to_str(row.diagnostic_started_at),
            diagnostic_completed_at=_datetime_to_str(row.diagnostic_completed_at),
        )


@dataclass
class StoredTopic:
    topic_id: str
    exam_id: str
    label: str
    created_at: str
    info: Dict[str, Any]

    @classmethod
    def from_orm(cls, row) -> "StoredTopic":
        return cls(
            topic_id=row.topic_id,
            exam_id=row.exam_id,
            label=row.label,
            created_at=_datetime_to_str(row.created_at),
            info=row.info or {},
        )


@dataclass
class StoredCard:
    card_id: str
    exam_id: str
    topic_id: str
    question: str
    answer: str
    difficulty: int
    created_at: str
    status: str
    info: Dict[str, Any]
    card_type: str = "learning"

    @classmethod
    def from_orm(cls, row) -> "StoredCard":
        return cls(
            card_id=row.card_id,
            exam_id=row.exam_id,
            topic_id=row.topic_id,
            question=row.question,
            answer=row.answer,
            difficulty=row.difficulty,
            created_at=_datetime_to_str(row.created_at),
            status=row.status,
            info=row.info or {},
            card_type=row.card_type,
        )


@dataclass
class StoredCardProof:
    proof_id: str
    card_id: str
    doc_id: str
    page: Optional[int]
    start: int
    end: int
    text: str
    score: float

    @classmethod
    def from_orm(cls, row) -> "StoredCardProof":
        return cls(
            proof_id=row.proof_id,
            card_id=row.card_id,
            doc_id=row.doc_id,
            page=row.page,
            start=int(row.start or 0),
            end=int(row.end or 0),
            text=row.text or "",
            score=float(row.score or 0.0),
        )


@dataclass
class StoredCardTopic:
    card_id: str
    topic_id: str
    role: str
    weight: float
    created_at: str

    @classmethod
    def from_orm(cls, row) -> "StoredCardTopic":
        return cls(
            card_id=row.card_id,
            topic_id=row.topic_id,
            role=row.role,
            weight=float(row.weight),
            created_at=_datetime_to_str(row.created_at),
        )


@dataclass
class StoredCardScheduling:
    card_id: str
    due_at: str
    state: str
    interval_days: float
    ease: float
    reps: int
    lapses: int
    last_reviewed_at: str

    @classmethod
    def from_orm(cls, row) -> "StoredCardScheduling":
        return cls(
            card_id=row.card_id,
            due_at=_datetime_to_str(row.due_at),
            state=row.state,
            interval_days=float(row.interval_days),
            ease=float(row.ease),
            reps=int(row.reps),
            lapses=int(row.lapses),
            last_reviewed_at=_datetime_to_str(row.last_reviewed_at),
        )


@dataclass
class StoredTopicProficiency:
    user_id: str
    exam_id: str
    topic_id: str
    proficiency: float
    current_difficulty: int
    streak_up: int
    streak_down: int
    seen_count: int
    correctish_count: int
    last_updated_at: str
    info: Dict[str, Any]

    @classmethod
    def from_orm(cls, row) -> "StoredTopicProficiency":
        return cls(
            user_id=row.user_id,
            exam_id=row.exam_id,
            topic_id=row.topic_id,
            proficiency=float(row.proficiency),
            current_difficulty=int(row.current_difficulty),
            streak_up=int(row.streak_up),
            streak_down=int(row.streak_down),
            seen_count=int(row.seen_count),
            correctish_count=int(row.correctish_count),
            last_updated_at=_datetime_to_str(row.last_updated_at),
            info=row.info or {},
        )


@dataclass
class StoredStudentKnowledgeState:
    user_id: str
    exam_id: str
    topic_id: str
    memory: Dict[str, Any]
    updated_at: str

    @classmethod
    def from_orm(cls, row) -> "StoredStudentKnowledgeState":
        return cls(
            user_id=row.user_id,
            exam_id=row.exam_id,
            topic_id=row.topic_id,
            memory=row.memory or {},
            updated_at=_datetime_to_str(row.updated_at),
        )


@dataclass
class StoredExamSessionState:
    user_id: str
    exam_id: str
    last_served_card_id: Optional[str]
    last_presented_at: str
    updated_at: str

    @classmethod
    def from_orm(cls, row) -> "StoredExamSessionState":
        return cls(
            user_id=row.user_id,
            exam_id=row.exam_id,
            last_served_card_id=row.last_served_card_id,
            last_presented_at=_datetime_to_str(row.last_presented_at),
            updated_at=_datetime_to_str(row.updated_at),
        )


@dataclass
class StoredCardPresentation:
    user_id: str
    exam_id: str
    sequence_no: int
    card_id: str
    presented_at: str
    info: Dict[str, Any]

    @classmethod
    def from_orm(cls, row) -> "StoredCardPresentation":
        return cls(
            user_id=row.user_id,
            exam_id=row.exam_id,
            sequence_no=int(row.sequence_no),
            card_id=row.card_id,
            presented_at=_datetime_to_str(row.presented_at),
            info=row.info or {},
        )
