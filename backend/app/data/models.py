"""
SQLAlchemy ORM Models for FlashCards Project

This module defines all database models using SQLAlchemy ORM for easy
migration between SQLite (development) and PostgreSQL/MySQL (cloud).

Tables:
- users: User accounts
- documents: Ingested documents
- chunks: Document chunks with text
- embedding_cache: Cached embeddings
- vector_index_map: Stable mapping from vector index to chunk_id
- exams: Exam workspaces (like ChatGPT threads)
- exam_documents: Many-to-many relationship between exams and documents
- topics: Semantic topics per exam
- topic_chunks: Many-to-many relationship between topics and chunks
- topic_evidence: Evidence spans proving topic labels
- topic_vectors: Topic centroid vectors for routing
- cards: Flashcards per exam/topic
- card_proofs: Evidence/proofs attached to cards
- card_reviews: User ratings/reviews
- card_scheduling: SRS scheduling data (Phase 6)
- topic_proficiency: User proficiency per topic (Phase 6)
- card_topics: Topic linkage and role/weight per card
- exam_session_state: Per-user exam resume pointers
- card_presentation_log: Immutable per-user card presentation sequence
- student_knowledge_state: Bounded per-topic student memory state
- question_index: Question embeddings for deduplication (Phase 5)
- events: Immutable event log
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import uuid

from sqlalchemy import (
    String, Integer, Float, Text, LargeBinary,
    ForeignKey, Index, DateTime, text
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    sessionmaker, Session
)
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_id(length: int = 16) -> str:
    """Generate a random hex ID."""
    return uuid.uuid4().hex[:length]


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# USER & AUTH
# =============================================================================

class User(Base):
    """User accounts."""
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    exams: Mapped[List["Exam"]] = relationship("Exam", back_populates="user")
    exam_session_states: Mapped[List["ExamSessionState"]] = relationship(
        "ExamSessionState", back_populates="user", cascade="all, delete-orphan"
    )
    presentation_logs: Mapped[List["CardPresentationLog"]] = relationship(
        "CardPresentationLog", back_populates="user", cascade="all, delete-orphan"
    )
    student_knowledge_states: Mapped[List["StudentKnowledgeState"]] = relationship(
        "StudentKnowledgeState", back_populates="user", cascade="all, delete-orphan"
    )


# =============================================================================
# DOCUMENTS & CHUNKS
# =============================================================================

class Document(Base):
    """Ingested documents."""
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """Document chunks with text content."""
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id"), nullable=False, index=True
    )
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")


class EmbeddingCache(Base):
    """Cached embeddings to avoid recomputation."""
    __tablename__ = "embedding_cache"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class VectorIndexMap(Base):
    """Stable mapping from vector index position to chunk_id."""
    __tablename__ = "vector_index_map"

    vector_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.chunk_id"), nullable=False, index=True
    )


# =============================================================================
# EXAMS
# =============================================================================

class Exam(Base):
    """Exam workspaces (like ChatGPT threads)."""
    __tablename__ = "exams"

    exam_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), default="mastery")  # mastery | exam
    state: Mapped[str] = mapped_column(String(32), default="diagnostic", index=True)
    diagnostic_total: Mapped[int] = mapped_column(Integer, default=0)
    diagnostic_answered: Mapped[int] = mapped_column(Integer, default=0)
    diagnostic_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    diagnostic_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    job_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    job_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    job_attempts: Mapped[int] = mapped_column(Integer, default=0)
    job_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="exams")
    topics: Mapped[List["Topic"]] = relationship("Topic", back_populates="exam", cascade="all, delete-orphan")
    cards: Mapped[List["Card"]] = relationship("Card", back_populates="exam", cascade="all, delete-orphan")
    events: Mapped[List["Event"]] = relationship("Event", back_populates="exam", cascade="all, delete-orphan")
    session_states: Mapped[List["ExamSessionState"]] = relationship(
        "ExamSessionState", back_populates="exam", cascade="all, delete-orphan"
    )
    presentation_logs: Mapped[List["CardPresentationLog"]] = relationship(
        "CardPresentationLog", back_populates="exam", cascade="all, delete-orphan"
    )
    student_knowledge_states: Mapped[List["StudentKnowledgeState"]] = relationship(
        "StudentKnowledgeState", back_populates="exam", cascade="all, delete-orphan"
    )


class ExamDocument(Base):
    """Many-to-many relationship between exams and documents."""
    __tablename__ = "exam_documents"

    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), primary_key=True
    )
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id"), primary_key=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# =============================================================================
# TOPICS
# =============================================================================

class Topic(Base):
    """Semantic topics per exam, built by clustering."""
    __tablename__ = "topics"

    topic_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    exam: Mapped["Exam"] = relationship("Exam", back_populates="topics")
    evidence: Mapped[List["TopicEvidence"]] = relationship(
        "TopicEvidence", back_populates="topic", cascade="all, delete-orphan"
    )
    cards: Mapped[List["Card"]] = relationship("Card", back_populates="topic")
    card_topics: Mapped[List["CardTopic"]] = relationship(
        "CardTopic", back_populates="topic", cascade="all, delete-orphan"
    )
    student_knowledge_states: Mapped[List["StudentKnowledgeState"]] = relationship(
        "StudentKnowledgeState", back_populates="topic", cascade="all, delete-orphan"
    )


class TopicChunk(Base):
    """Many-to-many relationship between topics and chunks."""
    __tablename__ = "topic_chunks"

    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.chunk_id"), primary_key=True, index=True
    )


class TopicEvidence(Base):
    """Evidence spans proving topic labels are grounded in document text."""
    __tablename__ = "topic_evidence"

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), nullable=False, index=True
    )
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="evidence")


class TopicVector(Base):
    """Topic centroid vectors for fast routing."""
    __tablename__ = "topic_vectors"

    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), primary_key=True
    )
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


# =============================================================================
# CARDS
# =============================================================================

class Card(Base):
    """Flashcards per exam/topic."""
    __tablename__ = "cards"

    card_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), nullable=False, index=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    card_type: Mapped[str] = mapped_column(String(32), default="learning", index=True)
    retired_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    supersedes_card_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("cards.card_id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    exam: Mapped["Exam"] = relationship("Exam", back_populates="cards")
    topic: Mapped["Topic"] = relationship("Topic", back_populates="cards")
    proofs: Mapped[List["CardProof"]] = relationship(
        "CardProof", back_populates="card", cascade="all, delete-orphan"
    )
    reviews: Mapped[List["CardReview"]] = relationship(
        "CardReview", back_populates="card", cascade="all, delete-orphan"
    )
    scheduling: Mapped[Optional["CardScheduling"]] = relationship(
        "CardScheduling", back_populates="card", uselist=False, cascade="all, delete-orphan"
    )
    topic_links: Mapped[List["CardTopic"]] = relationship(
        "CardTopic", back_populates="card", cascade="all, delete-orphan"
    )
    supersedes_card: Mapped[Optional["Card"]] = relationship(
        "Card",
        remote_side=[card_id],
        back_populates="superseded_by_cards",
        foreign_keys=[supersedes_card_id],
    )
    superseded_by_cards: Mapped[List["Card"]] = relationship(
        "Card",
        back_populates="supersedes_card",
    )


class CardTopic(Base):
    """Card-topic mapping with role-based impact weighting."""
    __tablename__ = "card_topics"

    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_id"), primary_key=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(16), default="primary")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    card: Mapped["Card"] = relationship("Card", back_populates="topic_links")
    topic: Mapped["Topic"] = relationship("Topic", back_populates="card_topics")


class CardProof(Base):
    """Evidence/proofs attached to a card."""
    __tablename__ = "card_proofs"

    proof_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_id"), nullable=False, index=True
    )
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    card: Mapped["Card"] = relationship("Card", back_populates="proofs")


class CardReview(Base):
    """User ratings/reviews for cards."""
    __tablename__ = "card_reviews"

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), nullable=False, index=True
    )
    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_id"), nullable=False, index=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), nullable=False, index=True
    )
    # Rating values (Phase 6 semantics):
    # - i_knew_it
    # - almost_knew
    # - learned_now
    # - dont_understand
    rating: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    card: Mapped["Card"] = relationship("Card", back_populates="reviews")


# =============================================================================
# SRS SCHEDULING
# =============================================================================

class CardScheduling(Base):
    """SRS scheduling data for spaced repetition."""
    __tablename__ = "card_scheduling"

    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_id"), primary_key=True
    )
    due_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(16), default="new", index=True)
    interval_days: Mapped[float] = mapped_column(Float, default=1.0)
    ease: Mapped[float] = mapped_column(Float, default=2.5)  # SM-2 ease factor
    reps: Mapped[int] = mapped_column(Integer, default=0)
    lapses: Mapped[int] = mapped_column(Integer, default=0)
    last_reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    card: Mapped["Card"] = relationship("Card", back_populates="scheduling")


# =============================================================================
# TOPIC PROFICIENCY 
# =============================================================================

class TopicProficiency(Base):
    """User proficiency per topic for adaptive learning."""
    __tablename__ = "topic_proficiency"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), primary_key=True
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), primary_key=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), primary_key=True
    )
    proficiency: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0 to 1.0
    current_difficulty: Mapped[int] = mapped_column(Integer, default=1)
    streak_up: Mapped[int] = mapped_column(Integer, default=0)
    streak_down: Mapped[int] = mapped_column(Integer, default=0)
    seen_count: Mapped[int] = mapped_column(Integer, default=0)
    correctish_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class ExamSessionState(Base):
    """Per-user exam session pointer for resume behavior."""
    __tablename__ = "exam_session_state"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), primary_key=True
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), primary_key=True
    )
    last_served_card_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("cards.card_id"), nullable=True
    )
    last_presented_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="exam_session_states")
    exam: Mapped["Exam"] = relationship("Exam", back_populates="session_states")
    last_served_card: Mapped[Optional["Card"]] = relationship("Card")


class CardPresentationLog(Base):
    """Immutable card presentation sequence for a user+exam."""
    __tablename__ = "card_presentation_log"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), primary_key=True
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), primary_key=True
    )
    sequence_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_id"), nullable=False, index=True
    )
    presented_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="presentation_logs")
    exam: Mapped["Exam"] = relationship("Exam", back_populates="presentation_logs")
    card: Mapped["Card"] = relationship("Card")


class StudentKnowledgeState(Base):
    """Bounded per-topic memory state for student-model synthesis."""
    __tablename__ = "student_knowledge_state"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), primary_key=True
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), primary_key=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), primary_key=True
    )
    memory: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="student_knowledge_states")
    exam: Mapped["Exam"] = relationship("Exam", back_populates="student_knowledge_states")
    topic: Mapped["Topic"] = relationship("Topic", back_populates="student_knowledge_states")


# =============================================================================
# QUESTION INDEX 
# =============================================================================

class QuestionIndexEntry(Base):
    """Question embeddings for deduplication.
    
    Stores question text + embedding for semantic similarity search.
    """
    __tablename__ = "question_index"

    question_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), nullable=False, index=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("topics.topic_id"), nullable=False, index=True
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# =============================================================================
# EVENTS
# =============================================================================

class Event(Base):
    """Immutable event log for analytics and debugging."""
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False
    )
    exam_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("exams.exam_id"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relationships
    exam: Mapped["Exam"] = relationship("Exam", back_populates="events")


# =============================================================================
# INDEXES (additional composite indexes for common queries)
# =============================================================================

# Index for finding cards by exam and status
Index("idx_cards_exam_status", Card.exam_id, Card.status)

# Index for finding due cards
Index("idx_scheduling_due", CardScheduling.due_at)
Index("idx_scheduling_due_state", CardScheduling.state, CardScheduling.due_at)

# Index for finding reviews by user and date
Index("idx_reviews_user_date", CardReview.user_id, CardReview.created_at)

# Index for topic proficiency lookup
Index("idx_proficiency_user_exam", TopicProficiency.user_id, TopicProficiency.exam_id)
Index("idx_proficiency_exam_topic", TopicProficiency.exam_id, TopicProficiency.topic_id)

# Index for question deduplication by exam/topic
Index("idx_question_exam_topic", QuestionIndexEntry.exam_id, QuestionIndexEntry.topic_id)

# Indexes for exam lifecycle views and session resume/history
Index("idx_session_state_exam_user", ExamSessionState.exam_id, ExamSessionState.user_id)
Index("idx_presentation_exam_user_seq", CardPresentationLog.exam_id, CardPresentationLog.user_id, CardPresentationLog.sequence_no)
Index("idx_student_knowledge_exam_topic", StudentKnowledgeState.exam_id, StudentKnowledgeState.topic_id)

# One primary topic constraint per card (dialect-specific partial unique indexes)
Index(
    "uq_card_topics_primary_per_card_sqlite",
    CardTopic.card_id,
    unique=True,
    sqlite_where=text("role = 'primary'"),
)
# Index(
#     "uq_card_topics_primary_per_card_postgres",
#     CardTopic.card_id,
#     unique=True,
#     postgresql_where=text("role = 'primary'"),
# )
