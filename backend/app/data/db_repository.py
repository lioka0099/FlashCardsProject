"""
Database Repository Layer using SQLAlchemy ORM

This module provides the high-level database interface for the FlashCards application.
It uses SQLAlchemy ORM for all database operations, making it easy to
switch between SQLite (development) and PostgreSQL (cloud).

The dataclasses (StoredChunk, StoredExam, etc.) are kept for backwards
compatibility with existing code.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable, Sequence
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, delete, and_, func
from sqlalchemy.orm import Session

from app.data.db_engine import get_db, init_db, DATABASE_URL
from app.data.models import (
    User, Document, Chunk, EmbeddingCache, VectorIndexMap,
    Exam, ExamDocument, Topic, TopicChunk, TopicEvidence, TopicVector,
    Card, CardProof, CardReview, CardTopic, CardScheduling, TopicProficiency,
    ExamSessionState, CardPresentationLog, Event, QuestionIndexEntry,
    StudentKnowledgeState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str


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


@dataclass
class StoredTopic:
    topic_id: str
    exam_id: str
    label: str
    created_at: str
    info: Dict[str, Any]


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


@dataclass
class StoredCardTopic:
    card_id: str
    topic_id: str
    role: str
    weight: float
    created_at: str


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


@dataclass
class StoredStudentKnowledgeState:
    user_id: str
    exam_id: str
    topic_id: str
    memory: Dict[str, Any]
    updated_at: str


@dataclass
class StoredExamSessionState:
    user_id: str
    exam_id: str
    last_served_card_id: Optional[str]
    last_presented_at: str
    updated_at: str


@dataclass
class StoredCardPresentation:
    user_id: str
    exam_id: str
    sequence_no: int
    card_id: str
    presented_at: str
    info: Dict[str, Any]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _datetime_to_str(dt: Optional[datetime]) -> str:
    """Convert datetime to ISO string."""
    if dt is None:
        return ""
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


# =============================================================================
# DBRepository - Main Database Repository Class
# =============================================================================

class DBRepository:
    """
    Database repository layer using SQLAlchemy ORM.
    
    Works with any SQLAlchemy-supported database (SQLite, PostgreSQL, MySQL).
    
    Note: The actual database connection is configured via the DATABASE_URL
    environment variable. The db_path parameter is kept for backwards
    compatibility but the actual path is determined by DATABASE_URL or
    VECTOR_STORE_PATH environment variables.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to database file (kept for backwards compatibility).
                     Actual path determined by DATABASE_URL or VECTOR_STORE_PATH.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize tables using SQLAlchemy models
        init_db()
        
        # Log the actual database being used
        if DATABASE_URL.startswith("sqlite"):
            actual_path = DATABASE_URL.replace("sqlite:///", "")
            if str(db_path) != actual_path:
                logger.debug(
                    "Note: DBRepository initialized with path=%s but actual DB is %s "
                    "(controlled by DATABASE_URL/VECTOR_STORE_PATH env vars)",
                    db_path, actual_path
                )
        logger.info("SQLAlchemy database initialized")
    
    # def close(self):
    #     """Close the database connection."""
    #     # SQLAlchemy handles connection pooling, nothing to do here
    #     pass
    
    # =========================================================================
    # DOCUMENT METHODS
    # =========================================================================
    
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        """Add or update a document."""
        with get_db() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.path = path
                doc.title = title
                doc.info = info
            else:
                doc = Document(doc_id=doc_id, path=path, title=title, info=info)
                db.add(doc)

    def get_document_path(self, *, doc_id: str) -> Optional[str]:
        """Get absolute document path by document ID."""
        with get_db() as db:
            row = db.query(Document).filter(Document.doc_id == doc_id).first()
            if row is None:
                return None
            return row.path
    
    def add_chunks(self, chunk_rows: Iterable[StoredChunk]):
        """Add or update chunks."""
        with get_db() as db:
            for c in chunk_rows:
                chunk = db.query(Chunk).filter(Chunk.chunk_id == c.chunk_id).first()
                if chunk:
                    chunk.doc_id = c.doc_id
                    chunk.page = c.page
                    chunk.start = c.start
                    chunk.end = c.end
                    chunk.text = c.text
                else:
                    chunk = Chunk(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        page=c.page,
                        start=c.start,
                        end=c.end,
                        text=c.text,
                    )
                    db.add(chunk)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Get a chunk by its ID."""
        with get_db() as db:
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            return StoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page or 0,
                start=chunk.start or 0,
                end=chunk.end or 0,
                text=chunk.text or "",
            )

    def get_chunk_by_vector_index(self, vector_index: int) -> Optional[StoredChunk]:
        """Get chunk by vector index using the mapping table."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.vector_index == vector_index
            ).first()
            if not mapping:
                return None
            # Extract chunk_id while session is still open
            chunk_id = mapping.chunk_id
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            # Extract all values while session is still open
            return StoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page or 0,
                start=chunk.start or 0,
                end=chunk.end or 0,
                text=chunk.text or "",
            )
    
    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        """List all chunks for a document."""
        with get_db() as db:
            chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(
                Chunk.page, Chunk.start
            ).all()
            return [
                StoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    page=c.page or 0,
                    start=c.start or 0,
                    end=c.end or 0,
                    text=c.text or "",
                )
                for c in chunks
            ]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> Dict[str, StoredChunk]:
        """Fetch chunks by IDs in one query (used by Pinecone-backed retrieval)."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            rows = db.query(Chunk).filter(Chunk.chunk_id.in_(ids)).all()
            out: Dict[str, StoredChunk] = {}
            for chunk in rows:
                out[chunk.chunk_id] = StoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page=chunk.page or 0,
                    start=chunk.start or 0,
                    end=chunk.end or 0,
                    text=chunk.text or "",
                )
            return out
    
    # =========================================================================
    # VECTOR INDEX MAPPING METHODS
    # =========================================================================
    
    def add_vector_index_mapping(self, *, start_index: int, chunk_ids: Sequence[str]) -> None:
        """Add mapping from vector indices to chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return
        with get_db() as db:
            for i, cid in enumerate(ids):
                idx = start_index + i
                existing = db.query(VectorIndexMap).filter(
                    VectorIndexMap.vector_index == idx
                ).first()
                if existing:
                    existing.chunk_id = cid
                else:
                    db.add(VectorIndexMap(vector_index=idx, chunk_id=cid))
    
    def get_vector_index_by_chunk_id(self, chunk_id: str) -> Optional[int]:
        """Get vector index for a chunk ID."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id == chunk_id
            ).first()
            return mapping.vector_index if mapping else None
    
    def list_vector_indices_by_chunk_ids(self, chunk_ids: Sequence[str]) -> Dict[str, int]:
        """Get vector indices for multiple chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            mappings = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id.in_(ids)
            ).all()
            # Extract values while session is still open
            return {m.chunk_id: m.vector_index for m in mappings}
    
    # =========================================================================
    # USER METHODS
    # =========================================================================
    
    def ensure_user(self, user_id: str, *, name: Optional[str] = None) -> None:
        """Create user if not exists, optionally update name."""
        with get_db() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                if name is not None:
                    user.name = name
            else:
                db.add(User(user_id=user_id, name=name))
    
    # =========================================================================
    # EXAM METHODS
    # =========================================================================
    
    def create_exam(
        self,
        *,
        user_id: str,
        title: str,
        mode: str = "mastery",
        info: Optional[Dict[str, Any]] = None,
        exam_id: Optional[str] = None,
    ) -> str:
        """Create an exam workspace. Returns exam_id."""
        self.ensure_user(user_id)
        eid = exam_id or uuid.uuid4().hex[:16]
        with get_db() as db:
            existing = db.query(Exam).filter(Exam.exam_id == eid).first()
            if existing:
                existing.user_id = user_id
                existing.title = title
                existing.mode = mode
                existing.info = info or {}
            else:
                db.add(Exam(
                    exam_id=eid,
                    user_id=user_id,
                    title=title,
                    mode=mode,
                    info=info or {},
                ))
        return eid
    
    def update_exam(self, exam_id: str) -> None:
        """Update exam's updated_at timestamp."""
        with get_db() as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if exam:
                exam.updated_at = datetime.now(timezone.utc)
    
    def get_exam(
        self,
        exam_id: str,
        *,
        session: Optional[Session] = None,
    ) -> Optional[StoredExam]:
        """Get exam by ID."""
        if session is not None:
            exam = session.query(Exam).filter(Exam.exam_id == exam_id).first()
            if not exam:
                return None
            return StoredExam(
                exam_id=exam.exam_id,
                user_id=exam.user_id,
                title=exam.title,
                mode=exam.mode,
                created_at=_datetime_to_str(exam.created_at),
                updated_at=_datetime_to_str(exam.updated_at),
                info=exam.info or {},
                state=exam.state,
                diagnostic_total=int(exam.diagnostic_total or 0),
                diagnostic_answered=int(exam.diagnostic_answered or 0),
                diagnostic_started_at=_datetime_to_str(exam.diagnostic_started_at),
                diagnostic_completed_at=_datetime_to_str(exam.diagnostic_completed_at),
            )

        with get_db() as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if not exam:
                return None
            return StoredExam(
                exam_id=exam.exam_id,
                user_id=exam.user_id,
                title=exam.title,
                mode=exam.mode,
                created_at=_datetime_to_str(exam.created_at),
                updated_at=_datetime_to_str(exam.updated_at),
                info=exam.info or {},
                state=exam.state,
                diagnostic_total=int(exam.diagnostic_total or 0),
                diagnostic_answered=int(exam.diagnostic_answered or 0),
                diagnostic_started_at=_datetime_to_str(exam.diagnostic_started_at),
                diagnostic_completed_at=_datetime_to_str(exam.diagnostic_completed_at),
            )
    
    def list_exams(self, *, user_id: str, limit: int = 50) -> List[StoredExam]:
        """List exams for a user."""
        with get_db() as db:
            exams = db.query(Exam).filter(Exam.user_id == user_id).order_by(
                Exam.updated_at.desc()
            ).limit(limit).all()
            return [
                StoredExam(
                    exam_id=e.exam_id,
                    user_id=e.user_id,
                    title=e.title,
                    mode=e.mode,
                    created_at=_datetime_to_str(e.created_at),
                    updated_at=_datetime_to_str(e.updated_at),
                    info=e.info or {},
                    state=e.state,
                    diagnostic_total=int(e.diagnostic_total or 0),
                    diagnostic_answered=int(e.diagnostic_answered or 0),
                    diagnostic_started_at=_datetime_to_str(e.diagnostic_started_at),
                    diagnostic_completed_at=_datetime_to_str(e.diagnostic_completed_at),
                )
                for e in exams
            ]

    def update_exam_lifecycle(
        self,
        *,
        exam_id: str,
        state: Optional[str] = None,
        diagnostic_total: Optional[int] = None,
        diagnostic_answered: Optional[int] = None,
        diagnostic_started_at: Optional[datetime] = None,
        diagnostic_completed_at: Optional[datetime] = None,
        info_patch: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> None:
        """Update exam lifecycle fields and optionally merge info payload."""

        def _apply(db: Session) -> None:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is None:
                raise ValueError(f"Exam not found: {exam_id}")
            if state is not None:
                row.state = state
            if diagnostic_total is not None:
                row.diagnostic_total = max(0, int(diagnostic_total))
            if diagnostic_answered is not None:
                row.diagnostic_answered = max(0, int(diagnostic_answered))
            if diagnostic_started_at is not None:
                row.diagnostic_started_at = diagnostic_started_at
            if diagnostic_completed_at is not None:
                row.diagnostic_completed_at = diagnostic_completed_at
            if info_patch:
                merged = dict(row.info or {})
                merged.update(info_patch)
                row.info = merged
            row.updated_at = datetime.now(timezone.utc)

        if session is not None:
            _apply(session)
            return
        with get_db() as db:
            _apply(db)

    def increment_exam_diagnostic_answered(
        self,
        *,
        exam_id: str,
        by: int = 1,
        session: Optional[Session] = None,
    ) -> int:
        """Increment exam diagnostic_answered and return the new value."""

        def _apply(db: Session) -> int:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is None:
                raise ValueError(f"Exam not found: {exam_id}")
            row.diagnostic_answered = max(0, int(row.diagnostic_answered or 0) + int(by))
            row.updated_at = datetime.now(timezone.utc)
            return int(row.diagnostic_answered)

        if session is not None:
            return _apply(session)
        with get_db() as db:
            return _apply(db)
    
    def attach_documents_to_exam(self, *, exam_id: str, doc_ids: Sequence[str]) -> None:
        """Attach documents to an exam."""
        ids = [d for d in doc_ids if d]
        if not ids:
            return
        with get_db() as db:
            for doc_id in ids:
                existing = db.query(ExamDocument).filter(
                    ExamDocument.exam_id == exam_id,
                    ExamDocument.doc_id == doc_id
                ).first()
                if not existing:
                    db.add(ExamDocument(exam_id=exam_id, doc_id=doc_id))
        self.update_exam(exam_id)
    
    def list_exam_documents(self, *, exam_id: str) -> List[str]:
        """List document IDs attached to an exam."""
        with get_db() as db:
            docs = db.query(ExamDocument).filter(
                ExamDocument.exam_id == exam_id
            ).order_by(ExamDocument.added_at).all()
            return [d.doc_id for d in docs]

    def delete_exam_cascade(self, *, exam_id: str) -> None:
        """
        Delete an exam and all related exam-scoped rows.

        This is intended for compensation cleanup when bootstrap fails.
        """
        with get_db() as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if exam is None:
                return

            # Resolve document IDs currently linked to this exam.
            doc_ids = [
                row.doc_id
                for row in db.query(ExamDocument)
                .filter(ExamDocument.exam_id == exam_id)
                .all()
            ]

            # Card-scoped tables.
            card_ids = [
                row.card_id
                for row in db.query(Card).filter(Card.exam_id == exam_id).all()
            ]
            if card_ids:
                db.query(CardProof).filter(CardProof.card_id.in_(card_ids)).delete(
                    synchronize_session=False
                )
                db.query(CardScheduling).filter(CardScheduling.card_id.in_(card_ids)).delete(
                    synchronize_session=False
                )
                db.query(CardTopic).filter(CardTopic.card_id.in_(card_ids)).delete(
                    synchronize_session=False
                )
                db.query(CardReview).filter(CardReview.card_id.in_(card_ids)).delete(
                    synchronize_session=False
                )
                db.query(CardPresentationLog).filter(
                    CardPresentationLog.exam_id == exam_id
                ).delete(synchronize_session=False)
                db.query(Card).filter(Card.card_id.in_(card_ids)).delete(
                    synchronize_session=False
                )

            # Topic-scoped tables.
            topic_ids = [
                row.topic_id
                for row in db.query(Topic).filter(Topic.exam_id == exam_id).all()
            ]
            if topic_ids:
                db.query(QuestionIndexEntry).filter(
                    QuestionIndexEntry.topic_id.in_(topic_ids)
                ).delete(synchronize_session=False)
                db.query(TopicEvidence).filter(TopicEvidence.topic_id.in_(topic_ids)).delete(
                    synchronize_session=False
                )
                db.query(TopicChunk).filter(TopicChunk.topic_id.in_(topic_ids)).delete(
                    synchronize_session=False
                )
                db.query(TopicVector).filter(TopicVector.topic_id.in_(topic_ids)).delete(
                    synchronize_session=False
                )
                db.query(StudentKnowledgeState).filter(
                    StudentKnowledgeState.exam_id == exam_id
                ).delete(synchronize_session=False)
                db.query(TopicProficiency).filter(
                    TopicProficiency.exam_id == exam_id
                ).delete(synchronize_session=False)
                db.query(Topic).filter(Topic.topic_id.in_(topic_ids)).delete(
                    synchronize_session=False
                )

            # Remaining exam-scoped tables.
            db.query(QuestionIndexEntry).filter(
                QuestionIndexEntry.exam_id == exam_id
            ).delete(synchronize_session=False)
            db.query(Event).filter(Event.exam_id == exam_id).delete(
                synchronize_session=False
            )
            db.query(ExamSessionState).filter(
                ExamSessionState.exam_id == exam_id
            ).delete(synchronize_session=False)
            db.query(ExamDocument).filter(ExamDocument.exam_id == exam_id).delete(
                synchronize_session=False
            )
            db.query(Exam).filter(Exam.exam_id == exam_id).delete(
                synchronize_session=False
            )

            # Best-effort document/chunk cleanup for docs no longer attached anywhere.
            if doc_ids:
                still_attached = {
                    row.doc_id
                    for row in db.query(ExamDocument.doc_id)
                    .filter(ExamDocument.doc_id.in_(doc_ids))
                    .all()
                }
                orphan_doc_ids = [d for d in doc_ids if d not in still_attached]
                if orphan_doc_ids:
                    orphan_chunk_ids = [
                        row.chunk_id
                        for row in db.query(Chunk.chunk_id)
                        .filter(Chunk.doc_id.in_(orphan_doc_ids))
                        .all()
                    ]
                    if orphan_chunk_ids:
                        db.query(VectorIndexMap).filter(
                            VectorIndexMap.chunk_id.in_(orphan_chunk_ids)
                        ).delete(synchronize_session=False)
                        db.query(Chunk).filter(Chunk.chunk_id.in_(orphan_chunk_ids)).delete(
                            synchronize_session=False
                        )
                    db.query(Document).filter(Document.doc_id.in_(orphan_doc_ids)).delete(
                        synchronize_session=False
                    )
    
    # =========================================================================
    # TOPIC METHODS
    # =========================================================================
    
    def upsert_topic(
        self,
        *,
        topic_id: str,
        exam_id: str,
        label: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a topic."""
        with get_db() as db:
            topic = db.query(Topic).filter(Topic.topic_id == topic_id).first()
            if topic:
                topic.exam_id = exam_id
                topic.label = label
                topic.info = info or {}
            else:
                db.add(Topic(
                    topic_id=topic_id,
                    exam_id=exam_id,
                    label=label,
                    info=info or {},
                ))
    
    def list_topics(self, *, exam_id: str) -> List[StoredTopic]:
        """List topics for an exam."""
        with get_db() as db:
            topics = db.query(Topic).filter(Topic.exam_id == exam_id).order_by(
                Topic.created_at
            ).all()
            return [
                StoredTopic(
                    topic_id=t.topic_id,
                    exam_id=t.exam_id,
                    label=t.label,
                    created_at=_datetime_to_str(t.created_at),
                    info=t.info or {},
                )
                for t in topics
            ]
    
    def delete_topics_for_exam(self, *, exam_id: str) -> None:
        """Delete all topics and related data for an exam."""
        topics = self.list_topics(exam_id=exam_id)
        if not topics:
            return
        topic_ids = [t.topic_id for t in topics]
        with get_db() as db:
            # Delete related records first
            db.query(TopicEvidence).filter(TopicEvidence.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(TopicChunk).filter(TopicChunk.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(TopicVector).filter(TopicVector.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(Topic).filter(Topic.exam_id == exam_id).delete(
                synchronize_session=False
            )
    
    def replace_topic_chunks(self, *, topic_id: str, chunk_ids: Sequence[str]) -> None:
        """Replace all chunk assignments for a topic."""
        ids = [c for c in chunk_ids if c]
        with get_db() as db:
            db.query(TopicChunk).filter(TopicChunk.topic_id == topic_id).delete(
                synchronize_session=False
            )
            for cid in ids:
                db.add(TopicChunk(topic_id=topic_id, chunk_id=cid))
    
    def list_chunk_ids_for_topic(self, *, topic_id: str) -> List[str]:
        """Get chunk IDs for a topic."""
        with get_db() as db:
            chunks = db.query(TopicChunk).filter(TopicChunk.topic_id == topic_id).all()
            return [c.chunk_id for c in chunks]
    
    def list_topic_chunks_for_exam(self, *, exam_id: str) -> Dict[str, List[str]]:
        """Get all topic→chunks mappings for an exam."""
        with get_db() as db:
            results = db.query(TopicChunk.topic_id, TopicChunk.chunk_id).join(
                Topic, Topic.topic_id == TopicChunk.topic_id
            ).filter(Topic.exam_id == exam_id).all()
            out: Dict[str, List[str]] = {}
            for topic_id, chunk_id in results:
                out.setdefault(topic_id, []).append(chunk_id)
            return out
    
    def replace_topic_evidence(
        self,
        *,
        topic_id: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> None:
        """Replace all evidence for a topic."""
        with get_db() as db:
            db.query(TopicEvidence).filter(TopicEvidence.topic_id == topic_id).delete(
                synchronize_session=False
            )
            for ev in evidence:
                db.add(TopicEvidence(
                    evidence_id=ev.get("evidence_id") or uuid.uuid4().hex[:20],
                    topic_id=topic_id,
                    doc_id=str(ev.get("doc_id") or ""),
                    page=int(ev.get("page") or 0),
                    start=int(ev.get("start") or 0),
                    end=int(ev.get("end") or 0),
                    text=str(ev.get("text") or ""),
                ))
    
    # =========================================================================
    # TOPIC VECTOR METHODS
    # =========================================================================
    
    def upsert_topic_vector(self, *, topic_id: str, vector: np.ndarray) -> None:
        """Store or update topic centroid vector."""
        arr = vector.astype("float32", copy=False).reshape(-1)
        with get_db() as db:
            existing = db.query(TopicVector).filter(TopicVector.topic_id == topic_id).first()
            if existing:
                existing.dim = int(arr.size)
                existing.vector = arr.tobytes()
            else:
                db.add(TopicVector(
                    topic_id=topic_id,
                    dim=int(arr.size),
                    vector=arr.tobytes(),
                ))
    
    def get_topic_vector(self, *, topic_id: str) -> Optional[np.ndarray]:
        """Get topic centroid vector."""
        with get_db() as db:
            tv = db.query(TopicVector).filter(TopicVector.topic_id == topic_id).first()
            if not tv:
                return None
            try:
                arr = np.frombuffer(tv.vector, dtype="float32")
                if int(tv.dim) != arr.size:
                    return None
                return arr
            except Exception:
                return None
    
    def list_topic_vectors_for_exam(self, *, exam_id: str) -> Dict[str, np.ndarray]:
        """Get all topic vectors for an exam."""
        with get_db() as db:
            results = db.query(TopicVector.topic_id, TopicVector.dim, TopicVector.vector).join(
                Topic, Topic.topic_id == TopicVector.topic_id
            ).filter(Topic.exam_id == exam_id).all()
            out: Dict[str, np.ndarray] = {}
            for topic_id, dim, blob in results:
                try:
                    arr = np.frombuffer(blob, dtype="float32")
                    if int(dim) == arr.size:
                        out[topic_id] = arr
                except Exception:
                    continue
            return out
    
    # =========================================================================
    # CARD METHODS
    # =========================================================================
    
    def upsert_card(
        self,
        *,
        card_id: str,
        exam_id: str,
        topic_id: str,
        question: str,
        answer: str,
        difficulty: int = 1,
        card_type: str = "learning",
        status: str = "active",
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a card."""
        with get_db() as db:
            card = db.query(Card).filter(Card.card_id == card_id).first()
            if card:
                card.exam_id = exam_id
                card.topic_id = topic_id
                card.question = question
                card.answer = answer
                card.difficulty = difficulty
                card.card_type = card_type
                card.status = status
                card.info = info or {}
            else:
                db.add(Card(
                    card_id=card_id,
                    exam_id=exam_id,
                    topic_id=topic_id,
                    question=question,
                    answer=answer,
                    difficulty=difficulty,
                    card_type=card_type,
                    status=status,
                    info=info or {},
                ))
    
    def replace_card_proofs(self, *, card_id: str, proofs: Sequence[Dict[str, Any]]) -> None:
        """Replace all proofs for a card."""
        with get_db() as db:
            db.query(CardProof).filter(CardProof.card_id == card_id).delete(
                synchronize_session=False
            )
            for p in proofs:
                db.add(CardProof(
                    proof_id=p.get("proof_id") or uuid.uuid4().hex[:20],
                    card_id=card_id,
                    doc_id=str(p.get("doc_id") or ""),
                    page=int(p.get("page") or 0),
                    start=int(p.get("start") or 0),
                    end=int(p.get("end") or 0),
                    text=str(p.get("text") or ""),
                    score=float(p.get("score") or 0.0),
                ))

    def list_card_proofs(self, *, card_id: str) -> List[StoredCardProof]:
        """List proofs for a card."""
        with get_db() as db:
            rows = (
                db.query(CardProof)
                .filter(CardProof.card_id == card_id)
                .order_by(CardProof.doc_id.asc(), CardProof.page.asc(), CardProof.start.asc())
                .all()
            )
            return [
                StoredCardProof(
                    proof_id=r.proof_id,
                    card_id=r.card_id,
                    doc_id=r.doc_id,
                    page=r.page,
                    start=int(r.start or 0),
                    end=int(r.end or 0),
                    text=r.text or "",
                    score=float(r.score or 0.0),
                )
                for r in rows
            ]
    
    def list_cards_for_exam(self, *, exam_id: str, limit: int = 200) -> List[StoredCard]:
        """List cards for an exam."""
        with get_db() as db:
            cards = db.query(Card).filter(Card.exam_id == exam_id).order_by(
                Card.created_at
            ).limit(limit).all()
            return [
                StoredCard(
                    card_id=c.card_id,
                    exam_id=c.exam_id,
                    topic_id=c.topic_id,
                    question=c.question,
                    answer=c.answer,
                    difficulty=c.difficulty,
                    created_at=_datetime_to_str(c.created_at),
                    status=c.status,
                    info=c.info or {},
                    card_type=c.card_type,
                )
                for c in cards
            ]
    
    def list_cards_for_topic(
        self, *, topic_id: str, status: str = "active", limit: int = 200
    ) -> List[StoredCard]:
        """List cards for a topic."""
        with get_db() as db:
            cards = db.query(Card).filter(
                Card.topic_id == topic_id,
                Card.status == status
            ).order_by(Card.created_at).limit(limit).all()
            return [
                StoredCard(
                    card_id=c.card_id,
                    exam_id=c.exam_id,
                    topic_id=c.topic_id,
                    question=c.question,
                    answer=c.answer,
                    difficulty=c.difficulty,
                    created_at=_datetime_to_str(c.created_at),
                    status=c.status,
                    info=c.info or {},
                    card_type=c.card_type,
                )
                for c in cards
            ]
    
    def get_cards_with_question_vector_idx(
        self, *, exam_id: str
    ) -> Dict[int, StoredCard]:
        """Get cards with question_vector_idx in their info."""
        cards = self.list_cards_for_exam(exam_id=exam_id, limit=10000)
        mapping: Dict[int, StoredCard] = {}
        for card in cards:
            vector_idx = card.info.get("question_vector_idx")
            if vector_idx is not None:
                try:
                    mapping[int(vector_idx)] = card
                except (ValueError, TypeError):
                    continue
        return mapping

    def get_card(self, *, card_id: str) -> Optional[StoredCard]:
        """Get a single card by ID."""
        with get_db() as db:
            c = db.query(Card).filter(Card.card_id == card_id).first()
            if not c:
                return None
            return StoredCard(
                card_id=c.card_id,
                exam_id=c.exam_id,
                topic_id=c.topic_id,
                question=c.question,
                answer=c.answer,
                difficulty=c.difficulty,
                created_at=_datetime_to_str(c.created_at),
                status=c.status,
                info=c.info or {},
                card_type=c.card_type,
            )

    # =========================================================================
    # CARD TOPIC METHODS
    # =========================================================================

    def replace_card_topics(
        self,
        *,
        card_id: str,
        topics: Sequence[Dict[str, Any]],
        session: Optional[Session] = None,
    ) -> None:
        """
        Replace all topic links for a card.

        Enforces one-primary-topic policy at repository layer before write.
        """
        rows: List[Dict[str, Any]] = []
        for t in topics:
            topic_id = str(t.get("topic_id") or "").strip()
            if not topic_id:
                continue
            role = str(t.get("role") or "primary").strip().lower()
            weight = float(t.get("weight") if t.get("weight") is not None else 1.0)
            rows.append(
                {"topic_id": topic_id, "role": role, "weight": weight}
            )
        primary_count = sum(1 for t in rows if t["role"] == "primary")
        if primary_count != 1:
            raise ValueError("card_topics requires exactly one primary topic")

        if session is not None:
            db = session
            db.query(CardTopic).filter(CardTopic.card_id == card_id).delete(
                synchronize_session=False
            )
            for t in rows:
                db.add(
                    CardTopic(
                        card_id=card_id,
                        topic_id=t["topic_id"],
                        role=t["role"],
                        weight=t["weight"],
                    )
                )
            # SessionLocal uses autoflush=False; flush so immediate follow-up reads
            # in the same transaction can see the newly inserted topic links.
            db.flush()
            return

        with get_db() as db:
            db.query(CardTopic).filter(CardTopic.card_id == card_id).delete(
                synchronize_session=False
            )
            for t in rows:
                db.add(
                    CardTopic(
                        card_id=card_id,
                        topic_id=t["topic_id"],
                        role=t["role"],
                        weight=t["weight"],
                    )
                )

    def list_card_topics(
        self,
        *,
        card_id: str,
        session: Optional[Session] = None,
    ) -> List[StoredCardTopic]:
        """List card-topic links for a card (primary first)."""
        if session is not None:
            rows = (
                session.query(CardTopic)
                .filter(CardTopic.card_id == card_id)
                .order_by(CardTopic.role.asc(), CardTopic.created_at.asc())
                .all()
            )
            return [
                StoredCardTopic(
                    card_id=r.card_id,
                    topic_id=r.topic_id,
                    role=r.role,
                    weight=float(r.weight),
                    created_at=_datetime_to_str(r.created_at),
                )
                for r in rows
            ]
        with get_db() as db:
            rows = (
                db.query(CardTopic)
                .filter(CardTopic.card_id == card_id)
                .order_by(CardTopic.role.asc(), CardTopic.created_at.asc())
                .all()
            )
            return [
                StoredCardTopic(
                    card_id=r.card_id,
                    topic_id=r.topic_id,
                    role=r.role,
                    weight=float(r.weight),
                    created_at=_datetime_to_str(r.created_at),
                )
                for r in rows
            ]

    def list_card_ids_for_topic(
        self,
        *,
        topic_id: str,
        role: Optional[str] = None,
    ) -> List[str]:
        """List card IDs linked to a topic, optionally filtered by role."""
        with get_db() as db:
            q = db.query(CardTopic).filter(CardTopic.topic_id == topic_id)
            if role:
                q = q.filter(CardTopic.role == role)
            rows = q.order_by(CardTopic.created_at.asc()).all()
            return [r.card_id for r in rows]

    # =========================================================================
    # CARD SCHEDULING METHODS
    # =========================================================================

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
            return StoredCardScheduling(
                card_id=row.card_id,
                due_at=_datetime_to_str(row.due_at),
                state=row.state,
                interval_days=float(row.interval_days),
                ease=float(row.ease),
                reps=int(row.reps),
                lapses=int(row.lapses),
                last_reviewed_at=_datetime_to_str(row.last_reviewed_at),
            )

        with get_db() as db:
            row = db.query(CardScheduling).filter(CardScheduling.card_id == card_id).first()
            if not row:
                return None
            return StoredCardScheduling(
                card_id=row.card_id,
                due_at=_datetime_to_str(row.due_at),
                state=row.state,
                interval_days=float(row.interval_days),
                ease=float(row.ease),
                reps=int(row.reps),
                lapses=int(row.lapses),
                last_reviewed_at=_datetime_to_str(row.last_reviewed_at),
            )

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
                StoredCardScheduling(
                    card_id=r.card_id,
                    due_at=_datetime_to_str(r.due_at),
                    state=r.state,
                    interval_days=float(r.interval_days),
                    ease=float(r.ease),
                    reps=int(r.reps),
                    lapses=int(r.lapses),
                    last_reviewed_at=_datetime_to_str(r.last_reviewed_at),
                )
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
                StoredCardScheduling(
                    card_id=r.card_id,
                    due_at=_datetime_to_str(r.due_at),
                    state=r.state,
                    interval_days=float(r.interval_days),
                    ease=float(r.ease),
                    reps=int(r.reps),
                    lapses=int(r.lapses),
                    last_reviewed_at=_datetime_to_str(r.last_reviewed_at),
                )
                for r in rows
            ]

    # =========================================================================
    # TOPIC PROFICIENCY METHODS
    # =========================================================================

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
            return StoredTopicProficiency(
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
            return StoredTopicProficiency(
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
                StoredTopicProficiency(
                    user_id=r.user_id,
                    exam_id=r.exam_id,
                    topic_id=r.topic_id,
                    proficiency=float(r.proficiency),
                    current_difficulty=int(r.current_difficulty),
                    streak_up=int(r.streak_up),
                    streak_down=int(r.streak_down),
                    seen_count=int(r.seen_count),
                    correctish_count=int(r.correctish_count),
                    last_updated_at=_datetime_to_str(r.last_updated_at),
                    info=r.info or {},
                )
                for r in rows
            ]

    # =========================================================================
    # STUDENT KNOWLEDGE STATE METHODS
    # =========================================================================

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
            return StoredStudentKnowledgeState(
                user_id=row.user_id,
                exam_id=row.exam_id,
                topic_id=row.topic_id,
                memory=row.memory or {},
                updated_at=_datetime_to_str(row.updated_at),
            )

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
            return StoredStudentKnowledgeState(
                user_id=row.user_id,
                exam_id=row.exam_id,
                topic_id=row.topic_id,
                memory=row.memory or {},
                updated_at=_datetime_to_str(row.updated_at),
            )

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
                StoredStudentKnowledgeState(
                    user_id=r.user_id,
                    exam_id=r.exam_id,
                    topic_id=r.topic_id,
                    memory=r.memory or {},
                    updated_at=_datetime_to_str(r.updated_at),
                )
                for r in rows
            ]

    # =========================================================================
    # SESSION STATE + PRESENTATION LOG METHODS
    # =========================================================================

    def get_exam_session_state(
        self,
        *,
        user_id: str,
        exam_id: str,
        session: Optional[Session] = None,
    ) -> Optional[StoredExamSessionState]:
        """Get user exam session state."""
        if session is not None:
            row = (
                session.query(ExamSessionState)
                .filter(
                    ExamSessionState.user_id == user_id,
                    ExamSessionState.exam_id == exam_id,
                )
                .first()
            )
            if not row:
                return None
            return StoredExamSessionState(
                user_id=row.user_id,
                exam_id=row.exam_id,
                last_served_card_id=row.last_served_card_id,
                last_presented_at=_datetime_to_str(row.last_presented_at),
                updated_at=_datetime_to_str(row.updated_at),
            )

        with get_db() as db:
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
            return StoredExamSessionState(
                user_id=row.user_id,
                exam_id=row.exam_id,
                last_served_card_id=row.last_served_card_id,
                last_presented_at=_datetime_to_str(row.last_presented_at),
                updated_at=_datetime_to_str(row.updated_at),
            )

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
        if session is not None:
            row = (
                session.query(ExamSessionState)
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
                session.add(
                    ExamSessionState(
                        user_id=user_id,
                        exam_id=exam_id,
                        last_served_card_id=last_served_card_id,
                        last_presented_at=last_presented_at,
                    )
                )
            return

        with get_db() as db:
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

        if session is not None:
            max_seq = (
                session.query(func.max(CardPresentationLog.sequence_no))
                .filter(
                    CardPresentationLog.user_id == user_id,
                    CardPresentationLog.exam_id == exam_id,
                )
                .scalar()
            )
            next_seq = int(max_seq or 0) + 1
            session.add(
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

        with get_db() as db:
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
            return StoredCardPresentation(
                user_id=row.user_id,
                exam_id=row.exam_id,
                sequence_no=int(row.sequence_no),
                card_id=row.card_id,
                presented_at=_datetime_to_str(row.presented_at),
                info=row.info or {},
            )

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
            return StoredCardPresentation(
                user_id=row.user_id,
                exam_id=row.exam_id,
                sequence_no=int(row.sequence_no),
                card_id=row.card_id,
                presented_at=_datetime_to_str(row.presented_at),
                info=row.info or {},
            )

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
                StoredCardPresentation(
                    user_id=row.user_id,
                    exam_id=row.exam_id,
                    sequence_no=int(row.sequence_no),
                    card_id=row.card_id,
                    presented_at=_datetime_to_str(row.presented_at),
                    info=row.info or {},
                )
                for row in rows
            ]

    # =========================================================================
    # CARD REVIEW METHODS
    # =========================================================================

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

    # =========================================================================
    # QUESTION INDEX METHODS (SQL audit/source-of-truth)
    # =========================================================================

    def add_question_index_entry(
        self,
        *,
        question_id: str,
        exam_id: str,
        topic_id: str,
        question_text: str,
        difficulty: Optional[int],
        embedding: np.ndarray,
    ) -> None:
        """Insert or update a question_index row for a successfully created card question."""
        arr = embedding.astype("float32", copy=False).reshape(-1)
        with get_db() as db:
            existing = db.query(QuestionIndexEntry).filter(
                QuestionIndexEntry.question_id == question_id
            ).first()
            if existing:
                existing.exam_id = exam_id
                existing.topic_id = topic_id
                existing.question_text = question_text
                existing.difficulty = difficulty
                existing.dim = int(arr.size)
                existing.embedding = arr.tobytes()
            else:
                db.add(
                    QuestionIndexEntry(
                        question_id=question_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        question_text=question_text,
                        difficulty=difficulty,
                        dim=int(arr.size),
                        embedding=arr.tobytes(),
                    )
                )
    
    def has_equivalent_question_text(self, *, exam_id: str, question_text: str) -> bool:
        """Return True when a normalized question text already exists in exam index."""
        normalized = " ".join(str(question_text or "").strip().lower().split())
        if not normalized:
            return False
        with get_db() as db:
            rows = (
                db.query(QuestionIndexEntry.question_text)
                .filter(QuestionIndexEntry.exam_id == exam_id)
                .all()
            )
            for (existing_text,) in rows:
                existing_norm = " ".join(str(existing_text or "").strip().lower().split())
                if existing_norm == normalized:
                    return True
            return False

    def list_question_texts_for_exam(self, *, exam_id: str, limit: int = 200) -> List[str]:
        """List recent indexed question texts for an exam."""
        with get_db() as db:
            rows = (
                db.query(QuestionIndexEntry.question_text)
                .filter(QuestionIndexEntry.exam_id == exam_id)
                .order_by(QuestionIndexEntry.created_at.desc())
                .limit(max(1, int(limit)))
                .all()
            )
            return [str(text or "") for (text,) in rows if str(text or "").strip()]

    def list_math_fingerprints_for_exam(self, *, exam_id: str, limit: int = 200) -> List[str]:
        """List recent structural fingerprints of math calculation cards for an exam.

        Used to enforce structural diversity across batches so the same problem
        *type* is not generated repeatedly with only different numbers.
        """
        with get_db() as db:
            cards = (
                db.query(Card.info)
                .filter(Card.exam_id == exam_id)
                .order_by(Card.created_at.desc())
                .limit(max(1, int(limit)))
                .all()
            )
        out: List[str] = []
        for (info,) in cards:
            if not isinstance(info, dict):
                continue
            fingerprint = str(info.get("math_fingerprint") or "").strip()
            if fingerprint and fingerprint not in out:
                out.append(fingerprint)
        return out

    # =========================================================================
    # EVENT METHODS
    # =========================================================================
    
    def add_event(
        self,
        *,
        user_id: str,
        exam_id: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Add an event to the exam history."""
        self.ensure_user(user_id)
        eid = event_id or uuid.uuid4().hex[:20]
        with get_db() as db:
            db.add(Event(
                event_id=eid,
                user_id=user_id,
                exam_id=exam_id,
                type=type,
                payload=payload or {},
            ))
        self.update_exam(exam_id)
        return eid
    
    # =========================================================================
    # EMBEDDING CACHE METHODS
    # =========================================================================
    
    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        """Get cached embeddings by hash."""
        if not hashes:
            return {}
        with get_db() as db:
            cached = db.query(EmbeddingCache).filter(
                EmbeddingCache.hash.in_(hashes)
            ).all()
            result: Dict[str, np.ndarray] = {}
            for c in cached:
                try:
                    arr = np.frombuffer(c.vector, dtype="float32")
                    if arr.size == c.dim:
                        result[c.hash] = arr
                except Exception:
                    continue
            return result
    
    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        """Cache embeddings."""
        if not mapping:
            return
        with get_db() as db:
            for key, vec in mapping.items():
                arr = vec.astype("float32", copy=False)
                existing = db.query(EmbeddingCache).filter(EmbeddingCache.hash == key).first()
                if existing:
                    existing.dim = arr.size
                    existing.vector = arr.tobytes()
                else:
                    db.add(EmbeddingCache(
                        hash=key,
                        dim=arr.size,
                        vector=arr.tobytes(),
                    ))
