"""Users, exams, exam-documents, jobs, events."""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.data.db_engine import get_db, db_scope
from app.data.models import (
    User, Document, Chunk, VectorIndexMap, Exam, ExamDocument,
    Topic, TopicChunk, TopicEvidence, TopicVector,
    Card, CardProof, CardReview, CardTopic, CardScheduling, TopicProficiency,
    ExamSessionState, CardPresentationLog, Event, QuestionIndexEntry, StudentKnowledgeState,
)
from app.data.db_repository.dtos import (
    JobData, StoredExam, MAX_JOB_ATTEMPTS,
)


class ExamRepo:
    def ensure_user(self, user_id: str, *, name: Optional[str] = None) -> None:
        """Create user if not exists, optionally update name."""
        with get_db() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                if name is not None:
                    user.name = name
            else:
                db.add(User(user_id=user_id, name=name))
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
        with db_scope(session) as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if not exam:
                return None
            return StoredExam.from_orm(exam)
    
    def list_exams(self, *, user_id: str, limit: int = 50) -> List[StoredExam]:
        """List exams for a user."""
        with get_db() as db:
            exams = db.query(Exam).filter(Exam.user_id == user_id).order_by(
                Exam.updated_at.desc()
            ).limit(limit).all()
            return [
                StoredExam.from_orm(e)
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
        with db_scope(session) as db:
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
                # ponytail: read-modify-write on the whole `info` JSON blob is
                # last-write-wins. Safe under the single embedded worker (writes are
                # sequential; the worker's reporter only bumps the heartbeat column,
                # not info). Multi-worker cloud needs a locked RMW or a JSON-field
                # update, else concurrent writers clobber sibling keys (progress vs
                # math_profile vs bootstrap_error).
                merged = dict(row.info or {})
                merged.update(info_patch)
                row.info = merged
            row.updated_at = datetime.now(timezone.utc)

    def enqueue_job(self, *, exam_id: str, payload: dict) -> None:
        with get_db() as db:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is None:
                raise ValueError(f"Exam not found: {exam_id}")
            row.job_status = "queued"
            row.job_payload = payload
            row.job_attempts = 0
            row.job_heartbeat_at = None

    def claim_next_job(self, *, stale_seconds: int = 300) -> Optional[JobData]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=stale_seconds)
        with get_db() as db:
            # ponytail: plain SELECT is correct for the single embedded worker.
            # For multi-worker cloud (Postgres), add .with_for_update(skip_locked=True).
            # ponytail: stale-running reclaim only works if ANOTHER worker polls.
            # In single-embedded-worker mode a thread death without process restart
            # leaves the exam stuck at state="processing" forever (the dead worker is
            # the only caller of this). Process restart self-heals after stale_seconds.
            # Pre-cloud follow-up: a max-age sweep flipping ancient processing -> failed,
            # plus a client-side "taking longer than expected" affordance.
            row = (
                db.query(Exam)
                .filter(
                    or_(
                        Exam.job_status == "queued",
                        and_(Exam.job_status == "running", Exam.job_heartbeat_at < cutoff),
                    )
                )
                .order_by(Exam.created_at.asc())
                .first()
            )
            if row is None:
                return None
            row.job_status = "running"
            row.job_attempts = int(row.job_attempts or 0) + 1
            row.job_heartbeat_at = now
            return JobData(exam_id=row.exam_id, user_id=row.user_id,
                           payload=dict(row.job_payload or {}), attempts=row.job_attempts)

    def heartbeat_job(self, exam_id: str) -> None:
        with get_db() as db:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is not None:
                row.job_heartbeat_at = datetime.now(timezone.utc)

    def finish_job(self, *, exam_id: str, ok: bool, error: Optional[str] = None) -> str:
        with get_db() as db:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is None:
                return "missing"
            if ok:
                row.job_status = "done"
            else:
                row.job_status = "failed" if int(row.job_attempts or 0) >= MAX_JOB_ATTEMPTS else "queued"
            return row.job_status

    def increment_exam_diagnostic_answered(
        self,
        *,
        exam_id: str,
        by: int = 1,
        session: Optional[Session] = None,
    ) -> int:
        """Increment exam diagnostic_answered and return the new value."""
        with db_scope(session) as db:
            row = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if row is None:
                raise ValueError(f"Exam not found: {exam_id}")
            row.diagnostic_answered = max(0, int(row.diagnostic_answered or 0) + int(by))
            row.updated_at = datetime.now(timezone.utc)
            return int(row.diagnostic_answered)
    
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
