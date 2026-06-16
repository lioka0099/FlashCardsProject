from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Dict, List, Optional, Sequence

from app.data.db_engine import get_db
from app.data.pinecone_backend import PineconeClient, pinecone_namespace
from app.data.db_repository import DBRepository, StoredExam
from app.data.vector_store import VectorStore
from app.services.cards import GeneratedCard
from app.services.document_math_profile import classify_document_math_profile
from app.services.exams import create_exam
from app.services.graph import generate_starter_cards_v2
from app.services.ingestion import UnsupportedDocumentTypeError, ingest_documents
from app.services.topics import build_topics_for_exam


@dataclass
class DiagnosticBootstrapResult:
    exam_id: str
    state: str
    diagnostic_total: int
    diagnostic_answered: int
    cards_generated: int
    topic_count: int


class DiagnosticBootstrapError(RuntimeError):
    pass


class DiagnosticLifecycleService:
    """
    Thin coordinator for diagnostic bootstrap.

    Graph remains the single source of truth for card generation.
    """

    def __init__(
        self,
        *,
        store: Optional[VectorStore] = None,
        repo: Optional[DBRepository] = None,
    ) -> None:
        self.store = store or VectorStore()
        self.repo = repo or self.store.db

    def _cleanup_failed_bootstrap(self, *, user_id: str, exam_id: str) -> None:
        # Best-effort external vector cleanup first.
        if self.store.vector_backend == "pinecone":
            try:
                ns = pinecone_namespace(user_id=user_id, exam_id=exam_id)
                pc = PineconeClient()
                pc.chunks.delete(delete_all=True, namespace=ns)
                pc.questions.delete(delete_all=True, namespace=ns)
            except Exception:
                # Do not mask original bootstrap failure.
                pass

        # SQL cleanup for partially created exam data.
        self.repo.delete_exam_cascade(exam_id=exam_id)

    def bootstrap_exam_from_upload(
        self,
        *,
        user_id: str,
        title: str,
        paths: Sequence[str],
        mode: str = "mastery",
        info: Optional[Dict[str, object]] = None,
    ) -> DiagnosticBootstrapResult:
        exam_id = create_exam(
            store=self.store,
            user_id=user_id,
            title=title,
            mode=mode,
            info=(info or {}),
        )
        self.repo.add_event(
            user_id=user_id,
            exam_id=exam_id,
            type="bootstrap_started",
            payload={"paths_count": len(paths)},
        )

        try:
            ingest_results = ingest_documents(
                paths,
                store=self.store,
                user_id=user_id,
                exam_id=exam_id,
            )
            doc_ids = [x.doc_id for x in ingest_results]
            self.repo.attach_documents_to_exam(exam_id=exam_id, doc_ids=doc_ids)
            chunks = []
            for doc_id in doc_ids:
                chunks.extend(self.repo.list_chunks_by_doc(doc_id))
            math_profile = classify_document_math_profile(chunks=chunks)
            self.repo.update_exam_lifecycle(
                exam_id=exam_id,
                info_patch={"math_profile": math_profile.to_info()},
            )
        except UnsupportedDocumentTypeError:
            self._cleanup_failed_bootstrap(user_id=user_id, exam_id=exam_id)
            raise
        except Exception as exc:
            self._cleanup_failed_bootstrap(user_id=user_id, exam_id=exam_id)
            raise

        topics = build_topics_for_exam(exam_id=exam_id, store=self.store, overwrite=True)
        topic_count = len(topics)
        self.repo.add_event(
            user_id=user_id,
            exam_id=exam_id,
            type="topics_built",
            payload={"topics_created": topic_count},
        )
        if topic_count <= 0:
            self._cleanup_failed_bootstrap(user_id=user_id, exam_id=exam_id)
            raise DiagnosticBootstrapError("No topics built for diagnostic bootstrap.")

        try:
            cards = generate_starter_cards_v2(
                exam_id=exam_id,
                user_id=user_id,
                n=topic_count,
                difficulty=1,
                card_type="diagnostic",
                store=self.store,
            )
        except Exception as exc:
            raise
        cards_generated = len(cards)
        self.repo.add_event(
            user_id=user_id,
            exam_id=exam_id,
            type="diagnostic_cards_generated",
            payload={"cards_generated": cards_generated, "topics_targeted": topic_count},
        )
        minimum_required_cards = max(1, math.ceil(topic_count * 0.8))
        if cards_generated < minimum_required_cards:
            self._cleanup_failed_bootstrap(user_id=user_id, exam_id=exam_id)
            raise DiagnosticBootstrapError(
                f"Only {cards_generated} diagnostic cards were created for {topic_count} topics; "
                f"at least {minimum_required_cards} are required."
            )

        now = datetime.now(timezone.utc)
        with get_db() as session:
            self.repo.update_exam_lifecycle(
                exam_id=exam_id,
                state="diagnostic",
                diagnostic_total=topic_count,
                diagnostic_answered=0,
                diagnostic_started_at=now,
                diagnostic_completed_at=None,
                info_patch={"bootstrap_error": None},
                session=session,
            )

        self.repo.add_event(
            user_id=user_id,
            exam_id=exam_id,
            type="diagnostic_started",
            payload={"diagnostic_total": topic_count},
        )

        exam = self.repo.get_exam(exam_id)
        if exam is None:
            raise RuntimeError(f"Exam not found after bootstrap: {exam_id}")
        return DiagnosticBootstrapResult(
            exam_id=exam.exam_id,
            state=exam.state,
            diagnostic_total=exam.diagnostic_total,
            diagnostic_answered=exam.diagnostic_answered,
            cards_generated=cards_generated,
            topic_count=topic_count,
        )


