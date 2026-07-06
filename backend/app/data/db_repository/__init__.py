"""Composition root: DBRepository = all aggregate mixins. Re-exports DTOs so
existing `from app.data.db_repository import DBRepository, Stored*` keep working."""
import logging
from pathlib import Path

from app.data.db_engine import init_db, DATABASE_URL
from app.data.db_repository.dtos import (
    MAX_JOB_ATTEMPTS, JobData, StoredChunk, StoredExam, StoredTopic, StoredCard, StoredCardProof, StoredCardTopic, StoredCardScheduling, StoredTopicProficiency, StoredStudentKnowledgeState, StoredExamSessionState, StoredCardPresentation, _datetime_to_str,
)
from app.data.db_repository.corpus import CorpusRepo
from app.data.db_repository.topics import TopicRepo
from app.data.db_repository.exams import ExamRepo
from app.data.db_repository.cards import CardRepo
from app.data.db_repository.reviews import ReviewRepo
from app.data.db_repository.sessions import SessionRepo

logger = logging.getLogger(__name__)


class DBRepository(
    CorpusRepo,
    TopicRepo,
    ExamRepo,
    CardRepo,
    ReviewRepo,
    SessionRepo,
):
    """SQLAlchemy repository composed from aggregate mixins. One stateless object
    over the global engine; the db_path arg is vestigial (kept for compatibility)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        init_db()
        if DATABASE_URL.startswith("sqlite"):
            actual_path = DATABASE_URL.replace("sqlite:///", "")
            if str(db_path) != actual_path:
                logger.debug(
                    "Note: DBRepository initialized with path=%s but actual DB is %s "
                    "(controlled by DATABASE_URL/VECTOR_STORE_PATH env vars)",
                    db_path, actual_path,
                )
        logger.info("SQLAlchemy database initialized")


__all__ = [
    "DBRepository", "MAX_JOB_ATTEMPTS", "JobData", "StoredChunk", "StoredExam", "StoredTopic", "StoredCard", "StoredCardProof", "StoredCardTopic", "StoredCardScheduling", "StoredTopicProficiency", "StoredStudentKnowledgeState", "StoredExamSessionState", "StoredCardPresentation",
]
