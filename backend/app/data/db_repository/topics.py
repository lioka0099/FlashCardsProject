"""Topics, topic-chunks, topic-evidence, topic-vectors."""
import numpy as np
import uuid
from typing import Any, Dict, List, Optional, Sequence

from app.data.db_engine import get_db
from app.data.models import (
    Topic, TopicChunk, TopicEvidence, TopicVector,
)
from app.data.db_repository.dtos import (
    StoredTopic,
)


class TopicRepo:
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
                StoredTopic.from_orm(t)
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
