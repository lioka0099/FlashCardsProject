"""Cards, card-proofs, card-topics, question-index."""
import numpy as np
import uuid
from typing import Any, Dict, List, Optional, Sequence
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.data.db_engine import get_db
from app.data.models import (
    Card, CardProof, CardTopic, QuestionIndexEntry,
)
from app.data.db_repository.dtos import (
    StoredCard, StoredCardProof, StoredCardTopic, _datetime_to_str,
)


class CardRepo:
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
                StoredCardProof.from_orm(r)
                for r in rows
            ]
    
    def list_cards_for_exam(self, *, exam_id: str, limit: int = 200) -> List[StoredCard]:
        """List cards for an exam."""
        with get_db() as db:
            cards = db.query(Card).filter(Card.exam_id == exam_id).order_by(
                Card.created_at
            ).limit(limit).all()
            return [
                StoredCard.from_orm(c)
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
                StoredCard.from_orm(c)
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
            return StoredCard.from_orm(c)
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
                StoredCardTopic.from_orm(r)
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
                StoredCardTopic.from_orm(r)
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
