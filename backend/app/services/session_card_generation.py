from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.data.db_repository import DBRepository, StoredCard, StoredCardPresentation, StoredTopic
from app.data.pinecone_backend import pinecone_namespace
from app.data.vector_store import VectorStore
from app.services.context_packs import build_diverse_chunk_pack
from app.services.graph import generate_single_card
from app.services.card_routing import classify_card_route
from app.services.diagnostic_coverage import all_topics_diagnosed, diagnosed_topic_ids, undiagnosed_topics
from app.services.difficulty_frameworks import clamp_difficulty, framework_for_route
from app.services.topic_route_proficiency import default_route_state, route_state_from_info

logger = logging.getLogger(__name__)

PREFETCH_STATUS_READY = "ready"
PREFETCH_STATUS_SERVED = "served"
PREFETCH_STATUS_STALE = "stale"
PREFETCH_STATUS_FAILED = "failed"

_REFILL_LOCK = threading.Lock()
_ACTIVE_REFILLS: Set[Tuple[str, str]] = set()


@dataclass
class TopicGenerationChoice:
    topic_id: str
    topic_label: str
    difficulty: int
    cached_route: Optional[Dict[str, object]] = None


@dataclass
class GeneratedSessionCard:
    card_id: str
    reason: str
    topic_id: str


class SessionCardGenerationService:
    """
    Generates the next learning card when the planner has no scheduled card.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        max_consecutive_topic_cards: int = 2,
        max_ready_prefetch_per_topic: int = 1,
        max_ready_prefetch_per_exam: int = 3,
    ) -> None:
        self.repo = repo or DBRepository(Path("store/meta.sqlite"))
        self.max_consecutive_topic_cards = max(1, int(max_consecutive_topic_cards))
        self.max_ready_prefetch_per_topic = max(1, int(max_ready_prefetch_per_topic))
        self.max_ready_prefetch_per_exam = max(1, int(max_ready_prefetch_per_exam))

    def select_next_topic(self, *, user_id: str, exam_id: str) -> Optional[TopicGenerationChoice]:
        topics = self.repo.list_topics(exam_id=exam_id)
        if not topics:
            return None

        recent = self.repo.list_presentations(
            user_id=user_id,
            exam_id=exam_id,
            ascending=False,
            limit=100,
        )
        blocked_topic = self._blocked_topic_from_recent(recent)
        presentation_counts = self._topic_counts(recent)
        generated_counts = self._learning_card_counts(exam_id=exam_id)
        proficiencies = {
            p.topic_id: p
            for p in self.repo.list_topic_proficiencies(user_id=user_id, exam_id=exam_id)
        }

        ordered = sorted(
            topics,
            key=lambda t: (
                presentation_counts.get(t.topic_id, 0),
                generated_counts.get(t.topic_id, 0),
                t.created_at,
                t.topic_id,
            ),
        )
        chosen = self._first_unblocked_topic(ordered, blocked_topic)
        if chosen is None:
            return None

        prof = proficiencies.get(chosen.topic_id)
        difficulty = int(prof.current_difficulty) if prof is not None else 1
        difficulty = min(5, max(1, difficulty))
        return TopicGenerationChoice(
            topic_id=chosen.topic_id,
            topic_label=chosen.label,
            difficulty=difficulty,
            cached_route=(chosen.info or {}).get("route_candidate") if isinstance(chosen.info, dict) else None,
        )

    def generate_next_card(
        self,
        *,
        user_id: str,
        exam_id: str,
        store: Optional[VectorStore] = None,
        prefetch: bool = False,
    ) -> Optional[GeneratedSessionCard]:
        store = store or VectorStore()
        exam = self.repo.get_exam(exam_id)
        if exam is None:
            return None
        raw_profile = (exam.info or {}).get("math_profile") if isinstance(exam.info, dict) else None
        document_math_profile = raw_profile if isinstance(raw_profile, dict) else {"kind": "non_math"}
        if store.vector_backend == "pinecone":
            store.set_namespace(pinecone_namespace(user_id=user_id, exam_id=exam_id))

        diagnosed = diagnosed_topic_ids(repo=self.repo, user_id=user_id, exam_id=exam_id)
        needs_diagnostic = not all_topics_diagnosed(repo=self.repo, user_id=user_id, exam_id=exam_id)
        card_type = "diagnostic" if needs_diagnostic else "learning"
        allowed_topic_ids = (
            {t.topic_id for t in undiagnosed_topics(repo=self.repo, user_id=user_id, exam_id=exam_id)}
            if needs_diagnostic
            else diagnosed
        )
        if not allowed_topic_ids:
            return None

        choices = self._ordered_topic_choices(
            user_id=user_id,
            exam_id=exam_id,
            allowed_topic_ids=allowed_topic_ids,
        )
        for choice in choices:
            if prefetch and not self._can_prefetch_topic(user_id=user_id, exam_id=exam_id, topic_id=choice.topic_id):
                continue
            chunk_ids = self.repo.list_chunk_ids_for_topic(topic_id=choice.topic_id)
            if not chunk_ids:
                continue
            try:
                context_pack = build_diverse_chunk_pack(
                    store=store,
                    chunk_ids=chunk_ids,
                    centroid=self.repo.get_topic_vector(topic_id=choice.topic_id),
                )
                if not context_pack:
                    continue
                route_decision = classify_card_route(
                    topic_label=choice.topic_label,
                    context_pack=context_pack,
                    cached_topic_route=choice.cached_route if isinstance(choice.cached_route, dict) else None,
                    document_math_profile=document_math_profile,
                    store=store,
                )
                if card_type == "diagnostic":
                    difficulty = 1
                else:
                    difficulty = self._difficulty_for_route(
                        user_id=user_id,
                        exam_id=exam_id,
                        topic_id=choice.topic_id,
                        card_route=route_decision.card_route,
                    )
                card = generate_single_card(
                    exam_id=exam_id,
                    topic_id=choice.topic_id,
                    topic_label=choice.topic_label,
                    allowed_chunk_ids=chunk_ids,
                    context_pack=context_pack,
                    difficulty=difficulty,
                    card_type=card_type,
                    user_id=user_id,
                    store=store,
                )
            except Exception:
                logger.exception(
                    "Failed to generate session card for exam %s topic %s",
                    exam_id,
                    choice.topic_id,
                )
                continue
            if card is None:
                continue
            self.repo.replace_card_topics(
                card_id=card.card_id,
                topics=[{"topic_id": choice.topic_id, "role": "primary", "weight": 1.0}],
            )
            if prefetch:
                self._mark_prefetched_card_ready(
                    card_id=card.card_id,
                    user_id=user_id,
                    choice=choice,
                )
            if card_type == "diagnostic":
                serve_reason = "diagnostic" if not prefetch else "prefetched"
            else:
                serve_reason = "prefetched" if prefetch else "generated"
            return GeneratedSessionCard(
                card_id=card.card_id,
                reason=serve_reason,
                topic_id=choice.topic_id,
            )
        return None

    def archive_invalid_prefetched_cards(self, *, user_id: str, exam_id: str) -> None:
        diagnosed = diagnosed_topic_ids(repo=self.repo, user_id=user_id, exam_id=exam_id)
        needs_diagnostic = not all_topics_diagnosed(repo=self.repo, user_id=user_id, exam_id=exam_id)
        for card in self.repo.list_cards_for_exam(exam_id=exam_id, limit=5000):
            if not self._is_ready_prefetch(card=card, user_id=user_id):
                continue
            topic_id = self._card_topic_id(card)
            if needs_diagnostic:
                if card.card_type != "diagnostic" or topic_id in diagnosed:
                    self._archive_prefetched_card(card=card, status=PREFETCH_STATUS_STALE)
            elif card.card_type == "diagnostic" or topic_id not in diagnosed:
                self._archive_prefetched_card(card=card, status=PREFETCH_STATUS_STALE)

    def get_fresh_prefetched_card(self, *, user_id: str, exam_id: str) -> Optional[GeneratedSessionCard]:
        self.archive_invalid_prefetched_cards(user_id=user_id, exam_id=exam_id)
        cards = self.repo.list_cards_for_exam(exam_id=exam_id, limit=5000)
        candidates: List[StoredCard] = []
        for card in cards:
            if not self._is_ready_prefetch(card=card, user_id=user_id):
                continue
            if self._has_card_been_presented(user_id=user_id, exam_id=exam_id, card_id=card.card_id):
                self.mark_prefetched_card_served(card_id=card.card_id, user_id=user_id)
                continue
            if self._is_prefetch_fresh(card=card, user_id=user_id, exam_id=exam_id):
                candidates.append(card)
            else:
                self._archive_prefetched_card(card=card, status=PREFETCH_STATUS_STALE)

        chosen = self._choose_card_with_topic_fairness(
            candidates=candidates,
            user_id=user_id,
            exam_id=exam_id,
        )
        if chosen is None:
            return None
        return GeneratedSessionCard(
            card_id=chosen.card_id,
            reason="prefetched",
            topic_id=self._card_topic_id(chosen),
        )

    def mark_prefetched_card_served(self, *, card_id: str, user_id: str) -> None:
        card = self.repo.get_card(card_id=card_id)
        if card is None or not self._is_ready_prefetch(card=card, user_id=user_id):
            return
        info = dict(card.info or {})
        info["prefetch_status"] = PREFETCH_STATUS_SERVED
        info["prefetch_served_at"] = datetime.now(timezone.utc).isoformat()
        self.repo.upsert_card(
            card_id=card.card_id,
            exam_id=card.exam_id,
            topic_id=card.topic_id,
            question=card.question,
            answer=card.answer,
            difficulty=card.difficulty,
            card_type=card.card_type,
            status=card.status,
            info=info,
        )

    def refill_prefetch_if_needed(self, *, user_id: str, exam_id: str) -> None:
        key = (user_id, exam_id)
        with _REFILL_LOCK:
            if key in _ACTIVE_REFILLS:
                return
            _ACTIVE_REFILLS.add(key)

        try:
            ready_before = self._ready_prefetch_count(user_id=user_id, exam_id=exam_id)
            if ready_before >= self.max_ready_prefetch_per_exam:
                return
            generated = self.generate_next_card(user_id=user_id, exam_id=exam_id, prefetch=True)
            if generated is None:
                logger.info("No prefetch card generated for exam %s", exam_id)
        finally:
            with _REFILL_LOCK:
                _ACTIVE_REFILLS.discard(key)

    def _ordered_topic_choices(
        self,
        *,
        user_id: str,
        exam_id: str,
        allowed_topic_ids: Optional[Set[str]] = None,
    ) -> List[TopicGenerationChoice]:
        topics = self.repo.list_topics(exam_id=exam_id)
        if not topics:
            return []
        if allowed_topic_ids is not None:
            topics = [t for t in topics if t.topic_id in allowed_topic_ids]
            if not topics:
                return []

        recent = self.repo.list_presentations(
            user_id=user_id,
            exam_id=exam_id,
            ascending=False,
            limit=100,
        )
        blocked_topic = self._blocked_topic_from_recent(recent)
        presentation_counts = self._topic_counts(recent)
        generated_counts = self._learning_card_counts(exam_id=exam_id)
        proficiencies = {
            p.topic_id: p
            for p in self.repo.list_topic_proficiencies(user_id=user_id, exam_id=exam_id)
        }

        ordered_topics = sorted(
            topics,
            key=lambda t: (
                presentation_counts.get(t.topic_id, 0),
                generated_counts.get(t.topic_id, 0),
                t.created_at,
                t.topic_id,
            ),
        )
        if blocked_topic is not None:
            unblocked = [t for t in ordered_topics if t.topic_id != blocked_topic]
            blocked = [t for t in ordered_topics if t.topic_id == blocked_topic]
            ordered_topics = unblocked + blocked

        choices: List[TopicGenerationChoice] = []
        for topic in ordered_topics:
            prof = proficiencies.get(topic.topic_id)
            difficulty = int(prof.current_difficulty) if prof is not None else 1
            choices.append(
                TopicGenerationChoice(
                    topic_id=topic.topic_id,
                    topic_label=topic.label,
                    difficulty=min(5, max(1, difficulty)),
                    cached_route=(topic.info or {}).get("route_candidate") if isinstance(topic.info, dict) else None,
                )
            )
        return choices

    def _can_prefetch_topic(self, *, user_id: str, exam_id: str, topic_id: str) -> bool:
        if self._ready_prefetch_count(user_id=user_id, exam_id=exam_id) >= self.max_ready_prefetch_per_exam:
            return False
        ready_for_topic = 0
        for card in self.repo.list_cards_for_exam(exam_id=exam_id, limit=5000):
            if self._is_ready_prefetch(card=card, user_id=user_id) and self._card_topic_id(card) == topic_id:
                ready_for_topic += 1
        return ready_for_topic < self.max_ready_prefetch_per_topic

    def _ready_prefetch_count(self, *, user_id: str, exam_id: str) -> int:
        return sum(
            1
            for card in self.repo.list_cards_for_exam(exam_id=exam_id, limit=5000)
            if self._is_ready_prefetch(card=card, user_id=user_id)
        )

    def _mark_prefetched_card_ready(
        self,
        *,
        card_id: str,
        user_id: str,
        choice: TopicGenerationChoice,
    ) -> None:
        card = self.repo.get_card(card_id=card_id)
        if card is None:
            return
        if self._has_card_been_presented(user_id=user_id, exam_id=card.exam_id, card_id=card.card_id):
            info = dict(card.info or {})
            info["prefetch_status"] = PREFETCH_STATUS_SERVED
            info["prefetch_served_at"] = datetime.now(timezone.utc).isoformat()
            info["prefetch_skipped_reason"] = "already_presented"
            self.repo.upsert_card(
                card_id=card.card_id,
                exam_id=card.exam_id,
                topic_id=card.topic_id,
                question=card.question,
                answer=card.answer,
                difficulty=card.difficulty,
                card_type=card.card_type,
                status=card.status,
                info=info,
            )
            return
        prof = self.repo.get_topic_proficiency(
            user_id=user_id,
            exam_id=card.exam_id,
            topic_id=choice.topic_id,
        )
        info = dict(card.info or {})
        info.update(
            {
                "prefetch_status": PREFETCH_STATUS_READY,
                "generated_for_user_id": user_id,
                "generated_difficulty": card.difficulty,
                "generated_card_route": info.get("card_route", "default"),
                "generated_difficulty_framework": info.get("difficulty_framework", "bloom"),
                "generated_topic_proficiency": float(prof.proficiency) if prof else None,
                "prefetch_created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.repo.upsert_card(
            card_id=card.card_id,
            exam_id=card.exam_id,
            topic_id=card.topic_id,
            question=card.question,
            answer=card.answer,
            difficulty=card.difficulty,
            card_type=card.card_type,
            status=card.status,
            info=info,
        )

    def _archive_prefetched_card(self, *, card: StoredCard, status: str) -> None:
        info = dict(card.info or {})
        info["prefetch_status"] = status
        info["prefetch_archived_at"] = datetime.now(timezone.utc).isoformat()
        self.repo.upsert_card(
            card_id=card.card_id,
            exam_id=card.exam_id,
            topic_id=card.topic_id,
            question=card.question,
            answer=card.answer,
            difficulty=card.difficulty,
            card_type=card.card_type,
            status="archived",
            info=info,
        )

    def _is_ready_prefetch(self, *, card: StoredCard, user_id: str) -> bool:
        info = card.info or {}
        return (
            card.status == "active"
            and card.card_type != "diagnostic"
            and info.get("prefetch_status") == PREFETCH_STATUS_READY
            and info.get("generated_for_user_id") == user_id
        )

    def _is_prefetch_fresh(self, *, card: StoredCard, user_id: str, exam_id: str) -> bool:
        generated_difficulty = int(card.info.get("generated_difficulty") or card.difficulty)
        topic_id = self._card_topic_id(card)
        card_route = str((card.info or {}).get("card_route") or "default")
        current_difficulty = self._difficulty_for_route(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
            card_route=card_route,
        )
        return abs(generated_difficulty - current_difficulty) <= 1

    def _difficulty_for_route(
        self,
        *,
        user_id: str,
        exam_id: str,
        topic_id: str,
        card_route: str,
    ) -> int:
        framework = framework_for_route(card_route)
        prof = self.repo.get_topic_proficiency(
            user_id=user_id,
            exam_id=exam_id,
            topic_id=topic_id,
        )
        if prof is None:
            return 1
        if card_route == "math_calculation":
            fallback = default_route_state(card_route="math_calculation", current_difficulty=1)
        elif card_route == "math_conceptual":
            fallback = default_route_state(
                card_route="math_conceptual",
                current_difficulty=prof.current_difficulty,
                seen_count=prof.seen_count,
                correctish_count=prof.correctish_count,
            )
        else:
            fallback = default_route_state(
                card_route="default",
                proficiency=prof.proficiency,
                current_difficulty=prof.current_difficulty,
                streak_up=prof.streak_up,
                streak_down=prof.streak_down,
                seen_count=prof.seen_count,
                correctish_count=prof.correctish_count,
            )
        route_state = route_state_from_info(
            info=prof.info or {},
            card_route=card_route,
            fallback=fallback,
        )
        return clamp_difficulty(framework, route_state.current_difficulty)

    def _choose_card_with_topic_fairness(
        self,
        *,
        candidates: Sequence[StoredCard],
        user_id: str,
        exam_id: str,
    ) -> Optional[StoredCard]:
        if not candidates:
            return None

        recent = self.repo.list_presentations(
            user_id=user_id,
            exam_id=exam_id,
            ascending=False,
            limit=100,
        )
        blocked_topic = self._blocked_topic_from_recent(recent)
        topic_counts = self._topic_counts(recent)

        grouped: Dict[str, List[StoredCard]] = {}
        for card in candidates:
            grouped.setdefault(self._card_topic_id(card), []).append(card)

        for topic_id in sorted(grouped.keys(), key=lambda t: (topic_counts.get(t, 0), t)):
            if blocked_topic is not None and topic_id == blocked_topic:
                continue
            return sorted(grouped[topic_id], key=lambda c: (c.created_at, c.card_id))[0]

        return sorted(candidates, key=lambda c: (c.created_at, c.card_id))[0]

    def _first_unblocked_topic(
        self,
        topics: Sequence[StoredTopic],
        blocked_topic: Optional[str],
    ) -> Optional[StoredTopic]:
        for topic in topics:
            if blocked_topic is not None and topic.topic_id == blocked_topic:
                continue
            return topic
        return topics[0] if topics else None

    def _learning_card_counts(self, *, exam_id: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for card in self.repo.list_cards_for_exam(exam_id=exam_id, limit=5000):
            if card.status != "active" or card.card_type == "diagnostic":
                continue
            topic_id = self._card_topic_id(card)
            counts[topic_id] = counts.get(topic_id, 0) + 1
        return counts

    def _blocked_topic_from_recent(
        self, recent: Sequence[StoredCardPresentation]
    ) -> Optional[str]:
        run_topic: Optional[str] = None
        run_len = 0
        for p in recent[: self.max_consecutive_topic_cards]:
            card = self.repo.get_card(card_id=p.card_id)
            if card is None:
                break
            topic_id = self._card_topic_id(card)
            if run_topic is None:
                run_topic = topic_id
                run_len = 1
                continue
            if topic_id == run_topic:
                run_len += 1
            else:
                break
        if run_topic is not None and run_len >= self.max_consecutive_topic_cards:
            return run_topic
        return None

    def _topic_counts(self, recent: Iterable[StoredCardPresentation]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in recent:
            card = self.repo.get_card(card_id=p.card_id)
            if card is None:
                continue
            topic_id = self._card_topic_id(card)
            counts[topic_id] = counts.get(topic_id, 0) + 1
        return counts

    def _card_topic_id(self, card: StoredCard) -> str:
        links = self.repo.list_card_topics(card_id=card.card_id)
        primaries = [x for x in links if x.role == "primary"]
        if len(primaries) == 1:
            return primaries[0].topic_id
        return card.topic_id

    def _has_card_been_presented(self, *, user_id: str, exam_id: str, card_id: str) -> bool:
        return any(
            presentation.card_id == card_id
            for presentation in self.repo.list_presentations(
                user_id=user_id,
                exam_id=exam_id,
                ascending=False,
                limit=5000,
            )
        )
