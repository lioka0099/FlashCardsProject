"""Session planner: decide which cards (due reviews vs new) to serve next in a study session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set

from app.data.db_repository import DBRepository, StoredCard, StoredCardPresentation
from app.deps import get_repo
from app.services.diagnostic.coverage import all_topics_diagnosed, diagnosed_topic_ids


@dataclass
class PlannedCard:
    card_id: str
    reason: str
    topic_id: str


class SessionPlannerService:
    """
    Deterministic session planner with queue priority and topic fairness.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        max_consecutive_topic_cards: int = 2,
    ) -> None:
        self.repo = repo or get_repo()
        self.max_consecutive_topic_cards = max(1, int(max_consecutive_topic_cards))

    def plan_next_card(
        self,
        *,
        user_id: str,
        exam_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[PlannedCard]:
        now_dt = now or datetime.now(timezone.utc)
        exam = self.repo.get_exam(exam_id=exam_id)
        if exam is not None and exam.state == "diagnostic":
            diagnostic = self._build_diagnostic_queue(user_id=user_id, exam_id=exam_id)
            chosen = self._choose_with_topic_fairness(
                candidates=diagnostic,
                user_id=user_id,
                exam_id=exam_id,
            )
            if chosen is not None:
                return PlannedCard(
                    card_id=chosen.card_id,
                    reason="diagnostic",
                    topic_id=self._card_topic_id(chosen),
                )
            if all_topics_diagnosed(repo=self.repo, user_id=user_id, exam_id=exam_id):
                self.repo.update_exam_lifecycle(
                    exam_id=exam_id,
                    state="active_learning",
                    diagnostic_completed_at=now_dt,
                )
            else:
                return None

        overdue = self._build_overdue_queue(user_id=user_id, exam_id=exam_id, now=now_dt)
        chosen = self._choose_with_topic_fairness(
            candidates=overdue,
            user_id=user_id,
            exam_id=exam_id,
        )
        if chosen is not None:
            return PlannedCard(card_id=chosen.card_id, reason="overdue", topic_id=self._card_topic_id(chosen))

        remediation = self._build_remediation_queue(user_id=user_id, exam_id=exam_id, now=now_dt)
        chosen = self._choose_with_topic_fairness(
            candidates=remediation,
            user_id=user_id,
            exam_id=exam_id,
        )
        if chosen is not None:
            return PlannedCard(card_id=chosen.card_id, reason="remediation", topic_id=self._card_topic_id(chosen))

        progression = self._build_progression_queue(user_id=user_id, exam_id=exam_id, now=now_dt)
        chosen = self._choose_with_topic_fairness(
            candidates=progression,
            user_id=user_id,
            exam_id=exam_id,
        )
        if chosen is not None:
            return PlannedCard(card_id=chosen.card_id, reason="progression", topic_id=self._card_topic_id(chosen))

        return None

    def _build_overdue_queue(
        self, *, user_id: str, exam_id: str, now: datetime
    ) -> List[StoredCard]:
        diagnosed = self._diagnosed_topic_ids(user_id=user_id, exam_id=exam_id)
        due = self.repo.list_due_cards(
            user_id=user_id,
            exam_id=exam_id,
            at_or_before=now,
            limit=300,
        )
        cards: List[StoredCard] = []
        for s in due:
            if s.state not in {"new", "learning", "review"}:
                continue
            card = self.repo.get_card(card_id=s.card_id)
            if card is None or card.status != "active":
                continue
            if card.card_type == "diagnostic":
                continue
            if self._card_topic_id(card) not in diagnosed:
                continue
            cards.append(card)
        return cards

    def _build_remediation_queue(self, *, user_id: str, exam_id: str, now: datetime) -> List[StoredCard]:
        diagnosed = self._diagnosed_topic_ids(user_id=user_id, exam_id=exam_id)
        rows = self.repo.list_card_scheduling_by_state(
            exam_id=exam_id,
            state="relearning",
            limit=300,
        )
        cards: List[StoredCard] = []
        for s in rows:
            if not self._is_due(s.due_at, now):
                continue
            card = self.repo.get_card(card_id=s.card_id)
            if card is None or card.status != "active":
                continue
            if card.card_type == "diagnostic":
                continue
            if self._card_topic_id(card) not in diagnosed:
                continue
            cards.append(card)
        return cards

    def _build_progression_queue(self, *, user_id: str, exam_id: str, now: datetime) -> List[StoredCard]:
        diagnosed = self._diagnosed_topic_ids(user_id=user_id, exam_id=exam_id)
        presented_card_ids = self._presented_card_ids(user_id=user_id, exam_id=exam_id)
        cards = self.repo.list_cards_for_exam(exam_id=exam_id, limit=1000)
        out: List[StoredCard] = []
        for card in cards:
            if card.status != "active":
                continue
            if card.card_type == "diagnostic":
                continue
            if self._card_topic_id(card) not in diagnosed:
                continue
            if (card.info or {}).get("prefetch_status") == "ready":
                continue
            sched = self.repo.get_card_scheduling(card_id=card.card_id)
            if sched is None:
                if card.card_id in presented_card_ids:
                    continue
                out.append(card)
                continue
            if sched.state in {"new", "learning", "review"} and self._is_due(sched.due_at, now):
                out.append(card)

        # Deterministic tie-breaks: difficulty asc, created_at asc, card_id asc.
        out.sort(key=lambda c: (int(c.difficulty), c.created_at, c.card_id))
        return out

    def _presented_card_ids(self, *, user_id: str, exam_id: str) -> Set[str]:
        return {
            presentation.card_id
            for presentation in self.repo.list_presentations(
                user_id=user_id,
                exam_id=exam_id,
                ascending=False,
                limit=5000,
            )
        }

    def _build_diagnostic_queue(self, *, user_id: str, exam_id: str) -> List[StoredCard]:
        cards = self.repo.list_cards_for_exam(exam_id=exam_id, limit=1000)
        proficiencies = self.repo.list_topic_proficiencies(user_id=user_id, exam_id=exam_id)
        answered_topic_ids = {p.topic_id for p in proficiencies if int(p.seen_count) > 0}
        out: List[StoredCard] = []
        for card in cards:
            if card.status != "active":
                continue
            if card.card_type != "diagnostic":
                continue
            if self._card_topic_id(card) in answered_topic_ids:
                continue
            out.append(card)

        out.sort(key=lambda c: (c.created_at, c.card_id))
        return out

    def _choose_with_topic_fairness(
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
            topic_id = self._card_topic_id(card)
            grouped.setdefault(topic_id, []).append(card)

        ordered_topics = sorted(grouped.keys(), key=lambda t: (topic_counts.get(t, 0), t))
        for topic_id in ordered_topics:
            if blocked_topic is not None and topic_id == blocked_topic:
                continue
            return grouped[topic_id][0]

        # If all candidates are from the blocked topic, allow fallback.
        return candidates[0]

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

    def _topic_counts(
        self, recent: Iterable[StoredCardPresentation]
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in recent:
            card = self.repo.get_card(card_id=p.card_id)
            if card is None:
                continue
            topic_id = self._card_topic_id(card)
            counts[topic_id] = counts.get(topic_id, 0) + 1
        return counts

    def _diagnosed_topic_ids(self, *, user_id: str, exam_id: str) -> Set[str]:
        return diagnosed_topic_ids(repo=self.repo, user_id=user_id, exam_id=exam_id)

    def _card_topic_id(self, card: StoredCard) -> str:
        links = self.repo.list_card_topics(card_id=card.card_id)
        primaries = [x for x in links if x.role == "primary"]
        if len(primaries) == 1:
            return primaries[0].topic_id
        return card.topic_id

    def _is_due(self, due_at: str, now: datetime) -> bool:
        if not due_at:
            return True
        try:
            due_dt = datetime.fromisoformat(due_at)
        except ValueError:
            return True
        if due_dt.tzinfo is None:
            due_dt = due_dt.replace(tzinfo=timezone.utc)
        return due_dt <= now
