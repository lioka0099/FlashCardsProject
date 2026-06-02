from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from app.data.db_repository import DBRepository, StoredTopic


def diagnosed_topic_ids(
    *,
    repo: DBRepository,
    user_id: str,
    exam_id: str,
) -> Set[str]:
    return {
        p.topic_id
        for p in repo.list_topic_proficiencies(user_id=user_id, exam_id=exam_id)
        if int(p.seen_count) > 0
    }


def undiagnosed_topics(
    *,
    repo: DBRepository,
    user_id: str,
    exam_id: str,
) -> List[StoredTopic]:
    diagnosed = diagnosed_topic_ids(repo=repo, user_id=user_id, exam_id=exam_id)
    return [t for t in repo.list_topics(exam_id=exam_id) if t.topic_id not in diagnosed]


def all_topics_diagnosed(
    *,
    repo: DBRepository,
    user_id: str,
    exam_id: str,
) -> bool:
    topics = repo.list_topics(exam_id=exam_id)
    if not topics:
        return True
    diagnosed = diagnosed_topic_ids(repo=repo, user_id=user_id, exam_id=exam_id)
    return all(t.topic_id in diagnosed for t in topics)


def default_repo() -> DBRepository:
    return DBRepository(Path("store/meta.sqlite"))
