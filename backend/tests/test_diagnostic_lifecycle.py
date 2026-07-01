import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase3_diagnostic_lifecycle_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from fastapi.testclient import TestClient

from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.vector_store import VectorStore
from app.services.exams import create_exam
from app.services.graph import GeneratedCard


def _write_text_doc(name: str, body: str) -> str:
    path = _TEST_ROOT / name
    path.write_text(body, encoding="utf-8")
    return str(path)


class DiagnosticLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.client = TestClient(app)
        from app.services import diagnostic_lifecycle as dl_mod
        from app.services import ingestion as ingestion_mod

        self._dl_mod = dl_mod
        self._ingestion_mod = ingestion_mod
        self._original_classifier = dl_mod.classify_document_math_profile
        self._original_embed_texts = ingestion_mod.embed_texts

        class FakeMathProfile:
            def to_info(self):
                return {
                    "kind": "non_math",
                    "label": "CONCEPTUAL",
                    "confidence": 0.95,
                    "reason": "unit test default",
                    "source": "test",
                }

        dl_mod.classify_document_math_profile = lambda **_kwargs: FakeMathProfile()  # type: ignore[assignment]
        ingestion_mod.embed_texts = lambda texts: np.ones((len(texts), 3072), dtype="float32")  # type: ignore[assignment]

    def tearDown(self) -> None:
        self._dl_mod.classify_document_math_profile = self._original_classifier  # type: ignore[assignment]
        self._ingestion_mod.embed_texts = self._original_embed_texts  # type: ignore[assignment]

    def test_bootstrap_and_atomic_transition(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            store.db.upsert_topic(topic_id="tA", exam_id=exam_id, label="Topic A", info={"n_chunks": 1})
            store.db.upsert_topic(topic_id="tB", exam_id=exam_id, label="Topic B", info={"n_chunks": 1})
            return [
                {"topic_id": "tA", "label": "Topic A"},
                {"topic_id": "tB", "label": "Topic B"},
            ]

        def fake_generate_starter_cards_v2(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            cards = []
            for i, tid in enumerate(["tA", "tB"][:n], start=1):
                cid = f"diag_card_{i}"
                store.db.upsert_card(
                    card_id=cid,
                    exam_id=exam_id,
                    topic_id=tid,
                    question=f"Q{i}",
                    answer=f"A{i}",
                    difficulty=1,
                    card_type="diagnostic",
                    status="active",
                    info={"source": "test"},
                )
                store.db.replace_card_topics(
                    card_id=cid,
                    topics=[{"topic_id": tid, "role": "primary", "weight": 1.0}],
                )
                cards.append(
                    GeneratedCard(
                        card_id=cid,
                        exam_id=exam_id,
                        topic_id=tid,
                        topic_label=f"Topic {tid}",
                        question=f"Q{i}",
                        answer=f"A{i}",
                        difficulty=1,
                        proofs=[],
                    )
                )
            return cards

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_starter_cards_v2
        try:
            d1 = _write_text_doc("p3_bootstrap_1.txt", "alpha beta gamma")
            from app.services.diagnostic_lifecycle import DiagnosticLifecycleService
            result = DiagnosticLifecycleService(store=VectorStore(basepath=str(_TEST_ROOT / "store"))).bootstrap_exam_from_upload(
                user_id="u1", title="Exam 1", mode="mastery", paths=[d1],
                info={"source": "from_upload", "filenames": ["p3_bootstrap_1.txt"]})
            exam_id = result.exam_id
            self.assertEqual(result.state, "diagnostic")
            self.assertEqual(result.diagnostic_total, 2)
            self.assertEqual(result.diagnostic_answered, 0)

            exam_resp = self.client.get(f"/exams/{exam_id}", params={"user_id": "u1"})
            self.assertEqual(exam_resp.status_code, 200)
            self.assertEqual(exam_resp.json()["state"], "diagnostic")

            r1 = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_1/review",
                data={"user_id": "u1", "rating": "almost_knew"},
                headers={"Idempotency-Key": "diag-r1"},
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.json()["diagnostic_answered"], 1)
            self.assertEqual(r1.json()["exam_state"], "diagnostic")

            r1_replay = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_1/review",
                data={"user_id": "u1", "rating": "almost_knew"},
                headers={"Idempotency-Key": "diag-r1"},
            )
            self.assertEqual(r1_replay.status_code, 200)
            self.assertTrue(r1_replay.json()["idempotent_replay"])
            self.assertEqual(r1_replay.json()["diagnostic_answered"], 1)

            r2 = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_2/review",
                data={"user_id": "u1", "rating": "i_knew_it"},
                headers={"Idempotency-Key": "diag-r2"},
            )
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.json()["diagnostic_answered"], 2)
            self.assertEqual(r2.json()["exam_state"], "active_learning")

            exam_resp2 = self.client.get(f"/exams/{exam_id}", params={"user_id": "u1"})
            self.assertEqual(exam_resp2.status_code, 200)
            self.assertEqual(exam_resp2.json()["state"], "active_learning")
            self.assertEqual(exam_resp2.json()["diagnostic_answered"], 2)
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate

    def test_bootstrap_persists_math_profile_before_topic_build(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2
        captured = {}

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            exam = store.db.get_exam(exam_id)
            captured["math_profile"] = (exam.info or {}).get("math_profile") if exam else None
            store.db.upsert_topic(topic_id="tA", exam_id=exam_id, label="Topic A", info={"n_chunks": 1})
            return [{"topic_id": "tA", "label": "Topic A"}]

        def fake_generate_starter_cards_v2(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            store.db.upsert_card(
                card_id="diag_card_profile",
                exam_id=exam_id,
                topic_id="tA",
                question="Q",
                answer="A",
                difficulty=1,
                card_type="diagnostic",
                status="active",
                info={"source": "test"},
            )
            store.db.replace_card_topics(
                card_id="diag_card_profile",
                topics=[{"topic_id": "tA", "role": "primary", "weight": 1.0}],
            )
            return [
                GeneratedCard(
                    card_id="diag_card_profile",
                    exam_id=exam_id,
                    topic_id="tA",
                    topic_label="Topic A",
                    question="Q",
                    answer="A",
                    difficulty=1,
                    proofs=[],
                )
            ]

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_starter_cards_v2
        try:
            d1 = _write_text_doc("p3_math_profile_timing.txt", "44 examples and 100 rows, but conceptual notes.")
            from app.services.diagnostic_lifecycle import DiagnosticLifecycleService
            DiagnosticLifecycleService(store=VectorStore(basepath=str(_TEST_ROOT / "store"))).bootstrap_exam_from_upload(
                user_id="u-profile", title="Profile timing", mode="mastery", paths=[d1],
                info={"source": "from_upload", "filenames": ["p3_math_profile_timing.txt"]})
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate

        self.assertIsNotNone(captured["math_profile"])
        self.assertEqual(captured["math_profile"]["kind"], "non_math")

    def test_failed_bootstrap_cleans_up_exam_data(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            store.db.upsert_topic(topic_id="tA", exam_id=exam_id, label="Topic A", info={"n_chunks": 1})
            store.db.upsert_topic(topic_id="tB", exam_id=exam_id, label="Topic B", info={"n_chunks": 1})
            return [
                {"topic_id": "tA", "label": "Topic A"},
                {"topic_id": "tB", "label": "Topic B"},
            ]

        def fake_generate_partial(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            # Generate fewer cards than topics to force failure.
            cid = "diag_card_partial"
            store.db.upsert_card(
                card_id=cid,
                exam_id=exam_id,
                topic_id="tA",
                question="Q?",
                answer="A",
                difficulty=1,
                card_type="diagnostic",
                status="active",
                info={"source": "test"},
            )
            store.db.replace_card_topics(
                card_id=cid,
                topics=[{"topic_id": "tA", "role": "primary", "weight": 1.0}],
            )
            return [
                GeneratedCard(
                    card_id=cid,
                    exam_id=exam_id,
                    topic_id="tA",
                    topic_label="Topic A",
                    question="Q?",
                    answer="A",
                    difficulty=1,
                    proofs=[],
                )
            ]

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_partial
        try:
            d1 = _write_text_doc("p3_bootstrap_fail.txt", "alpha beta gamma")
            from app.services.diagnostic_lifecycle import DiagnosticLifecycleService, DiagnosticBootstrapError
            with self.assertRaises(DiagnosticBootstrapError):
                DiagnosticLifecycleService(store=VectorStore(basepath=str(_TEST_ROOT / "store"))).bootstrap_exam_from_upload(
                    user_id="u2", title="Exam fail", mode="mastery", paths=[d1],
                    info={"source": "from_upload", "filenames": ["p3_bootstrap_fail.txt"]})

            store = VectorStore(basepath=str(_TEST_ROOT / "store"))
            exams = store.db.list_exams(user_id="u2")
            self.assertEqual(len(exams), 0)
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate

    def test_bootstrap_allows_coverage_at_or_above_80_percent(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            topic_ids = ["tA", "tB", "tC", "tD", "tE"]
            for tid in topic_ids:
                store.db.upsert_topic(topic_id=tid, exam_id=exam_id, label=f"Topic {tid}", info={"n_chunks": 1})
            return [{"topic_id": tid, "label": f"Topic {tid}"} for tid in topic_ids]

        def fake_generate_partial_but_ok(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            cards = []
            # 4/5 coverage = 80% and should be accepted.
            for i, tid in enumerate(["tA", "tB", "tC", "tD"], start=1):
                cid = f"diag_card_{i}"
                store.db.upsert_card(
                    card_id=cid,
                    exam_id=exam_id,
                    topic_id=tid,
                    question=f"Q{i}",
                    answer=f"A{i}",
                    difficulty=1,
                    card_type="diagnostic",
                    status="active",
                    info={"source": "test"},
                )
                store.db.replace_card_topics(
                    card_id=cid,
                    topics=[{"topic_id": tid, "role": "primary", "weight": 1.0}],
                )
                cards.append(
                    GeneratedCard(
                        card_id=cid,
                        exam_id=exam_id,
                        topic_id=tid,
                        topic_label=f"Topic {tid}",
                        question=f"Q{i}",
                        answer=f"A{i}",
                        difficulty=1,
                        proofs=[],
                    )
                )
            return cards

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_partial_but_ok
        try:
            d1 = _write_text_doc("p3_bootstrap_80pct.txt", "alpha beta gamma")
            from app.services.diagnostic_lifecycle import DiagnosticLifecycleService
            result = DiagnosticLifecycleService(store=VectorStore(basepath=str(_TEST_ROOT / "store"))).bootstrap_exam_from_upload(
                user_id="u3", title="Exam 80%", mode="mastery", paths=[d1],
                info={"source": "from_upload", "filenames": ["p3_bootstrap_80pct.txt"]})
            self.assertEqual(result.state, "diagnostic")
            self.assertEqual(result.diagnostic_total, 5)
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate


if __name__ == "__main__":
    unittest.main()

