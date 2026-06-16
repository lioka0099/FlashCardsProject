import unittest

from app.data.db_repository import StoredChunk
from app.services import document_math_profile as profile_mod
from app.services.document_math_profile import (
    build_document_math_sample,
    classify_document_math_profile,
    normalize_document_math_kind,
)


def _chunk(text: str, *, page: int = 1) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"c{page}",
        doc_id="doc1",
        page=page,
        start=0,
        end=len(text),
        text=text,
    )


class DocumentMathProfileTests(unittest.TestCase):
    def test_normalizes_llm_labels_and_internal_kinds(self) -> None:
        self.assertEqual(normalize_document_math_kind({"label": "MATHEMATICAL"}), "math")
        self.assertEqual(normalize_document_math_kind({"label": "CONCEPTUAL"}), "non_math")
        self.assertEqual(normalize_document_math_kind({"label": "BOTH"}), "mixed")
        self.assertEqual(normalize_document_math_kind({"kind": "non_math"}), "non_math")

    def test_teacher_style_numeric_document_falls_back_to_non_math(self) -> None:
        original = profile_mod.chat_completions_create

        def fail_llm(**_kwargs):
            raise RuntimeError("no llm in unit test")

        teacher_text = (
            "Computer adaptive testing enables individualization of tests and better accuracy of knowledge level "
            "determination. Bachelor students answered 44 programming questions and estimated difficulty. "
            "The results include first, second, and third year students, p-values, averages, tables, and charts. "
            "The learning task is understanding code-tracing skills, calibration of questions, and educational findings."
        )
        try:
            profile_mod.chat_completions_create = fail_llm  # type: ignore[assignment]
            profile = classify_document_math_profile(chunks=[_chunk(teacher_text)])
        finally:
            profile_mod.chat_completions_create = original  # type: ignore[assignment]

        self.assertEqual(profile.kind, "non_math")
        self.assertEqual(profile.label, "CONCEPTUAL")

    def test_llm_conceptual_label_maps_to_non_math(self) -> None:
        original = profile_mod.chat_completions_create

        class Message:
            content = (
                '{"label":"CONCEPTUAL","confidence":0.93,'
                '"reason":"The document is about educational testing, not calculation.",'
                '"evidence":["Computer adaptive testing","44 programming questions"]}'
            )

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        try:
            profile_mod.chat_completions_create = lambda **_kwargs: Response()  # type: ignore[assignment]
            profile = classify_document_math_profile(chunks=[_chunk("44 programming questions and p-values")])
        finally:
            profile_mod.chat_completions_create = original  # type: ignore[assignment]

        self.assertEqual(profile.kind, "non_math")
        self.assertEqual(profile.source, "llm")
        self.assertGreaterEqual(profile.confidence, 0.9)

    def test_fallback_detects_math_task_document(self) -> None:
        original = profile_mod.chat_completions_create

        def fail_llm(**_kwargs):
            raise RuntimeError("no llm in unit test")

        math_text = "Solve equations such as f(x)=x^2+3x and calculate derivatives using d/dx x^n = n*x^(n-1)."
        try:
            profile_mod.chat_completions_create = fail_llm  # type: ignore[assignment]
            profile = classify_document_math_profile(chunks=[_chunk(math_text)])
        finally:
            profile_mod.chat_completions_create = original  # type: ignore[assignment]

        self.assertEqual(profile.kind, "math")
        self.assertEqual(profile.label, "MATHEMATICAL")

    def test_document_sample_spreads_across_chunks(self) -> None:
        chunks = [_chunk(f"section {i}", page=i) for i in range(1, 21)]
        sample = build_document_math_sample(chunks, max_chunks=4)

        self.assertIn("section 1", sample)
        self.assertIn("section 20", sample)


if __name__ == "__main__":
    unittest.main()
