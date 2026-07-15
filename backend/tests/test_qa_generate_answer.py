import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.schemas import ProofSpan
from app.services.corpus.qa import generate_answer


def _fake_response(content: str):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _proof_pool(n: int = 8):
    return [
        ProofSpan(doc_id="doc", page=1, start=0, end=10, text="A fact about the topic.", score=0.9)
        for _ in range(n)
    ]


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_answer_from_well_formed_json(self):
        with patch(
            "app.services.corpus.qa.chat_completions_create",
            return_value=_fake_response(
                '{"answer": "The answer.", "score": 0.9, "critique": ""}'
            ),
        ):
            result = generate_answer("What is it?", prefetched_pool=_proof_pool())
        self.assertEqual(result.answer, "The answer.")

    def test_raises_instead_of_leaking_raw_text_on_truncated_json(self):
        # A response cut off mid-string (e.g. by a token limit or a model that
        # degenerated into repetition) is exactly the shape that used to leak
        # the whole raw JSON blob into the card's answer field.
        truncated = '{"answer": "Some long partial answer that never closes'
        with patch(
            "app.services.corpus.qa.chat_completions_create",
            return_value=_fake_response(truncated),
        ):
            with self.assertRaises(ValueError):
                generate_answer("What is it?", prefetched_pool=_proof_pool())


if __name__ == "__main__":
    unittest.main()
