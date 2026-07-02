import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.qa import summarize_evidence


def _fake_response(quote: str):
    message = SimpleNamespace(content=json.dumps({"quote": quote}))
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class SummarizeEvidenceTests(unittest.TestCase):
    def test_returns_verbatim_span_when_it_matches_the_passage(self):
        passage = "The client queries the root server to find the .com DNS server."
        with patch(
            "app.services.qa.chat_completions_create",
            return_value=_fake_response("The client queries the root server"),
        ):
            result = summarize_evidence(answer="How does DNS start?", proof_texts=[passage])
        self.assertEqual(result, ["The client queries the root server"])

    def test_accepts_span_even_when_llm_drops_bullet_glyphs(self):
        passage = "An application-layer protocol defines: ▪ types of messages ▪ message syntax"
        with patch(
            "app.services.qa.chat_completions_create",
            return_value=_fake_response("types of messages message syntax"),
        ):
            result = summarize_evidence(answer="what does a protocol define", proof_texts=[passage])
        self.assertEqual(result, ["types of messages message syntax"])

    def test_falls_back_to_real_segment_when_llm_paraphrases(self):
        passage = "Alpha beta gamma. ▪ Delta epsilon zeta"
        with patch(
            "app.services.qa.chat_completions_create",
            return_value=_fake_response("completely invented wording not present"),
        ):
            result = summarize_evidence(answer="delta epsilon", proof_texts=[passage])
        # Not a substring -> deterministic fallback picks the overlapping segment, verbatim.
        self.assertEqual(result, ["Delta epsilon zeta"])

    def test_falls_back_to_verbatim_segment_when_llm_raises(self):
        passage = "Alpha beta gamma. ▪ Delta epsilon zeta"
        with patch(
            "app.services.qa.chat_completions_create",
            side_effect=RuntimeError("boom"),
        ):
            result = summarize_evidence(answer="alpha beta", proof_texts=[passage])
        self.assertEqual(result, ["Alpha beta gamma."])

    def test_blank_proof_returns_empty_string_without_calling_llm(self):
        with patch("app.services.qa.chat_completions_create") as mock_call:
            result = summarize_evidence(answer="x", proof_texts=["   ", ""])
        self.assertEqual(result, ["", ""])
        mock_call.assert_not_called()

    def test_empty_input_returns_empty_list(self):
        with patch("app.services.qa.chat_completions_create") as mock_call:
            result = summarize_evidence(answer="x", proof_texts=[])
        self.assertEqual(result, [])
        mock_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
