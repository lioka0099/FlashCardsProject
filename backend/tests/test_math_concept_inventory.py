import unittest

from app.services.math.concept_inventory import (
    ConceptInventory,
    MathConceptInventoryService,
    parse_inventory_payload,
)


class ConceptInventoryParsingTests(unittest.TestCase):
    def test_parse_full_payload(self) -> None:
        inv = parse_inventory_payload(
            {
                "concepts": [{"name": "Eigenvalue", "summary": "scalar lambda with Av=lambda v"}],
                "formulas": [{"name": "char poly", "expression": "det(A - lambda*I)", "source_quote": "..."}],
                "methods": [{"name": "diagonalization", "when_to_use": "when A has n independent eigenvectors"}],
                "archetypes": [{"name": "eigen-and-definiteness", "description": "find eigenvalues then classify"}],
            }
        )
        self.assertEqual(len(inv.concepts), 1)
        self.assertEqual(len(inv.formulas), 1)
        self.assertEqual(len(inv.methods), 1)
        self.assertEqual(len(inv.archetypes), 1)
        self.assertFalse(inv.is_empty())

    def test_parse_dedupes_and_handles_strings(self) -> None:
        inv = parse_inventory_payload(
            {"concepts": ["slope", "slope", {"name": "intercept", "summary": "b"}]}
        )
        names = [c["name"] for c in inv.concepts]
        self.assertEqual(names, ["slope", "intercept"])

    def test_empty_payload(self) -> None:
        self.assertTrue(parse_inventory_payload({}).is_empty())
        self.assertTrue(parse_inventory_payload(None).is_empty())

    def test_as_prompt_renders_sections(self) -> None:
        inv = ConceptInventory(concepts=[{"name": "derivative", "summary": "rate of change"}])
        prompt = inv.as_prompt()
        self.assertIn("CONCEPTS", prompt)
        self.assertIn("derivative", prompt)
        self.assertIn("FORMULAS", prompt)

    def test_get_or_build_uses_cache_without_llm(self) -> None:
        cached = {
            "concepts": [{"name": "limit", "summary": "value approached"}],
            "version": 1,
        }
        # If this hit the LLM it would raise (no API client configured in tests).
        inv = MathConceptInventoryService().get_or_build(
            topic_label="Limits", context_pack="...", cached=cached
        )
        self.assertEqual(len(inv.concepts), 1)
        self.assertEqual(inv.concepts[0]["name"], "limit")


if __name__ == "__main__":
    unittest.main()
