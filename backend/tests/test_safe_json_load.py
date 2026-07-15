import unittest

from app.services.llm import safe_json_load


class SafeJsonLoadTests(unittest.TestCase):
    def test_parses_well_formed_json(self):
        self.assertEqual(safe_json_load('{"answer": "ok"}'), {"answer": "ok"})

    def test_repairs_stray_unescaped_latex_backslash(self):
        # Model forgot to double the backslash before a LaTeX delimiter/command,
        # which is invalid JSON on its own.
        raw = r'{"answer": "Consider \( f(x) = x^2 \)"}'
        self.assertEqual(
            safe_json_load(raw), {"answer": "Consider \\( f(x) = x^2 \\)"}
        )

    def test_leaves_correctly_escaped_backslashes_untouched(self):
        # A matrix row separator \\ (LaTeX) is legitimately double-backslash
        # once JSON-escaped to \\\\; repair must not touch it.
        raw = r'{"answer": "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}"}'
        payload = safe_json_load(raw)
        self.assertEqual(
            payload["answer"], "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}"
        )

    def test_tolerates_literal_control_characters_in_strings(self):
        raw = '{"answer": "Step 1\nStep 2"}'
        self.assertEqual(safe_json_load(raw), {"answer": "Step 1\nStep 2"})

    def test_returns_empty_dict_on_unrecoverable_garbage(self):
        self.assertEqual(safe_json_load("not json at all {"), {})

    def test_returns_empty_dict_for_non_object_json(self):
        self.assertEqual(safe_json_load("[1, 2, 3]"), {})


if __name__ == "__main__":
    unittest.main()
