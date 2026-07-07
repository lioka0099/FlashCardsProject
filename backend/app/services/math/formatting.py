"""Shared prompt instruction so generated card text renders as real math.

The frontend renders inline LaTeX (\\( ... \\)) with KaTeX; without this the LLM
emits bare ASCII like ``x^3`` inconsistently, which reads poorly.
"""

MATH_DISPLAY_INSTRUCTION = (
    "Format every mathematical expression, variable, symbol, and formula using "
    "inline LaTeX wrapped in \\( ... \\) (use \\[ ... \\] for standalone display "
    "equations). For example write \\(f(x) = x^3 - 3x^2 + 4\\), never bare ASCII "
    "like x^3 - 3x^2 + 4. Leave non-mathematical prose as plain text."
)
