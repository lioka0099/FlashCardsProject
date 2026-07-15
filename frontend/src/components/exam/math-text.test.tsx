import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MathText } from "@/components/exam/math-text";

describe("MathText", () => {
  it("renders correctly-escaped LaTeX delimiters", () => {
    const { container } = render(<MathText text="Consider \\( f(x) = x^2 \\)" />);
    expect(container.querySelector(".katex")).not.toBeNull();
  });

  it("renders LaTeX even when the delimiter was double-escaped", () => {
    // Occasional LLM slip: an extra backslash before the delimiter (\\\\( instead
    // of \\(), which KaTeX's auto-render would otherwise fail to recognize.
    const { container } = render(<MathText text="Consider \\\\( f(x) = x^2 \\\\)" />);
    const katex = container.querySelector(".katex");
    expect(katex).not.toBeNull();
    expect(container.textContent).not.toContain("\\\\(");
  });

  it("wraps a bare LaTeX command left with no delimiters at all", () => {
    // Occasional LLM slip: the whole clause never got \\( \\) at all, just a
    // bare (and here also double-escaped) command dropped into plain prose.
    const { container } = render(
      <MathText text="Solve Ax = b for each b in \\\\mathbb{R}^n?" />,
    );
    expect(container.querySelector(".katex")).not.toBeNull();
    // KaTeX's accessibility annotation legitimately echoes the (now-normalized,
    // single-backslash) source, so assert the ORIGINAL double backslash is gone
    // rather than asserting "\mathbb" is absent entirely.
    expect(container.textContent).not.toContain("\\\\mathbb");
  });

  it("leaves plain prose with no LaTeX untouched", () => {
    const { container } = render(<MathText text="What is the rank of a matrix?" />);
    expect(container.querySelector(".katex")).toBeNull();
    expect(container.textContent).toBe("What is the rank of a matrix?");
  });
});
