import { describe, expect, it } from "vitest";
import { toUnicodeSuperscripts } from "@/components/exam/lib/math-superscript";

describe("toUnicodeSuperscripts", () => {
  it("converts simple integer exponents", () => {
    expect(toUnicodeSuperscripts("f(x) = x^3 - 3x^2 + 4")).toBe(
      "f(x) = x³ - 3x² + 4",
    );
  });

  it("handles braces, signs, and letter exponents", () => {
    expect(toUnicodeSuperscripts("x^{-1}")).toBe("x⁻¹");
    expect(toUnicodeSuperscripts("e^n")).toBe("eⁿ");
    expect(toUnicodeSuperscripts("2^{10}")).toBe("2¹⁰");
  });

  it("leaves text without carets unchanged", () => {
    expect(toUnicodeSuperscripts("no math here")).toBe("no math here");
  });

  it("leaves a match untouched when an exponent char has no superscript", () => {
    // 'q' has no Unicode superscript, so the whole exponent is left as-is.
    expect(toUnicodeSuperscripts("x^{q}")).toBe("x^{q}");
  });
});
