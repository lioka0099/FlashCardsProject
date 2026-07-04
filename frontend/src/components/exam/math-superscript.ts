// Converts ASCII caret-exponents in prose (e.g. "x^3", "3x^2", "x^{-1}", "e^n")
// to Unicode superscripts ("x³", "3x²", "x⁻¹", "eⁿ"). Used for cards whose math
// arrives as plain text instead of LaTeX. Leaves a match untouched if any
// exponent character has no superscript form, so nothing is half-converted.

const SUP: Record<string, string> = {
  "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
  "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
  a: "ᵃ", b: "ᵇ", c: "ᶜ", d: "ᵈ", e: "ᵉ", f: "ᶠ", g: "ᵍ", h: "ʰ",
  i: "ⁱ", j: "ʲ", k: "ᵏ", l: "ˡ", m: "ᵐ", n: "ⁿ", o: "ᵒ", p: "ᵖ",
  r: "ʳ", s: "ˢ", t: "ᵗ", u: "ᵘ", v: "ᵛ", w: "ʷ", x: "ˣ", y: "ʸ", z: "ᶻ",
};

// ^{...} or ^ followed by a run of exponent-ish chars.
const EXPONENT = /\^(?:\{([^}]+)\}|([0-9a-z+\-=()]+))/gi;

export function toUnicodeSuperscripts(text: string): string {
  return text.replace(EXPONENT, (match, braced?: string, bare?: string) => {
    const exp = (braced ?? bare ?? "").toLowerCase();
    let out = "";
    for (const ch of exp) {
      if (!(ch in SUP)) return match; // unmappable char → leave original
      out += SUP[ch];
    }
    return out;
  });
}
