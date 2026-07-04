"use client";

import { useEffect, useRef } from "react";
import renderMathInElement from "katex/contrib/auto-render";
import "katex/dist/katex.min.css";
import { toUnicodeSuperscripts } from "@/components/exam/math-superscript";

// Standard delimiters the card LLM emits. Keep display ($$, \[ \]) before the
// inline forms so the longer openers win.
const DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "\\(", right: "\\)", display: false },
  { left: "$", right: "$", display: false },
];

type MathTextProps = {
  text: string;
  className?: string;
};

// Renders text that may contain inline/display LaTeX. Sets the raw text, then
// lets KaTeX auto-render replace the math spans in place.
export function MathText({ text, className }: MathTextProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.textContent = text;
    renderMathInElement(ref.current, {
      delimiters: DELIMITERS,
      throwOnError: false,
    });
    // KaTeX handled delimited math; superscript ASCII carets in the remaining
    // prose (cards whose math arrived as plain text), skipping rendered math.
    const walker = document.createTreeWalker(ref.current, NodeFilter.SHOW_TEXT);
    const textNodes: Text[] = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      textNodes.push(node as Text);
    }
    for (const node of textNodes) {
      if ((node.parentElement as Element | null)?.closest(".katex")) continue;
      const converted = toUnicodeSuperscripts(node.nodeValue ?? "");
      if (converted !== node.nodeValue) node.nodeValue = converted;
    }
  }, [text]);

  return <div ref={ref} className={className} />;
}
