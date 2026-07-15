"use client";

import { useEffect, useRef } from "react";
import renderMathInElement from "katex/contrib/auto-render";
import "katex/dist/katex.min.css";
import { toUnicodeSuperscripts } from "@/components/exam/lib/math-superscript";

// Standard delimiters the card LLM emits. Keep display ($$, \[ \]) before the
// inline forms so the longer openers win.
const DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "\\(", right: "\\)", display: false },
  { left: "$", right: "$", display: false },
];

// The card LLM occasionally double-escapes a delimiter (\\( instead of \()
// when authoring JSON, since JSON itself requires backslashes doubled — most
// of the time it un-escapes correctly, but the rare miss leaves KaTeX unable
// to recognize the delimiter. Collapse that one extra backslash back down.
const OVER_ESCAPED_DELIMITER = /\\\\(?=[()[\]])/g;

// Rarer still: the LLM drops the \( \) delimiters around a clause entirely,
// leaving a bare command like \\mathbb{R}^n sitting in plain prose with
// nothing for KaTeX to key off. Only kick in when the text has no delimiters
// anywhere (so a normal, already-working card is never touched), and wrap
// each bare command run in delimiters, normalizing 1+ leading backslashes to
// one. ponytail: single-command heuristic, not a LaTeX parser — won't catch a
// bare command mixed into an otherwise-delimited card; widen if that shows up.
const HAS_DELIMITER = /\\\(|\\\[|\$/;
const BARE_LATEX_COMMAND = /\\+([a-zA-Z]+(?:\{[^{}]*\})*(?:[\^_](?:\{[^{}]*\}|[a-zA-Z0-9]+))*)/g;

function wrapBareLatexCommands(text: string): string {
  if (HAS_DELIMITER.test(text)) return text;
  return text.replace(BARE_LATEX_COMMAND, (_match, body: string) => `\\(\\${body}\\)`);
}

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
    const normalized = wrapBareLatexCommands(text.replace(OVER_ESCAPED_DELIMITER, "\\"));
    ref.current.textContent = normalized;
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
