const BULLET_RE = /[▪•●■◦‣·]+/g;

function normalize(text: string): string {
  return text
    .replace(BULLET_RE, " ")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Given the pdf.js text-layer item strings for a page and the citation quote,
 * return the set of item indices whose text overlaps the quote. Matching is
 * done on normalized text (lowercased, bullet glyphs stripped, whitespace
 * collapsed) because the stored quote is cleaned the same way. Boundary items
 * are returned whole. Empty set when the quote can't be located.
 */
export function computeHighlightItemIndices(
  itemStrings: string[],
  quote: string,
): Set<number> {
  const result = new Set<number>();
  const normQuote = normalize(quote);
  if (!normQuote) {
    return result;
  }

  const ranges: Array<{ index: number; start: number; end: number }> = [];
  let page = "";
  itemStrings.forEach((raw, index) => {
    const norm = normalize(raw);
    if (!norm) {
      return;
    }
    if (page.length > 0) {
      page += " ";
    }
    const start = page.length;
    page += norm;
    ranges.push({ index, start, end: page.length });
  });

  const matchStart = page.indexOf(normQuote);
  if (matchStart === -1) {
    return result;
  }
  const matchEnd = matchStart + normQuote.length;

  for (const r of ranges) {
    if (r.start < matchEnd && r.end > matchStart) {
      result.add(r.index);
    }
  }
  return result;
}
