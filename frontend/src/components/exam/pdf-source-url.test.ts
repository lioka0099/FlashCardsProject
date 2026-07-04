import { describe, expect, it } from "vitest";
import { buildSourceUrl } from "@/components/exam/pdf-source-url";
import type { Card, ProofSpan } from "@/lib/api/client";

const card = { exam_id: "exam-1" } as Card;

function proof(overrides: Partial<ProofSpan>): ProofSpan {
  return { doc_id: "d", page: null, start: 0, end: 0, text: "", score: 0, ...overrides };
}

describe("buildSourceUrl", () => {
  it("returns the doc_id verbatim when it is an http URL", () => {
    const url = buildSourceUrl(
      proof({ doc_id: "https://example.com/source.pdf", page: 3, start: 12, end: 46 }),
      card,
      "guest",
    );
    expect(url).toContain("https://example.com/source.pdf");
    expect(url).toContain("#page=3");
  });

  it("builds an internal API URL with exam_id and user_id for a local doc_id", () => {
    const url = buildSourceUrl(proof({ doc_id: "local_file.pdf" }), card, "guest");
    expect(url).toContain(
      "/documents/local_file.pdf/source?exam_id=exam-1&user_id=guest",
    );
  });
});
