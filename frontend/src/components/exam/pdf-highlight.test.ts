import { describe, expect, it } from "vitest";
import { computeHighlightItemIndices } from "@/components/exam/pdf-highlight";

describe("computeHighlightItemIndices", () => {
  it("matches an exact multi-item quote", () => {
    const items = ["TCP guarantees ", "in-order and ", "reliable delivery."];
    const result = computeHighlightItemIndices(
      items,
      "TCP guarantees in-order and reliable delivery.",
    );
    expect(result).toEqual(new Set([0, 1, 2]));
  });

  it("matches after bullet glyphs are stripped", () => {
    const items = ["▪ types of messages", "▪ message syntax"];
    const result = computeHighlightItemIndices(
      items,
      "types of messages message syntax",
    );
    expect(result).toEqual(new Set([0, 1]));
  });

  it("matches after collapsing whitespace", () => {
    const items = ["client   queries", "root  server"];
    const result = computeHighlightItemIndices(items, "client queries root server");
    expect(result).toEqual(new Set([0, 1]));
  });

  it("returns an empty set when the quote is absent", () => {
    const items = ["alpha beta", "gamma delta"];
    expect(computeHighlightItemIndices(items, "totally absent phrase")).toEqual(
      new Set(),
    );
  });

  it("returns an empty set for a blank quote", () => {
    expect(computeHighlightItemIndices(["alpha"], "   ")).toEqual(new Set());
  });

  it("highlights only the overlapping items", () => {
    const items = ["intro noise ", "reliable delivery ", "trailing noise"];
    const result = computeHighlightItemIndices(items, "reliable delivery");
    expect(result).toEqual(new Set([1]));
  });
});
