import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Card, ProofSpan } from "@/lib/api/client";

const fetchSourceBlobMock = vi.fn();
vi.mock("@/components/exam/lib/fetch-source-blob", () => ({
  fetchSourceBlob: (...args: unknown[]) => fetchSourceBlobMock(...args),
}));

import { TextSourceModal } from "@/components/exam/study/text-source-modal";

const card = { exam_id: "exam-1" } as Card;
const proof: ProofSpan = {
  doc_id: "notes.txt",
  page: null,
  start: 0,
  end: 0,
  text: "reliable delivery",
  score: 0.9,
};

function textBlob(text: string): Blob {
  return { text: () => Promise.resolve(text) } as unknown as Blob;
}

describe("TextSourceModal", () => {
  beforeEach(() => {
    fetchSourceBlobMock.mockReset();
  });

  it("fetches the source through the authenticated helper and highlights the cited paragraph", async () => {
    fetchSourceBlobMock.mockResolvedValue(
      textBlob("intro noise\nTCP guarantees reliable delivery\ntrailing noise"),
    );

    render(<TextSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    expect(fetchSourceBlobMock).toHaveBeenCalledWith(
      expect.stringContaining("notes.txt"),
    );
    const highlighted = await screen.findByText("reliable delivery", { exact: false });
    expect(highlighted.tagName.toLowerCase()).toBe("mark");
    expect(screen.getByText("intro noise")).toBeInTheDocument();
  });

  it("shows an error message when the fetch fails", async () => {
    fetchSourceBlobMock.mockRejectedValue(new Error("Failed to fetch source document (401)"));

    render(<TextSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    expect(
      await screen.findByText("Could not load the source document."),
    ).toBeInTheDocument();
  });

  it("calls onClose when the backdrop is clicked", () => {
    fetchSourceBlobMock.mockResolvedValue(textBlob("some text"));
    const onClose = vi.fn();
    render(<TextSourceModal proof={proof} card={card} userId="guest" onClose={onClose} />);
    fireEvent.click(screen.getByRole("presentation"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
