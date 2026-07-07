import { fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Card, ProofSpan } from "@/lib/api/client";

// react-pdf needs canvas/DOMMatrix (absent in jsdom), so mock it. The Document
// mock echoes the file URL and reports a 3-page document via onLoadSuccess
// (which is what drives how many Page children the modal renders); the Page
// mock echoes its page number.
vi.mock("react-pdf", () => ({
  Document: ({
    file,
    children,
    onLoadSuccess,
  }: {
    file: string;
    children: React.ReactNode;
    onLoadSuccess?: (info: { numPages: number }) => void;
  }) => {
    useEffect(() => {
      onLoadSuccess?.({ numPages: 3 });
    }, [onLoadSuccess]);
    return (
      <div data-testid="pdf-document" data-file={file}>
        {children}
      </div>
    );
  },
  Page: ({ pageNumber }: { pageNumber: number }) => (
    <div data-testid="pdf-page">page {pageNumber}</div>
  ),
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" }, version: "10.0.0" },
}));

import { PdfSourceModal } from "@/components/exam/study/pdf-source-modal";

const card = { exam_id: "exam-1" } as Card;
const proof: ProofSpan = {
  doc_id: "https://example.com/source.pdf",
  page: 3,
  start: 0,
  end: 0,
  text: "reliable delivery",
  score: 0.9,
};

describe("PdfSourceModal", () => {
  it("renders every page of the document and passes the source URL", async () => {
    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);
    expect(screen.getByTestId("pdf-document")).toHaveAttribute(
      "data-file",
      expect.stringContaining("https://example.com/source.pdf"),
    );
    // onLoadSuccess reports 3 pages, so all three render (not just the cited one).
    const pages = await screen.findAllByTestId("pdf-page");
    expect(pages.map((p) => p.textContent)).toEqual(["page 1", "page 2", "page 3"]);
  });

  it("calls onClose when the backdrop is clicked", () => {
    const onClose = vi.fn();
    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={onClose} />);
    fireEvent.click(screen.getByRole("presentation"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
