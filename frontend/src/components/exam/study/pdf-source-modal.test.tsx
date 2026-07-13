import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Card, ProofSpan } from "@/lib/api/client";

// Shared with the react-pdf mock below via vi.hoisted so the mock factory
// (which vitest hoists above other module code) can close over them.
const { documentFiles, loadErrorFlag } = vi.hoisted(() => ({
  documentFiles: [] as unknown[],
  loadErrorFlag: { value: false },
}));

vi.mock("react-pdf", () => ({
  Document: ({
    file,
    children,
    onLoadSuccess,
    onLoadError,
  }: {
    file: { data: ArrayBuffer };
    children: React.ReactNode;
    onLoadSuccess?: (info: { numPages: number }) => void;
    onLoadError?: (error: Error) => void;
  }) => {
    documentFiles.push(file);
    useEffect(() => {
      if (loadErrorFlag.value) {
        onLoadError?.(new Error("bad pdf"));
      } else {
        onLoadSuccess?.({ numPages: 3 });
      }
    }, [onLoadSuccess, onLoadError]);
    return (
      <div data-testid="pdf-document" data-byte-length={file.data.byteLength}>
        {children}
      </div>
    );
  },
  Page: ({
    pageNumber,
    onGetTextSuccess,
  }: {
    pageNumber: number;
    onGetTextSuccess?: (textContent: { items: { str: string }[] }) => void;
  }) => {
    // Simulates the cited page's text extraction completing, which in the
    // real component drives the highlight computation and a forceRepaint —
    // the re-render path that must not recreate the `file` prop.
    useEffect(() => {
      onGetTextSuccess?.({ items: [{ str: "reliable delivery" }] });
    }, [onGetTextSuccess]);
    return <div data-testid="pdf-page">page {pageNumber}</div>;
  },
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" }, version: "10.0.0" },
}));

const fetchSourceBlobMock = vi.fn();
vi.mock("@/components/exam/lib/fetch-source-blob", () => ({
  fetchSourceBlob: (...args: unknown[]) => fetchSourceBlobMock(...args),
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
  beforeEach(() => {
    fetchSourceBlobMock.mockReset();
    documentFiles.length = 0;
    loadErrorFlag.value = false;
  });

  it("fetches the source through the authenticated helper and renders every page", async () => {
    fetchSourceBlobMock.mockResolvedValue({
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
    });

    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    expect(fetchSourceBlobMock).toHaveBeenCalledWith(
      expect.stringContaining("https://example.com/source.pdf"),
    );
    expect(await screen.findByTestId("pdf-document")).toHaveAttribute(
      "data-byte-length",
      "8",
    );
    const pages = await screen.findAllByTestId("pdf-page");
    expect(pages.map((p) => p.textContent)).toEqual(["page 1", "page 2", "page 3"]);
  });

  it("shows an error message when the fetch fails (e.g. 401)", async () => {
    fetchSourceBlobMock.mockRejectedValue(new Error("Failed to fetch source document (401)"));

    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    expect(
      await screen.findByText("Could not load the source document."),
    ).toBeInTheDocument();
  });

  it("shows an error message when the fetch succeeds but pdf.js fails to parse the bytes", async () => {
    loadErrorFlag.value = true;
    fetchSourceBlobMock.mockResolvedValue({
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
    });

    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    expect(
      await screen.findByText("Could not load the source document."),
    ).toBeInTheDocument();
  });

  it("keeps the file object reference stable across re-renders", async () => {
    fetchSourceBlobMock.mockResolvedValue({
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
    });

    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={() => {}} />);

    await screen.findByTestId("pdf-document");
    // Wait for the highlight-driven forceRepaint (see the Page mock above) to
    // produce a second render of <Document>.
    await waitFor(() => expect(documentFiles.length).toBeGreaterThan(1));

    const [first, ...rest] = documentFiles;
    for (const later of rest) {
      expect(later).toBe(first);
    }
  });

  it("calls onClose when the backdrop is clicked", async () => {
    fetchSourceBlobMock.mockResolvedValue({
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
    });
    const onClose = vi.fn();
    render(<PdfSourceModal proof={proof} card={card} userId="guest" onClose={onClose} />);
    await screen.findByTestId("pdf-document");
    fireEvent.click(screen.getByRole("presentation"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
