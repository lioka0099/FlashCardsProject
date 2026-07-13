import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProofsDialog } from "@/components/exam/study/proofs-dialog";
import type { Card } from "@/lib/api/client";

vi.mock("@/components/exam/study/pdf-source-modal", () => ({
  PdfSourceModal: ({ proof }: { proof: { doc_id: string } }) => (
    <div data-testid="pdf-modal">modal:{proof.doc_id}</div>
  ),
}));

vi.mock("@/components/exam/study/text-source-modal", () => ({
  TextSourceModal: ({ proof }: { proof: { doc_id: string } }) => (
    <div data-testid="text-modal">modal:{proof.doc_id}</div>
  ),
}));

const fetchSourceBlobMock = vi.fn();
vi.mock("@/components/exam/lib/fetch-source-blob", () => ({
  fetchSourceBlob: (...args: unknown[]) => fetchSourceBlobMock(...args),
}));

function buildCard(overrides?: Partial<Card>): Card {
  return {
    card_id: "card-1",
    exam_id: "exam-1",
    topic_id: "topic-1",
    topic_label: "Networks",
    question: "What does TCP guarantee?",
    answer: "Reliable ordered delivery.",
    difficulty: 2,
    created_at: "2026-01-01T00:00:00Z",
    status: "active",
    proofs: [
      {
        doc_id: "https://example.com/source.pdf",
        page: 3,
        start: 12,
        end: 46,
        text: "TCP guarantees in-order and reliable delivery.",
        score: 0.92,
      },
    ],
    info: {},
    ...overrides,
  };
}

describe("ProofsDialog", () => {
  it("renders source metadata and opens the source modal", async () => {
    render(
      <ProofsDialog isOpen card={buildCard()} userId="guest" onClose={() => {}} />,
    );

    expect(
      screen.getByRole("dialog", { name: /evidence and source context/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    expect(screen.getByText("Page: 3")).toBeInTheDocument();
    expect(
      screen.getByText("TCP guarantees in-order and reliable delivery."),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));
    // PdfSourceModal is loaded via next/dynamic (client-only), so it resolves async.
    expect(await screen.findByTestId("pdf-modal")).toHaveTextContent(
      "modal:https://example.com/source.pdf",
    );
  });

  it("opens the modal with the internal doc_id", async () => {
    render(
      <ProofsDialog
        isOpen
        card={buildCard({
          proofs: [
            {
              doc_id: "local_file.pdf",
              page: null,
              start: 0,
              end: 0,
              text: "Source text",
              score: 0.6,
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));
    expect(await screen.findByTestId("pdf-modal")).toHaveTextContent(
      "modal:local_file.pdf",
    );
  });

  it("closes when backdrop is clicked", () => {
    const onClose = vi.fn();
    render(
      <ProofsDialog
        isOpen
        card={buildCard()}
        userId="guest"
        onClose={onClose}
      />,
    );
    fireEvent.click(screen.getByRole("presentation"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("allows multiple proof accordion items to stay open", () => {
    render(
      <ProofsDialog
        isOpen
        card={buildCard({
          proofs: [
            {
              doc_id: "first.pdf",
              page: 1,
              start: 10,
              end: 20,
              text: "First proof text",
              score: 0.8,
            },
            {
              doc_id: "second.pdf",
              page: 2,
              start: 30,
              end: 60,
              text: "Second proof text",
              score: 0.85,
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /source 2/i }));

    expect(screen.getByText("First proof text")).toBeInTheDocument();
    expect(screen.getByText("Second proof text")).toBeInTheDocument();
  });

  it("closing the dialog dismisses the PDF modal (FIX 2)", async () => {
    render(
      <ProofsDialog isOpen card={buildCard()} userId="guest" onClose={() => {}} />,
    );

    // Open accordion and launch the PDF modal
    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));
    expect(await screen.findByTestId("pdf-modal")).toBeInTheDocument();

    // Close the evidence dialog via the Close button
    fireEvent.click(screen.getByRole("button", { name: /^close$/i }));

    // The PDF modal must be gone
    expect(screen.queryByTestId("pdf-modal")).toBeNull();
  });

  it("opens TXT sources in the text modal", async () => {
    render(
      <ProofsDialog
        isOpen
        card={buildCard({
          proofs: [
            {
              doc_id: "notes.txt",
              page: null,
              start: 0,
              end: 0,
              text: "Some notes",
              score: 0.5,
              is_pdf: false,
              is_txt: true,
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));

    expect(await screen.findByTestId("text-modal")).toHaveTextContent("modal:notes.txt");
  });

  it("downloads DOCX sources instead of opening a modal", async () => {
    const blob = new Blob(["docx bytes"]);
    fetchSourceBlobMock.mockResolvedValue(blob);
    // URL.createObjectURL/revokeObjectURL don't exist in jsdom by default, so
    // they're assigned directly rather than spied on or stubbed as a whole
    // (spreading `URL` would lose its constructor, breaking `new URL(...)`
    // inside buildSourceUrl).
    const createObjectURL = vi.fn().mockReturnValue("blob:fake-url");
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    const clickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreateElement(tag);
      if (tag === "a") el.click = clickSpy;
      return el;
    });

    render(
      <ProofsDialog
        isOpen
        card={buildCard({
          proofs: [
            {
              doc_id: "notes.docx",
              page: null,
              start: 0,
              end: 0,
              text: "Some notes",
              score: 0.5,
              is_pdf: false,
              is_txt: false,
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));

    await waitFor(() => expect(clickSpy).toHaveBeenCalledTimes(1));
    expect(fetchSourceBlobMock).toHaveBeenCalledWith(
      expect.stringContaining("/documents/notes.docx/source?exam_id=exam-1&user_id=guest"),
    );
    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:fake-url");

    expect(screen.queryByTestId("pdf-modal")).toBeNull();
    expect(screen.queryByTestId("text-modal")).toBeNull();

    vi.restoreAllMocks();
    delete (URL as { createObjectURL?: unknown }).createObjectURL;
    delete (URL as { revokeObjectURL?: unknown }).revokeObjectURL;
  });

  it("shows an inline error when the DOCX download fails", async () => {
    fetchSourceBlobMock.mockRejectedValue(new Error("Failed to fetch source document (401)"));

    render(
      <ProofsDialog
        isOpen
        card={buildCard({
          proofs: [
            {
              doc_id: "notes.docx",
              page: null,
              start: 0,
              end: 0,
              text: "Some notes",
              score: 0.5,
              is_pdf: false,
              is_txt: false,
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));

    expect(await screen.findByText("Could not download the source.")).toBeInTheDocument();
  });
});
