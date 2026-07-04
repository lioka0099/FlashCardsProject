import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProofsDialog } from "@/components/exam/proofs-dialog";
import type { Card } from "@/lib/api/client";

vi.mock("@/components/exam/pdf-source-modal", () => ({
  PdfSourceModal: ({ proof }: { proof: { doc_id: string } }) => (
    <div data-testid="pdf-modal">modal:{proof.doc_id}</div>
  ),
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

  it("opens non-PDF sources in a new tab instead of the modal", () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
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
            },
          ],
        })}
        userId="guest"
        onClose={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /open the source/i }));

    expect(screen.queryByTestId("pdf-modal")).toBeNull();
    expect(openSpy).toHaveBeenCalledTimes(1);
    expect(openSpy.mock.calls[0][0]).toContain(
      "/documents/notes.docx/source?exam_id=exam-1&user_id=guest",
    );
    openSpy.mockRestore();
  });
});
