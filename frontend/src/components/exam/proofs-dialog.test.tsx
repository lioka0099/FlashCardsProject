import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProofsDialog } from "@/components/exam/proofs-dialog";
import type { Card } from "@/lib/api/client";

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
  it("renders source metadata and jump link", () => {
    render(
      <ProofsDialog
        isOpen
        card={buildCard()}
        userId="guest"
        onClose={() => {}}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: /evidence and source context/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /source 1/i }));
    expect(
      screen.getByText("Page: 3 • Span: 12-46"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /open the source/i }),
    ).toHaveAttribute(
      "href",
      expect.stringContaining("https://example.com/source.pdf"),
    );
  });

  it("builds an API source link when doc_id is internal", () => {
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
    expect(
      screen.getByRole("link", { name: /open the source/i }),
    ).toHaveAttribute(
      "href",
      expect.stringContaining(
        "/documents/local_file.pdf/source?exam_id=exam-1&user_id=guest",
      ),
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
});
