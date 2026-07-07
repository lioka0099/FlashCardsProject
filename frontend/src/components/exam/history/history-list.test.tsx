import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HistoryList } from "@/components/exam/history/history-list";
import type { Card } from "@/lib/api/client";

const getExamByIdMock = vi.fn();
const getPresentedHistoryMock = vi.fn();

vi.mock("@/lib/api/client", () => ({
  getExamById: (...args: unknown[]) => getExamByIdMock(...args),
  getPresentedHistory: (...args: unknown[]) => getPresentedHistoryMock(...args),
}));

function renderHistory() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <HistoryList examId="exam-1" />
    </QueryClientProvider>,
  );
}

function buildCard(id: string, question: string, overrides?: Partial<Card>): Card {
  return {
    card_id: id,
    exam_id: "exam-1",
    topic_id: "topic-1",
    topic_label: "Networking",
    question,
    answer: `${question} answer`,
    difficulty: 2,
    created_at: "2026-01-01T00:00:00Z",
    status: "active",
    proofs: [],
    info: {},
    ...overrides,
  };
}

describe("HistoryList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Exam A",
      mode: "mastery",
      state: "active_learning",
      diagnostic_total: 0,
      diagnostic_answered: 0,
      diagnostic_started_at: null,
      diagnostic_completed_at: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      info: {},
    });
  });

  it("shows cards newest first and selects newest by default", async () => {
    getPresentedHistoryMock.mockResolvedValue([
      buildCard("c3", "Question 3"),
      buildCard("c2", "Question 2"),
      buildCard("c1", "Question 1"),
    ]);

    renderHistory();

    await screen.findByRole("button", { name: /question 3/i });
    const list = screen.getByRole("list", { name: "Presented cards list" });
    const historyButtons = within(list).getAllByRole("button");
    expect(historyButtons[0]).toHaveTextContent("Question 3");
    expect(screen.getByRole("button", { name: /question 3/i })).toHaveAttribute("aria-current", "true");
  });

  it("updates active card when selecting another item", async () => {
    getPresentedHistoryMock.mockResolvedValue([
      buildCard("c3", "Question 3"),
      buildCard("c2", "Question 2"),
    ]);

    renderHistory();

    await screen.findByText("Question 2");
    fireEvent.click(screen.getByRole("button", { name: /question 2/i }));
    expect(screen.getByRole("button", { name: /question 2/i })).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: /show answer side/i })).toBeInTheDocument();
  });

  it("shows rating labels from history card metadata", async () => {
    getPresentedHistoryMock.mockResolvedValue([
      buildCard("c1", "Question 1", { info: { rating: "learned_now" } }),
    ]);

    renderHistory();

    expect(await screen.findAllByText("Just learned it")).not.toHaveLength(0);
  });
});
