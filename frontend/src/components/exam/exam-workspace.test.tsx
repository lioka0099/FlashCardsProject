import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExamWorkspace } from "@/components/exam/exam-workspace";
import type { Card, ReviewRating } from "@/lib/api/client";

const getExamByIdMock = vi.fn();
const getSessionNextCardMock = vi.fn();
const getPresentedHistoryMock = vi.fn();
const submitCardReviewMock = vi.fn();
const logSessionEventMock = vi.fn();
const replaceMock = vi.fn();

vi.mock("@/lib/api/client", () => ({
  getExamById: (...args: unknown[]) => getExamByIdMock(...args),
  getSessionNextCard: (...args: unknown[]) => getSessionNextCardMock(...args),
  getPresentedHistory: (...args: unknown[]) => getPresentedHistoryMock(...args),
  submitCardReview: (...args: unknown[]) => submitCardReviewMock(...args),
  logSessionEvent: (...args: unknown[]) => logSessionEventMock(...args),
  assertActiveRateableCard: (cardId: string, activeCardId: string | null) => {
    if (!activeCardId || cardId !== activeCardId) {
      throw new Error("Only the active session card can be rated.");
    }
  },
}));

vi.mock("@/components/exam/progress-panel", () => ({
  ProgressPanel: () => null,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: replaceMock, push: vi.fn() }),
}));

vi.mock("@/components/exam/proofs-dialog", () => ({
  ProofsDialog: () => null,
}));

function buildCard(id: string, question: string): Card {
  return {
    card_id: id,
    exam_id: "exam-1",
    topic_id: "topic-1",
    topic_label: "Networks",
    question,
    answer: `${question} answer`,
    difficulty: 2,
    created_at: "2026-01-01T00:00:00Z",
    status: "active",
    proofs: [],
    info: {},
  };
}

function nextCardResponse(card: Card | null) {
  return {
    card,
    reason: card ? "review" : null,
    no_cards_available: !card,
    message: null,
  };
}

function renderWorkspace() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ExamWorkspace examId="exam-1" />
    </QueryClientProvider>,
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

describe("ExamWorkspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    replaceMock.mockReset();
    getPresentedHistoryMock.mockResolvedValue([]);
  });

  it("does not allow previous on first loaded card", async () => {
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Test Exam",
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
    getSessionNextCardMock.mockResolvedValueOnce(nextCardResponse(buildCard("c1", "Question 1")));
    logSessionEventMock.mockResolvedValue({ event_id: "evt-1" });

    renderWorkspace();

    await screen.findByText("Question 1");
    const previousButton = screen.getAllByRole("button", { name: "Previous" })[0];
    expect(previousButton).toBeDisabled();
    fireEvent.click(previousButton);

    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(getSessionNextCardMock).toHaveBeenCalledTimes(1);
  });

  it("hydrates previous navigation from persisted presented history", async () => {
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Test Exam",
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
    getSessionNextCardMock.mockResolvedValueOnce(nextCardResponse(buildCard("c3", "Question 3")));
    getPresentedHistoryMock.mockResolvedValueOnce([
      buildCard("c3", "Question 3"),
      buildCard("c2", "Question 2"),
      buildCard("c1", "Question 1"),
    ]);
    logSessionEventMock.mockResolvedValue({ event_id: "evt-1" });

    renderWorkspace();

    await screen.findByText("Question 3");
    const previousButton = screen.getAllByRole("button", { name: "Previous" })[0];
    await waitFor(() => expect(previousButton).toBeEnabled());

    fireEvent.click(previousButton);
    await screen.findByText("Question 2");

    expect(getPresentedHistoryMock).toHaveBeenCalledWith("exam-1", "guest");
  });

  it("keeps rating visible on previous card and reuses local forward history", async () => {
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Test Exam",
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
    getSessionNextCardMock
      .mockResolvedValueOnce(nextCardResponse(buildCard("c1", "Question 1")))
      .mockResolvedValueOnce(nextCardResponse(buildCard("c2", "Question 2")));
    submitCardReviewMock.mockImplementation(
      async (_examId: string, cardId: string, _userId: string, rating: ReviewRating) => ({
        review_id: `review-${cardId}`,
        card_id: cardId,
        rating,
        due_at: null,
        interval_days: 1,
        ease: 2.5,
        topic_proficiency: 0.5,
        diagnostic_answered: 0,
        diagnostic_total: 0,
        exam_state: "active_learning",
        idempotent_replay: false,
      }),
    );
    logSessionEventMock.mockResolvedValue({ event_id: "evt-1" });

    renderWorkspace();

    await screen.findByText("Question 1");
    fireEvent.click(screen.getByRole("button", { name: /I knew it/i }));
    await waitFor(() => expect(submitCardReviewMock).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getAllByRole("button", { name: "Next" })[0]);
    await screen.findByText("Question 2");

    fireEvent.click(screen.getAllByRole("button", { name: "Previous" })[0]);
    await screen.findByText("Question 1");

    expect(screen.getByRole("button", { name: /I knew it/i })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /Almost knew it/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Next" })[0]);
    await screen.findByText("Question 2");
    expect(getSessionNextCardMock).toHaveBeenCalledTimes(2);
  });

  it("keeps current card visible while preparing the next card", async () => {
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Test Exam",
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
    const pendingNext = deferred<ReturnType<typeof nextCardResponse>>();
    getSessionNextCardMock
      .mockResolvedValueOnce(nextCardResponse(buildCard("c1", "Question 1")))
      .mockReturnValueOnce(pendingNext.promise);
    submitCardReviewMock.mockResolvedValue({
      review_id: "review-c1",
      card_id: "c1",
      rating: "i_knew_it",
      due_at: null,
      interval_days: 1,
      ease: 2.5,
      topic_proficiency: 0.5,
      diagnostic_answered: 0,
      diagnostic_total: 0,
      exam_state: "active_learning",
      idempotent_replay: false,
    });
    logSessionEventMock.mockResolvedValue({ event_id: "evt-1" });

    renderWorkspace();

    await screen.findByText("Question 1");
    fireEvent.click(screen.getByRole("button", { name: /I knew it/i }));
    await waitFor(() => expect(submitCardReviewMock).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getAllByRole("button", { name: "Next" })[0]);

    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Preparing..." })).toBeDisabled();
    expect(screen.getByText(/Crafting your next card/i)).toBeInTheDocument();

    pendingNext.resolve(nextCardResponse(buildCard("c2", "Question 2")));
    await screen.findByText("Question 2");
  });

  it("redirects to creating-test when exam state is processing", async () => {
    getExamByIdMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      title: "Test Exam",
      mode: "mastery",
      state: "processing",
      diagnostic_total: 0,
      diagnostic_answered: 0,
      diagnostic_started_at: null,
      diagnostic_completed_at: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      info: {},
    });

    renderWorkspace();

    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/exams/exam-1/creating"));
  });
});
