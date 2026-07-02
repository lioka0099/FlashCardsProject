import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProgressPanel } from "@/components/exam/progress-panel";

const getExamProgressMock = vi.fn();

vi.mock("@/lib/api/client", () => ({
  getExamProgress: (...args: unknown[]) => getExamProgressMock(...args),
}));

function renderPanel() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ProgressPanel examId="exam-1" userId="guest" />
    </QueryClientProvider>,
  );
}

describe("ProgressPanel", () => {
  it("renders progress summary and topic list", async () => {
    getExamProgressMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      overall_proficiency: 0.63,
      topics: [
        {
          topic_id: "topic-1",
          topic_label: "Transport Layer",
          proficiency: 0.7,
          last_updated_at: "2026-01-01T00:00:00Z",
          n_reviews: 4,
        },
      ],
    });

    renderPanel();

    await waitFor(() => {
      expect(screen.getByText("Your Progress")).toBeInTheDocument();
      expect(screen.getByText("63%")).toBeInTheDocument();
      expect(screen.getByText("Transport Layer")).toBeInTheDocument();
      expect(screen.getByText("70%")).toBeInTheDocument();
    });
  });

  it("renders empty state when there is no topic progress", async () => {
    getExamProgressMock.mockResolvedValue({
      exam_id: "exam-1",
      user_id: "guest",
      overall_proficiency: null,
      topics: [],
    });

    renderPanel();

    await waitFor(() => {
      expect(screen.getByText(/No topic progress yet/i)).toBeInTheDocument();
    });
  });

  it("renders error state when api call fails", async () => {
    getExamProgressMock.mockRejectedValue(new Error("bad network"));

    renderPanel();

    await waitFor(
      () => {
        expect(screen.getByText(/Could not load progress/i)).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });
});
