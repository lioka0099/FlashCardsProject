import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreatingTestScreen } from "@/components/exam/creating-test-screen";

const replaceMock = vi.fn();
const getExamByIdMock = vi.fn();

vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: replaceMock, push: vi.fn() }) }));
vi.mock("@/lib/api/client", async (orig) => ({
  ...(await orig<typeof import("@/lib/api/client")>()),
  getExamById: (...a: unknown[]) => getExamByIdMock(...a),
}));
vi.mock("@/lib/session/guest-session", () => ({ useGuestSession: () => ({ userId: "guest", mode: "guest" }) }));

function renderScreen() {
  const qc = new QueryClient();
  return render(<QueryClientProvider client={qc}><CreatingTestScreen examId="ex-1" /></QueryClientProvider>);
}

const progress = { updated_at: "x", steps: [
  { key: "uploading", label: "Uploading document", detail: "d", status: "active" as const },
] };

describe("CreatingTestScreen", () => {
  beforeEach(() => { replaceMock.mockReset(); getExamByIdMock.mockReset(); });

  it("redirects to the exam when ready", async () => {
    getExamByIdMock.mockResolvedValue({ state: "diagnostic", title: "T", info: { progress, filenames: ["a.pdf"] } });
    renderScreen();
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/exams/ex-1"));
  });

  it("shows an error view when failed", async () => {
    getExamByIdMock.mockResolvedValue({ state: "failed", title: "T", info: { bootstrap_error: "API request failed (422): {\"error\":\"x\"}" } });
    renderScreen();
    await waitFor(() => expect(screen.getByRole("button", { name: /back home/i })).toBeInTheDocument());
  });
});
