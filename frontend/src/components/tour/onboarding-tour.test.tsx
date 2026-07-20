import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";

const push = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  usePathname: () => "/",
}));

const getMe = vi.fn();
const markOnboarded = vi.fn();
vi.mock("@/lib/api/auth", () => ({
  getMe: () => getMe(),
  markOnboarded: () => markOnboarded(),
}));

import { OnboardingTourProvider } from "@/components/tour/onboarding-tour";

function renderWithClient() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <OnboardingTourProvider>
        <div id="acct" data-tour="account">Account</div>
        <div id="mt" data-tour="my-tests">Tests</div>
        <div id="dn" data-tour="deck-name">Deck</div>
        <div id="up" data-tour="upload">Upload</div>
      </OnboardingTourProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  Element.prototype.getBoundingClientRect = vi.fn(() => ({
    width: 100, height: 40, top: 100, left: 100, right: 200, bottom: 140, x: 100, y: 100,
    toJSON: () => {},
  })) as unknown as typeof Element.prototype.getBoundingClientRect;
});

describe("OnboardingTourProvider", () => {
  it("does not auto-start when the account is already onboarded", async () => {
    getMe.mockResolvedValue({ user_id: "u", email: null, name: null, onboarded: true });
    renderWithClient();
    // Give the query a tick to resolve, then assert no tour dialog.
    await waitFor(() => expect(getMe).toHaveBeenCalled());
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("auto-starts for a not-yet-onboarded account on the dashboard", async () => {
    getMe.mockResolvedValue({ user_id: "u", email: null, name: null, onboarded: false });
    renderWithClient();
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(await screen.findByText("Your decks")).toBeInTheDocument();
  });

  it("marks onboarded when the tour is skipped", async () => {
    getMe.mockResolvedValue({ user_id: "u", email: null, name: null, onboarded: false });
    markOnboarded.mockResolvedValue({ user_id: "u", email: null, name: null, onboarded: true });
    renderWithClient();
    (await screen.findByRole("button", { name: "Skip" })).click();
    await waitFor(() => expect(markOnboarded).toHaveBeenCalled());
  });

  it("stays closed after skip even when the onboarded write fails", async () => {
    // If markOnboarded rejects, `me.onboarded` never flips to true — so the
    // auto-start effect would re-open the tour forever without a session guard.
    getMe.mockResolvedValue({ user_id: "u", email: null, name: null, onboarded: false });
    markOnboarded.mockRejectedValue(new Error("network"));
    renderWithClient();
    (await screen.findByRole("button", { name: "Skip" })).click();
    await waitFor(() => expect(markOnboarded).toHaveBeenCalled());
    // Give the auto-start effect several ticks to (not) re-fire, then assert closed.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
