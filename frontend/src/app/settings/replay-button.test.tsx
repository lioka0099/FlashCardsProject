import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ReplayWalkthroughCard } from "@/app/settings/replay-button";

const startTour = vi.fn();
vi.mock("@/components/tour/onboarding-tour", () => ({
  useTour: () => ({ startTour }),
}));

describe("ReplayWalkthroughCard", () => {
  it("starts the tour when clicked", () => {
    render(<ReplayWalkthroughCard />);
    fireEvent.click(screen.getByRole("button", { name: /replay walkthrough/i }));
    expect(startTour).toHaveBeenCalledOnce();
  });
});
