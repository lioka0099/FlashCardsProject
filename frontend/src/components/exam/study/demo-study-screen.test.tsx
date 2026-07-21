import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { DemoStudyScreen } from "@/components/exam/study/demo-study-screen";

// next/link renders an <a>; stub it so the component renders in isolation.
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

describe("DemoStudyScreen", () => {
  it("renders a sample card question and the tour targets", () => {
    render(<DemoStudyScreen />);
    expect(screen.getAllByText(/photosynthesis/i).length).toBeGreaterThan(0);
    expect(document.querySelector('[data-tour="flashcard"]')).not.toBeNull();
    expect(document.querySelector('[data-tour="sources"]')).not.toBeNull();
    expect(document.querySelector('[data-tour="progress"]')).not.toBeNull();
  });

  it("reveals the answer when the flip button is clicked", async () => {
    render(<DemoStudyScreen />);
    fireEvent.click(screen.getByRole("button", { name: /flip card/i }));
    // FlashcardPlayer flips via an async framer-motion animation, so the
    // answer face mounts a beat after the click — wait for it.
    await waitFor(() => expect(screen.getByText(/carbon dioxide/i)).toBeInTheDocument());
    // Rating controls (a tour target) appear on a rateable card.
    expect(document.querySelector('[data-tour="rating"]')).not.toBeNull();
  });
});
