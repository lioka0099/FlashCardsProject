import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RatingControls } from "@/components/exam/rating-controls";

describe("RatingControls", () => {
  it("renders all four rating buttons", () => {
    render(
      <RatingControls
        canRateCurrentCard
        selectedRating={null}
        isPending={false}
        onRate={() => {}}
      />,
    );

    expect(screen.getByRole("button", { name: /I knew it/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Almost knew it/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Just learned it/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Didn't get it/i })).toBeInTheDocument();
  });

  it("calls onRate with the selected rating", () => {
    const onRate = vi.fn();
    render(
      <RatingControls
        canRateCurrentCard
        selectedRating={null}
        isPending={false}
        onRate={onRate}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Almost knew it/i }));
    expect(onRate).toHaveBeenCalledWith("almost_knew");
  });

  it("disables all buttons when card is not rateable", () => {
    render(
      <RatingControls
        canRateCurrentCard={false}
        selectedRating={null}
        isPending={false}
        onRate={() => {}}
      />,
    );
    const buttons = screen.getAllByRole("button");
    expect(buttons.every((button) => button.hasAttribute("disabled"))).toBe(true);
  });

  it("shows only selected bubble when rating is locked", () => {
    render(
      <RatingControls
        canRateCurrentCard
        selectedRating="learned_now"
        isPending={false}
        onRate={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /Just learned it/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /I knew it/i })).not.toBeInTheDocument();
  });
});
