import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FlashcardPlayer } from "@/components/exam/flashcard-player";
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
    proofs: [],
    info: {},
    ...overrides,
  };
}

describe("FlashcardPlayer", () => {
  it("shows question side first with the type pill and flip hint", () => {
    render(
      <FlashcardPlayer
        card={buildCard()}
        isAnswerVisible={false}
        canRateCurrentCard
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled={false}
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    expect(screen.getByText("What does TCP guarantee?")).toBeInTheDocument();
    expect(screen.getByText("Try to answer before flipping the card")).toBeInTheDocument();
    expect(screen.getByText("Networks")).toBeInTheDocument();
    expect(screen.getByText("Difficulty 2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Flip Card/i })).toBeInTheDocument();
  });

  it("shows answer and proofs action when flipped", async () => {
    render(
      <FlashcardPlayer
        card={buildCard()}
        isAnswerVisible
        canRateCurrentCard
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled={false}
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    expect(screen.getByText("Answer")).toBeInTheDocument();
    expect(screen.getByText("Reliable ordered delivery.")).toBeInTheDocument();
    expect(screen.getByText("Back to the question")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show evidence" })).toBeInTheDocument();
  });

  it("opens proofs without flipping when tapping view proofs", () => {
    const onToggleAnswer = vi.fn();
    const onShowProofs = vi.fn();
    const card = buildCard();
    render(
      <FlashcardPlayer
        card={card}
        isAnswerVisible
        canRateCurrentCard
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled={false}
        statusMessage={null}
        onToggleAnswer={onToggleAnswer}
        onShowProofs={onShowProofs}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Show evidence" }));
    expect(onShowProofs).toHaveBeenCalledWith(card);
    expect(onToggleAnswer).not.toHaveBeenCalled();
  });

  it("renders read-only hint when current card is not rateable", () => {
    render(
      <FlashcardPlayer
        card={buildCard()}
        isAnswerVisible
        canRateCurrentCard={false}
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    expect(screen.getByText(/read-only/i)).toBeInTheDocument();
  });

  it("triggers rating callback from controls", () => {
    const onRate = vi.fn();
    render(
      <FlashcardPlayer
        card={buildCard()}
        isAnswerVisible
        canRateCurrentCard
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled={false}
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={onRate}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /I knew it/i }));
    expect(onRate).toHaveBeenCalledWith("i_knew_it");
  });

  it("shows the card type in the pill for diagnostic cards", () => {
    render(
      <FlashcardPlayer
        card={buildCard({ info: { card_type: "diagnostic" } })}
        isAnswerVisible={false}
        canRateCurrentCard
        selectedRating={null}
        isRatingPending={false}
        isPreparingNextCard={false}
        isNextEnabled={false}
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    expect(screen.getByText("Diagnostic")).toBeInTheDocument();
  });

  it("shows next-card preparation feedback while pending", () => {
    render(
      <FlashcardPlayer
        card={buildCard()}
        isAnswerVisible={false}
        canRateCurrentCard
        selectedRating="i_knew_it"
        isRatingPending={false}
        isPreparingNextCard
        isNextEnabled
        statusMessage={null}
        onToggleAnswer={() => {}}
        onShowProofs={() => {}}
        onRate={() => {}}
        onLoadPrevious={() => {}}
        onLoadNext={() => {}}
        isPreviousEnabled
      />,
    );

    expect(screen.getByRole("button", { name: "Preparing..." })).toBeDisabled();
    expect(screen.getByText(/Preparing a card that matches your current level/i)).toBeInTheDocument();
  });
});
