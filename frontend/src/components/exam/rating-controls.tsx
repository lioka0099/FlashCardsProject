"use client";

import { Frown, Laugh, Meh, Smile, type LucideIcon } from "lucide-react";
import type { ReviewRating } from "@/lib/api/client";

type RatingOption = {
  value: ReviewRating;
  title: string;
  tone: "green" | "yellow" | "blue" | "red";
  Icon: LucideIcon;
};

const RATING_OPTIONS: RatingOption[] = [
  { value: "i_knew_it", title: "I knew it", tone: "green", Icon: Laugh },
  { value: "almost_knew", title: "Almost knew it", tone: "yellow", Icon: Smile },
  { value: "learned_now", title: "Just learned it", tone: "blue", Icon: Meh },
  { value: "dont_understand", title: "Didn't get it", tone: "red", Icon: Frown },
];

type RatingControlsProps = {
  canRateCurrentCard: boolean;
  selectedRating: ReviewRating | null;
  isPending: boolean;
  onRate: (rating: ReviewRating) => void;
};

export function RatingControls({
  canRateCurrentCard,
  selectedRating,
  isPending,
  onRate,
}: RatingControlsProps) {
  function renderCard(option: RatingOption, isSelected: boolean) {
    const isDisabled = isSelected || !canRateCurrentCard || isPending;
    return (
      <button
        key={option.value}
        className={`rate__card rate__card--${option.tone}${isSelected ? " rate__card--selected" : ""}`}
        type="button"
        onClick={() => onRate(option.value)}
        disabled={isDisabled}
      >
        <span className="rate__face" aria-hidden="true">
          <option.Icon size={22} />
        </span>
        <span className="rate__title">{option.title}</span>
      </button>
    );
  }

  const selectedOption = selectedRating
    ? (RATING_OPTIONS.find((option) => option.value === selectedRating) ?? null)
    : null;

  return (
    <div className="rate-wrap">
      {!selectedOption ? <p className="rate-prompt">How well did you know this?</p> : null}
      <div
        className={`rate${selectedOption ? " rate--single" : ""}`}
        role="group"
        aria-label="Rate how well you knew this card"
      >
        {selectedOption
          ? renderCard(selectedOption, true)
          : RATING_OPTIONS.map((option) => renderCard(option, false))}
      </div>
    </div>
  );
}
