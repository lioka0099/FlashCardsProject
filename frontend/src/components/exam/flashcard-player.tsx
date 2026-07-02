"use client";

import { useEffect, useState } from "react";
import { motion, useAnimationControls } from "framer-motion";
import { ArrowLeft, ArrowRight, Brain, Quote, RotateCcw, Sparkles } from "lucide-react";
import type { Card } from "@/lib/api/client";
import { RatingControls } from "@/components/exam/rating-controls";
import type { ReviewRating } from "@/lib/api/client";
import "./study.css";

type FlashcardPlayerProps = {
  card: Card | null;
  isAnswerVisible: boolean;
  canRateCurrentCard: boolean;
  selectedRating: ReviewRating | null;
  isRatingPending: boolean;
  isPreparingNextCard: boolean;
  isNextEnabled: boolean;
  statusMessage: string | null;
  onToggleAnswer: () => void;
  onShowProofs: (card: Card) => void;
  onRate: (rating: ReviewRating) => void;
  onLoadPrevious: () => void;
  onLoadNext: () => void;
  isPreviousEnabled: boolean;
};

export function FlashcardPlayer({
  card,
  isAnswerVisible,
  canRateCurrentCard,
  selectedRating,
  isRatingPending,
  isPreparingNextCard,
  isNextEnabled,
  statusMessage,
  onToggleAnswer,
  onShowProofs,
  onRate,
  onLoadPrevious,
  onLoadNext,
  isPreviousEnabled,
}: FlashcardPlayerProps) {
  const [displayedAnswerVisible, setDisplayedAnswerVisible] = useState(isAnswerVisible);
  const cardControls = useAnimationControls();

  useEffect(() => {
    let isCancelled = false;

    async function runFlip() {
      if (displayedAnswerVisible === isAnswerVisible) {
        await cardControls.start({
          rotateY: 0,
          transition: { duration: 0.12, ease: "easeOut" },
        });
        return;
      }

      const direction = isAnswerVisible ? 1 : -1;
      await cardControls.start({
        rotateY: direction * 90,
        scale: 0.985,
        transition: { duration: 0.18, ease: "easeIn" },
      });
      if (isCancelled) {
        return;
      }
      setDisplayedAnswerVisible(isAnswerVisible);
      cardControls.set({ rotateY: direction * -90 });
      await cardControls.start({
        rotateY: 0,
        scale: 1,
        transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] },
      });
    }

    void runFlip();

    return () => {
      isCancelled = true;
    };
  }, [cardControls, displayedAnswerVisible, isAnswerVisible]);

  if (!card) {
    return (
      <motion.section
        className="stage stage--empty"
        aria-live="polite"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
      >
        <h1 className="stage__empty-title">No cards on stage yet</h1>
        <p className="stage__status">
          {statusMessage ?? "This deck is still warming up its vocabulary."}
        </p>
        <button
          className="card__flip"
          type="button"
          onClick={onLoadNext}
          disabled={isPreparingNextCard}
        >
          {isPreparingNextCard ? "Preparing a clever card..." : "Next card"}
        </button>
      </motion.section>
    );
  }

  const cardType = typeof card.info?.card_type === "string" ? card.info.card_type : null;
  const isDiagnostic = cardType === "diagnostic";
  const pillLabel = cardType ? cardType.charAt(0).toUpperCase() + cardType.slice(1) : "Concept";

  return (
    <motion.section
      className="stage"
      aria-label="Flashcard player"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.38, ease: "easeOut" }}
    >
      <div className="stage__cardrow">
        <button
          className="stage__arrow stage__arrow--prev"
          type="button"
          onClick={onLoadPrevious}
          disabled={!isPreviousEnabled}
        >
          <span className="stage__arrow-icon"><ArrowLeft size={20} /></span>
          Previous
        </button>

        <div className="card-shell">
          <motion.article
            animate={cardControls}
            className={`card-face${displayedAnswerVisible ? " card-face--back" : " card-face--front"}`}
            role="button"
            tabIndex={0}
            initial={false}
            style={{
              backfaceVisibility: "hidden",
              transformOrigin: "center center",
              transformPerspective: 1200,
            }}
            onClick={(event) => {
              if ((event.target as HTMLElement).closest("[data-no-flip='true']")) {
                return;
              }
              onToggleAnswer();
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onToggleAnswer();
              }
            }}
            aria-label={isAnswerVisible ? "Show question side" : "Show answer side"}
          >
            {!displayedAnswerVisible ? (
              <>
                <div className="card__meta">
                  <span className="card__pill"><Brain size={16} aria-hidden="true" /> {pillLabel}</span>
                  {!isDiagnostic ? (
                    <span className="card__pill card__pill--difficulty">Difficulty {card.difficulty}</span>
                  ) : null}
                </div>
                {card.topic_label ? <p className="card__topic">{card.topic_label}</p> : null}
                <p className="card__q">{card.question}</p>
                <hr className="card__divider" />
                <p className="card__hint">
                  <Sparkles size={18} aria-hidden="true" /> Try to answer before flipping the card
                </p>
                <button
                  className="card__flip"
                  type="button"
                  data-no-flip="true"
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggleAnswer();
                  }}
                >
                  <RotateCcw size={18} aria-hidden="true" /> Flip Card
                </button>
              </>
            ) : (
              <>
                <div className="card__meta">
                  <span className="card__pill"><Brain size={16} aria-hidden="true" /> Answer</span>
                  {!isDiagnostic ? (
                    <span className="card__pill card__pill--difficulty">Difficulty {card.difficulty}</span>
                  ) : null}
                </div>
                {card.topic_label ? <p className="card__topic">{card.topic_label}</p> : null}
                <div className="card__a">{card.answer}</div>
                <hr className="card__divider" />
                <button
                  className="card__receipts"
                  type="button"
                  data-no-flip="true"
                  onClick={(event) => {
                    event.stopPropagation();
                    onShowProofs(card);
                  }}
                >
                  <Quote size={18} aria-hidden="true" /> Show evidence
                </button>
                <button
                  className="card__flip"
                  type="button"
                  data-no-flip="true"
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggleAnswer();
                  }}
                >
                  <RotateCcw size={18} aria-hidden="true" /> Back to the question
                </button>
              </>
            )}
          </motion.article>
        </div>

        <button
          className="stage__arrow stage__arrow--next"
          type="button"
          onClick={onLoadNext}
          disabled={!isNextEnabled || isRatingPending || isPreparingNextCard}
        >
          <span className="stage__arrow-icon"><ArrowRight size={20} /></span>
          {isPreparingNextCard ? "Preparing..." : "Next"}
        </button>
      </div>

      {canRateCurrentCard || selectedRating ? (
        <RatingControls
          canRateCurrentCard={canRateCurrentCard}
          selectedRating={selectedRating}
          isPending={isRatingPending || isPreparingNextCard}
          onRate={onRate}
        />
      ) : null}
      {!canRateCurrentCard && !selectedRating ? (
        <p className="stage__readonly">
          This read-only card is a museum piece today, so ratings are closed.
        </p>
      ) : null}

      {statusMessage ? <p className="stage__status">{statusMessage}</p> : null}
      {isPreparingNextCard ? (
        <p className="stage__status" aria-live="polite">
          Preparing a card that matches your current level. Tiny academic chef is plating.
        </p>
      ) : null}
    </motion.section>
  );
}
