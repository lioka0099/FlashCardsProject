"use client";

import { useState, type CSSProperties } from "react";
import Link from "next/link";
import { FileText, Home } from "lucide-react";
import type { Card, ReviewRating } from "@/lib/api/client";
import { FlashcardPlayer } from "@/components/exam/study/flashcard-player";
import "@/components/exam/study/study.css";

// Hardcoded sample deck. Purely local — no API calls, no history writes. It
// exists so the onboarding tour can show real study controls before the user
// has created any deck of their own.
const DEMO_CARDS: Card[] = [
  {
    card_id: "demo-1",
    exam_id: "demo",
    topic_id: "demo-topic",
    topic_label: "Photosynthesis",
    question: "What gas do plants absorb from the air during photosynthesis?",
    answer: "Carbon dioxide (CO₂).",
    difficulty: 1,
    created_at: "2026-01-01T00:00:00Z",
    status: "active",
    proofs: [],
    info: { card_type: "concept" },
  },
  {
    card_id: "demo-2",
    exam_id: "demo",
    topic_id: "demo-topic",
    topic_label: "Photosynthesis",
    question: "Where in the plant cell does photosynthesis mainly happen?",
    answer: "In the chloroplasts.",
    difficulty: 1,
    created_at: "2026-01-01T00:00:00Z",
    status: "active",
    proofs: [],
    info: { card_type: "concept" },
  },
];

const progressStyle: CSSProperties = {
  alignSelf: "center",
  margin: "0.5rem 0 1rem",
  padding: "0.4rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid var(--line, rgba(148,163,184,0.3))",
  fontSize: "0.85rem",
  fontWeight: 600,
};

export function DemoStudyScreen() {
  const [index, setIndex] = useState(0);
  const [isAnswerVisible, setIsAnswerVisible] = useState(false);
  const [ratings, setRatings] = useState<Record<string, ReviewRating>>({});

  const card = DEMO_CARDS[index];
  const rating = ratings[card.card_id] ?? null;

  return (
    <div className="study">
      <header className="study__topbar">
        <Link className="study__back" href="/">
          <Home size={18} aria-hidden="true" />
          Back to Home
        </Link>
        <div className="study__deck">
          <h1 className="study__deck-title">Sample deck</h1>
          <p className="study__deck-sub">{DEMO_CARDS.length} cards</p>
        </div>
        <div className="study__sources" data-tour="sources">
          <button type="button" className="study__sources-trigger" aria-haspopup="menu">
            <FileText size={16} aria-hidden="true" />
            <span>1 file uploaded</span>
          </button>
        </div>
      </header>

      <div className="study__body">
        <div data-tour="progress" style={progressStyle}>
          Progress: {index + 1} / {DEMO_CARDS.length}
        </div>

        <FlashcardPlayer
          card={card}
          isAnswerVisible={isAnswerVisible}
          canRateCurrentCard={rating === null}
          selectedRating={rating}
          isRatingPending={false}
          isPreparingNextCard={false}
          isNextEnabled
          statusMessage={null}
          onToggleAnswer={() => setIsAnswerVisible((v) => !v)}
          onShowProofs={() => {}}
          onRate={(r) => setRatings((prev) => ({ ...prev, [card.card_id]: r }))}
          onLoadPrevious={() => {
            setIndex((i) => Math.max(0, i - 1));
            setIsAnswerVisible(false);
          }}
          onLoadNext={() => {
            setIndex((i) => Math.min(DEMO_CARDS.length - 1, i + 1));
            setIsAnswerVisible(false);
          }}
          isPreviousEnabled={index > 0}
        />
      </div>
    </div>
  );
}
