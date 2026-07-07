"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Home } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  assertActiveRateableCard,
  getExamById,
  getPresentedHistory,
  getSessionNextCard,
  logSessionEvent,
  submitCardReview,
  type Card,
  type ReviewRating,
} from "@/lib/api/client";
import { mapApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";
import { InlineError } from "@/components/common/inline-error";
import { FlashcardPlayer } from "@/components/exam/study/flashcard-player";
import { ProofsDialog } from "@/components/exam/study/proofs-dialog";
import { ProgressPanel } from "@/components/exam/study/progress-panel";

type ExamWorkspaceProps = {
  examId: string;
};

function coalesceConsecutiveCards(cards: Card[]) {
  return cards.reduce<Card[]>((history, card) => {
    if (history.at(-1)?.card_id === card.card_id) {
      return history;
    }
    return [...history, card];
  }, []);
}

export function ExamWorkspace({ examId }: ExamWorkspaceProps) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { userId } = useGuestSession();
  const [isAnswerVisible, setIsAnswerVisible] = useState(false);
  const [historyCards, setHistoryCards] = useState<Card[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [activeCardId, setActiveCardId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null | undefined>(undefined);
  const [selectedRatings, setSelectedRatings] = useState<Record<string, ReviewRating>>({});
  const [proofsCard, setProofsCard] = useState<Card | null>(null);

  const sessionQuery = useQuery({
    queryKey: ["session-next-card", examId, userId],
    queryFn: () => getSessionNextCard(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const examQuery = useQuery({
    queryKey: ["exam-by-id", examId, userId],
    queryFn: () => getExamById(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const presentedHistoryQuery = useQuery({
    queryKey: ["presented-history", examId, userId],
    queryFn: () => getPresentedHistory(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (examQuery.data?.state === "processing") {
      router.replace(`/exams/${examId}/creating`);
    }
  }, [examQuery.data?.state, examId, router]);

  const initialCard = (sessionQuery.data?.card as Card | null) ?? null;
  const seededHistoryCards = historyCards.length > 0 ? historyCards : initialCard ? [initialCard] : [];
  const seededHistoryIndex = historyCards.length > 0 ? historyIndex : initialCard ? 0 : -1;
  const currentCard =
    seededHistoryIndex >= 0 && seededHistoryIndex < seededHistoryCards.length
      ? seededHistoryCards[seededHistoryIndex]
      : null;
  const resolvedActiveCardId = activeCardId ?? initialCard?.card_id ?? null;
  const isViewingPreviousCard = Boolean(
    currentCard && resolvedActiveCardId && currentCard.card_id !== resolvedActiveCardId,
  );
  const currentMessage = statusMessage ?? sessionQuery.data?.message ?? null;

  useEffect(() => {
    void logSessionEvent(examId, userId, "session_start", { source: "exam_workspace" });
  }, [examId, userId]);

  useEffect(() => {
    const persistedHistory = presentedHistoryQuery.data ?? [];
    if (historyCards.length > 0 || persistedHistory.length === 0) {
      return;
    }

    const chronologicalHistory = coalesceConsecutiveCards([...persistedHistory].reverse());
    setHistoryCards(chronologicalHistory);
    setHistoryIndex(chronologicalHistory.length - 1);
  }, [historyCards.length, presentedHistoryQuery.data]);

  const nextCardMutation = useMutation({
    mutationFn: () => getSessionNextCard(examId, userId),
    onSuccess: (response) => {
      const nextCard = (response.card as Card | null) ?? null;
      setActiveCardId(nextCard?.card_id ?? null);
      setStatusMessage(response.message);
      setIsAnswerVisible(false);
      if (!nextCard) {
        return;
      }
      setHistoryCards((previous) => {
        const seeded = previous.length > 0 ? previous : initialCard ? [initialCard] : [];
        const lastCard = seeded.at(-1);
        const nextHistory =
          lastCard && lastCard.card_id === nextCard.card_id ? seeded : [...seeded, nextCard];
        setHistoryIndex(nextHistory.length - 1);
        return nextHistory;
      });
    },
  });

  const reviewMutation = useMutation({
    mutationFn: ({ cardId, rating }: { cardId: string; rating: ReviewRating }) =>
      submitCardReview(examId, cardId, userId, rating),
  });

  const canRateCurrentCard = useMemo(() => {
    if (!currentCard || !resolvedActiveCardId || isViewingPreviousCard) {
      return false;
    }
    return currentCard.card_id === resolvedActiveCardId;
  }, [currentCard, isViewingPreviousCard, resolvedActiveCardId]);
  const currentCardRating = currentCard ? (selectedRatings[currentCard.card_id] ?? null) : null;
  const isHistoryForwardAvailable =
    seededHistoryIndex >= 0 && seededHistoryIndex < seededHistoryCards.length - 1;
  const isNextEnabled = isHistoryForwardAvailable
    ? true
    : currentCard
      ? !canRateCurrentCard || currentCardRating !== null
      : true;
  const isPreviousEnabled = seededHistoryIndex > 0;

  async function handleRate(rating: ReviewRating) {
    const selectedCard = currentCard;
    if (!selectedCard || !canRateCurrentCard) {
      return;
    }
    if (selectedRatings[selectedCard.card_id]) {
      return;
    }

    setSelectedRatings((previous) => ({
      ...previous,
      [selectedCard.card_id]: rating,
    }));
    setStatusMessage(null);
    try {
      assertActiveRateableCard(selectedCard.card_id, resolvedActiveCardId);
      await reviewMutation.mutateAsync({ cardId: selectedCard.card_id, rating });
      await queryClient.invalidateQueries({ queryKey: ["exam-progress", examId, userId] });
    } catch (error) {
      setSelectedRatings((previous) => {
        const next = { ...previous };
        delete next[selectedCard.card_id];
        return next;
      });
      setStatusMessage(mapApiError(error, "exam.workspace.review_card").message);
    }
  }

  const handleNext = useCallback(async () => {
    // Block re-entry while a card is generating: a second mutate would race and
    // corrupt the active-card/history state.
    if (nextCardMutation.isPending) {
      return;
    }
    if (isHistoryForwardAvailable) {
      setHistoryIndex((previous) => previous + 1);
      setIsAnswerVisible(false);
      return;
    }
    try {
      await nextCardMutation.mutateAsync();
    } catch (error) {
      setStatusMessage(mapApiError(error, "exam.workspace.next_card").message);
    }
  }, [isHistoryForwardAvailable, nextCardMutation]);

  const handlePrevious = useCallback(async () => {
    // Freeze history navigation while generating (mouse and keyboard both route
    // here) so we don't swap the current card out from under the mutation.
    if (!isPreviousEnabled || nextCardMutation.isPending) {
      return;
    }
    setHistoryIndex((previous) => previous - 1);
    setIsAnswerVisible(false);
  }, [isPreviousEnabled, nextCardMutation.isPending]);

  function handleToggleAnswer() {
    if (!currentCard) {
      return;
    }
    setIsAnswerVisible((previous) => !previous);
  }

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      if (
        event.target instanceof HTMLElement &&
        (event.target.tagName === "INPUT" ||
          event.target.tagName === "TEXTAREA" ||
          event.target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        void handleNext();
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        void handlePrevious();
      }
    }

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [handleNext, handlePrevious]);

  const isLoading = sessionQuery.isLoading || nextCardMutation.isPending;
  const isPreparingNextCard = nextCardMutation.isPending;
  const sessionLoadError = sessionQuery.isError
    ? mapApiError(sessionQuery.error, "exam.workspace.load_session")
    : null;
  const examLoadError = examQuery.isError
    ? mapApiError(examQuery.error, "exam.workspace.load_exam")
    : null;

  const cardCountRaw = (examQuery.data?.info as Record<string, unknown> | undefined)?.card_count;
  const cardCount = typeof cardCountRaw === "number" ? cardCountRaw : null;

  return (
    <motion.div
      className="study"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <header className="study__topbar">
        <Link
          className={`study__back${isPreparingNextCard ? " study__back--disabled" : ""}`}
          href="/"
          aria-disabled={isPreparingNextCard || undefined}
          tabIndex={isPreparingNextCard ? -1 : undefined}
          onClick={(event) => {
            if (isPreparingNextCard) event.preventDefault();
          }}
        >
          <Home size={18} aria-hidden="true" />
          Back to Home
        </Link>
        <div className="study__deck">
          <h1 className="study__deck-title">
            {examQuery.data?.title ?? (examQuery.isLoading ? "Preparing your deck..." : "Study deck")}
          </h1>
          {cardCount !== null ? <p className="study__deck-sub">{cardCount} cards</p> : null}
        </div>
      </header>

      <div className="study__body">
        {examLoadError ? (
          <section className="study__banner">
            <InlineError
              message={examLoadError.message}
              onRetry={examLoadError.canRetry ? () => void examQuery.refetch() : undefined}
              messageClassName=""
              retryClassName="study__retry"
            />
          </section>
        ) : null}

        <ProgressPanel examId={examId} userId={userId} navDisabled={isPreparingNextCard} />

        {sessionLoadError ? (
          <section className="study__banner">
            <InlineError
              message={sessionLoadError.message}
              onRetry={sessionLoadError.canRetry ? () => void sessionQuery.refetch() : undefined}
              messageClassName=""
              retryClassName="study__retry"
            />
          </section>
        ) : (
          <FlashcardPlayer
            card={currentCard}
            isAnswerVisible={isAnswerVisible}
            canRateCurrentCard={canRateCurrentCard}
            selectedRating={currentCardRating}
            isRatingPending={reviewMutation.isPending || isLoading}
            isPreparingNextCard={isPreparingNextCard}
            isNextEnabled={isNextEnabled}
            statusMessage={currentMessage}
            onToggleAnswer={handleToggleAnswer}
            onShowProofs={(card) => setProofsCard(card)}
            onRate={(rating) => void handleRate(rating)}
            onLoadPrevious={() => void handlePrevious()}
            onLoadNext={() => void handleNext()}
            isPreviousEnabled={isPreviousEnabled && !isPreparingNextCard}
          />
        )}
      </div>

      <ProofsDialog
        isOpen={Boolean(proofsCard)}
        card={proofsCard}
        userId={userId}
        onClose={() => setProofsCard(null)}
      />
    </motion.div>
  );
}
