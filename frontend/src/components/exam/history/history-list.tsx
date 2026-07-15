"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  Brain,
  ChevronDown,
  Frown,
  Home,
  Meh,
  Quote,
  RotateCcw,
  Search,
  Smile,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { InlineError } from "@/components/common/inline-error";
import { downloadSource } from "@/components/exam/lib/download-source";
import { MathText } from "@/components/exam/math-text";
import { TextSourceModal } from "@/components/exam/study/text-source-modal";
import {
  getExamById,
  getPresentedHistory,
  type Card,
  type ProofSpan,
  type ReviewRating,
} from "@/lib/api/client";
import { mapApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";
import { getTopicColor } from "@/lib/topic-colors";
import "@/components/exam/study/study.css";
import "./history.css";

// react-pdf touches browser-only globals at module load; load client-only.
const PdfSourceModal = dynamic(
  () => import("@/components/exam/study/pdf-source-modal").then((m) => m.PdfSourceModal),
  { ssr: false },
);

type HistoryListProps = {
  examId: string;
};

const RATING_UI: Record<ReviewRating, { label: string; icon: typeof Smile }> = {
  i_knew_it: { label: "Nailed it", icon: Smile },
  almost_knew: { label: "Nearly there", icon: Smile },
  learned_now: { label: "Just learned it", icon: Meh },
  dont_understand: { label: "Still mysterious", icon: Frown },
};
const RATING_ORDER: ReviewRating[] = ["i_knew_it", "almost_knew", "learned_now", "dont_understand"];

function formatTime(value: string | null | undefined): { short: string; full: string } {
  if (!value) return { short: "—", full: "Unknown time" };
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return { short: "—", full: "Unknown time" };
  const isToday = new Date().toDateString() === date.toDateString();
  const short = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    ...(isToday ? {} : { month: "short", day: "numeric" }),
  }).format(date);
  const full = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
  return { short, full };
}

function presentedAtFromCard(card: Card): string {
  const presentedAt = card.info?.presented_at;
  return typeof presentedAt === "string" ? presentedAt : card.created_at;
}

function ratingFromCard(card: Card): ReviewRating | null {
  const raw =
    (typeof card.info?.rating === "string" && card.info.rating) ||
    (typeof card.info?.last_rating === "string" && card.info.last_rating) ||
    (typeof card.info?.review_rating === "string" && card.info.review_rating) ||
    null;
  return raw && raw in RATING_UI ? (raw as ReviewRating) : null;
}

export function HistoryList({ examId }: HistoryListProps) {
  const { userId } = useGuestSession();
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [isAnswerVisible, setIsAnswerVisible] = useState(false);
  const [activeProofIndex, setActiveProofIndex] = useState<number | null>(0);
  const [openSourceModal, setOpenSourceModal] = useState<
    { kind: "pdf" | "text"; proof: ProofSpan } | null
  >(null);
  const [downloadErrorKey, setDownloadErrorKey] = useState<string | null>(null);

  const [topicFilter, setTopicFilter] = useState("all");
  const [ratingFilter, setRatingFilter] = useState("all");
  const [search, setSearch] = useState("");

  const examQuery = useQuery({
    queryKey: ["exam-by-id", examId, userId],
    queryFn: () => getExamById(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const historyQuery = useQuery({
    queryKey: ["presented-history", examId, userId],
    queryFn: () => getPresentedHistory(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const cards = useMemo(() => historyQuery.data ?? [], [historyQuery.data]);

  const topics = useMemo(() => {
    const set = new Set<string>();
    for (const card of cards) {
      if (card.topic_label) set.add(card.topic_label);
    }
    return Array.from(set).sort();
  }, [cards]);

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    return cards.filter((card) => {
      if (topicFilter !== "all" && card.topic_label !== topicFilter) return false;
      if (ratingFilter !== "all") {
        const rating = ratingFromCard(card) ?? "none";
        if (rating !== ratingFilter) return false;
      }
      if (term && !`${card.question} ${card.answer}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [cards, topicFilter, ratingFilter, search]);

  const resolvedSelectedCardId = useMemo(() => {
    if (selectedCardId && filtered.some((c) => c.card_id === selectedCardId)) {
      return selectedCardId;
    }
    return filtered[0]?.card_id ?? null;
  }, [filtered, selectedCardId]);
  const selectedCard = useMemo(
    () => filtered.find((c) => c.card_id === resolvedSelectedCardId) ?? null,
    [filtered, resolvedSelectedCardId],
  );

  const selectedRating = selectedCard ? ratingFromCard(selectedCard) : null;
  const selectedRatingUi = selectedRating ? RATING_UI[selectedRating] : null;
  const cardType =
    selectedCard && typeof selectedCard.info?.card_type === "string" ? selectedCard.info.card_type : null;
  const pillLabel = cardType ? cardType.charAt(0).toUpperCase() + cardType.slice(1) : "Concept";

  const examError = examQuery.isError ? mapApiError(examQuery.error, "exam.workspace.load_exam") : null;
  const historyError = historyQuery.isError ? mapApiError(historyQuery.error, "exam.workspace.load_session") : null;

  const cardCountRaw = (examQuery.data?.info as Record<string, unknown> | undefined)?.card_count;
  const cardCount = typeof cardCountRaw === "number" ? cardCountRaw : null;

  return (
    <section className="history" aria-label="Exam history">
      <header className="hist-topbar">
        <Link className="hist-back" href="/">
          <Home size={18} aria-hidden="true" />
          Back to Home
        </Link>
        <div className="hist-deck">
          <h1 className="hist-deck__title">
            {examQuery.data?.title ?? (examQuery.isLoading ? "Loading deck…" : "Study deck")}
          </h1>
          {cardCount !== null ? <p className="hist-deck__sub">{cardCount} cards</p> : null}
        </div>
        <Link className="hist-resume" href={`/exams/${examId}`}>
          <ArrowLeft size={18} aria-hidden="true" />
          Back to exam
        </Link>
      </header>

      <div className="history__grid">
        <aside className="hist-panel">
          <h1 className="hist-panel__title">Card History</h1>
          <p className="hist-panel__sub">See all the cards you&rsquo;ve studied and how you performed.</p>

          <div className="hist-filters">
            <label className="hist-select">
              <select value={topicFilter} onChange={(e) => setTopicFilter(e.target.value)} aria-label="Filter by topic">
                <option value="all">All Topics</option>
                {topics.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <ChevronDown size={16} className="hist-select__chev" aria-hidden="true" />
            </label>
            <label className="hist-select">
              <select value={ratingFilter} onChange={(e) => setRatingFilter(e.target.value)} aria-label="Filter by rating">
                <option value="all">All Ratings</option>
                {RATING_ORDER.map((r) => (
                  <option key={r} value={r}>{RATING_UI[r].label}</option>
                ))}
                <option value="none">Not rated</option>
              </select>
              <ChevronDown size={16} className="hist-select__chev" aria-hidden="true" />
            </label>
            <div className="hist-search">
              <Search size={16} className="hist-search__icon" aria-hidden="true" />
              <input
                type="search"
                placeholder="Search cards..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search cards"
              />
            </div>
          </div>

          {examError ? (
            <InlineError message={examError.message} onRetry={examError.canRetry ? () => void examQuery.refetch() : undefined} />
          ) : null}
          {historyError ? (
            <InlineError message={historyError.message} onRetry={historyError.canRetry ? () => void historyQuery.refetch() : undefined} />
          ) : null}

          {!historyQuery.isError && historyQuery.isLoading ? (
            <p className="hist-note">Gathering the cards you have seen&hellip;</p>
          ) : null}
          {!historyQuery.isLoading && !historyQuery.isError && cards.length === 0 ? (
            <p className="hist-note">No cards have walked across the stage yet.</p>
          ) : null}
          {!historyQuery.isLoading && !historyQuery.isError && cards.length > 0 && filtered.length === 0 ? (
            <p className="hist-note">No cards match these filters.</p>
          ) : null}

          {filtered.length > 0 ? (
            <ol className="hist-list" aria-label="Presented cards list">
              {filtered.map((card, index) => {
                const rating = ratingFromCard(card);
                const ratingUi = rating ? RATING_UI[rating] : null;
                const FaceIcon = ratingUi?.icon ?? Meh;
                const time = formatTime(presentedAtFromCard(card));
                const isActive = resolvedSelectedCardId === card.card_id;
                const tc = card.topic_label ? getTopicColor(card.topic_label) : undefined;
                return (
                  <li
                    key={`${card.card_id}-${index}`}
                    className={`hist-item rc--${rating ?? "none"}`}
                    style={tc ? ({ ["--tc" as string]: tc }) : undefined}
                  >
                    <button
                      type="button"
                      className="hist-item__btn"
                      aria-current={isActive ? "true" : undefined}
                      onClick={() => {
                        setSelectedCardId(card.card_id);
                        setIsAnswerVisible(false);
                        setActiveProofIndex(0);
                      }}
                    >
                      <span
                        className="hist-item__face"
                        role="img"
                        aria-label={ratingUi?.label ?? "Not rated"}
                        title={ratingUi?.label ?? "Not rated"}
                      >
                        <FaceIcon size={20} />
                      </span>
                      <span className="hist-item__body">
                        <span className="hist-item__head">
                          <span className="hist-item__topic">{card.topic_label ?? "General topic"}</span>
                          <span className="hist-item__time" title={time.full}>{time.short}</span>
                        </span>
                        <MathText className="hist-item__q" text={card.question} />
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          ) : null}
        </aside>

        <main className="hist-stage" aria-live="polite">
          {!selectedCard ? (
            <div className="hist-empty">
              <p className="hist-empty__title">Nothing to review yet</p>
              <p>Pick a card from the list to stroll through your past brilliance.</p>
            </div>
          ) : (
            <>
            <div className={`hist-shell rc--${selectedRating ?? "none"}`}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={selectedCard.card_id}
                  style={{ position: "absolute", inset: 0, transformStyle: "preserve-3d" }}
                  initial={{ opacity: 0 }}
                  animate={{ rotateY: isAnswerVisible ? 180 : 0, opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, ease: "easeInOut" }}
                  role="button"
                  tabIndex={0}
                  aria-label={isAnswerVisible ? "Show question side" : "Show answer side"}
                  onClick={() => setIsAnswerVisible((p) => !p)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setIsAnswerVisible((p) => !p);
                    }
                  }}
                >
                  <article className="hist-face">
                    <div className="hist-face__meta">
                      <span className="hist-pill"><Brain size={16} aria-hidden="true" /> {pillLabel}</span>
                      {selectedRatingUi ? <span className="hist-pill hist-pill--rate">{selectedRatingUi.label}</span> : null}
                    </div>
                    {selectedCard.topic_label ? <p className="hist-face__topic">{selectedCard.topic_label}</p> : null}
                    <div className="hist-face__q">
                      <MathText className="math-text" text={selectedCard.question} />
                    </div>
                    <hr className="hist-divider" />
                    <button
                      className="hist-flip"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setIsAnswerVisible(true);
                      }}
                    >
                      <RotateCcw size={18} aria-hidden="true" /> Flip Card
                    </button>
                  </article>

                  <article className="hist-face" style={{ transform: "rotateY(180deg)" }}>
                    <div className="hist-face__meta">
                      <span className="hist-pill"><Brain size={16} aria-hidden="true" /> Answer</span>
                      {selectedRatingUi ? <span className="hist-pill hist-pill--rate">{selectedRatingUi.label}</span> : null}
                    </div>
                    <div className="hist-face__a">
                      <MathText className="math-text" text={selectedCard.answer} />
                      <p className="hist-proofs__title"><Quote size={14} aria-hidden="true" /> Supporting evidence</p>
                      {selectedCard.proofs.length > 0 ? (
                        <div className="hist-proofs">
                          {selectedCard.proofs.map((proof, proofIndex) => {
                            const pageLabel = proof.page !== null ? String(proof.page) : "Unavailable";
                            const proofKey = `${proof.doc_id}:${proof.page ?? "na"}:${proof.start}:${proof.end}`;
                            return (
                              <div className="hist-proof" key={proofKey}>
                                <button
                                  type="button"
                                  className="hist-proof__btn"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setActiveProofIndex((prev) => (prev === proofIndex ? null : proofIndex));
                                  }}
                                >
                                  Source {proofIndex + 1}
                                  <ChevronDown
                                    size={16}
                                    className={`hist-proof__chev${activeProofIndex === proofIndex ? " hist-proof__chev--open" : ""}`}
                                    aria-hidden="true"
                                  />
                                </button>
                                <AnimatePresence initial={false}>
                                  {activeProofIndex === proofIndex ? (
                                    <motion.div
                                      initial={{ height: 0, opacity: 0 }}
                                      animate={{ height: "auto", opacity: 1 }}
                                      exit={{ height: 0, opacity: 0 }}
                                      transition={{ duration: 0.2, ease: "easeInOut" }}
                                      style={{ overflow: "hidden" }}
                                    >
                                      <div className="hist-proof__body">
                                        <p className="hist-proof__text">{proof.text}</p>
                                        <p className="evidence__meta">Page: {pageLabel}</p>
                                        <button
                                          className="evidence__jump"
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            if (proof.is_pdf !== false) {
                                              setOpenSourceModal({ kind: "pdf", proof });
                                            } else if (proof.is_txt) {
                                              setOpenSourceModal({ kind: "text", proof });
                                            } else {
                                              setDownloadErrorKey(null);
                                              downloadSource(proof, selectedCard, userId).catch(() =>
                                                setDownloadErrorKey(proofKey),
                                              );
                                            }
                                          }}
                                        >
                                          Open the source
                                        </button>
                                        {downloadErrorKey === proofKey ? (
                                          <p className="evidence__download-error">Could not download the source.</p>
                                        ) : null}
                                      </div>
                                    </motion.div>
                                  ) : null}
                                </AnimatePresence>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="hist-note">No evidence attached to this card.</p>
                      )}
                      <button
                        className="hist-flip"
                        type="button"
                        style={{ marginTop: "1rem", alignSelf: "flex-start" }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setIsAnswerVisible(false);
                        }}
                      >
                        <RotateCcw size={18} aria-hidden="true" /> Back to the question
                      </button>
                    </div>
                  </article>
                </motion.div>
              </AnimatePresence>
            </div>
            {openSourceModal?.kind === "pdf" ? (
              <PdfSourceModal
                proof={openSourceModal.proof}
                card={selectedCard}
                userId={userId}
                onClose={() => setOpenSourceModal(null)}
              />
            ) : null}
            {openSourceModal?.kind === "text" ? (
              <TextSourceModal
                proof={openSourceModal.proof}
                card={selectedCard}
                userId={userId}
                onClose={() => setOpenSourceModal(null)}
              />
            ) : null}
            </>
          )}
        </main>
      </div>
    </section>
  );
}
