"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Layers, Plus, Search } from "lucide-react";
import { InlineError } from "@/components/common/inline-error";
import { listRecentExams, type ExamListItem } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";

type ExamHistorySidebarProps = {
  className?: string;
  onNavigate?: () => void;
  onNewTest?: () => void;
};

const ICON_TINTS = ["#6f42c1", "#2db7c5", "#f59e0b", "#ef4444", "#3b82f6"];

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Recently";
  }
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

function fileType(info: Record<string, unknown>) {
  const filenames = info.filenames;
  const first = Array.isArray(filenames) ? filenames[0] : undefined;
  if (typeof first === "string" && first.includes(".")) {
    return first.split(".").pop()?.toUpperCase() ?? "DOC";
  }
  return "DOC";
}

function progressPct(info: Record<string, unknown>) {
  // Overall proficiency (0..1), same score the exam's progress ring shows.
  const proficiency = info.overall_proficiency;
  if (proficiency == null) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(Number(proficiency) * 100)));
}

export function ExamHistorySidebar({ className, onNavigate, onNewTest }: ExamHistorySidebarProps) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const { userId } = useGuestSession();

  const {
    data: exams,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["recent-exams", userId],
    queryFn: () => listRecentExams(userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const recentExams = useMemo(() => exams ?? [], [exams]);
  const filteredExams = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) {
      return recentExams;
    }
    return recentExams.filter((exam) =>
      (exam.title || "").toLowerCase().includes(term),
    );
  }, [recentExams, query]);
  const recentExamsError = isError
    ? mapHomeApiError(error, "home.sidebar.list_recent_exams")
    : null;

  function handleExamSelect(examId: string) {
    onNavigate?.();
    router.push(`/exams/${examId}`);
  }

  function cardLabel(exam: ExamListItem) {
    return exam.title || "Untitled study quest";
  }

  return (
    <aside
      id="home-sidebar"
      className={className}
      aria-label="Test history sidebar"
    >
      <Link className="dash-brand" href="/" aria-label="Flashcards home">
        <span className="dash-brand__mark" aria-hidden="true">
          <Layers size={22} />
        </span>
        <span>
          <span className="dash-brand__name">Flashcards</span>
          <span className="dash-brand__tag">AI-Powered Learning</span>
        </span>
      </Link>

      <div className="dash-side__head">
        <h2 className="dash-side__title">My Tests</h2>
        <button className="dash-newtest" type="button" onClick={onNewTest}>
          <Plus size={16} />
          New Test
        </button>
      </div>

      <div className="dash-search">
        <Search className="dash-search__icon" size={16} aria-hidden="true" />
        <input
          className="dash-search__input"
          type="search"
          placeholder="Search tests..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Search tests"
        />
      </div>

      {isLoading ? <p className="dash-side__msg">Finding your latest brain snacks...</p> : null}

      {isError ? (
        <InlineError
          message={recentExamsError?.message ?? "Something went wrong while loading your exams. Please try again."}
          onRetry={recentExamsError?.canRetry ? () => void refetch() : undefined}
        />
      ) : null}

      {!isLoading && !isError && recentExams.length === 0 ? (
        <p className="dash-side__msg">No decks yet. Your comeback arc starts with one upload.</p>
      ) : null}

      {!isLoading && !isError && recentExams.length > 0 && filteredExams.length === 0 ? (
        <p className="dash-side__msg">No tests match &ldquo;{query}&rdquo;.</p>
      ) : null}

      {!isLoading && !isError && filteredExams.length > 0 ? (
        <ul className="dash-list" aria-label="Recent tests">
          {filteredExams.map((exam, index) => {
            const pct = progressPct(exam.info);
            return (
              <li key={exam.exam_id}>
                <motion.button
                  className="dash-card"
                  type="button"
                  onClick={() => handleExamSelect(exam.exam_id)}
                  whileHover={{ x: 2 }}
                  whileTap={{ scale: 0.99 }}
                  aria-label={cardLabel(exam)}
                >
                  <span
                    className="dash-card__icon"
                    style={{ background: ICON_TINTS[index % ICON_TINTS.length] }}
                    aria-hidden="true"
                  >
                    {fileType(exam.info).slice(0, 1)}
                  </span>
                  <span className="dash-card__body">
                    <span className="dash-card__row">
                      <span className="dash-card__title">{cardLabel(exam)}</span>
                      <span className="dash-card__pct">{pct}%</span>
                    </span>
                    <span className="dash-card__meta">
                      {fileType(exam.info)} • {formatDate(exam.updated_at)}
                    </span>
                    <span className="dash-card__bar">
                      <span className="dash-card__fill" style={{ width: `${pct}%` }} />
                    </span>
                  </span>
                </motion.button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </aside>
  );
}
