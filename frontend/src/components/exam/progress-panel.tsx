"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronRight, History } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getExamProgress } from "@/lib/api/client";
import { mapApiError } from "@/lib/api/ui-error";
import { getTopicColor } from "@/lib/topic-colors";
import { InlineError } from "@/components/common/inline-error";

type ProgressPanelProps = {
  examId: string;
  userId: string;
  // Freeze navigation (the History link) while the next card is generating, so
  // the user can't leave mid-mutation and corrupt the session state.
  navDisabled?: boolean;
};

function toPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

export function ProgressPanel({ examId, userId, navDisabled = false }: ProgressPanelProps) {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["exam-progress", examId, userId],
    queryFn: () => getExamProgress(examId, userId),
    retry: 1,
  });
  const progressError = isError ? mapApiError(error, "exam.progress.load") : null;
  const overall = toPercent(data?.overall_proficiency ?? null);

  // Scroll-affordance fades: show a top fade once scrolled down and a bottom
  // fade while there's more list below, so it's obvious the topics scroll.
  const scrollRef = useRef<HTMLDivElement>(null);
  const [fade, setFade] = useState({ top: false, bottom: false });
  const topicCount = data?.topics.length ?? 0;

  const updateFades = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const overflow = el.scrollHeight - el.clientHeight;
    setFade({
      top: overflow > 1 && el.scrollTop > 1,
      bottom: overflow > 1 && el.scrollTop < overflow - 1,
    });
  }, []);

  useEffect(() => {
    updateFades();
    window.addEventListener("resize", updateFades);
    return () => window.removeEventListener("resize", updateFades);
  }, [updateFades, topicCount]);

  return (
    <aside className="side" aria-label="Study progress">
      <div className="side__panel">
        <h2 className="side__title">Your Progress</h2>

        {isLoading ? <p className="side__hint">Measuring your intellectual sparkle...</p> : null}

        {progressError ? (
          <InlineError
            message={progressError.message}
            onRetry={progressError.canRetry ? () => void refetch() : undefined}
            messageClassName="side__error"
            retryClassName="study__retry"
          />
        ) : null}

        {!isLoading && !isError ? (
          <>
            <div className="side__ring-wrap">
              <div className="ring" style={{ ["--pct" as string]: overall }} aria-hidden="true">
                <span className="ring__label">
                  <span className="ring__pct">{overall}%</span>
                  <span className="ring__cap">Overall</span>
                </span>
              </div>
            </div>

            {(data?.topics.length ?? 0) === 0 ? (
              <p className="side__hint">No topic progress yet. Rate a few cards and the meter will wake up.</p>
            ) : (
              <>
                <h3 className="side__subtitle">By Topic</h3>
                <div
                  className="side__scroll-wrap"
                  data-fade-top={fade.top ? "true" : "false"}
                  data-fade-bottom={fade.bottom ? "true" : "false"}
                >
                  <div className="side__scroll" ref={scrollRef} onScroll={updateFades}>
                    <ul className="side__topics">
                      {data?.topics.map((topic) => {
                        const pct = toPercent(topic.proficiency);
                        const color = getTopicColor(topic.topic_label);
                        return (
                          <li key={topic.topic_id} style={{ ["--c" as string]: color }}>
                            <div className="topic__head">
                              <span className="topic__dot" aria-hidden="true" />
                              <span className="topic__name">{topic.topic_label}</span>
                              <span className="topic__pct">{pct}%</span>
                            </div>
                            <div className="topic__track" aria-hidden="true">
                              <span className="topic__fill" style={{ width: `${pct}%` }} />
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              </>
            )}
          </>
        ) : null}

        <Link
          className={`side__history${navDisabled ? " side__history--disabled" : ""}`}
          href={`/exams/${examId}/history`}
          aria-disabled={navDisabled || undefined}
          tabIndex={navDisabled ? -1 : undefined}
          onClick={(event) => {
            if (navDisabled) event.preventDefault();
          }}
        >
          <History size={18} aria-hidden="true" />
          <span>
            View History
            <span className="side__history-sub">See all studied cards</span>
          </span>
          <ChevronRight className="side__history-chev" size={18} aria-hidden="true" />
        </Link>
      </div>
    </aside>
  );
}
