"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getExamById, type BootstrapProgress } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";
import { CreatingTest } from "@/components/exam/creating/creating-test";

export function CreatingTestScreen({ examId }: { examId: string }) {
  const router = useRouter();
  const { userId } = useGuestSession();

  const { data } = useQuery({
    queryKey: ["exam-creating", examId, userId],
    queryFn: () => getExamById(examId, userId),
    refetchInterval: (query) => {
      const s = query.state.data?.state;
      return s === "diagnostic" || s === "failed" ? false : 1500;
    },
  });

  useEffect(() => {
    if (data?.state === "diagnostic") {
      router.replace(`/exams/${examId}`);
    }
  }, [data?.state, examId, router]);

  if (data?.state === "failed") {
    const message = mapHomeApiError(
      new Error(String((data.info as Record<string, unknown>)?.bootstrap_error ?? "")),
      "home.upload.create_exam",
    ).message;
    return (
      <div className="ct-screen">
        <div className="ct">
          <header className="ct__head">
            <h1 className="ct__title">We couldn&rsquo;t finish your test</h1>
            <p className="ct__sub">{message}</p>
            <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
              <button type="button" className="ct__btn" onClick={() => router.push("/")}>
                Back home
              </button>
            </div>
          </header>
        </div>
      </div>
    );
  }

  const info = (data?.info ?? {}) as Record<string, unknown>;
  const filenames = Array.isArray(info.filenames) ? (info.filenames as string[]) : [];
  return (
    <div className="ct-screen">
      <CreatingTest
        fileName={filenames[0] ?? data?.title}
        progress={info.progress as BootstrapProgress | undefined}
      />
    </div>
  );
}
