"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, Check, ChevronDown, FileText, Lightbulb, ListChecks, Loader2, Sparkles } from "lucide-react";
import type { BootstrapProgress, ProgressStepStatus } from "@/lib/api/client";
import "./creating-test.css";

const TIPS = [
  "Active recall is one of the most effective learning techniques. Our questions are designed to help you truly understand the material.",
  "Spacing your reviews over time beats cramming. We schedule cards so they come back right before you'd forget.",
  "Explaining a concept in your own words is the fastest way to find the gaps. The cards will nudge you to do exactly that.",
];

function StepIcon({ status }: { status: ProgressStepStatus }) {
  if (status === "done") {
    return <span className="ct-step__check" aria-hidden="true"><Check size={16} /></span>;
  }
  if (status === "active") {
    return <span className="ct-step__spin" aria-hidden="true"><Loader2 size={18} /></span>;
  }
  return <span className="ct-step__dot" aria-hidden="true" />;
}

export function CreatingTest({ fileNames, progress }: { fileNames?: string[]; progress?: BootstrapProgress }) {
  const steps = progress?.steps ?? [];
  const [tip, setTip] = useState(0);
  const [isFilesOpen, setIsFilesOpen] = useState(false);
  const filesRef = useRef<HTMLDivElement | null>(null);
  const fileCount = fileNames?.length ?? 0;

  useEffect(() => {
    const id = window.setInterval(() => setTip((t) => (t + 1) % TIPS.length), 5000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!isFilesOpen) {
      return;
    }
    function onPointerDown(event: MouseEvent) {
      if (filesRef.current?.contains(event.target as Node)) {
        return;
      }
      setIsFilesOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [isFilesOpen]);

  return (
    <div className="ct">
      <header className="ct__head">
        <span className="ct__sparkle" aria-hidden="true"><Sparkles size={28} /></span>
        <h1 className="ct__title">Creating your test...</h1>
        <p className="ct__sub">
          Our AI is analyzing your document{fileCount > 1 ? "s" : ""} and building a personalized study set just for you.
        </p>
        {fileCount > 0 ? (
          <div className="ct__files" ref={filesRef}>
            <button
              type="button"
              className="ct__files-trigger"
              aria-haspopup="menu"
              aria-expanded={isFilesOpen}
              onClick={() => setIsFilesOpen((open) => !open)}
            >
              <FileText size={16} aria-hidden="true" />
              <span>{fileCount} file{fileCount === 1 ? "" : "s"} uploaded</span>
              <ChevronDown
                size={16}
                aria-hidden="true"
                className={`ct__files-chevron${isFilesOpen ? " ct__files-chevron--open" : ""}`}
              />
            </button>
            <AnimatePresence>
              {isFilesOpen ? (
                <motion.div
                  className="ct__files-menu"
                  role="menu"
                  aria-label="Uploaded files"
                  initial={{ opacity: 0, y: 8, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: "easeOut" }}
                >
                  {fileNames?.map((name) => (
                    <span className="ct__file" key={name}>
                      <FileText size={16} aria-hidden="true" />
                      <span className="ct__file-name">{name}</span>
                    </span>
                  ))}
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        ) : null}
      </header>

      <section className="ct__card">
        <ol className="ct__steps">
          {steps.map((s, i) => (
            <li key={s.key} className="ct-step" data-status={s.status}>
              <span className="ct-step__rail" aria-hidden={i === steps.length - 1}>
                <StepIcon status={s.status} />
              </span>
              <span className="ct-step__body">
                <span className="ct-step__label">{s.label}</span>
                <span className="ct-step__detail">{s.detail}</span>
              </span>
            </li>
          ))}
        </ol>

        <aside className="ct__aside">
          <div className="ct__illus" aria-hidden="true">
            <span className="ct__orbit" />
            <span className="ct__spark ct__spark--a" />
            <span className="ct__spark ct__spark--b" />
            <span className="ct__spark ct__spark--c" />
            <div className="ct__doc">
              <FileText size={56} />
            </div>
            <span className="ct__tile ct__tile--list"><ListChecks size={18} /></span>
            <span className="ct__tile ct__tile--brain"><Brain size={20} /></span>
            <span className="ct__tile ct__tile--check"><Check size={18} /></span>
          </div>
          <div className="ct__callout">
            <span className="ct__callout-icon" aria-hidden="true"><Brain size={22} /></span>
            <div className="ct__callout-text">
              <strong>AI in action</strong>
              <p>We&rsquo;re not just reading your document — we&rsquo;re understanding it.</p>
            </div>
          </div>
        </aside>
      </section>

      <section className="ct__tip" aria-live="polite">
        <span className="ct__tip-icon" aria-hidden="true"><Lightbulb size={20} /></span>
        <p><strong>Did you know?</strong> {TIPS[tip]}</p>
        <span className="ct__tip-count">Tip {tip + 1} of {TIPS.length}</span>
      </section>

      <p className="ct__foot">You can close this window — we&rsquo;ll keep working in the background.</p>
    </div>
  );
}
