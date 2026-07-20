"use client";

import { useEffect, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import "./tour.css";

export type TourStep = {
  target: string;
  title: string;
  body: string;
};

type TourProps = {
  steps: TourStep[];
  stepIndex: number;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
  onFinish: () => void;
};

type Rect = { top: number; left: number; width: number; height: number };

const PAD = 8;
const FIND_BUDGET_MS = 1000;

export function Tour({ steps, stepIndex, onNext, onBack, onSkip, onFinish }: TourProps) {
  const step = steps[stepIndex];
  const [rect, setRect] = useState<Rect | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Locate the target; poll briefly if it isn't rendered yet, else auto-skip.
  useEffect(() => {
    if (!step) return;
    let raf = 0;
    let elapsed = 0;
    setRect(null);
    const tick = () => {
      const el = document.querySelector(step.target);
      if (el) {
        el.scrollIntoView?.({ behavior: "smooth", block: "center" });
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
          setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
          return;
        }
      }
      elapsed += 16;
      if (elapsed >= FIND_BUDGET_MS) {
        onNext();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step, onNext]);

  // Keep the spotlight aligned while scrolling/resizing.
  useEffect(() => {
    if (!step) return;
    function reposition() {
      const el = document.querySelector(step.target);
      if (!el) return;
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
    }
    window.addEventListener("scroll", reposition, { passive: true });
    window.addEventListener("resize", reposition);
    return () => {
      window.removeEventListener("scroll", reposition);
      window.removeEventListener("resize", reposition);
    };
  }, [step]);

  // Escape = skip.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onSkip();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onSkip]);

  if (!mounted || !step || !rect) return null;

  const isLast = stepIndex === steps.length - 1;
  const isFirst = stepIndex === 0;
  const advance = () => (isLast ? onFinish() : onNext());

  const spaceBelow = window.innerHeight - (rect.top + rect.height);
  const placeBelow = spaceBelow > 220;
  const tooltipStyle: CSSProperties = placeBelow
    ? { top: rect.top + rect.height + 12, left: Math.max(12, rect.left) }
    : { top: Math.max(12, rect.top - 12), left: Math.max(12, rect.left), transform: "translateY(-100%)" };

  return createPortal(
    <div className="tour" role="dialog" aria-modal="true" aria-label={step.title}>
      <div
        className="tour__spotlight"
        style={{
          top: rect.top - PAD,
          left: rect.left - PAD,
          width: rect.width + PAD * 2,
          height: rect.height + PAD * 2,
        }}
      />
      <div className="tour__catch" onClick={advance} />
      <div className="tour__tooltip" style={tooltipStyle}>
        <h2 className="tour__title">{step.title}</h2>
        <p className="tour__body">{step.body}</p>
        <div className="tour__row">
          <button type="button" className="tour__skip" onClick={onSkip}>
            Skip
          </button>
          <div className="tour__nav">
            <span className="tour__count">
              {stepIndex + 1} / {steps.length}
            </span>
            {!isFirst ? (
              <button type="button" className="tour__btn" onClick={onBack}>
                Back
              </button>
            ) : null}
            <button type="button" className="tour__btn tour__btn--primary" onClick={advance}>
              {isLast ? "Done" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
