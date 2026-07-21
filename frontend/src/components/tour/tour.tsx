"use client";

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
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
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tipH, setTipH] = useState(0);

  useEffect(() => setMounted(true), []);

  // Measure the tooltip so its top can be clamped fully into the viewport.
  useLayoutEffect(() => {
    if (tooltipRef.current) {
      setTipH(tooltipRef.current.offsetHeight);
    }
  }, [rect, stepIndex]);

  // Locate the target; poll briefly if it isn't rendered yet, else auto-skip.
  useEffect(() => {
    if (!step) return;
    let raf = 0;
    let elapsed = 0;
    setRect(null);
    const tick = () => {
      const el = document.querySelector(step.target);
      if (el) {
        // Instant (not smooth) so the rect we read below is the settled
        // position — a smooth scroll would still be animating and freeze the
        // spotlight at the pre-scroll spot.
        el.scrollIntoView?.({ block: "center" });
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
    // capture:true so scrolls inside an inner overflow container (the dashboard
    // scrolls a panel, not the window) still reposition the spotlight.
    window.addEventListener("scroll", reposition, { passive: true, capture: true });
    window.addEventListener("resize", reposition);
    return () => {
      window.removeEventListener("scroll", reposition, { capture: true });
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

  // Keep the tooltip fully on-screen: clamp both axes into the viewport so a
  // right-aligned or full-height target can't push it out of view. Prefer below
  // the target; fall back to above; then clamp using the measured tooltip height.
  const TOOLTIP_W = 320;
  const left = Math.min(Math.max(12, rect.left), window.innerWidth - TOOLTIP_W - 12);
  const spaceBelow = window.innerHeight - (rect.top + rect.height);
  const placeBelow = spaceBelow > tipH + 24;
  const preferredTop = placeBelow ? rect.top + rect.height + 12 : rect.top - tipH - 12;
  const top = Math.min(Math.max(12, preferredTop), Math.max(12, window.innerHeight - tipH - 12));
  const tooltipStyle: CSSProperties = { top, left };

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
      <div className="tour__tooltip" ref={tooltipRef} style={tooltipStyle}>
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
