"use client";

import { Sparkles } from "lucide-react";

// Shown in the card shell while the next card is being generated. Branded
// orbiting-sparkle motif (matches the creating-test screen) over a shimmer
// sweep, so the wait reads as "a fresh card is being drawn".
export function CardLoader() {
  return (
    <div className="card-loader" role="status" aria-live="polite">
      <div className="card-loader__shimmer" aria-hidden="true" />
      <div className="card-loader__orb" aria-hidden="true">
        <span className="card-loader__ring" />
        <span className="card-loader__spark" />
        <Sparkles size={30} className="card-loader__icon" />
      </div>
      <p className="card-loader__title">
        Crafting your next card
        <span className="card-loader__dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </p>
      <p className="card-loader__sub">Tuning it to your current level…</p>
    </div>
  );
}
