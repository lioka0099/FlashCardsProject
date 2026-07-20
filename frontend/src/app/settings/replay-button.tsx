"use client";

import { useTour } from "@/components/tour/onboarding-tour";

export function ReplayWalkthroughCard() {
  const { startTour } = useTour();
  return (
    <div className="settings__card">
      <h1 className="settings__title">Walkthrough</h1>
      <p className="settings__subtitle">Replay the guided tour of the app.</p>
      <button className="settings__submit" type="button" onClick={startTour}>
        Replay walkthrough
      </button>
    </div>
  );
}
