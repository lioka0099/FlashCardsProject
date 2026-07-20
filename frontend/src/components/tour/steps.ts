import type { TourStep } from "@/components/tour/tour";

export type OnboardingStep = TourStep & { page: string };

export const ONBOARDING_STEPS: OnboardingStep[] = [
  { page: "/", target: '[data-tour="my-tests"]', title: "Your decks", body: "All your study decks live here." },
  { page: "/", target: '[data-tour="deck-name"]', title: "Name a deck", body: "Give a new deck a name…" },
  { page: "/", target: '[data-tour="upload"]', title: "Add a document", body: "…then drop in a PDF or notes — we turn it into flashcards." },
  { page: "/", target: '[data-tour="account"]', title: "Your account", body: "Profile and settings live here." },
  { page: "/exams/demo", target: '[data-tour="flashcard"]', title: "Study a card", body: "Each flashcard hides the answer on its back — flip it to check yourself." },
  { page: "/exams/demo", target: '[data-tour="rating"]', title: "Rate yourself", body: "Rate how well you knew it — this schedules your next review." },
  { page: "/exams/demo", target: '[data-tour="sources"]', title: "See the source", body: "Every card links back to where it came from." },
  { page: "/exams/demo", target: '[data-tour="progress"]', title: "Track mastery", body: "Watch your progress grow as you study." },
];
