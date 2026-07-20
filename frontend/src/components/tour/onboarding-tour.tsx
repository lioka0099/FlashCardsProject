"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type PropsWithChildren,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getMe, markOnboarded } from "@/lib/api/auth";
import { Tour } from "@/components/tour/tour";
import { ONBOARDING_STEPS } from "@/components/tour/steps";

type TourContextValue = { startTour: () => void };

const TourContext = createContext<TourContextValue>({ startTour: () => {} });

export function useTour() {
  return useContext(TourContext);
}

export function OnboardingTourProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const pathname = usePathname();
  const queryClient = useQueryClient();
  const [active, setActive] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  // Guests get a 401 here -> query stays in error, me is undefined, no auto-start.
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: getMe, retry: false });

  // Auto-start once for a brand-new account landing on the dashboard.
  useEffect(() => {
    if (!active && me && me.onboarded === false && pathname === "/") {
      setStepIndex(0);
      setActive(true);
    }
  }, [active, me, pathname]);

  const step = active ? ONBOARDING_STEPS[stepIndex] : null;

  // Make sure we're on the page a step lives on before spotlighting it.
  useEffect(() => {
    if (step && pathname !== step.page) {
      router.push(step.page);
    }
  }, [step, pathname, router]);

  const finish = useCallback(async () => {
    setActive(false);
    setStepIndex(0);
    try {
      const updated = await markOnboarded();
      queryClient.setQueryData(["me"], updated);
    } catch {
      // Non-fatal: the tour simply reappears next session.
    }
    if (pathname !== "/") {
      router.push("/");
    }
  }, [pathname, queryClient, router]);

  const next = useCallback(() => {
    if (stepIndex >= ONBOARDING_STEPS.length - 1) {
      void finish();
      return;
    }
    setStepIndex(stepIndex + 1);
  }, [stepIndex, finish]);

  const back = useCallback(() => {
    setStepIndex((i) => Math.max(0, i - 1));
  }, []);

  const startTour = useCallback(() => {
    setStepIndex(0);
    setActive(true);
    router.push("/");
  }, [router]);

  return (
    <TourContext.Provider value={{ startTour }}>
      {children}
      {active && step ? (
        <Tour
          steps={ONBOARDING_STEPS}
          stepIndex={stepIndex}
          onNext={next}
          onBack={back}
          onSkip={() => void finish()}
          onFinish={() => void finish()}
        />
      ) : null}
    </TourContext.Provider>
  );
}
