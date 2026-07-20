"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type PropsWithChildren } from "react";
import { GuestSessionProvider } from "@/lib/session/guest-session";
import { OnboardingTourProvider } from "@/components/tour/onboarding-tour";

export function AppProviders({ children }: PropsWithChildren) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <GuestSessionProvider>
      <QueryClientProvider client={queryClient}>
        <OnboardingTourProvider>{children}</OnboardingTourProvider>
      </QueryClientProvider>
    </GuestSessionProvider>
  );
}
