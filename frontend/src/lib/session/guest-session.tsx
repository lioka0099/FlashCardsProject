"use client";

import { createContext, useContext, type PropsWithChildren } from "react";

type GuestSession = {
  userId: string;
  mode: "guest";
};

const DEFAULT_GUEST_SESSION: GuestSession = {
  userId: "guest",
  mode: "guest",
};

const GuestSessionContext = createContext<GuestSession>(DEFAULT_GUEST_SESSION);

export function GuestSessionProvider({ children }: PropsWithChildren) {
  return <GuestSessionContext.Provider value={DEFAULT_GUEST_SESSION}>{children}</GuestSessionContext.Provider>;
}

export function useGuestSession() {
  return useContext(GuestSessionContext);
}
