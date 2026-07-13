"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState, type PropsWithChildren } from "react";
import { getToken, decodeUserId } from "@/lib/session/token";

type Session = { userId: string };

const SessionContext = createContext<Session>({ userId: "" });

export function GuestSessionProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const pathname = usePathname();
  const [userId, setUserId] = useState<string>("");

  useEffect(() => {
    const token = getToken();
    const uid = token ? decodeUserId(token) : null;
    if (!uid) {
      if (pathname !== "/login") router.replace("/login");
      return;
    }
    setUserId(uid);
  }, [pathname, router]);

  return <SessionContext.Provider value={{ userId }}>{children}</SessionContext.Provider>;
}

// Name kept for backward compatibility with existing consumers.
export function useGuestSession() {
  return useContext(SessionContext);
}
