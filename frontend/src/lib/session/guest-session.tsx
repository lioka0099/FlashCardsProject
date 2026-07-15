"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState, type PropsWithChildren } from "react";
import { getToken, decodeUserId } from "@/lib/session/token";

type Session = { userId: string };

const SessionContext = createContext<Session>({ userId: "" });

export function GuestSessionProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const pathname = usePathname();
  // Primitive state (not a single session object) so React's same-value bail-out
  // stops the effect from re-rendering forever if it ever re-fires with an
  // unchanged result (e.g. router's reference isn't stable across renders).
  const [userId, setUserId] = useState<string>("");
  const [isResolved, setIsResolved] = useState(false);

  useEffect(() => {
    const token = getToken();
    const uid = token ? decodeUserId(token) : null;
    if (!uid) {
      if (pathname !== "/login") router.replace("/login");
      return;
    }
    setUserId(uid);
    setIsResolved(true);
  }, [pathname, router]);

  // The login page doesn't need a resolved session and is the redirect target
  // above, so it must render even while session resolution is still pending.
  if (pathname === "/login") {
    return <SessionContext.Provider value={{ userId: "" }}>{children}</SessionContext.Provider>;
  }
  if (!isResolved) {
    return null;
  }
  return <SessionContext.Provider value={{ userId }}>{children}</SessionContext.Provider>;
}

// Name kept for backward compatibility with existing consumers.
export function useGuestSession() {
  return useContext(SessionContext);
}
