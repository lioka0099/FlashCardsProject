"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  // Home, the test-creation loading screen, the study screen, and the history
  // screen each render their own full-bleed chrome (their own "Back to Home"
  // topbar), so the shared app header is hidden on them.
  const pathname = usePathname();
  const isStudyScreen = /^\/exams\/[^/]+$/.test(pathname);
  const isHistoryScreen = /^\/exams\/[^/]+\/history$/.test(pathname);
  const showHeader =
    pathname !== "/" && !pathname.endsWith("/creating") && !isStudyScreen && !isHistoryScreen;

  return (
    <div className="app-shell">
      {showHeader ? (
        <header className="app-shell__header">
          <Link href="/" className="app-shell__brand app-shell__brand-link cursor-pointer" aria-label="Go to home">
            FlashCards Studio
          </Link>
        </header>
      ) : null}
      <main className="app-shell__main">{children}</main>
    </div>
  );
}
