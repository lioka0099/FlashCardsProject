"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  // Home and the test-creation loading screen render their own full-bleed
  // chrome, so the shared app header is hidden on both.
  const pathname = usePathname();
  const showHeader = pathname !== "/" && !pathname.endsWith("/creating");

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
