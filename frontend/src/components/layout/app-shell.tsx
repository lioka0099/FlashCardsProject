import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  // Every route renders its own full-bleed chrome (its own "Back to Home"
  // topbar), so there is no shared app header.
  return (
    <div className="app-shell">
      <main className="app-shell__main">{children}</main>
    </div>
  );
}
