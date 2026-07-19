"use client";

// Catches errors thrown in the root layout itself; it replaces <html>/<body>,
// so it can't rely on the normal layout chrome. Kept minimal but on-brand.
import "./globals.css";
import "./status.css";

export default function GlobalError({ reset }: { error: Error; reset: () => void }) {
  return (
    <html lang="en">
      <body>
        <main className="status">
          <div className="status__card">
            {/* eslint-disable-next-line @next/next/no-img-element -- small fixed-size logo, optimizer not needed */}
            <img className="status__brand-mark" src="/logo.png" alt="" aria-hidden />
            <span className="status__code">Oops</span>
            <h1 className="status__title">Something went wrong</h1>
            <p className="status__text">
              An unexpected error occurred. Please try again.
            </p>
            <div className="status__actions">
              <button className="status__btn" type="button" onClick={reset}>
                Try again
              </button>
            </div>
          </div>
        </main>
      </body>
    </html>
  );
}
