"use client";

import Link from "next/link";
import "./status.css";

export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <main className="status">
      <div className="status__card">
        {/* eslint-disable-next-line @next/next/no-img-element -- small fixed-size logo, optimizer not needed */}
        <img className="status__brand-mark" src="/logo.png" alt="" aria-hidden />
        <span className="status__code">Oops</span>
        <h1 className="status__title">Something went wrong</h1>
        <p className="status__text">
          An unexpected error occurred. You can try again or head back home.
        </p>
        <div className="status__actions">
          <button className="status__btn" type="button" onClick={reset}>
            Try again
          </button>
          <Link className="status__btn status__btn--ghost" href="/">
            Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
