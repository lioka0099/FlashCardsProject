import Link from "next/link";
import "./status.css";

export default function NotFound() {
  return (
    <main className="status">
      <div className="status__card">
        {/* eslint-disable-next-line @next/next/no-img-element -- small fixed-size logo, optimizer not needed */}
        <img className="status__brand-mark" src="/logo.png" alt="" aria-hidden />
        <span className="status__code">404</span>
        <h1 className="status__title">Page not found</h1>
        <p className="status__text">
          The page you&rsquo;re looking for doesn&rsquo;t exist or has moved.
        </p>
        <div className="status__actions">
          <Link className="status__btn" href="/">
            Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
