"use client";

import { useQuery } from "@tanstack/react-query";
import { getMe, logout } from "@/lib/api/auth";
import "./profile.css";

export default function ProfilePage() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["me"], queryFn: getMe });
  const initial = (data?.name ?? data?.email ?? "?").charAt(0).toUpperCase();

  return (
    <section className="profile">
      <div className="profile__card">
        <div className="profile__head">
          <span className="profile__avatar" aria-hidden>{initial}</span>
          <div>
            <h1 className="profile__title">Your account</h1>
            <p className="profile__subtitle">Manage your FlashCards profile.</p>
          </div>
        </div>

        {isLoading && <p className="profile__muted">Loading…</p>}
        {isError && <p className="profile__error">Could not load your account.</p>}
        {data && (
          <dl className="profile__details">
            <div className="profile__row"><dt>Name</dt><dd>{data.name ?? "—"}</dd></div>
            <div className="profile__row"><dt>Email</dt><dd>{data.email ?? "—"}</dd></div>
          </dl>
        )}

        <button className="profile__logout" type="button" onClick={logout}>
          Log out
        </button>
      </div>
    </section>
  );
}
