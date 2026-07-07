"use client";

import { useQuery } from "@tanstack/react-query";
import { getMe, logout } from "@/lib/api/auth";
import "./profile.css";

export default function ProfilePage() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["me"], queryFn: getMe });

  return (
    <section className="profile">
      <h1 className="profile__title">Your account</h1>
      {isLoading && <p>Loading…</p>}
      {isError && <p className="profile__error">Could not load your account.</p>}
      {data && (
        <dl className="profile__details">
          <div><dt>Name</dt><dd>{data.name ?? "—"}</dd></div>
          <div><dt>Email</dt><dd>{data.email ?? "—"}</dd></div>
        </dl>
      )}
      <button className="profile__logout" type="button" onClick={logout}>
        Log out
      </button>
    </section>
  );
}
