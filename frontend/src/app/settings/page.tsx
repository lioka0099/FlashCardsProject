"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { changePassword, getMe, updateMe } from "@/lib/api/auth";
import { ApiRequestError } from "@/lib/api/client";
import "./settings.css";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, isError } = useQuery({ queryKey: ["me"], queryFn: getMe });

  const profileMutation = useMutation({
    mutationFn: (fields: { name: string; email: string }) => updateMe(fields),
    onSuccess: (me) => queryClient.setQueryData(["me"], me),
  });

  const passwordMutation = useMutation({
    mutationFn: ({ current, next }: { current: string; next: string }) => changePassword(current, next),
  });

  function onSubmitProfile(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    profileMutation.mutate({
      name: String(formData.get("name") ?? "").trim(),
      email: String(formData.get("email") ?? "").trim(),
    });
  }

  function onSubmitPassword(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    passwordMutation.mutate(
      {
        current: String(formData.get("current_password") ?? ""),
        next: String(formData.get("new_password") ?? ""),
      },
      { onSuccess: () => form.reset() },
    );
  }

  return (
    <section className="settings">
      <div className="settings__bar">
        <Link href="/" className="settings__back">
          <span aria-hidden>←</span> Back to home
        </Link>
      </div>

      <div className="settings__stack">
        <div className="settings__card">
          <h1 className="settings__title">Account</h1>
          <p className="settings__subtitle">Update your display name and email.</p>

          {isLoading && <p className="settings__muted">Loading…</p>}
          {isError && <p className="settings__error">Could not load your account.</p>}

          {data && (
            <form className="settings__form" onSubmit={onSubmitProfile} key={data.email}>
              <label className="settings__field">
                <span>Name</span>
                <input name="name" defaultValue={data.name ?? ""} autoComplete="name" />
              </label>
              <label className="settings__field">
                <span>Email</span>
                <input name="email" type="email" required defaultValue={data.email ?? ""} autoComplete="email" />
              </label>

              {profileMutation.isError && (
                <p className="settings__error" role="alert">
                  {profileMutation.error instanceof ApiRequestError && profileMutation.error.status === 409
                    ? "That email is already registered."
                    : "Something went wrong. Please try again."}
                </p>
              )}
              {profileMutation.isSuccess && <p className="settings__success">Saved.</p>}

              <button className="settings__submit" type="submit" disabled={profileMutation.isPending}>
                {profileMutation.isPending ? "…" : "Save changes"}
              </button>
            </form>
          )}
        </div>

        <div className="settings__card">
          <h1 className="settings__title">Password</h1>
          <p className="settings__subtitle">Change your password.</p>

          <form className="settings__form" onSubmit={onSubmitPassword}>
            <label className="settings__field">
              <span>Current password</span>
              <input name="current_password" type="password" required autoComplete="current-password" />
            </label>
            <label className="settings__field">
              <span>New password</span>
              <input name="new_password" type="password" required minLength={6} autoComplete="new-password" />
            </label>

            {passwordMutation.isError && (
              <p className="settings__error" role="alert">
                {passwordMutation.error instanceof ApiRequestError && passwordMutation.error.status === 401
                  ? "Current password is incorrect."
                  : "Something went wrong. Please try again."}
              </p>
            )}
            {passwordMutation.isSuccess && <p className="settings__success">Password updated.</p>}

            <button className="settings__submit" type="submit" disabled={passwordMutation.isPending}>
              {passwordMutation.isPending ? "…" : "Update password"}
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}
