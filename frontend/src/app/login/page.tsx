"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { login, register } from "@/lib/api/auth";
import { ApiRequestError } from "@/lib/api/client";
import "./login.css";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "register") {
        await register(email, password, name || undefined);
      } else {
        await login(email, password);
      }
      router.replace("/");
    } catch (err) {
      if (err instanceof ApiRequestError && err.status === 409) {
        setError("That email is already registered.");
      } else if (err instanceof ApiRequestError && err.status === 401) {
        setError("Invalid email or password.");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="login__card" onSubmit={onSubmit}>
        <h1 className="login__title">{mode === "login" ? "Welcome back" : "Create your account"}</h1>

        {mode === "register" && (
          <label className="login__field">
            <span>Name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
          </label>
        )}
        <label className="login__field">
          <span>Email</span>
          <input
            type="email" required value={email}
            onChange={(e) => setEmail(e.target.value)} autoComplete="email"
          />
        </label>
        <label className="login__field">
          <span>Password</span>
          <input
            type="password" required minLength={6} value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
          />
        </label>

        {error && <p className="login__error" role="alert">{error}</p>}

        <button className="login__submit" type="submit" disabled={busy}>
          {busy ? "…" : mode === "login" ? "Log in" : "Sign up"}
        </button>

        <button
          type="button" className="login__toggle"
          onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}
        >
          {mode === "login" ? "Need an account? Sign up" : "Have an account? Log in"}
        </button>
      </form>
    </main>
  );
}
