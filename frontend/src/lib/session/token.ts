const TOKEN_KEY = "flashcards_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
}

// Reads the `sub` claim from our HMAC token WITHOUT verifying the signature.
// The server verifies; the client only needs the user_id for display/params.
export function decodeUserId(token: string): string | null {
  try {
    const payload = token.split(".")[0];
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(b64.padEnd(b64.length + ((4 - (b64.length % 4)) % 4), "="));
    const data = JSON.parse(json);
    return typeof data.sub === "string" ? data.sub : null;
  } catch {
    return null;
  }
}
