import { apiRequest } from "@/lib/api/client";
import { apiEndpoints } from "@/lib/api/endpoints";
import { setToken, clearToken } from "@/lib/session/token";

type AuthResponse = { token: string; user_id: string };
export type Me = { user_id: string; email: string | null; name: string | null; onboarded: boolean };

export async function register(email: string, password: string, name?: string): Promise<AuthResponse> {
  const res = await apiRequest<AuthResponse>(apiEndpoints.authRegister, {
    method: "POST",
    body: { email, password, name },
  });
  setToken(res.token);
  return res;
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const res = await apiRequest<AuthResponse>(apiEndpoints.authLogin, {
    method: "POST",
    body: { email, password },
  });
  setToken(res.token);
  return res;
}

export async function getMe(): Promise<Me> {
  return apiRequest<Me>(apiEndpoints.authMe);
}

export async function markOnboarded(): Promise<Me> {
  return apiRequest<Me>(apiEndpoints.authOnboarded, { method: "POST" });
}

export async function updateMe(fields: { name?: string; email?: string }): Promise<Me> {
  return apiRequest<Me>(apiEndpoints.authMe, { method: "PATCH", body: fields });
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  await apiRequest<{ ok: boolean }>(apiEndpoints.authChangePassword, {
    method: "POST",
    body: { current_password: currentPassword, new_password: newPassword },
  });
}

export function logout(): void {
  clearToken();
  if (typeof window !== "undefined") window.location.href = "/login";
}
