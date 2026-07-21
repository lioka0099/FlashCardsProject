import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("@/lib/api/client", () => ({
  apiRequest: vi.fn(),
}));
vi.mock("@/lib/session/token", () => ({
  setToken: vi.fn(),
  clearToken: vi.fn(),
}));

import { apiRequest } from "@/lib/api/client";
import { markOnboarded } from "@/lib/api/auth";

describe("markOnboarded", () => {
  beforeEach(() => vi.clearAllMocks());

  it("POSTs to /auth/me/onboarded and returns the updated Me", async () => {
    const me = { user_id: "u1", email: "a@b.co", name: "Ada", onboarded: true };
    vi.mocked(apiRequest).mockResolvedValue(me);

    const result = await markOnboarded();

    expect(apiRequest).toHaveBeenCalledWith("/auth/me/onboarded", { method: "POST" });
    expect(result).toEqual(me);
  });
});
