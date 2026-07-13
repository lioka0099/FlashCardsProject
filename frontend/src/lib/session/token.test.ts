import { describe, expect, it } from "vitest";
import { decodeUserId } from "./token";

// Mirrors backend issue_token payload: base64url(JSON).sig
function fakeToken(sub: string): string {
  const payload = btoa(JSON.stringify({ sub, exp: 9999999999 }))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return `${payload}.signaturenotchecked`;
}

describe("decodeUserId", () => {
  it("extracts sub from a token payload", () => {
    expect(decodeUserId(fakeToken("user-abc"))).toBe("user-abc");
  });
  it("returns null for garbage", () => {
    expect(decodeUserId("garbage")).toBeNull();
    expect(decodeUserId("")).toBeNull();
  });
});
