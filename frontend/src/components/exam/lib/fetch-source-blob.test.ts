import { afterEach, describe, expect, it, vi } from "vitest";
import * as tokenModule from "@/lib/session/token";
import { fetchSourceBlob } from "@/components/exam/lib/fetch-source-blob";

describe("fetchSourceBlob", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("attaches the Authorization header when a token is present", async () => {
    vi.spyOn(tokenModule, "getToken").mockReturnValue("abc123");
    const body = new Blob(["hello"]);
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await fetchSourceBlob("http://example.com/documents/doc1/source");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0];
    expect(calledUrl).toBe("http://example.com/documents/doc1/source");
    const headers = calledInit.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer abc123");
  });

  it("does not set an Authorization header when there is no token", async () => {
    vi.spyOn(tokenModule, "getToken").mockReturnValue(null);
    const fetchMock = vi.fn().mockResolvedValue(new Response(new Blob(["x"]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await fetchSourceBlob("http://example.com/documents/doc1/source");

    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("Authorization")).toBeNull();
  });

  it("throws when the response is not ok", async () => {
    vi.spyOn(tokenModule, "getToken").mockReturnValue(null);
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchSourceBlob("http://example.com/x")).rejects.toThrow("401");
  });
});
