import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GuestSessionProvider, useGuestSession } from "@/lib/session/guest-session";

const replaceMock = vi.fn();
const routerMock = { replace: replaceMock };
let pathname = "/exams/exam-1";

vi.mock("next/navigation", () => ({
  useRouter: () => routerMock,
  usePathname: () => pathname,
}));

const getTokenMock = vi.fn();
vi.mock("@/lib/session/token", () => ({
  getToken: () => getTokenMock(),
  // Mirrors backend issue_token payload: base64url(JSON).sig
  decodeUserId: (token: string) => {
    try {
      return JSON.parse(atob(token.split(".")[0])).sub ?? null;
    } catch {
      return null;
    }
  },
}));

function fakeToken(sub: string): string {
  return `${btoa(JSON.stringify({ sub }))}.signaturenotchecked`;
}

function Probe() {
  const { userId } = useGuestSession();
  return <div data-testid="probe">{userId}</div>;
}

describe("GuestSessionProvider", () => {
  beforeEach(() => {
    replaceMock.mockReset();
    getTokenMock.mockReset();
    pathname = "/exams/exam-1";
  });

  afterEach(() => {
    cleanup();
  });

  it("does not render children on a protected route until a valid token resolves", () => {
    getTokenMock.mockReturnValue(null);
    render(
      <GuestSessionProvider>
        <Probe />
      </GuestSessionProvider>,
    );
    expect(screen.queryByTestId("probe")).not.toBeInTheDocument();
    expect(replaceMock).toHaveBeenCalledWith("/login");
  });

  it("renders children with the resolved userId once a valid token is found", () => {
    getTokenMock.mockReturnValue(fakeToken("user-42"));
    render(
      <GuestSessionProvider>
        <Probe />
      </GuestSessionProvider>,
    );
    expect(screen.getByTestId("probe")).toHaveTextContent("user-42");
    expect(replaceMock).not.toHaveBeenCalled();
  });

  it("renders children immediately on /login even without a resolved session", () => {
    pathname = "/login";
    getTokenMock.mockReturnValue(null);
    render(
      <GuestSessionProvider>
        <Probe />
      </GuestSessionProvider>,
    );
    expect(screen.getByTestId("probe")).toBeInTheDocument();
    expect(replaceMock).not.toHaveBeenCalled();
  });
});
