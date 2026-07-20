import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { Tour, type TourStep } from "@/components/tour/tour";

const STEPS: TourStep[] = [
  { target: "#a", title: "First", body: "First body" },
  { target: "#b", title: "Second", body: "Second body" },
];

function mountTargets() {
  document.body.innerHTML = `<div id="a">A</div><div id="b">B</div>`;
}

beforeEach(() => {
  mountTargets();
  Element.prototype.scrollIntoView = vi.fn();
  Element.prototype.getBoundingClientRect = vi.fn(() => ({
    width: 100, height: 40, top: 100, left: 100, right: 200, bottom: 140, x: 100, y: 100,
    toJSON: () => {},
  })) as unknown as typeof Element.prototype.getBoundingClientRect;
});

describe("Tour", () => {
  const noop = () => {};

  it("renders the current step's title and body", async () => {
    render(<Tour steps={STEPS} stepIndex={0} onNext={noop} onBack={noop} onSkip={noop} onFinish={noop} />);
    expect(await screen.findByText("First")).toBeInTheDocument();
    expect(screen.getByText("First body")).toBeInTheDocument();
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
  });

  it("calls onNext when the Next button is clicked", async () => {
    const onNext = vi.fn();
    render(<Tour steps={STEPS} stepIndex={0} onNext={onNext} onBack={noop} onSkip={noop} onFinish={noop} />);
    fireEvent.click(await screen.findByRole("button", { name: "Next" }));
    expect(onNext).toHaveBeenCalledOnce();
  });

  it("shows Done and calls onFinish on the last step", async () => {
    const onFinish = vi.fn();
    render(<Tour steps={STEPS} stepIndex={1} onNext={noop} onBack={noop} onSkip={noop} onFinish={onFinish} />);
    fireEvent.click(await screen.findByRole("button", { name: "Done" }));
    expect(onFinish).toHaveBeenCalledOnce();
  });

  it("calls onSkip when Skip is clicked and on Escape", async () => {
    const onSkip = vi.fn();
    render(<Tour steps={STEPS} stepIndex={0} onNext={noop} onBack={noop} onSkip={onSkip} onFinish={noop} />);
    fireEvent.click(await screen.findByRole("button", { name: "Skip" }));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onSkip).toHaveBeenCalledTimes(2);
  });

  it("auto-skips (onNext) when the target is missing", async () => {
    document.body.innerHTML = ""; // no #a
    const onNext = vi.fn();
    render(<Tour steps={STEPS} stepIndex={0} onNext={onNext} onBack={noop} onSkip={noop} onFinish={noop} />);
    await waitFor(() => expect(onNext).toHaveBeenCalled(), { timeout: 2000 });
  });
});
