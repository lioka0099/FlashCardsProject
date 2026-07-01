import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CreatingTest } from "@/components/exam/creating-test";

const progress = {
  updated_at: "x",
  steps: [
    { key: "uploading", label: "Uploading document", detail: "Document uploaded successfully", status: "done" as const },
    { key: "reading", label: "Reading pages", detail: "Reading the document", status: "active" as const },
    { key: "understanding", label: "Understanding concepts", detail: "AI is identifying key concepts", status: "pending" as const },
    { key: "topics", label: "Finding important topics", detail: "Extracting relevant topics", status: "pending" as const },
    { key: "questions", label: "Creating questions", detail: "Generating high-quality questions", status: "pending" as const },
    { key: "finalizing", label: "Building your study set", detail: "Organizing and optimizing your flashcards", status: "pending" as const },
  ],
};

describe("CreatingTest", () => {
  it("renders header, file chip and all six steps", () => {
    render(<CreatingTest fileName="ML.pdf" progress={progress} />);
    expect(screen.getByText("Creating your test...")).toBeInTheDocument();
    expect(screen.getByText("ML.pdf")).toBeInTheDocument();
    expect(screen.getByText("Uploading document")).toBeInTheDocument();
    expect(screen.getByText("Building your study set")).toBeInTheDocument();
  });

  it("marks done/active steps via data-status for styling", () => {
    const { container } = render(<CreatingTest progress={progress} />);
    expect(container.querySelector('[data-status="done"]')).toBeTruthy();
    expect(container.querySelector('[data-status="active"]')).toBeTruthy();
    expect(container.querySelectorAll('[data-status="pending"]').length).toBe(4);
  });
});
