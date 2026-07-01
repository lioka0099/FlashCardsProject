import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UploadExamForm } from "@/components/home/upload-exam-form";

const pushMock = vi.fn();
const createExamFromUploadMock = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

vi.mock("@/lib/api/client", () => ({
  createExamFromUpload: (...args: unknown[]) => createExamFromUploadMock(...args),
}));

vi.mock("@/lib/session/guest-session", () => ({
  useGuestSession: () => ({ userId: "guest", mode: "guest" }),
}));

function renderWithQueryClient() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <UploadExamForm />
    </QueryClientProvider>,
  );
}

describe("UploadExamForm", () => {
  beforeEach(() => {
    createExamFromUploadMock.mockReset();
    pushMock.mockReset();
  });

  it("renders the create-test hero content and dropzone", () => {
    renderWithQueryClient();

    expect(
      screen.getByRole("heading", { name: "Upload your document and create a smart test" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Our AI will analyze your document/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Give your test a name")).toBeInTheDocument();
    expect(screen.getByText("Drag & drop your file here")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Test" })).toBeInTheDocument();
    expect(screen.getByText("PDF, DOCX, TXT up to 20MB")).toBeInTheDocument();
    expect(screen.getByText("AI-Powered")).toBeInTheDocument();
    expect(screen.getByText("Track Progress")).toBeInTheDocument();
  });

  it("uploads selected files and redirects to exam page", async () => {
    createExamFromUploadMock.mockResolvedValue({
      exam_id: "exam-123",
      state: "processing",
    });

    const { container } = renderWithQueryClient();
    const fileInput = container.querySelector('input[type="file"]');
    if (!fileInput) {
      throw new Error("Missing file input");
    }

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["hello"], "notes.txt", { type: "text/plain" })],
      },
    });

    fireEvent.change(screen.getByLabelText("Give your test a name"), {
      target: { value: "Networks exam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create Test" }));

    await waitFor(() => {
      expect(createExamFromUploadMock).toHaveBeenCalledTimes(1);
      expect(pushMock).toHaveBeenCalledWith("/exams/exam-123/creating");
    });
  });

  it("keeps previously selected files when adding more", async () => {
    const { container } = renderWithQueryClient();
    const fileInput = container.querySelector('input[type="file"]');
    if (!fileInput) {
      throw new Error("Missing file input");
    }

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["one"], "first.pdf", { type: "application/pdf" })],
      },
    });
    fireEvent.change(fileInput, {
      target: {
        files: [new File(["two"], "second.txt", { type: "text/plain" })],
      },
    });

    await waitFor(() => {
      expect(screen.getByText("first.pdf")).toBeInTheDocument();
      expect(screen.getByText("second.txt")).toBeInTheDocument();
      expect(screen.getByText("2 files ready to turn into a test")).toBeInTheDocument();
    });
  });

  it("adds dropped files to the existing list", async () => {
    const { container } = renderWithQueryClient();
    const uploadCard = container.querySelector(".dash-create");
    if (!uploadCard) {
      throw new Error("Missing upload card");
    }

    fireEvent.drop(uploadCard, {
      dataTransfer: {
        files: [new File(["first"], "a.pdf", { type: "application/pdf" })],
      },
    });
    fireEvent.drop(uploadCard, {
      dataTransfer: {
        files: [new File(["second"], "b.docx", { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" })],
      },
    });

    await waitFor(() => {
      expect(screen.getByText("a.pdf")).toBeInTheDocument();
      expect(screen.getByText("b.docx")).toBeInTheDocument();
      expect(screen.getByText("2 files ready to turn into a test")).toBeInTheDocument();
    });
  });

  it("keeps create button disabled until files and title are provided", async () => {
    const { container } = renderWithQueryClient();
    const fileInput = container.querySelector('input[type="file"]');
    if (!fileInput) {
      throw new Error("Missing upload input");
    }

    const createMagicButton = screen.getByRole("button", { name: "Create Test" });
    expect(createMagicButton).toBeDisabled();

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["content"], "ready.pdf", { type: "application/pdf" })],
      },
    });
    expect(createMagicButton).toBeDisabled();

    fireEvent.change(screen.getByLabelText("Give your test a name"), {
      target: { value: "Ready exam title" },
    });
    expect(createMagicButton).not.toBeDisabled();
  });

  it("shows a friendly message for 422 upload errors", async () => {
    createExamFromUploadMock.mockRejectedValue(
      Object.assign(
        new Error(
          'API request failed (422): {"error":"diagnostic_bootstrap_failed","message":"Only 2 diagnostic cards were created for 3 topics; at least 3 are required."}',
        ),
        { status: 422 },
      ),
    );

    const { container } = renderWithQueryClient();
    const fileInput = container.querySelector('input[type="file"]');
    if (!fileInput) {
      throw new Error("Missing file input");
    }

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["hello"], "notes.txt", { type: "text/plain" })],
      },
    });
    fireEvent.change(screen.getByLabelText("Give your test a name"), {
      target: { value: "Networks exam" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create Test" }));

    await waitFor(() => {
      expect(screen.getByText("Please review your files and title, then try again.")).toBeInTheDocument();
    });
  });
});
