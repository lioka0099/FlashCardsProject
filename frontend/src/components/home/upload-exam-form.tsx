"use client";

import { useRouter } from "next/navigation";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type RefObject,
} from "react";
import { motion } from "framer-motion";
import { useMutation } from "@tanstack/react-query";
import { Brain, Sparkles, Target, TrendingUp, UploadCloud } from "lucide-react";
import { createExamFromUpload } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { InlineError } from "@/components/common/inline-error";
import { useGuestSession } from "@/lib/session/guest-session";

const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];

function buildFileKey(file: File) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function fileTag(fileName: string) {
  return fileName.split(".").pop()?.toUpperCase() ?? "FILE";
}

type SelectedUpload = {
  id: string;
  file: File;
};

type UploadExamFormProps = {
  id?: string;
  sectionRef?: RefObject<HTMLElement | null>;
  onFocusWithin?: () => void;
  deckNameInputId?: string;
};

export function UploadExamForm({
  id,
  sectionRef,
  onFocusWithin,
  deckNameInputId = "dash-deck-name",
}: UploadExamFormProps) {
  const router = useRouter();
  const filesInputRef = useRef<HTMLInputElement | null>(null);
  const uploadIdRef = useRef(0);
  const [isDragActive, setIsDragActive] = useState(false);
  const [selectedUploads, setSelectedUploads] = useState<SelectedUpload[]>([]);
  const [title, setTitle] = useState("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const uploadErrorTimeoutRef = useRef<number | null>(null);
  const { userId } = useGuestSession();

  const uploadMutation = useMutation({
    mutationFn: createExamFromUpload,
    onSuccess: (result) => {
      router.push(`/exams/${result.exam_id}/creating`);
    },
  });

  const fileHint = useMemo(() => {
    if (selectedUploads.length === 0) {
      return "No files yet. Drop your study material to get started.";
    }
    if (selectedUploads.length === 1) {
      return `Ready: ${selectedUploads[0].file.name}`;
    }
    return `${selectedUploads.length} files ready to turn into a test`;
  }, [selectedUploads]);
  const titleValue = title.trim();
  const hasFiles = selectedUploads.length > 0;
  const hasTitle = titleValue.length > 0;

  useEffect(() => {
    return () => {
      if (uploadErrorTimeoutRef.current !== null) {
        window.clearTimeout(uploadErrorTimeoutRef.current);
      }
    };
  }, []);

  function clearUploadError() {
    if (uploadErrorTimeoutRef.current !== null) {
      window.clearTimeout(uploadErrorTimeoutRef.current);
      uploadErrorTimeoutRef.current = null;
    }
    setUploadError(null);
  }

  function showUploadError(message: string) {
    if (uploadErrorTimeoutRef.current !== null) {
      window.clearTimeout(uploadErrorTimeoutRef.current);
    }
    setUploadError(message);
    uploadErrorTimeoutRef.current = window.setTimeout(() => {
      setUploadError(null);
      uploadErrorTimeoutRef.current = null;
    }, 4_500);
  }

  function isSupportedUpload(file: File) {
    const loweredName = file.name.toLowerCase();
    return SUPPORTED_EXTENSIONS.some((extension) => loweredName.endsWith(extension));
  }

  function updateFiles(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }

    const droppedFiles = Array.from(files);
    const validFiles = droppedFiles.filter(isSupportedUpload);
    const rejectedCount = droppedFiles.length - validFiles.length;

    const nextUploads = validFiles.map((file) => {
      uploadIdRef.current += 1;
      return {
        id: `${buildFileKey(file)}:${uploadIdRef.current}`,
        file,
      };
    });

    if (nextUploads.length > 0) {
      setSelectedUploads((previous) => [...previous, ...nextUploads]);
      uploadMutation.reset();
      if (rejectedCount === 0) {
        clearUploadError();
      }
    }

    if (rejectedCount > 0) {
      showUploadError("Some files were ignored. Only PDF, DOCX, and TXT files are supported.");
    }
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragActive(false);
    updateFiles(event.dataTransfer.files);
  }

  function onDragOver(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragActive) {
      setIsDragActive(true);
    }
  }

  function onDragLeave(event: DragEvent<HTMLElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node)) {
      return;
    }
    setIsDragActive(false);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (selectedUploads.length === 0 || titleValue.length === 0) {
      return;
    }

    try {
      await uploadMutation.mutateAsync({
        userId,
        title: titleValue,
        files: selectedUploads.map((upload) => upload.file),
        mode: "mastery",
      });
    } catch {
      // Error is already reflected in mutation state for UI messaging.
    }
  }

  return (
    <section
      id={id}
      ref={sectionRef}
      className="dash-create"
      tabIndex={-1}
      onFocus={onFocusWithin}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      aria-label="Create test from files"
    >
      <div className="dash-create__card">
        <form className="dash-create__grid" onSubmit={onSubmit}>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
          >
            <span className="dash-create__eyebrow">
              <Sparkles size={16} />
              Create a new test
            </span>
            <h2 className="dash-create__title">
              Upload your document and create a <em>smart test</em>
            </h2>
            <p className="dash-create__desc">
              Our AI will analyze your document and generate high-quality
              flashcards, tailored to your goals.
            </p>

            <label className="dash-create__label" htmlFor={deckNameInputId}>
              Give your test a name
            </label>
            <input
              className="dash-create__input"
              id={deckNameInputId}
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Enter test name..."
            />

            {uploadError ? (
              <p className="dash-msg dash-msg--err" role="alert">
                {uploadError}
              </p>
            ) : null}

            {uploadMutation.isError ? (
              <InlineError
                message={mapHomeApiError(uploadMutation.error, "home.upload.create_exam").message}
                messageClassName="dash-msg dash-msg--err"
              />
            ) : null}

            <button
              className="dash-submit"
              type="submit"
              disabled={!hasFiles || !hasTitle || uploadMutation.isPending}
            >
              <Sparkles size={18} />
              {uploadMutation.isPending ? "Creating…" : "Create Test"}
            </button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: "easeOut", delay: 0.08 }}
          >
            <div
              className={`dash-drop${isDragActive ? " dash-drop--active" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => filesInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  filesInputRef.current?.click();
                }
              }}
              aria-label="Drop files here or click to choose files"
            >
              <UploadCloud className="dash-drop__icon" size={56} aria-hidden="true" />
              <p className="dash-drop__title">
                {isDragActive ? "Drop them here" : "Drag & drop your file here"}
              </p>
              <p className="dash-drop__hint">PDF, DOCX, TXT up to 20MB</p>

              {selectedUploads.length > 0 ? (
                <ul className="dash-files" aria-label="Selected files">
                  {selectedUploads.map((upload) => (
                    <li key={upload.id} className="dash-file">
                      <span className="dash-file__tag">{fileTag(upload.file.name)}</span>
                      <span className="dash-file__name">{upload.file.name}</span>
                      <button
                        className="dash-file__remove"
                        type="button"
                        aria-label={`Remove ${upload.file.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedUploads((previous) =>
                            previous.filter((item) => item.id !== upload.id),
                          );
                        }}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <p className="dash-filehint" aria-live="polite">
              {fileHint}
            </p>
          </motion.div>

          <input
            ref={filesInputRef}
            type="file"
            hidden
            multiple
            onChange={(event) => {
              updateFiles(event.target.files);
              event.currentTarget.value = "";
            }}
            accept={SUPPORTED_EXTENSIONS.join(",")}
          />
        </form>

        <div className="dash-features">
          <article className="dash-feature">
            <span className="dash-feature__icon" aria-hidden="true">
              <Brain size={20} />
            </span>
            <h3 className="dash-feature__title">AI-Powered</h3>
            <p className="dash-feature__text">
              Advanced AI creates questions that match your document&rsquo;s
              content and your learning goals.
            </p>
          </article>
          <article className="dash-feature">
            <span className="dash-feature__icon" aria-hidden="true">
              <Target size={20} />
            </span>
            <h3 className="dash-feature__title">Smart &amp; Adaptive</h3>
            <p className="dash-feature__text">
              Questions adapt to your understanding and help you focus on what
              matters.
            </p>
          </article>
          <article className="dash-feature">
            <span className="dash-feature__icon" aria-hidden="true">
              <TrendingUp size={20} />
            </span>
            <h3 className="dash-feature__title">Track Progress</h3>
            <p className="dash-feature__text">
              Monitor your progress and see how your understanding improves over
              time.
            </p>
          </article>
        </div>
      </div>
    </section>
  );
}
