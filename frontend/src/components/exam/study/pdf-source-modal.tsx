"use client";

import { motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import type { TextContent, TextItem } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import type { Card, ProofSpan } from "@/lib/api/client";
import { buildSourceUrl } from "@/components/exam/lib/pdf-source-url";
import { computeHighlightItemIndices } from "@/components/exam/lib/pdf-highlight";
import { fetchSourceBlob } from "@/components/exam/lib/fetch-source-blob";

// Self-host the pdf.js worker from the installed pdfjs-dist so its version
// always matches and there is no network/CDN dependency. Turbopack/Webpack
// resolve `new URL(..., import.meta.url)` to a bundled asset.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

// ponytail: fixed render width — every page renders at the same width so the
// stacked heights settle predictably. Make it responsive (ResizeObserver) only
// if small screens need it.
const PAGE_WIDTH = 760;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

type PdfSourceModalProps = {
  proof: ProofSpan;
  card: Card;
  userId: string;
  onClose: () => void;
};

export function PdfSourceModal({ proof, card, userId, onClose }: PdfSourceModalProps) {
  const fileUrl = buildSourceUrl(proof, card, userId);
  const citedPage = proof.page ?? 1;

  const [fileData, setFileData] = useState<ArrayBuffer | null>(null);
  const [loadError, setLoadError] = useState(false);
  // react-pdf resets/reloads the document whenever `file` changes by
  // *reference* (see react-pdf's own Document source), so this must stay
  // referentially stable across re-renders that don't change the bytes —
  // otherwise every setNumPages/forceRepaint re-render would re-trigger a
  // full PDF reload.
  const file = useMemo(() => (fileData ? { data: fileData } : null), [fileData]);
  const [numPages, setNumPages] = useState(0);
  const markSet = useRef<Set<number>>(new Set());
  const computed = useRef(false);
  const [, forceRepaint] = useState(0);

  // Auto-scroll to the cited page once every page has rendered (so the heights
  // above it are settled and the target offset is correct).
  const citedRef = useRef<HTMLDivElement>(null);
  const renderedCount = useRef(0);
  const scrolled = useRef(false);

  useEffect(() => {
    computed.current = false;
    markSet.current = new Set();
    renderedCount.current = 0;
    scrolled.current = false;
  }, [proof.text, fileUrl]);

  useEffect(() => {
    let cancelled = false;
    setFileData(null);
    setLoadError(false);
    fetchSourceBlob(fileUrl)
      .then((blob) => blob.arrayBuffer())
      .then((buffer) => {
        if (!cancelled) setFileData(buffer);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [fileUrl]);

  const handleTextSuccess = useCallback(
    (textContent: TextContent) => {
      if (computed.current) {
        return;
      }
      computed.current = true;
      const textItems: Array<{ str: string; original: number }> = [];
      textContent.items.forEach((item, i) => {
        if ("str" in item) {
          textItems.push({ str: (item as TextItem).str, original: i });
        }
      });
      const compact = computeHighlightItemIndices(
        textItems.map((t) => t.str),
        proof.text,
      );
      markSet.current = new Set([...compact].map((ci) => textItems[ci].original));
      // Repaint once so the text layer re-runs customTextRenderer with the
      // now-populated mark set.
      forceRepaint((n) => n + 1);
    },
    [proof.text],
  );

  const handlePageRender = useCallback(() => {
    renderedCount.current += 1;
    if (!scrolled.current && numPages > 0 && renderedCount.current >= numPages) {
      scrolled.current = true;
      citedRef.current?.scrollIntoView({ block: "start" });
    }
  }, [numPages]);

  return (
    <motion.div
      className="pdf-backdrop"
      role="presentation"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className="pdf-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Source document preview"
        onClick={(event) => event.stopPropagation()}
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
      >
        <header className="pdf-modal__header">
          <h2 className="pdf-modal__title">Source document</h2>
          <button className="pdf-modal__close" type="button" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="pdf-modal__body">
          {loadError ? (
            "Could not load the source document."
          ) : !file ? (
            "Loading document…"
          ) : (
            <Document
              file={file}
              onLoadSuccess={({ numPages: n }) => setNumPages(n)}
              onLoadError={() => setLoadError(true)}
            >
              {Array.from({ length: numPages }, (_, i) => {
                const pageNumber = i + 1;
                const isCited = pageNumber === citedPage;
                return (
                  <div
                    key={pageNumber}
                    ref={isCited ? citedRef : undefined}
                    className="pdf-page-wrap"
                  >
                    <Page
                      pageNumber={pageNumber}
                      width={PAGE_WIDTH}
                      onRenderSuccess={handlePageRender}
                      onGetTextSuccess={isCited ? handleTextSuccess : undefined}
                      customTextRenderer={
                        isCited
                          ? ({ str, itemIndex }) =>
                              markSet.current.has(itemIndex)
                                ? `<mark class="pdf-highlight">${escapeHtml(str)}</mark>`
                                : escapeHtml(str)
                          : undefined
                      }
                    />
                  </div>
                );
              })}
            </Document>
          )}
        </div>
      </motion.section>
    </motion.div>
  );
}
