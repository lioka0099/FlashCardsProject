"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { Card, ProofSpan } from "@/lib/api/client";
import { buildSourceUrl } from "@/components/exam/lib/pdf-source-url";
import { fetchSourceBlob } from "@/components/exam/lib/fetch-source-blob";
import { computeHighlightItemIndices } from "@/components/exam/lib/pdf-highlight";

type TextSourceModalProps = {
  proof: ProofSpan;
  card: Card;
  userId: string;
  onClose: () => void;
};

export function TextSourceModal({ proof, card, userId, onClose }: TextSourceModalProps) {
  const fileUrl = buildSourceUrl(proof, card, userId);
  const [paragraphs, setParagraphs] = useState<string[] | null>(null);
  const [loadError, setLoadError] = useState(false);
  const firstMatchRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    let cancelled = false;
    setParagraphs(null);
    setLoadError(false);
    fetchSourceBlob(fileUrl)
      .then((blob) => blob.text())
      .then((text) => {
        if (!cancelled) setParagraphs(text.split("\n"));
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [fileUrl]);

  const matchIndices = paragraphs
    ? computeHighlightItemIndices(paragraphs, proof.text)
    : new Set<number>();
  const firstMatchIndex = matchIndices.size > 0 ? Math.min(...matchIndices) : null;

  useEffect(() => {
    if (firstMatchIndex !== null && firstMatchRef.current) {
      if (typeof firstMatchRef.current.scrollIntoView === 'function') {
        firstMatchRef.current.scrollIntoView({ block: "start" });
      }
    }
  }, [paragraphs, firstMatchIndex]);

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
        <div className="pdf-modal__body text-modal__body">
          {loadError
            ? "Could not load the source document."
            : !paragraphs
              ? "Loading document…"
              : paragraphs.map((paragraph, index) => (
                  <p
                    key={index}
                    ref={index === firstMatchIndex ? firstMatchRef : undefined}
                    className="text-modal__paragraph"
                  >
                    {matchIndices.has(index) ? (
                      <mark className="pdf-highlight">{paragraph}</mark>
                    ) : (
                      paragraph
                    )}
                  </p>
                ))}
        </div>
      </motion.section>
    </motion.div>
  );
}
