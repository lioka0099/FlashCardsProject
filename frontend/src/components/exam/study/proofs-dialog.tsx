"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { Card, ProofSpan } from "@/lib/api/client";
import dynamic from "next/dynamic";
import { buildSourceUrl } from "@/components/exam/lib/pdf-source-url";

// react-pdf (pdf.js) touches browser-only globals (DOMMatrix) at module load,
// so it must never evaluate during SSR. Load the modal client-only.
const PdfSourceModal = dynamic(
  () => import("@/components/exam/study/pdf-source-modal").then((m) => m.PdfSourceModal),
  { ssr: false },
);

type ProofsDialogProps = {
  isOpen: boolean;
  card: Card | null;
  userId: string;
  onClose: () => void;
};

export function ProofsDialog({
  isOpen,
  card,
  userId,
  onClose,
}: ProofsDialogProps) {
  const [openProofKeys, setOpenProofKeys] = useState<Set<string>>(new Set());
  const [sourceProof, setSourceProof] = useState<ProofSpan | null>(null);

  const closeDialog = useCallback(() => {
    setOpenProofKeys(new Set());
    setSourceProof(null);
    onClose();
  }, [onClose]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeDialog();
      }
    }
    document.addEventListener("keydown", handleEscape);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = previousOverflow;
    };
  }, [closeDialog, isOpen]);

  if (!isOpen || !card) {
    return null;
  }

  function toggleProof(key: string) {
    setOpenProofKeys((previous) => {
      const next = new Set(previous);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  return (
    <motion.div
      className="evidence-backdrop"
      role="presentation"
      onClick={closeDialog}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className="evidence"
        role="dialog"
        aria-modal="true"
        aria-label="Evidence and source context"
        onClick={(event) => event.stopPropagation()}
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
      >
        <header className="evidence__header">
          <div>
            <h2 className="evidence__title">Supporting evidence</h2>
            <p className="evidence__question">{card.question}</p>
          </div>
          <button
            className="evidence__close"
            type="button"
            onClick={closeDialog}
          >
            Close
          </button>
        </header>

        <div className="evidence__content">
          {card.proofs.length === 0 ? (
            <p className="evidence__empty">
              No supporting evidence was found for this card.
            </p>
          ) : (
            <ul className="evidence__list">
              {card.proofs.map((proof, proofIndex) => {
                const pageLabel = proof.page !== null ? String(proof.page) : "Unavailable";
                const proofKey = `${proof.doc_id}:${proof.page ?? "na"}:${proof.start}:${proof.end}`;
                const panelId = `proof-panel-${proofIndex}`;
                const isOpenProof = openProofKeys.has(proofKey);
                return (
                  <li key={proofKey} className="evidence__item">
                    <button
                      type="button"
                      className="evidence__trigger"
                      aria-expanded={isOpenProof}
                      aria-controls={panelId}
                      onClick={() => toggleProof(proofKey)}
                    >
                      <span className="evidence__trigger-title">
                        Source {proofIndex + 1}
                      </span>
                      <ChevronDown
                        size={16}
                        className={`evidence__chevron${
                          isOpenProof ? " evidence__chevron--open" : ""
                        }`}
                        aria-hidden="true"
                      />
                    </button>
                    <AnimatePresence initial={false}>
                      {isOpenProof ? (
                        <motion.div
                          id={panelId}
                          className="evidence__panel"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2, ease: "easeInOut" }}
                        >
                          <div className="evidence__panel-inner">
                            <blockquote className="evidence__quote">
                              {proof.text}
                            </blockquote>
                            <p className="evidence__meta">
                              Page: {pageLabel}
                            </p>
                            <button
                              className="evidence__jump"
                              type="button"
                              onClick={() =>
                                proof.is_pdf === false
                                  ? window.open(
                                      buildSourceUrl(proof, card, userId),
                                      "_blank",
                                      "noopener,noreferrer",
                                    )
                                  : setSourceProof(proof)
                              }
                            >
                              Open the source
                            </button>
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        {sourceProof ? (
          <PdfSourceModal
            proof={sourceProof}
            card={card}
            userId={userId}
            onClose={() => setSourceProof(null)}
          />
        ) : null}
      </motion.section>
    </motion.div>
  );
}
