import type { Card, ProofSpan } from "@/lib/api/client";
import { buildSourceUrl } from "@/components/exam/lib/pdf-source-url";
import { fetchSourceBlob } from "@/components/exam/lib/fetch-source-blob";

function downloadFilename(docId: string): string {
  const base = docId.split("/").pop() || "document";
  return base.toLowerCase().endsWith(".docx") ? base : `${base}.docx`;
}

export async function downloadSource(proof: ProofSpan, card: Card, userId: string): Promise<void> {
  const blob = await fetchSourceBlob(buildSourceUrl(proof, card, userId));
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadFilename(proof.doc_id);
  try {
    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();
    URL.revokeObjectURL(url);
  }
}
