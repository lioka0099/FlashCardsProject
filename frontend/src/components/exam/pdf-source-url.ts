import type { Card, ProofSpan } from "@/lib/api/client";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE_URL;

export function hasExactOffsets(proof: ProofSpan): boolean {
  return (
    Number.isFinite(proof.start) &&
    Number.isFinite(proof.end) &&
    proof.end > proof.start
  );
}

export function buildSourceUrl(
  proof: ProofSpan,
  card: Card,
  userId: string,
): string {
  const isHttpUrl = /^https?:\/\//i.test(proof.doc_id);
  const url = isHttpUrl
    ? new URL(proof.doc_id)
    : new URL(
        `/documents/${encodeURIComponent(proof.doc_id)}/source`,
        apiBaseUrl,
      );
  if (!isHttpUrl) {
    url.searchParams.set("exam_id", card.exam_id);
    url.searchParams.set("user_id", userId);
  }
  if (proof.page !== null) {
    url.hash = `page=${proof.page}`;
  }
  if (hasExactOffsets(proof)) {
    url.searchParams.set("start", String(proof.start));
    url.searchParams.set("end", String(proof.end));
  }
  return url.toString();
}
