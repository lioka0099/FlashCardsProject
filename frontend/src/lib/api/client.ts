import { apiEndpoints, assertEndpointAllowed } from "@/lib/api/endpoints";

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
};

export class ApiRequestError extends Error {
  status: number;
  responseText: string;

  constructor(status: number, responseText: string) {
    super(`API request failed (${status}): ${responseText}`);
    this.name = "ApiRequestError";
    this.status = status;
    this.responseText = responseText;
  }
}

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE_URL;

function buildUrl(path: string) {
  assertEndpointAllowed(path);
  return new URL(path, apiBaseUrl).toString();
}

export async function apiRequest<TResponse>(path: string, options: RequestOptions = {}): Promise<TResponse> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");

  let body: BodyInit | undefined;
  if (options.body !== undefined) {
    if (options.body instanceof FormData) {
      body = options.body;
    } else {
      headers.set("Content-Type", "application/json");
      body = JSON.stringify(options.body);
    }
  }

  const response = await fetch(buildUrl(path), {
    ...options,
    headers,
    body,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new ApiRequestError(response.status, errorText);
  }

  if (response.status === 204) {
    return undefined as TResponse;
  }

  return (await response.json()) as TResponse;
}

export type ExamListItem = {
  exam_id: string;
  title: string;
  updated_at: string;
  created_at: string;
  mode: string;
  info: Record<string, unknown>;
};

type ListExamsResponse = {
  exams: ExamListItem[];
};

export type ExamDetails = {
  exam_id: string;
  user_id: string;
  title: string;
  mode: string;
  state: string;
  diagnostic_total: number;
  diagnostic_answered: number;
  diagnostic_started_at: string | null;
  diagnostic_completed_at: string | null;
  created_at: string;
  updated_at: string;
  info: Record<string, unknown>;
};

type CreateExamFromUploadPayload = {
  userId: string;
  title: string;
  files: File[];
  mode?: string;
};

export type ProgressStepStatus = "pending" | "active" | "done" | "failed";

export type ProgressStep = {
  key: string;
  label: string;
  detail: string;
  status: ProgressStepStatus;
};

export type BootstrapProgress = {
  steps: ProgressStep[];
  updated_at: string;
};

export type CreateExamFromUploadResponse = {
  exam_id: string;
  state: string;
};

export type ProofSpan = {
  proof_id?: string;
  doc_id: string;
  page: number | null;
  start: number;
  end: number;
  text: string;
  score: number;
};

export type Card = {
  card_id: string;
  exam_id: string;
  topic_id: string;
  topic_label: string | null;
  question: string;
  answer: string;
  difficulty: number;
  created_at: string;
  status: string;
  proofs: ProofSpan[];
  info: Record<string, unknown>;
};

type NextCardResponse = {
  card: Card | null;
  reason: string | null;
  no_cards_available: boolean;
  message: string | null;
};

type CardListResponse = {
  cards: Card[];
  total: number;
};

type SessionEventResponse = {
  event_id: string;
};

export type ReviewRating =
  | "i_knew_it"
  | "almost_knew"
  | "learned_now"
  | "dont_understand";

export type ReviewCardResponse = {
  review_id: string;
  card_id: string;
  rating: ReviewRating;
  due_at: string | null;
  interval_days: number | null;
  ease: number | null;
  topic_proficiency: number | null;
  diagnostic_answered: number;
  diagnostic_total: number;
  exam_state: string;
  idempotent_replay: boolean;
};

export type TopicProgressItem = {
  topic_id: string;
  topic_label: string;
  proficiency: number;
  last_updated_at: string;
  n_reviews: number | null;
};

export type ExamProgressResponse = {
  exam_id: string;
  user_id: string;
  topics: TopicProgressItem[];
  overall_proficiency: number | null;
};

export async function listRecentExams(userId: string, limit = 15): Promise<ExamListItem[]> {
  const query = new URLSearchParams({
    user_id: userId,
    limit: String(limit),
  });
  const response = await apiRequest<ListExamsResponse>(`${apiEndpoints.exams}?${query.toString()}`);
  return response.exams;
}

export async function createExamFromUpload(
  payload: CreateExamFromUploadPayload,
): Promise<CreateExamFromUploadResponse> {
  const formData = new FormData();
  formData.set("user_id", payload.userId);
  formData.set("title", payload.title.trim() || "New exam");
  formData.set("mode", payload.mode ?? "mastery");
  payload.files.forEach((file) => {
    formData.append("files", file);
  });

  return apiRequest<CreateExamFromUploadResponse>(apiEndpoints.examFromUpload, {
    method: "POST",
    body: formData,
  });
}

export async function getExamById(examId: string, userId: string): Promise<ExamDetails> {
  const query = new URLSearchParams({ user_id: userId });
  return apiRequest<ExamDetails>(`${apiEndpoints.examById(examId)}?${query.toString()}`);
}

export async function getSessionNextCard(examId: string, userId: string): Promise<NextCardResponse> {
  const query = new URLSearchParams({ user_id: userId });
  return apiRequest<NextCardResponse>(`${apiEndpoints.sessionNextCard(examId)}?${query.toString()}`);
}

export async function getSessionPreviousCard(examId: string, userId: string): Promise<NextCardResponse> {
  const query = new URLSearchParams({ user_id: userId });
  return apiRequest<NextCardResponse>(`${apiEndpoints.sessionPreviousCard(examId)}?${query.toString()}`);
}

export async function getPresentedHistory(
  examId: string,
  userId: string,
  limit = 500,
): Promise<Card[]> {
  const query = new URLSearchParams({ user_id: userId, limit: String(limit) });
  const response = await apiRequest<CardListResponse>(
    `${apiEndpoints.presentedHistory(examId)}?${query.toString()}`,
  );
  return response.cards;
}

export async function submitCardReview(
  examId: string,
  cardId: string,
  userId: string,
  rating: ReviewRating,
): Promise<ReviewCardResponse> {
  return apiRequest<ReviewCardResponse>(apiEndpoints.reviewCard(examId, cardId), {
    method: "POST",
    body: {
      user_id: userId,
      rating,
    },
  });
}

/**
 * Client-side safety guard used by workspace navigation:
 * only the active card can be rated, while previous/history cards stay read-only.
 */
export function assertActiveRateableCard(cardId: string, activeCardId: string | null) {
  if (!activeCardId || cardId !== activeCardId) {
    throw new Error("Only the active session card can be rated.");
  }
}

export async function getExamProgress(examId: string, userId: string): Promise<ExamProgressResponse> {
  const query = new URLSearchParams({ user_id: userId });
  return apiRequest<ExamProgressResponse>(`${apiEndpoints.examProgress(examId)}?${query.toString()}`);
}

export async function logSessionEvent(
  examId: string,
  userId: string,
  eventType: string,
  payload?: Record<string, unknown>,
): Promise<SessionEventResponse> {
  return apiRequest<SessionEventResponse>(apiEndpoints.sessionEvents(examId), {
    method: "POST",
    body: {
      user_id: userId,
      event_type: eventType,
      payload,
    },
  });
}
