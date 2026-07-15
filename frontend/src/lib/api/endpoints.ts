const FORBIDDEN_GENERATE_ENDPOINT = /\/exams\/[^/]+\/topics\/[^/]+\/cards\/generate\/?$/;

export const apiEndpoints = {
  authRegister: "/auth/register",
  authLogin: "/auth/login",
  authMe: "/auth/me",
  authChangePassword: "/auth/change-password",
  exams: "/exams",
  examById: (examId: string) => `/exams/${examId}`,
  examFromUpload: "/exams/from-upload",
  sessionNextCard: (examId: string) => `/exams/${examId}/session/next-card`,
  sessionEvents: (examId: string) => `/exams/${examId}/session/event`,
  examProgress: (examId: string) => `/exams/${examId}/progress`,
  presentedHistory: (examId: string) => `/exams/${examId}/cards/presented-history`,
  reviewCard: (examId: string, cardId: string) => `/exams/${examId}/cards/${cardId}/review`,
} as const;

/**
 * API guardrail: Phase 8 frontend must never call the excluded endpoint:
 * POST /exams/{exam_id}/topics/{topic_id}/cards/generate
 */
export function assertEndpointAllowed(path: string) {
  if (FORBIDDEN_GENERATE_ENDPOINT.test(path)) {
    throw new Error("Blocked API endpoint: /exams/{exam_id}/topics/{topic_id}/cards/generate");
  }
}
