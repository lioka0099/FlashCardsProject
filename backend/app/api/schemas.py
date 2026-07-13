"""
API Request/Response Schemas for FlashCards Application

This module defines Pydantic models for all API endpoints, ensuring
type safety and automatic validation.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# EXAM SCHEMAS
# =============================================================================

class ExamCreateRequest(BaseModel):
    """Request to create a new exam workspace."""
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Exam title")
    mode: str = Field(default="mastery", description="Learning mode: 'mastery' or 'exam'")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class RegisterRequest(BaseModel):
    """Request to create an account."""
    email: str = Field(..., description="User email (unique)")
    password: str = Field(..., min_length=6, description="Plaintext password (hashed server-side)")
    name: Optional[str] = Field(default=None, description="Display name")


class LoginRequest(BaseModel):
    """Request to log in."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="Plaintext password")


class ExamResponse(BaseModel):
    """Response for exam data."""
    exam_id: str
    user_id: str
    title: str
    mode: str
    created_at: str
    updated_at: str
    info: Dict[str, Any] = Field(default_factory=dict)


class ExamListResponse(BaseModel):
    """Response for listing exams."""
    exams: List[ExamResponse]


# =============================================================================
# DOCUMENT INGESTION SCHEMAS
# =============================================================================

class DocumentIngestResponse(BaseModel):
    """Response for document ingestion."""
    doc_id: str
    filename: str
    num_chunks: int

class IngestDocumentsResponse(BaseModel):
    """Response for document ingestion."""
    # ok: bool = True
    exam_id: str
    documents: List[DocumentIngestResponse]


# =============================================================================
# TOPIC SCHEMAS
# =============================================================================

class EvidenceSpan(BaseModel):
    """Evidence span proving topic label is grounded."""
    # Evidence IDs are generated when persisting to DB; may be absent in computed outputs.
    evidence_id: Optional[str] = None
    doc_id: str
    page: Optional[int] = None
    start: int
    end: int
    text: str


class TopicResponse(BaseModel):
    """Response for topic data."""
    topic_id: str
    exam_id: str
    label: str
    created_at: str
    n_chunks: Optional[int] = Field(None, description="Number of chunks assigned to this topic")
    info: Dict[str, Any] = Field(default_factory=dict)


class TopicWithEvidenceResponse(TopicResponse):
    """Topic response including evidence spans."""
    evidence: List[EvidenceSpan] = Field(default_factory=list)


class BuildTopicsRequest(BaseModel):
    """Request to build topics for an exam."""
    # exam_id is a path parameter: POST /exams/{exam_id}/topics/build
    overwrite: bool = Field(default=False, description="Delete existing topics if True")
    n_topics: Optional[int] = Field(None, description="Target number of topics (auto if None)")


class BuildTopicsResponse(BaseModel):
    """Response for topic building."""
    # ok: bool = True
    exam_id: str
    topics_created: int
    topics: List[TopicResponse]


class TopicListResponse(BaseModel):
    """Response for listing topics."""
    exam_id: str
    topics: List[TopicResponse]


# =============================================================================
# CARD SCHEMAS
# =============================================================================

class ProofSpan(BaseModel):
    """Proof/evidence span for a card answer."""
    # Proof IDs are generated when persisting to DB; retrieval outputs don't include IDs.
    proof_id: Optional[str] = None
    doc_id: str
    page: Optional[int] = None
    start: int
    end: int
    text: str
    score: float = 0.0
    is_pdf: bool = True


class CardResponse(BaseModel):
    """Response for card data."""
    card_id: str
    exam_id: str
    topic_id: str
    topic_label: Optional[str] = None
    question: str
    answer: str
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level; framework is in info.difficulty_framework")
    created_at: str
    status: str = Field(default="active", description="Card status: 'active' or 'archived'")
    proofs: List[ProofSpan] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


class GenerateStarterCardsRequest(BaseModel):
    """Request to generate starter cards."""
    # exam_id is a path parameter: POST /exams/{exam_id}/cards/generate
    user_id: str = Field(..., description="User ID")
    n: int = Field(default=5, ge=1, le=50, description="Number of cards to generate")
    difficulty: int = Field(default=1, ge=1, le=5, description="Target difficulty level; framework is selected by card route")


class GenerateStarterCardsResponse(BaseModel):
    """Response for starter card generation."""
    # ok: bool = True
    exam_id: str
    cards_generated: int
    cards: List[CardResponse]


class GenerateSingleCardRequest(BaseModel):
    """Request to generate a single card for a topic."""
    # exam_id and topic_id are path parameters: POST /exams/{exam_id}/topics/{topic_id}/cards/generate
    difficulty: int = Field(default=1, ge=1, le=5, description="Target difficulty level; framework is selected by card route")


class GenerateSingleCardResponse(BaseModel):
    """Response for single card generation."""
    # ok: bool = True
    card: Optional[CardResponse] = None
    error: Optional[str] = None

class CardListResponse(BaseModel):
    """Response for listing cards."""
    cards: List[CardResponse]
    total: int


# =============================================================================
# REVIEW SCHEMAS
# =============================================================================

class ReviewCardRequest(BaseModel):
    """Request to review/rate a card."""
    # exam_id and card_id are path parameters: POST /exams/{exam_id}/cards/{card_id}/review
    rating: Literal["i_knew_it", "almost_knew", "learned_now", "dont_understand"] = Field(
        ...,
        description=(
            "Rating button:\n"
            "- i_knew_it: retire exact card; schedule distant variant; store key facts (topic-scoped)\n"
            "- almost_knew: schedule soon; near-variant soon\n"
            "- learned_now: schedule soon-ish; mark low stability\n"
            "- dont_understand: trigger remediation sequence; lower difficulty within topic; retry later"
        ),
    )


class ReviewCardResponse(BaseModel):
    """Response for card review."""
    # ok: bool = True
    review_id: str
    card_id: str
    rating: Literal["i_knew_it", "almost_knew", "learned_now", "dont_understand"]
    # SRS scheduling data (updated after review)
    due_at: Optional[str] = None
    interval_days: Optional[float] = None
    ease: Optional[float] = None
    # Topic proficiency (updated after review)
    topic_proficiency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated proficiency for this topic")


# =============================================================================
# SESSION PLANNER SCHEMAS 
# =============================================================================

class NextCardResponse(BaseModel):
    """Response for next card selection."""
    card: Optional[CardResponse] = None
    reason: Optional[str] = Field(None, description="Why this card was selected")
    # If no card available
    no_cards_available: bool = False
    message: Optional[str] = None


class SessionEventRequest(BaseModel):
    """Request to log a session event."""
    # exam_id is a path parameter: POST /exams/{exam_id}/session/event
    event_type: str = Field(..., description="Event type: 'session_start', 'session_end', 'card_served'")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Additional event data")


class SessionEventResponse(BaseModel):
    """Response for session event logging."""
    # ok: bool = True
    event_id: str


# =============================================================================
# TOPIC PROFICIENCY SCHEMAS
# =============================================================================

class TopicProficiencyResponse(BaseModel):
    """Response for topic proficiency data."""
    topic_id: str
    topic_label: str
    proficiency: float = Field(..., ge=0.0, le=1.0, description="Proficiency score (0.0 to 1.0)")
    last_updated_at: str
    n_reviews: Optional[int] = Field(None, description="Number of reviews contributing to this proficiency")
    current_difficulty: Optional[int] = Field(None, description="Projected current difficulty for this topic")
    by_route: Optional[Dict[str, Any]] = Field(None, description="Route-specific proficiency state")


class TopicProgressResponse(BaseModel):
    """Response for topic progress overview."""
    exam_id: str
    user_id: str
    topics: List[TopicProficiencyResponse]
    overall_proficiency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average proficiency across all topics")


# =============================================================================
# DUE CARDS SCHEMAS
# =============================================================================

class DueCardsResponse(BaseModel):
    """Response for due cards."""
    exam_id: str
    due_count: int
    cards: List[CardResponse]