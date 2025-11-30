"""
Consultation Schemas

법률 상담 관련 요청/응답 스키마
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import BaseSchema, TimestampSchema


class ConsultationCategory(str, Enum):
    """Legal consultation categories."""

    GENERAL = "general"
    CONTRACT = "contract"
    INTELLECTUAL_PROPERTY = "intellectual-property"
    LABOR = "labor"
    CRIMINAL = "criminal"
    ADMINISTRATIVE = "administrative"
    CORPORATE = "corporate"
    FAMILY = "family"
    REAL_ESTATE = "real-estate"


class ConsultationStatus(str, Enum):
    """Consultation status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Processing stage for real-time updates."""

    INITIALIZED = "initialized"
    ANALYZING_QUERY = "analyzing_query"
    SEARCHING_DOCUMENTS = "searching_documents"
    COLLECTING_OPINIONS = "collecting_opinions"
    PEER_REVIEWING = "peer_reviewing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Model Opinion Schemas
# ============================================================================


class ModelOpinionResponse(TimestampSchema):
    """Model opinion response schema."""

    id: UUID
    model_name: str
    model_version: str
    opinion_text: str
    legal_basis: str | None
    risk_assessment: str | None
    recommendations: str | None
    confidence_level: str | None
    tokens_input: int
    tokens_output: int
    processing_time_ms: int | None


class PeerReviewResponse(TimestampSchema):
    """Peer review response schema."""

    id: UUID
    reviewer_model: str
    review_text: str
    accuracy_score: int | None
    completeness_score: int | None
    practicality_score: int | None
    legal_basis_score: int | None
    overall_score: float | None
    strengths: str | None
    weaknesses: str | None
    suggestions: str | None


# ============================================================================
# Citation Schemas (referenced from document)
# ============================================================================


class CitationInTurn(BaseSchema):
    """Citation in consultation turn response."""

    id: UUID
    title: str
    content: str
    source: str
    source_url: str | None
    doc_type: str | None
    case_number: str | None
    law_number: str | None
    article_number: str | None
    relevance_score: float


# ============================================================================
# Consultation Turn Schemas
# ============================================================================


class ConsultationTurnCreate(BaseModel):
    """Create new consultation turn (follow-up question)."""

    query: str = Field(min_length=1, max_length=10000)
    attachment_ids: list[UUID] = Field(default_factory=list)
    document_ids: list[str] | None = None


class ConsultationTurnResponse(TimestampSchema):
    """Consultation turn response schema."""

    id: UUID
    turn_number: int
    user_query: str
    attached_document_ids: list[UUID] | None
    chairman_response: str | None
    status: ConsultationStatus
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    processing_time_ms: int | None
    tokens_used: int
    estimated_cost: float

    model_opinions: list[ModelOpinionResponse] = Field(default_factory=list)
    peer_reviews: list[PeerReviewResponse] = Field(default_factory=list)
    citations: list[CitationInTurn] = Field(default_factory=list)


# ============================================================================
# Consultation Schemas
# ============================================================================


class ConsultationCreate(BaseModel):
    """Create new consultation request."""

    query: str = Field(min_length=10, max_length=10000)
    category: ConsultationCategory = ConsultationCategory.GENERAL
    attachment_ids: list[UUID] = Field(default_factory=list)
    title: str | None = Field(default=None, max_length=255)


class ConsultationUpdate(BaseModel):
    """Update consultation schema."""

    title: str | None = Field(default=None, max_length=255)
    category: ConsultationCategory | None = None


class ConsultationResponse(TimestampSchema):
    """Full consultation response schema."""

    id: UUID
    user_id: UUID
    title: str
    category: ConsultationCategory
    status: ConsultationStatus
    summary: str | None
    turn_count: int
    total_tokens_used: int
    total_cost: float

    turns: list[ConsultationTurnResponse] = Field(default_factory=list)


class ConsultationListResponse(TimestampSchema):
    """Consultation list item schema (lighter than full response)."""

    id: UUID
    title: str
    category: ConsultationCategory
    status: ConsultationStatus
    summary: str | None
    turn_count: int
    total_cost: float


class ConsultationDetailResponse(BaseModel):
    """Detailed consultation response with all turns."""

    id: str
    title: str
    category: str
    status: str
    summary: str | None
    turn_count: int
    created_at: str
    updated_at: str
    turns: list[dict] = Field(default_factory=list)


# ============================================================================
# Progress Streaming Schemas
# ============================================================================


class ConsultationProgress(BaseModel):
    """Real-time consultation progress update (for SSE)."""

    consultation_id: UUID
    turn_id: UUID
    stage: ProcessingStage
    progress_percent: int = Field(ge=0, le=100)
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    current_model: str | None = None
    models_completed: int = 0
    models_total: int = 4
    citations_found: int = 0

    error: str | None = None
    error_code: str | None = None


class StreamingOpinion(BaseModel):
    """Streaming model opinion (for real-time display)."""

    model_name: str
    chunk: str
    is_complete: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamingReview(BaseModel):
    """Streaming peer review (for real-time display)."""

    reviewer_model: str
    target_model: str
    chunk: str
    is_complete: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamingSynthesis(BaseModel):
    """Streaming chairman synthesis (for real-time display)."""

    chunk: str
    section: str | None = None
    is_complete: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Analysis Result Schemas (v4.1 Format)
# ============================================================================


class LegalAnalysisResult(BaseSchema):
    """Structured legal analysis result from v4.1 prompt."""

    issue_summary: str
    legal_basis: list[str]
    applicable_laws: list[str]

    risk_level: str
    risk_factors: list[str]
    mitigation_strategies: list[str]

    recommendations: list[str]
    action_items: list[str]
    timeline_suggestions: str | None

    confidence_level: str
    confidence_explanation: str
    limitations: list[str]

    additional_considerations: list[str]
    related_cases: list[str]


class ChairmanSynthesis(BaseSchema):
    """Chairman's final synthesis following v4.1 format."""

    final_conclusion: str
    consensus_areas: list[str]
    divergent_views: list[str]

    integrated_analysis: LegalAnalysisResult

    priority_actions: list[str]
    long_term_recommendations: list[str]

    models_consulted: list[str]
    total_processing_time_ms: int
    total_tokens_used: int
