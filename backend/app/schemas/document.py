"""
Document Schemas

문서 업로드 및 검색 관련 스키마
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import BaseSchema, TimestampSchema


class DocumentType(str, Enum):
    """Document type values."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    WORD = "word"
    EXCEL = "excel"


class CitationType(str, Enum):
    """Citation document type."""

    LAW = "law"  # 법률
    PRECEDENT = "precedent"  # 판례
    REGULATION = "regulation"  # 시행령/규칙
    GUIDELINE = "guideline"  # 지침/가이드라인
    ARTICLE = "article"  # 논문/기사
    OTHER = "other"


# ============================================================================
# Document Schemas
# ============================================================================


class DocumentCreate(BaseModel):
    """Document upload metadata."""

    description: str | None = Field(default=None, max_length=500)


class DocumentUpdate(BaseModel):
    """Document update schema."""

    description: str | None = Field(default=None, max_length=500)


class DocumentResponse(TimestampSchema):
    """Document response schema."""

    id: UUID
    user_id: UUID
    file_name: str
    original_name: str
    file_type: DocumentType
    mime_type: str
    file_size: int
    description: str | None

    # OCR results
    extracted_text: str | None
    ocr_completed: bool
    ocr_confidence: float | None
    page_count: int | None

    # Status
    is_processed: bool

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return round(self.file_size / (1024 * 1024), 2)


class DocumentUploadResponse(BaseModel):
    """Document upload response with presigned URL."""

    document_id: UUID
    upload_url: str  # Presigned S3 URL
    expires_in: int  # Seconds until URL expires


class DocumentListResponse(BaseSchema):
    """Document list item (lighter than full response)."""

    id: UUID
    original_name: str
    file_type: DocumentType
    file_size: int
    is_processed: bool
    ocr_completed: bool
    created_at: datetime


# ============================================================================
# Citation Schemas
# ============================================================================


class CitationResponse(TimestampSchema):
    """Citation response schema."""

    id: UUID
    turn_id: UUID
    title: str
    content: str
    source: str
    source_url: str | None
    doc_type: CitationType | None
    category: str | None

    # Legal identifiers
    case_number: str | None
    law_number: str | None
    article_number: str | None

    # Search metadata
    relevance_score: float
    display_order: int


# ============================================================================
# Search Schemas
# ============================================================================


class SearchRequest(BaseModel):
    """Legal document search request."""

    query: str = Field(min_length=3, max_length=1000)
    doc_types: list[CitationType] | None = None
    categories: list[str] | None = None
    max_results: int = Field(default=10, ge=1, le=50)

    # Optional filters
    date_from: datetime | None = None
    date_to: datetime | None = None
    include_content: bool = True


class SearchResultItem(BaseSchema):
    """Individual search result item."""

    title: str
    content: str
    source: str
    source_url: str | None
    doc_type: CitationType
    category: str | None

    # Legal identifiers
    case_number: str | None
    law_number: str | None
    article_number: str | None

    # Search metadata
    relevance_score: float
    snippet: str  # Highlighted snippet

    # Additional metadata from Google File Search
    file_id: str | None = None
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    """Search response schema."""

    query: str
    results: list[SearchResultItem]
    total_results: int
    processing_time_ms: int

    # Search metadata
    search_id: str | None = None
    model_used: str | None = None


class SimilarCaseRequest(BaseModel):
    """Similar case search request."""

    case_description: str = Field(min_length=20, max_length=5000)
    category: str | None = None
    max_results: int = Field(default=5, ge=1, le=20)


class SimilarCaseResult(BaseSchema):
    """Similar case search result."""

    case_number: str
    case_title: str
    summary: str
    relevance_score: float
    decision_date: datetime | None
    court: str | None
    case_type: str | None
    key_points: list[str]
    source_url: str | None


class RelevantLawRequest(BaseModel):
    """Relevant law search request."""

    query: str = Field(min_length=5, max_length=1000)
    law_types: list[str] | None = None  # 법률, 시행령, 규칙 등
    max_results: int = Field(default=10, ge=1, le=30)


class RelevantLawResult(BaseSchema):
    """Relevant law search result."""

    law_name: str
    law_number: str | None
    article_number: str
    article_title: str
    content: str
    relevance_score: float
    effective_date: datetime | None
    amendment_date: datetime | None
    source_url: str | None


# ============================================================================
# RAG Statistics Schemas
# ============================================================================


class RAGStatsResponse(BaseModel):
    """RAG service statistics."""

    total_documents: int
    total_chunks: int
    corpus_id: str | None
    last_updated: datetime | None

    # By document type
    documents_by_type: dict[str, int]

    # Search statistics
    total_searches_today: int
    average_relevance_score: float
