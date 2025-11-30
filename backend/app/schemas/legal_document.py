"""
Legal Document Schemas for RAG System

Google File Search 기반 법률 문서 검색 시스템용 스키마.
Phase 6: backend-rag 구현.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LegalDocumentType(str, Enum):
    """
    법률 문서 타입.

    Google File Search Corpus에 저장되는 문서 분류.
    """

    LAW = "law"  # 법률/법령 (국가법령정보센터)
    PRECEDENT = "precedent"  # 판례 (대법원, 고등법원)
    CONSTITUTIONAL = "constitutional"  # 헌법재판소 결정
    ARTICLE = "article"  # 법학 논문 (KCI, RISS)
    COMMENTARY = "commentary"  # 법률 해설 (법제처)


class LegalCategory(str, Enum):
    """법률 분야 카테고리."""

    CIVIL = "civil"  # 민사
    CRIMINAL = "criminal"  # 형사
    ADMINISTRATIVE = "administrative"  # 행정
    CONSTITUTIONAL = "constitutional"  # 헌법
    LABOR = "labor"  # 노동
    TAX = "tax"  # 세법
    COMMERCIAL = "commercial"  # 상사
    FAMILY = "family"  # 가사
    REAL_ESTATE = "real_estate"  # 부동산
    INTELLECTUAL_PROPERTY = "intellectual_property"  # 지식재산권
    OTHER = "other"  # 기타


class CourtType(str, Enum):
    """법원 종류."""

    SUPREME = "supreme"  # 대법원
    HIGH = "high"  # 고등법원
    DISTRICT = "district"  # 지방법원
    FAMILY = "family"  # 가정법원
    ADMINISTRATIVE = "administrative"  # 행정법원
    PATENT = "patent"  # 특허법원
    CONSTITUTIONAL = "constitutional"  # 헌법재판소


# ============================================================================
# Legal Document Metadata
# ============================================================================


class LegalDocumentMetadata(BaseModel):
    """
    법률 문서 메타데이터.

    Google File Search에 저장되는 문서의 메타데이터.
    검색 필터링 및 결과 표시에 사용됩니다.
    """

    doc_id: str = Field(..., description="문서 고유 식별자")
    doc_type: LegalDocumentType = Field(..., description="문서 타입")
    title: str = Field(..., max_length=500, description="문서 제목")
    source: str = Field(..., description="출처 (대법원, 국가법령정보센터 등)")
    source_url: Optional[str] = Field(default=None, description="출처 URL")

    # Law specific fields
    law_number: Optional[str] = Field(
        default=None,
        description="법률 번호 (예: 법률 제18846호)"
    )
    article_number: Optional[str] = Field(
        default=None,
        description="조문 번호 (예: 제23조 제1항)"
    )
    law_name: Optional[str] = Field(
        default=None,
        description="법률명 (예: 근로기준법)"
    )

    # Precedent specific fields
    case_number: Optional[str] = Field(
        default=None,
        description="사건번호 (예: 2023다12345)"
    )
    court: Optional[CourtType] = Field(
        default=None,
        description="판결 법원"
    )
    decision_date: Optional[date] = Field(
        default=None,
        description="판결/결정일"
    )
    case_type: Optional[str] = Field(
        default=None,
        description="사건 유형 (예: 손해배상(기), 해고무효확인)"
    )

    # Constitutional decision specific
    constitutional_type: Optional[str] = Field(
        default=None,
        description="헌법재판 유형 (위헌심판, 헌법소원 등)"
    )

    # Article/Commentary specific
    author: Optional[str] = Field(default=None, description="저자")
    journal: Optional[str] = Field(default=None, description="학술지명")
    publication_date: Optional[date] = Field(default=None, description="발행일")

    # Common fields
    effective_date: Optional[date] = Field(
        default=None,
        description="시행일/효력 발생일"
    )
    category: LegalCategory = Field(
        default=LegalCategory.OTHER,
        description="법률 분야"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="검색 키워드"
    )
    summary: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="요약"
    )

    class Config:
        """Pydantic config."""

        use_enum_values = True


# ============================================================================
# Legal Document
# ============================================================================


class LegalDocument(BaseModel):
    """
    Google File Search에 업로드할 법률 문서.

    문서 내용과 메타데이터를 포함합니다.
    """

    doc_id: str = Field(..., description="문서 고유 식별자")
    content: str = Field(..., description="문서 전체 내용")
    metadata: LegalDocumentMetadata = Field(..., description="문서 메타데이터")
    file_name: str = Field(..., description="파일명")
    mime_type: str = Field(
        default="text/plain",
        description="MIME 타입"
    )

    class Config:
        """Pydantic config."""

        use_enum_values = True


class LegalDocumentCreate(BaseModel):
    """법률 문서 생성 요청."""

    content: str = Field(..., min_length=10, description="문서 내용")
    title: str = Field(..., max_length=500, description="문서 제목")
    doc_type: LegalDocumentType = Field(..., description="문서 타입")
    source: str = Field(..., description="출처")
    source_url: Optional[str] = Field(default=None, description="출처 URL")

    # Optional metadata
    law_number: Optional[str] = None
    article_number: Optional[str] = None
    law_name: Optional[str] = None
    case_number: Optional[str] = None
    court: Optional[CourtType] = None
    decision_date: Optional[date] = None
    category: LegalCategory = LegalCategory.OTHER
    keywords: list[str] = Field(default_factory=list)
    summary: Optional[str] = None


class LegalDocumentResponse(BaseModel):
    """법률 문서 응답."""

    doc_id: str
    file_id: str  # Google File Search file ID
    corpus_id: str  # Google Corpus ID
    title: str
    doc_type: LegalDocumentType
    source: str
    category: LegalCategory
    uploaded_at: datetime
    status: str = "active"

    class Config:
        """Pydantic config."""

        use_enum_values = True


# ============================================================================
# RAG Search Schemas
# ============================================================================


class RAGSearchRequest(BaseModel):
    """RAG 검색 요청."""

    query: str = Field(..., min_length=3, max_length=2000, description="검색 질의")
    doc_types: Optional[list[LegalDocumentType]] = Field(
        default=None,
        description="문서 타입 필터"
    )
    categories: Optional[list[LegalCategory]] = Field(
        default=None,
        description="카테고리 필터"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="최대 결과 수"
    )
    min_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="최소 관련성 점수"
    )

    # Date filters
    date_from: Optional[date] = None
    date_to: Optional[date] = None

    # Advanced options
    include_grounding: bool = Field(
        default=True,
        description="Grounding generation 포함 여부"
    )


class RAGSearchResult(BaseModel):
    """RAG 검색 결과 항목."""

    doc_id: str
    title: str
    content: str  # 관련 청크 내용
    doc_type: LegalDocumentType
    source: str
    source_url: Optional[str] = None

    # Relevance
    relevance_score: float = Field(ge=0.0, le=1.0)
    snippet: str  # 하이라이트된 스니펫

    # Legal identifiers
    case_number: Optional[str] = None
    law_number: Optional[str] = None
    article_number: Optional[str] = None
    court: Optional[CourtType] = None
    decision_date: Optional[date] = None

    # Metadata
    category: LegalCategory
    keywords: list[str] = Field(default_factory=list)

    # Google File Search metadata
    file_id: Optional[str] = None
    chunk_index: Optional[int] = None

    class Config:
        """Pydantic config."""

        use_enum_values = True


class RAGSearchResponse(BaseModel):
    """RAG 검색 응답."""

    query: str
    results: list[RAGSearchResult]
    total_results: int
    processing_time_ms: int

    # Grounding generation (if requested)
    grounded_answer: Optional[str] = Field(
        default=None,
        description="RAG 기반 생성 답변"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="인용 출처 목록"
    )

    # Search metadata
    search_id: Optional[str] = None
    model_used: Optional[str] = None
    corpus_id: Optional[str] = None


# ============================================================================
# Grounded Generation Schemas
# ============================================================================


class GroundedGenerationRequest(BaseModel):
    """Grounded generation 요청."""

    query: str = Field(..., min_length=10, max_length=5000)
    context_prompt: Optional[str] = Field(
        default=None,
        description="추가 컨텍스트/지시사항"
    )
    max_results: int = Field(default=10, ge=1, le=30)
    doc_types: Optional[list[LegalDocumentType]] = None
    categories: Optional[list[LegalCategory]] = None

    # Generation options
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=4096, ge=100, le=16000)


class GroundedGenerationResponse(BaseModel):
    """Grounded generation 응답."""

    query: str
    generated_answer: str
    grounding_chunks: list[RAGSearchResult]
    citations: list[str]

    # Confidence and metadata
    confidence_score: Optional[float] = None
    processing_time_ms: int
    model_used: str
    corpus_id: str


# ============================================================================
# Corpus Management Schemas
# ============================================================================


class CorpusInfo(BaseModel):
    """Corpus 정보."""

    corpus_id: str
    display_name: str
    description: Optional[str] = None
    document_count: int
    created_at: datetime
    last_updated: Optional[datetime] = None


class CorpusStats(BaseModel):
    """Corpus 통계."""

    corpus_id: str
    total_documents: int
    total_chunks: int
    documents_by_type: dict[str, int]
    documents_by_category: dict[str, int]
    last_upload: Optional[datetime] = None
    storage_size_bytes: Optional[int] = None
