"""
Search API Endpoints

법률 문서 검색 API (Google File Search 기반 RAG)
"""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.services.llm.rag_service import get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class CitationResponse(BaseModel):
    """Legal citation response."""

    title: str
    content: str
    source: str
    source_url: str | None = None
    doc_type: str | None = None
    category: str | None = None
    case_number: str | None = None
    law_number: str | None = None
    relevance_score: float = 0.0


class SearchResponse(BaseModel):
    """Search results response."""

    query: str
    results: list[CitationResponse]
    total: int
    search_latency_ms: int = 0


class AnswerWithCitationsResponse(BaseModel):
    """RAG answer with citations."""

    query: str
    answer: str
    citations: list[CitationResponse]
    model: str
    grounded: bool = False


class CorpusStatsResponse(BaseModel):
    """Corpus statistics response."""

    corpus_name: str
    total_documents: int = 0
    documents_by_type: dict[str, int] = {}
    documents_by_category: dict[str, int] = {}


# =============================================================================
# Helper Functions
# =============================================================================


def _document_to_citation(doc) -> CitationResponse:
    """Convert LegalDocument to CitationResponse."""
    return CitationResponse(
        title=doc.title,
        content=doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content,
        source=doc.source,
        source_url=doc.source_url,
        doc_type=doc.doc_type,
        category=doc.category,
        case_number=doc.case_number,
        law_number=doc.law_number,
        relevance_score=doc.relevance_score,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/legal", response_model=SearchResponse)
async def search_legal_documents(
    query: str = Query(..., min_length=5, max_length=500, description="검색 쿼리"),
    category: str | None = Query(None, description="법률 분야 필터"),
    doc_type: str | None = Query(None, description="문서 유형 필터 (판례, 법률, 조문 등)"),
    top_k: int = Query(10, ge=1, le=50, description="결과 수"),
) -> SearchResponse:
    """
    Search legal documents (Google File Search 기반).

    - **query**: 검색 쿼리 (최소 5자)
    - **category**: 법률 분야 필터 (contract, labor, criminal 등)
    - **doc_type**: 문서 유형 필터 (판례, 법률, 조문, 해설 등)
    - **top_k**: 반환할 결과 수 (1-50)

    Returns:
        검색 결과 목록과 메타데이터
    """
    try:
        rag_service = get_rag_service()

        result = await rag_service.search(
            query=query,
            category=category,
            doc_type=doc_type,
            top_k=top_k,
        )

        citations = [_document_to_citation(doc) for doc in result.documents]

        return SearchResponse(
            query=query,
            results=citations,
            total=result.total_found,
            search_latency_ms=result.search_latency_ms,
        )

    except Exception as e:
        logger.error(f"Legal search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="검색 서비스를 사용할 수 없습니다.",
        )


@router.get("/legal/answer", response_model=AnswerWithCitationsResponse)
async def search_with_answer(
    query: str = Query(..., min_length=5, max_length=1000, description="법률 질문"),
    category: str = Query("general", description="법률 분야"),
) -> AnswerWithCitationsResponse:
    """
    Search and generate answer with citations (Grounded Generation).

    Google File Search의 grounding 기능을 사용하여
    검색된 문서를 기반으로 답변을 생성합니다.

    - **query**: 법률 질문 (최소 5자)
    - **category**: 법률 분야 (general, contract, labor 등)

    Returns:
        생성된 답변과 인용 문서 목록
    """
    try:
        rag_service = get_rag_service()

        result = await rag_service.generate_with_grounding(
            query=query,
            category=category,
        )

        # Convert citations
        citations = []
        for citation in result.get("citations", []):
            citations.append(CitationResponse(
                title="",
                content=citation.get("content", ""),
                source=citation.get("source", ""),
                source_url=citation.get("url"),
            ))

        return AnswerWithCitationsResponse(
            query=query,
            answer=result.get("answer", ""),
            citations=citations,
            model=result.get("model", "unknown"),
            grounded=result.get("grounded", False),
        )

    except Exception as e:
        logger.error(f"Grounded generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="답변 생성 서비스를 사용할 수 없습니다.",
        )


@router.get("/similar-cases", response_model=SearchResponse)
async def find_similar_cases(
    description: str = Query(..., min_length=10, max_length=2000, description="사건 설명"),
    top_k: int = Query(5, ge=1, le=20, description="결과 수"),
) -> SearchResponse:
    """
    Find similar legal precedents.

    사건 설명을 기반으로 유사한 판례를 검색합니다.

    - **description**: 사건 설명 (최소 10자)
    - **top_k**: 반환할 결과 수 (1-20)

    Returns:
        유사한 판례 목록
    """
    try:
        rag_service = get_rag_service()

        documents = await rag_service.search_similar_cases(
            case_description=description,
            top_k=top_k,
        )

        citations = [_document_to_citation(doc) for doc in documents]

        return SearchResponse(
            query=description,
            results=citations,
            total=len(citations),
        )

    except Exception as e:
        logger.error(f"Similar cases search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="판례 검색 서비스를 사용할 수 없습니다.",
        )


@router.get("/relevant-laws", response_model=SearchResponse)
async def find_relevant_laws(
    topic: str = Query(..., min_length=5, max_length=500, description="법률 주제"),
    category: str | None = Query(None, description="법률 분야"),
    top_k: int = Query(5, ge=1, le=20, description="결과 수"),
) -> SearchResponse:
    """
    Find relevant laws and statutes.

    주제를 기반으로 관련 법률 조문을 검색합니다.

    - **topic**: 법률 주제 (최소 5자)
    - **category**: 법률 분야 필터
    - **top_k**: 반환할 결과 수 (1-20)

    Returns:
        관련 법률 조문 목록
    """
    try:
        rag_service = get_rag_service()

        documents = await rag_service.search_relevant_laws(
            topic=topic,
            category=category,
            top_k=top_k,
        )

        citations = [_document_to_citation(doc) for doc in documents]

        return SearchResponse(
            query=topic,
            results=citations,
            total=len(citations),
        )

    except Exception as e:
        logger.error(f"Relevant laws search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="법률 검색 서비스를 사용할 수 없습니다.",
        )


@router.get("/stats", response_model=CorpusStatsResponse)
async def get_corpus_stats() -> CorpusStatsResponse:
    """
    Get legal document corpus statistics.

    저장된 문서 수, 유형별/분야별 분포를 반환합니다.

    Returns:
        코퍼스 통계 정보
    """
    try:
        rag_service = get_rag_service()

        stats = await rag_service.get_corpus_stats()

        return CorpusStatsResponse(
            corpus_name=stats.get("corpus_name", ""),
            total_documents=stats.get("total_documents", 0),
            documents_by_type=stats.get("documents_by_type", {}),
            documents_by_category=stats.get("documents_by_category", {}),
        )

    except Exception as e:
        logger.error(f"Corpus stats error: {e}")
        return CorpusStatsResponse(
            corpus_name="legal-documents-corpus",
            total_documents=0,
            documents_by_type={},
            documents_by_category={},
        )
