"""
Consultations API Endpoints

법률 상담 관련 API
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import (
    CurrentActiveUser,
    DBSession,
    ConsultationLimitedUser,
    get_redis,
    require_tier,
)
from app.core.exceptions import (
    ConsultationLimitExceededError,
    ConsultationNotFoundError,
    UnauthorizedError,
)
from app.models.user import User
from app.services.consultation_service import ConsultationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consultations", tags=["Consultations"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class MemoryContext(BaseModel):
    """v4.1 Memory system context for LLM prompts."""

    session_memory: str | None = Field(
        default=None,
        max_length=50000,
        description="현재 세션의 대화 맥락",
    )
    short_term_memory: str | None = Field(
        default=None,
        max_length=50000,
        description="최근 7일 내 상담 이력",
    )
    long_term_memory: str | None = Field(
        default=None,
        max_length=100000,
        description="전체 클라이언트 이력 및 과거 자문",
    )


class TaskComplexityLevel(str):
    """Task complexity for v4.1 parameter optimization."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class CreateConsultationRequest(BaseModel):
    """Create consultation request with v4.1 memory system support."""

    title: str = Field(min_length=1, max_length=255, description="상담 제목")
    category: str = Field(
        default="general",
        pattern="^(general|contract|intellectual_property|labor|criminal|family|real_estate|corporate|tax|other)$",
        description="법률 카테고리",
    )
    initial_query: str = Field(
        min_length=10,
        max_length=10000,
        description="초기 법률 질문 (10-10000자)",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="첨부 문서 ID 목록",
    )
    # v4.1 Memory System
    memory_context: MemoryContext | None = Field(
        default=None,
        description="v4.1 메모리 시스템 컨텍스트 (세션/단기/장기 기억)",
    )
    complexity: str = Field(
        default="medium",
        pattern="^(simple|medium|complex)$",
        description="작업 복잡도 (simple/medium/complex) - v4.1 파라미터 최적화용",
    )


class ConsultationTurnRequest(BaseModel):
    """Send message in consultation with v4.1 memory support."""

    query: str = Field(
        min_length=1,
        max_length=10000,
        description="후속 질문",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="첨부 문서 ID 목록",
    )
    # v4.1 Memory System
    memory_context: MemoryContext | None = Field(
        default=None,
        description="v4.1 메모리 시스템 컨텍스트 (업데이트된 세션/단기/장기 기억)",
    )
    complexity: str = Field(
        default="medium",
        pattern="^(simple|medium|complex)$",
        description="작업 복잡도 - v4.1 파라미터 최적화용",
    )


class ModelOpinionResponse(BaseModel):
    """Model opinion response."""

    model_name: str
    opinion_text: str
    legal_basis: str | None = None
    risk_assessment: str | None = None
    recommendations: str | None = None
    confidence_level: str | None = None


class PeerReviewResponse(BaseModel):
    """Peer review response."""

    reviewer_model: str
    reviewed_model: str
    review_text: str
    overall_score: float | None = None


class CitationResponse(BaseModel):
    """Citation response."""

    title: str
    content: str
    source: str
    source_url: str | None = None
    doc_type: str | None = None
    case_number: str | None = None
    law_number: str | None = None
    relevance_score: float = 0.0


class ConsultationTurnResponse(BaseModel):
    """Consultation turn response."""

    id: str
    turn_number: int
    user_query: str
    chairman_response: str | None = None
    model_opinions: list[ModelOpinionResponse] = []
    peer_reviews: list[PeerReviewResponse] = []
    citations: list[CitationResponse] = []
    status: str
    processing_time_ms: int | None = None


class ConsultationResponse(BaseModel):
    """Consultation response."""

    id: str
    title: str
    category: str
    status: str
    summary: str | None = None
    turn_count: int
    created_at: str
    updated_at: str


class ConsultationDetailResponse(ConsultationResponse):
    """Consultation detail with turns."""

    turns: list[ConsultationTurnResponse] = []


class ConsultationListResponse(BaseModel):
    """Paginated consultation list."""

    items: list[ConsultationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ConsultationLimitResponse(BaseModel):
    """Consultation limit information."""

    limit: int
    used: int
    remaining: int
    is_unlimited: bool


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "",
    response_model=ConsultationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="상담 생성",
    description="새로운 법률 상담을 생성합니다.",
)
async def create_consultation(
    data: CreateConsultationRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: ConsultationLimitedUser,
) -> ConsultationResponse:
    """
    Create a new legal consultation.

    This will:
    1. Create a new consultation session
    2. Start processing the initial query through the LLM council
    3. Return the consultation with pending status

    The processing happens in background and can be monitored via SSE stream.
    """
    redis = await get_redis()
    service = ConsultationService(db, redis)

    try:
        # Create consultation
        from app.schemas.consultation import ConsultationCreate

        consultation_data = ConsultationCreate(
            title=data.title,
            category=data.category,
            initial_query=data.initial_query,
            document_ids=data.document_ids,
        )

        consultation = await service.create_consultation(
            user_id=current_user.id,
            data=consultation_data,
        )

        # Build v4.1 memory context
        from app.services.consultation_service import MemoryContext as ServiceMemoryContext

        memory_ctx = None
        if data.memory_context:
            memory_ctx = ServiceMemoryContext(
                session_memory=data.memory_context.session_memory or "",
                short_term_memory=data.memory_context.short_term_memory or "",
                long_term_memory=data.memory_context.long_term_memory or "",
            )

        # Start background processing with v4.1 parameters
        background_tasks.add_task(
            service.start_turn_processing,
            consultation.id,
            1,  # First turn
            current_user.id,  # user_id for memory system
            memory_ctx,
            data.complexity,
        )

        return ConsultationResponse(
            id=str(consultation.id),
            title=consultation.title,
            category=consultation.category.value if consultation.category else "general",
            status=consultation.status.value if consultation.status else "pending",
            summary=consultation.summary,
            turn_count=consultation.turn_count or 1,
            created_at=consultation.created_at.isoformat(),
            updated_at=consultation.updated_at.isoformat(),
        )

    except ConsultationLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )


@router.get(
    "",
    response_model=ConsultationListResponse,
    summary="상담 목록 조회",
    description="사용자의 상담 목록을 페이지네이션으로 조회합니다.",
)
async def list_consultations(
    db: DBSession,
    current_user: CurrentActiveUser,
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(10, ge=1, le=100, description="페이지당 항목 수"),
    status: str | None = Query(None, description="상태 필터"),
    category: str | None = Query(None, description="카테고리 필터"),
) -> ConsultationListResponse:
    """
    List user's consultations with pagination.
    """
    service = ConsultationService(db)

    result = await service.list_consultations(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
        category=category,
    )

    return ConsultationListResponse(**result)


@router.get(
    "/limit",
    response_model=ConsultationLimitResponse,
    summary="상담 한도 조회",
    description="현재 사용자의 월간 상담 한도를 조회합니다.",
)
async def get_consultation_limit(
    db: DBSession,
    current_user: CurrentActiveUser,
) -> ConsultationLimitResponse:
    """
    Get current user's consultation limit.
    """
    service = ConsultationService(db)
    limit_info = await service.get_remaining_consultations(current_user)

    return ConsultationLimitResponse(**limit_info)


@router.get(
    "/{consultation_id}",
    response_model=ConsultationDetailResponse,
    summary="상담 상세 조회",
    description="상담의 상세 정보와 모든 대화 내역을 조회합니다.",
)
async def get_consultation(
    consultation_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> ConsultationDetailResponse:
    """
    Get consultation details with all turns.
    """
    service = ConsultationService(db)

    try:
        result = await service.get_consultation_detail(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="상담을 찾을 수 없습니다.",
            )

        return result

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/{consultation_id}/turns",
    response_model=ConsultationTurnResponse,
    status_code=status.HTTP_201_CREATED,
    summary="후속 질문 추가",
    description="기존 상담에 후속 질문을 추가합니다.",
)
async def add_turn(
    consultation_id: UUID,
    data: ConsultationTurnRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> ConsultationTurnResponse:
    """
    Add a new turn (follow-up question) to the consultation.

    This will process the query through the full LLM pipeline:
    1. Stage 1: 4 LLMs provide parallel opinions
    2. Stage 2: Claude Sonnet performs blind peer review
    3. Chairman: Claude Opus synthesizes final response
    """
    redis = await get_redis()
    service = ConsultationService(db, redis)

    try:
        from app.schemas.consultation import ConsultationTurnCreate

        turn_data = ConsultationTurnCreate(
            query=data.query,
            document_ids=data.document_ids,
        )

        turn = await service.add_turn(
            consultation_id=consultation_id,
            user_id=current_user.id,
            data=turn_data,
        )

        # Build v4.1 memory context
        from app.services.consultation_service import MemoryContext as ServiceMemoryContext

        memory_ctx = None
        if data.memory_context:
            memory_ctx = ServiceMemoryContext(
                session_memory=data.memory_context.session_memory or "",
                short_term_memory=data.memory_context.short_term_memory or "",
                long_term_memory=data.memory_context.long_term_memory or "",
            )

        # Start background processing with v4.1 parameters
        background_tasks.add_task(
            service.start_turn_processing,
            consultation_id,
            turn.turn_number,
            current_user.id,  # user_id for memory system
            memory_ctx,
            data.complexity,
        )

        return ConsultationTurnResponse(
            id=str(turn.id),
            turn_number=turn.turn_number,
            user_query=turn.user_query,
            chairman_response=None,
            model_opinions=[],
            peer_reviews=[],
            citations=[],
            status=turn.status.value if turn.status else "pending",
            processing_time_ms=None,
        )

    except ConsultationNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/{consultation_id}/stream",
    summary="진행 상황 스트리밍",
    description="SSE를 통해 상담 처리 진행 상황을 실시간으로 수신합니다.",
)
async def stream_consultation_progress(
    consultation_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> EventSourceResponse:
    """
    Stream consultation processing progress using Server-Sent Events (SSE).

    Events:
    - progress: Processing progress (step, percentage)
    - log: Processing log message
    - complete: Processing complete
    - error: Error occurred
    """
    redis = await get_redis()
    service = ConsultationService(db, redis)

    # Verify ownership
    try:
        consultation = await service.get_consultation(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )

        if not consultation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="상담을 찾을 수 없습니다.",
            )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    async def event_generator():
        async for event in service.stream_progress(consultation_id):
            yield {
                "event": event["type"],
                "data": event["data"] if isinstance(event["data"], str) else str(event["data"]),
            }

    return EventSourceResponse(event_generator())


@router.get(
    "/{consultation_id}/turns/{turn_id}",
    response_model=ConsultationTurnResponse,
    summary="특정 턴 조회",
    description="특정 턴의 상세 정보를 조회합니다.",
)
async def get_turn(
    consultation_id: UUID,
    turn_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> ConsultationTurnResponse:
    """
    Get specific turn details.
    """
    service = ConsultationService(db)

    try:
        consultation = await service.get_consultation(consultation_id, current_user.id)

        if not consultation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="상담을 찾을 수 없습니다.",
            )

        # Find specific turn
        turn = next((t for t in consultation.turns if t.id == turn_id), None)

        if not turn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="턴을 찾을 수 없습니다.",
            )

        return await service._build_turn_response(turn)

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.delete(
    "/{consultation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="상담 삭제",
    description="상담과 관련된 모든 데이터를 삭제합니다.",
)
async def delete_consultation(
    consultation_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> None:
    """
    Delete a consultation.
    """
    service = ConsultationService(db)

    try:
        await service.delete_consultation(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )
    except ConsultationNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/{consultation_id}/export",
    summary="PDF 내보내기",
    description="상담 내용을 PDF로 내보냅니다. Pro/Enterprise 전용.",
)
async def export_consultation_pdf(
    consultation_id: UUID,
    db: DBSession,
    current_user: User = Depends(require_tier(["pro", "enterprise"])),
):
    """
    Export consultation as PDF document.
    Requires Pro or Enterprise tier.
    """
    from fastapi.responses import StreamingResponse

    service = ConsultationService(db)

    try:
        pdf_bytes = await service.export_to_pdf(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )

        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=consultation_{consultation_id}.pdf"
            },
        )

    except ConsultationNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )
