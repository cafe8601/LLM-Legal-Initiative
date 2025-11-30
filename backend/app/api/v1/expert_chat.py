"""
Expert Chat API Endpoints

단일 전문가 LLM 채팅 API.
위원회 모드보다 비용 효율적인 1:1 법률 자문.
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.expert_chat_service import (
    ExpertChatService,
    ExpertProvider,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/expert-chat", tags=["Expert Chat"])


# =============================================================================
# Schemas
# =============================================================================


class ExpertListResponse(BaseModel):
    """전문가 목록 응답."""
    experts: list[dict]
    cost_comparison: dict


class CreateSessionRequest(BaseModel):
    """채팅 세션 생성 요청."""
    expert: Literal["openai", "anthropic", "google", "xai"] = Field(
        ...,
        description="선택할 전문가 LLM",
    )
    initial_query: Optional[str] = Field(
        None,
        description="초기 질문 (도메인 자동 감지용)",
        max_length=5000,
    )


class CreateSessionResponse(BaseModel):
    """채팅 세션 생성 응답."""
    session_id: str
    expert: str
    expert_name: str
    domain: str
    created_at: str


class ChatRequest(BaseModel):
    """채팅 메시지 요청."""
    message: str = Field(
        ...,
        description="사용자 메시지",
        min_length=1,
        max_length=10000,
    )
    stream: bool = Field(
        True,
        description="스트리밍 응답 여부",
    )


class ChatResponse(BaseModel):
    """채팅 응답 (비스트리밍용)."""
    response: str
    session_id: str


class SessionCostResponse(BaseModel):
    """세션 비용 정보."""
    session_id: str
    expert: str
    expert_name: str
    message_count: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    estimated_cost_krw: int


class CloseSessionResponse(BaseModel):
    """세션 종료 응답."""
    session_id: str
    closed_at: str
    total_messages: int
    final_cost_usd: float
    final_cost_krw: int


# =============================================================================
# Dependencies
# =============================================================================


def get_expert_service() -> ExpertChatService:
    """전문가 채팅 서비스 의존성."""
    return ExpertChatService()


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/experts", response_model=ExpertListResponse)
async def list_experts(
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    사용 가능한 전문가 LLM 목록 조회.

    각 전문가의 특징, 강점, 비용 정보를 반환합니다.
    위원회 모드와의 비용 비교 정보도 포함됩니다.
    """
    experts = service.get_available_experts()
    cost_comparison = service.get_council_vs_expert_cost_comparison()

    return ExpertListResponse(
        experts=experts,
        cost_comparison=cost_comparison,
    )


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    새 채팅 세션 생성.

    선택한 전문가 LLM과의 1:1 채팅 세션을 시작합니다.
    초기 질문을 제공하면 법률 도메인을 자동으로 감지합니다.
    """
    try:
        expert = ExpertProvider(request.expert)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid expert: {request.expert}. "
                   f"Available: openai, anthropic, google, xai",
        )

    # TODO: 실제 구현에서는 인증된 사용자 ID 사용
    user_id = "demo_user"

    session = await service.create_session(
        user_id=user_id,
        expert=expert,
        initial_query=request.initial_query,
    )

    from app.services.expert_chat_service import EXPERT_INFO

    expert_info = EXPERT_INFO[expert]

    return CreateSessionResponse(
        session_id=session.id,
        expert=session.expert.value,
        expert_name=expert_info.display_name,
        domain=session.domain.value,
        created_at=session.created_at.isoformat(),
    )


@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    request: ChatRequest,
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    전문가와 채팅.

    스트리밍 모드(기본값)에서는 Server-Sent Events로 응답합니다.
    비스트리밍 모드에서는 전체 응답을 JSON으로 반환합니다.
    """
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.stream:
        async def generate():
            try:
                async for chunk in await service.chat(
                    session_id=session_id,
                    message=request.message,
                    stream=True,
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        response = await service.chat(
            session_id=session_id,
            message=request.message,
            stream=False,
        )

        return ChatResponse(
            response=response,
            session_id=session_id,
        )


@router.get("/sessions/{session_id}/cost", response_model=SessionCostResponse)
async def get_session_cost(
    session_id: str,
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    세션 비용 정보 조회.

    현재까지의 토큰 사용량과 예상 비용을 반환합니다.
    """
    cost_info = await service.get_session_cost(session_id)

    if "error" in cost_info:
        raise HTTPException(status_code=404, detail=cost_info["error"])

    return SessionCostResponse(**cost_info)


@router.delete("/sessions/{session_id}", response_model=CloseSessionResponse)
async def close_session(
    session_id: str,
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    채팅 세션 종료.

    세션을 종료하고 최종 비용 요약을 반환합니다.
    """
    result = await service.close_session(session_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return CloseSessionResponse(**result)


@router.get("/modes/comparison")
async def get_mode_comparison(
    service: ExpertChatService = Depends(get_expert_service),
):
    """
    위원회 모드 vs 전문가 모드 비교.

    각 모드의 특징과 비용을 비교합니다.
    사용자가 적합한 모드를 선택할 수 있도록 도움을 줍니다.
    """
    cost_comparison = service.get_council_vs_expert_cost_comparison()

    return {
        "modes": [
            {
                "id": "council",
                "name": "AI 법률 자문 위원회",
                "description": "4개의 AI 모델이 독립적으로 분석 후 상호 검증하고 의장이 최종 합성",
                "features": [
                    "4개 AI 모델의 다양한 관점",
                    "블라인드 피어리뷰로 품질 보증",
                    "의장 AI의 종합적인 최종 답변",
                    "높은 신뢰도와 정확성",
                ],
                "best_for": [
                    "복잡한 법률 분쟁",
                    "높은 금액이 걸린 사안",
                    "정확성이 중요한 경우",
                    "여러 관점이 필요한 경우",
                ],
                "estimated_cost": cost_comparison["council_mode"],
            },
            {
                "id": "expert",
                "name": "단일 전문가 채팅",
                "description": "선택한 1개의 AI 전문가와 1:1 대화",
                "features": [
                    "빠른 응답 속도",
                    "비용 효율적",
                    "자연스러운 대화 흐름",
                    "전문가 선택 가능",
                ],
                "best_for": [
                    "간단한 법률 질문",
                    "일반적인 법률 상담",
                    "비용이 제한된 경우",
                    "빠른 답변이 필요한 경우",
                ],
                "estimated_cost": cost_comparison["expert_mode"],
            },
        ],
        "recommendation": {
            "high_stakes": "council",
            "quick_question": "expert",
            "budget_conscious": "expert",
            "complex_issue": "council",
        },
    }
