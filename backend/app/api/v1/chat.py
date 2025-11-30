"""
Individual Chat API Endpoints

개인 채팅 API - CascadeFlow 기반 비용 최적화 법률 상담.
위원회 시스템과 동일한 RAG/Memory/Learning 인프라 사용.
"""

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.individual_chat_service import (
    IndividualChatService,
    ChatProvider,
    ChatConfig,
    get_chat_service,
)
from app.services.llm.legal_prompts_v4_3 import LegalDomain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Individual Chat"])


# =============================================================================
# Schemas
# =============================================================================


class ProviderInfo(BaseModel):
    """LLM 제공자 정보."""
    id: str
    name: str
    description: str
    models: dict


class ProvidersResponse(BaseModel):
    """제공자 목록 응답."""
    providers: list[ProviderInfo]
    default: str = "claude"


class CreateChatSessionRequest(BaseModel):
    """채팅 세션 생성 요청."""
    provider: Literal["claude", "gpt", "gemini", "grok"] = Field(
        default="claude",
        description="LLM 제공자 선택",
    )
    domain: Optional[str] = Field(
        default=None,
        description="법률 분야 (자동 감지 가능)",
    )


class ChatSessionResponse(BaseModel):
    """채팅 세션 응답."""
    session_id: str
    user_id: str
    provider: str
    provider_name: str
    domain: str
    created_at: str
    message_count: int = 0


class SendMessageRequest(BaseModel):
    """메시지 전송 요청."""
    content: str = Field(
        ...,
        description="사용자 메시지",
        min_length=1,
        max_length=10000,
    )
    provider: Optional[Literal["claude", "gpt", "gemini", "grok"]] = Field(
        default=None,
        description="제공자 변경 (선택)",
    )
    stream: bool = Field(
        default=True,
        description="스트리밍 응답 여부",
    )


class MessageResponse(BaseModel):
    """단일 메시지 응답."""
    id: str
    role: str
    content: str
    model: str
    provider: str
    timestamp: str
    tokens_used: int
    cascade_tier: Optional[str] = None


class ChatResponseModel(BaseModel):
    """채팅 응답 (비스트리밍)."""
    message: MessageResponse
    session_id: str
    rag_context_used: bool
    memory_context_used: bool
    learning_recorded: bool


class SessionHistoryResponse(BaseModel):
    """세션 히스토리 응답."""
    session_id: str
    provider: str
    domain: str
    messages: list[MessageResponse]
    total_tokens: int
    total_cost: float


class ChatStatsResponse(BaseModel):
    """채팅 서비스 통계."""
    total_messages: int
    drafter_success: int
    verifier_calls: int
    drafter_success_rate: str
    rag_hits: int
    learning_records: int
    active_sessions: int


# =============================================================================
# Dependencies
# =============================================================================

# 채팅 서비스 인스턴스 캐시
_chat_services: dict[str, IndividualChatService] = {}


async def get_chat_service_instance(
    db: AsyncSession = Depends(get_db),
) -> IndividualChatService:
    """채팅 서비스 인스턴스 반환."""
    # 간단한 싱글톤 패턴 (실제로는 사용자별로 관리해야 함)
    if "default" not in _chat_services:
        _chat_services["default"] = get_chat_service(db)
    return _chat_services["default"]


def get_current_user_id() -> str:
    """현재 사용자 ID 반환 (임시: 실제로는 인증에서 가져와야 함)."""
    # TODO: 실제 인증 시스템과 연동
    return "anonymous-user"


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers(
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """
    사용 가능한 LLM 제공자 목록 조회.

    각 제공자의 특징과 사용 모델(drafter/verifier) 정보 제공.
    """
    providers = chat_service.get_available_providers()
    return ProvidersResponse(
        providers=[ProviderInfo(**p) for p in providers],
        default="claude",
    )


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    request: CreateChatSessionRequest,
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
    user_id: str = Depends(get_current_user_id),
):
    """
    새 채팅 세션 생성.

    선택한 LLM 제공자와 법률 분야로 새 세션을 시작합니다.
    """
    try:
        provider = ChatProvider(request.provider)

        domain = LegalDomain.GENERAL_CIVIL
        if request.domain:
            try:
                domain = LegalDomain(request.domain)
            except ValueError:
                pass

        session = chat_service.create_session(
            user_id=user_id,
            provider=provider,
            domain=domain,
        )

        provider_info = {
            ChatProvider.CLAUDE: "Claude (Anthropic)",
            ChatProvider.GPT: "GPT (OpenAI)",
            ChatProvider.GEMINI: "Gemini (Google)",
            ChatProvider.GROK: "Grok (xAI)",
        }

        return ChatSessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            provider=session.provider.value,
            provider_name=provider_info[session.provider],
            domain=session.domain.value,
            created_at=session.created_at.isoformat(),
            message_count=len(session.messages),
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: str,
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """채팅 세션 조회."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    provider_info = {
        ChatProvider.CLAUDE: "Claude (Anthropic)",
        ChatProvider.GPT: "GPT (OpenAI)",
        ChatProvider.GEMINI: "Gemini (Google)",
        ChatProvider.GROK: "Grok (xAI)",
    }

    return ChatSessionResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        provider=session.provider.value,
        provider_name=provider_info[session.provider],
        domain=session.domain.value,
        created_at=session.created_at.isoformat(),
        message_count=len(session.messages),
    )


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """
    메시지 전송 및 응답 받기.

    CascadeFlow 기반으로 비용을 최적화하면서 응답을 생성합니다.
    - stream=True: SSE 스트리밍 응답
    - stream=False: 전체 응답 반환
    """
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 제공자 변경
    provider = None
    if request.provider:
        provider = ChatProvider(request.provider)

    if request.stream:
        # 스트리밍 응답
        async def generate():
            try:
                async for chunk in chat_service.send_message_stream(
                    session_id=session_id,
                    content=request.content,
                    provider=provider,
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
        # 비스트리밍 응답
        try:
            response = await chat_service.send_message(
                session_id=session_id,
                content=request.content,
                provider=provider,
            )

            return ChatResponseModel(
                message=MessageResponse(
                    id=response.message.id,
                    role=response.message.role,
                    content=response.message.content,
                    model=response.message.model,
                    provider=response.message.provider.value,
                    timestamp=response.message.timestamp.isoformat(),
                    tokens_used=response.message.tokens_used,
                    cascade_tier=response.message.cascade_tier,
                ),
                session_id=session_id,
                rag_context_used=response.rag_context is not None,
                memory_context_used=response.memory_context is not None,
                learning_recorded=response.learning_recorded,
            )
        except Exception as e:
            logger.error(f"Message error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """세션 대화 히스토리 조회."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = chat_service.get_session_history(session_id)

    return SessionHistoryResponse(
        session_id=session_id,
        provider=session.provider.value,
        domain=session.domain.value,
        messages=[MessageResponse(**msg) for msg in history],
        total_tokens=session.total_tokens,
        total_cost=session.total_cost,
    )


@router.patch("/sessions/{session_id}/provider")
async def change_provider(
    session_id: str,
    provider: Literal["claude", "gpt", "gemini", "grok"],
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """세션의 LLM 제공자 변경."""
    session = chat_service.update_session_provider(
        session_id=session_id,
        provider=ChatProvider(provider),
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "provider": provider, "message": "Provider changed"}


@router.get("/stats", response_model=ChatStatsResponse)
async def get_stats(
    chat_service: IndividualChatService = Depends(get_chat_service_instance),
):
    """채팅 서비스 통계 조회."""
    stats = chat_service.get_stats()
    return ChatStatsResponse(**stats)


# =============================================================================
# Combined Mode Endpoint
# =============================================================================


class ModeInfo(BaseModel):
    """상담 모드 정보."""
    id: str
    name: str
    description: str
    icon: str
    features: list[str]


class ModesResponse(BaseModel):
    """상담 모드 목록."""
    modes: list[ModeInfo]


@router.get("/modes", response_model=ModesResponse)
async def get_consultation_modes():
    """
    사용 가능한 상담 모드 조회.

    - council: 위원회 모드 (여러 AI 전문가 협의)
    - chat: 개인 채팅 모드 (단일 AI와 1:1 상담)
    """
    return ModesResponse(
        modes=[
            ModeInfo(
                id="council",
                name="AI 법률 자문 위원회",
                description="여러 AI 법률 전문가가 협의하여 종합적인 법률 자문을 제공합니다.",
                icon="users",
                features=[
                    "4명의 AI 전문가 협의",
                    "교차 검증으로 높은 정확도",
                    "다양한 관점의 법률 분석",
                    "의장의 최종 종합 의견",
                ],
            ),
            ModeInfo(
                id="chat",
                name="개인 법률 상담",
                description="선택한 AI 전문가와 1:1로 실시간 법률 상담을 진행합니다.",
                icon="message-circle",
                features=[
                    "원하는 AI 모델 선택 가능",
                    "실시간 대화형 상담",
                    "빠른 응답 속도",
                    "비용 효율적인 상담",
                ],
            ),
        ]
    )
