"""
Expert Chat Service

단일 전문가 LLM과의 1:1 채팅 서비스.
위원회 모드보다 비용 효율적인 간단한 법률 자문용.

v4.3.1 특징:
- 메모리 시스템 통합 (세션/단기/장기 메모리)
- 대화 히스토리 효율적 저장
- 키워드 기반 법률 도메인 자동 감지
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Optional
from uuid import UUID, uuid4

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.base import LegalContext, StreamingLLMClient
from app.services.llm.legal_prompts_v4_3 import (
    LegalDomain,
    detect_domains,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class ExpertProvider(str, Enum):
    """사용 가능한 전문가 LLM 제공자."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"


@dataclass
class ExpertInfo:
    """전문가 LLM 정보."""
    provider: ExpertProvider
    display_name: str
    model_id: str
    description: str
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    strengths: list[str] = field(default_factory=list)


# 전문가 LLM 정보 (비용은 OpenRouter 기준 대략적 추정)
EXPERT_INFO: dict[ExpertProvider, ExpertInfo] = {
    ExpertProvider.OPENAI: ExpertInfo(
        provider=ExpertProvider.OPENAI,
        display_name="GPT-5.1",
        model_id="openai/gpt-4.1",
        description="OpenAI의 최신 추론 모델. 복잡한 법률 분석에 강점.",
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.010,
        strengths=["복잡한 추론", "다단계 분석", "논리적 구성"],
    ),
    ExpertProvider.ANTHROPIC: ExpertInfo(
        provider=ExpertProvider.ANTHROPIC,
        display_name="Claude 4.5 Sonnet",
        model_id="anthropic/claude-sonnet-4",
        description="Anthropic의 Claude 모델. 신중하고 균형잡힌 법률 자문.",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        strengths=["신중한 분석", "위험 평가", "윤리적 고려"],
    ),
    ExpertProvider.GOOGLE: ExpertInfo(
        provider=ExpertProvider.GOOGLE,
        display_name="Gemini 2.5 Pro",
        model_id="google/gemini-2.5-pro-preview",
        description="Google의 Gemini 모델. 광범위한 지식 기반 자문.",
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        strengths=["광범위한 지식", "비용 효율성", "빠른 응답"],
    ),
    ExpertProvider.XAI: ExpertInfo(
        provider=ExpertProvider.XAI,
        display_name="Grok 4",
        model_id="x-ai/grok-2",
        description="xAI의 Grok 모델. 실용적이고 직접적인 법률 조언.",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.010,
        strengths=["실용적 조언", "명확한 설명", "직접적 답변"],
    ),
}


@dataclass
class ChatMessage:
    """채팅 메시지."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens_used: int = 0


@dataclass
class ExpertChatSession:
    """전문가 채팅 세션."""
    id: str
    user_id: str
    expert: ExpertProvider
    domain: LegalDomain
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """예상 비용 (USD)."""
        info = EXPERT_INFO[self.expert]
        input_cost = (self.total_input_tokens / 1000) * info.cost_per_1k_input
        output_cost = (self.total_output_tokens / 1000) * info.cost_per_1k_output
        return input_cost + output_cost

    @property
    def estimated_cost_krw(self) -> int:
        """예상 비용 (KRW, 대략적 환율 1350원 기준)."""
        return int(self.estimated_cost_usd * 1350)


class ExpertChatService:
    """
    단일 전문가 LLM 채팅 서비스.

    메모리 시스템 통합:
    - 세션 메모리: 현재 대화 컨텍스트 (Redis)
    - 단기 메모리: 최근 7일 대화 요약 (DB)
    - 장기 메모리: 사용자 패턴 학습 (DB)
    """

    def __init__(
        self,
        db: AsyncSession | None = None,
        redis: Redis | None = None,
    ):
        self.db = db
        self.redis = redis
        self._sessions: dict[str, ExpertChatSession] = {}  # In-memory session store
        self._memory_service = None  # Lazy initialization

    async def _get_memory_service(self):
        """메모리 서비스 lazy 초기화."""
        if self._memory_service is None and self.db:
            from app.services.memory_service import MemoryService
            self._memory_service = MemoryService(self.db)
        return self._memory_service

    def get_available_experts(self) -> list[dict]:
        """사용 가능한 전문가 목록 반환."""
        return [
            {
                "provider": info.provider.value,
                "display_name": info.display_name,
                "description": info.description,
                "strengths": info.strengths,
                "cost_per_1k_input": info.cost_per_1k_input,
                "cost_per_1k_output": info.cost_per_1k_output,
            }
            for info in EXPERT_INFO.values()
        ]

    def get_council_vs_expert_cost_comparison(self) -> dict:
        """위원회 모드 vs 전문가 모드 비용 비교."""
        # 위원회 모드: 4개 LLM + 피어리뷰 + 의장 합성
        council_estimated_cost_1k = sum(
            info.cost_per_1k_input + info.cost_per_1k_output
            for info in EXPERT_INFO.values()
        ) * 1.5  # 피어리뷰 + 의장 합성 비용 추가 (약 50%)

        # 전문가 모드: 1개 LLM만 사용
        expert_costs = {
            info.provider.value: info.cost_per_1k_input + info.cost_per_1k_output
            for info in EXPERT_INFO.values()
        }

        cheapest_expert = min(expert_costs, key=expert_costs.get)

        return {
            "council_mode": {
                "description": "4개 LLM 위원회 + 피어리뷰 + 의장 합성",
                "estimated_cost_per_query_usd": round(council_estimated_cost_1k * 2, 4),  # 평균 2K 토큰 가정
                "estimated_cost_per_query_krw": int(council_estimated_cost_1k * 2 * 1350),
            },
            "expert_mode": {
                "description": "단일 LLM 전문가 1:1 채팅",
                "costs_by_provider": {
                    provider: {
                        "cost_per_query_usd": round(cost * 2, 4),
                        "cost_per_query_krw": int(cost * 2 * 1350),
                    }
                    for provider, cost in expert_costs.items()
                },
                "cheapest": cheapest_expert,
                "savings_vs_council": f"{int((1 - expert_costs[cheapest_expert] / council_estimated_cost_1k) * 100)}%",
            },
        }

    async def create_session(
        self,
        user_id: str,
        expert: ExpertProvider,
        initial_query: str | None = None,
    ) -> ExpertChatSession:
        """새 채팅 세션 생성."""
        # 질문에서 도메인 자동 감지
        domain = LegalDomain.GENERAL_CIVIL
        if initial_query:
            detected = detect_domains(initial_query)
            if detected:
                domain = detected[0]

        session = ExpertChatSession(
            id=str(uuid4()),
            user_id=user_id,
            expert=expert,
            domain=domain,
        )

        self._sessions[session.id] = session

        # Redis에 세션 저장 (선택적)
        if self.redis:
            await self.redis.setex(
                f"expert_session:{session.id}",
                3600 * 24,  # 24시간 TTL
                json.dumps({
                    "id": session.id,
                    "user_id": session.user_id,
                    "expert": session.expert.value,
                    "domain": session.domain.value,
                    "created_at": session.created_at.isoformat(),
                }),
            )

        logger.info(f"Expert chat session created: {session.id} with {expert.value}")
        return session

    async def get_session(self, session_id: str) -> ExpertChatSession | None:
        """세션 조회."""
        return self._sessions.get(session_id)

    async def chat(
        self,
        session_id: str,
        message: str,
        stream: bool = True,
    ) -> AsyncGenerator[str, None] | str:
        """
        전문가와 채팅.

        Args:
            session_id: 세션 ID
            message: 사용자 메시지
            stream: 스트리밍 응답 여부

        Yields/Returns:
            스트리밍 시 토큰별 응답, 아니면 전체 응답
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        # 메시지 추가
        user_msg = ChatMessage(role="user", content=message)
        session.messages.append(user_msg)

        # LLM 클라이언트 생성
        client = await self._create_expert_client(session.expert, session.domain)

        # 대화 히스토리 구성
        conversation = [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages[:-1]  # 마지막 메시지 제외
        ]

        # 법률 컨텍스트 구성
        context = LegalContext(
            query=message,
            category=session.domain.value,
            conversation_history=conversation,
            rag_context=[],  # 단일 전문가 모드에서는 RAG 미사용 (비용 절감)
            user_tier="pro",
        )

        if stream:
            async def stream_response():
                full_response = ""
                input_tokens = 0
                output_tokens = 0

                async for chunk in client.stream_response(context):
                    full_response += chunk
                    output_tokens += 1  # 대략적 추정
                    yield chunk

                # 응답 완료 후 메시지 저장
                assistant_msg = ChatMessage(
                    role="assistant",
                    content=full_response,
                    tokens_used=output_tokens,
                )
                session.messages.append(assistant_msg)

                # 토큰 사용량 업데이트
                session.total_input_tokens += len(message.split()) * 1.3  # 대략적 추정
                session.total_output_tokens += output_tokens

                # 메모리 시스템에 대화 저장
                await self._save_to_memory(
                    session=session,
                    user_message=message,
                    assistant_message=full_response,
                )

                logger.info(
                    f"Expert chat response: session={session_id}, "
                    f"expert={session.expert.value}, tokens={output_tokens}"
                )

            return stream_response()
        else:
            # 비스트리밍 응답
            response = await client.generate_response(context)

            assistant_msg = ChatMessage(
                role="assistant",
                content=response,
            )
            session.messages.append(assistant_msg)

            # 메모리 시스템에 대화 저장
            await self._save_to_memory(
                session=session,
                user_message=message,
                assistant_message=response,
            )

            return response

    async def _create_expert_client(
        self,
        expert: ExpertProvider,
        domain: LegalDomain,
    ) -> StreamingLLMClient:
        """전문가 LLM 클라이언트 생성."""
        from app.services.llm.factory import LLMClientFactory
        from app.services.llm.legal_prompts_v4_3 import TaskComplexity

        # 전문가별 provider 매핑
        provider_map = {
            ExpertProvider.OPENAI: "openai",
            ExpertProvider.ANTHROPIC: "anthropic",
            ExpertProvider.GOOGLE: "google",
            ExpertProvider.XAI: "xai",
        }

        client = LLMClientFactory.create_council_member(
            provider=provider_map[expert],
            complexity=TaskComplexity.MEDIUM,
            domain=domain,
        )

        await client._initialize_client()
        return client

    async def _save_to_memory(
        self,
        session: ExpertChatSession,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        대화 내용을 메모리 시스템에 저장.

        저장 전략:
        1. Redis 세션 메모리: 현재 대화 컨텍스트 (실시간)
        2. DB 단기 메모리: 최근 대화 요약 (7일 보관)
        3. DB 장기 메모리: 사용자 패턴/선호도 학습
        """
        try:
            # 1. Redis 세션 메모리 업데이트 (빠른 컨텍스트 접근용)
            if self.redis:
                session_key = f"expert_chat_history:{session.id}"
                message_data = {
                    "turn": len(session.messages) // 2,
                    "user": user_message,
                    "assistant": assistant_message[:500],  # 요약 저장
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "expert": session.expert.value,
                    "domain": session.domain.value,
                }
                await self.redis.rpush(session_key, json.dumps(message_data))
                await self.redis.expire(session_key, 3600 * 24 * 7)  # 7일 TTL

            # 2. 메모리 서비스 연동 (DB 저장)
            memory_service = await self._get_memory_service()
            if memory_service:
                # 대화 턴 저장
                await memory_service.save_conversation_turn(
                    user_id=UUID(session.user_id) if len(session.user_id) == 36 else None,
                    consultation_id=None,  # 전문가 채팅은 별도 세션
                    turn_id=UUID(session.id),
                    turn_number=len(session.messages) // 2,
                    role="user",
                    content=user_message,
                    category=session.domain.value,
                    keywords=self._extract_keywords(user_message),
                )

                # 키워드 기반 학습 (장기 메모리)
                keywords = self._extract_keywords(user_message)
                if keywords:
                    await memory_service.learn_from_consultation(
                        user_id=UUID(session.user_id) if len(session.user_id) == 36 else None,
                        consultation_id=None,
                        category=session.domain.value,
                        keywords=keywords,
                    )

            logger.debug(f"Saved chat to memory: session={session.id}")

        except Exception as e:
            # 메모리 저장 실패는 채팅을 중단시키지 않음
            logger.warning(f"Failed to save to memory: {e}")

    def _extract_keywords(self, text: str) -> list[str]:
        """텍스트에서 법률 키워드 추출."""
        legal_terms = [
            "계약", "손해배상", "해고", "임대차", "상속", "이혼", "채권", "채무",
            "보증", "담보", "위자료", "양육권", "친권", "부동산", "등기", "소송",
            "재판", "항소", "상고", "조정", "중재", "형사", "민사", "행정",
            "노동", "근로", "임금", "퇴직금", "해약", "취소", "무효", "위반",
            "저작권", "특허", "상표", "영업비밀", "개인정보", "명예훼손",
        ]
        return [term for term in legal_terms if term in text][:10]

    async def get_session_cost(self, session_id: str) -> dict:
        """세션 비용 정보 조회."""
        session = await self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        expert_info = EXPERT_INFO[session.expert]

        return {
            "session_id": session_id,
            "expert": session.expert.value,
            "expert_name": expert_info.display_name,
            "message_count": len(session.messages),
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "estimated_cost_usd": round(session.estimated_cost_usd, 4),
            "estimated_cost_krw": session.estimated_cost_krw,
            "cost_rates": {
                "input_per_1k": expert_info.cost_per_1k_input,
                "output_per_1k": expert_info.cost_per_1k_output,
            },
        }

    async def close_session(self, session_id: str) -> dict:
        """세션 종료 및 비용 요약."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return {"error": "Session not found"}

        # Redis에서도 제거
        if self.redis:
            await self.redis.delete(f"expert_session:{session_id}")

        return {
            "session_id": session_id,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "total_messages": len(session.messages),
            "final_cost_usd": round(session.estimated_cost_usd, 4),
            "final_cost_krw": session.estimated_cost_krw,
        }
