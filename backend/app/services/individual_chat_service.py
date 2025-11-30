"""
Individual Chat Service - 개인 채팅 서비스

위원회 시스템과 동일한 인프라를 사용하는 1:1 법률 상담 채팅.
- CascadeFlow 기반 비용 최적화
- 법률 전문가 프롬프트 사용
- RAG/Memory/Learning 시스템 통합
- 모델 선택 기능 (Claude, GPT, Gemini, Grok)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Optional
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.cascade_service import (
    CascadeService,
    CascadeServiceFactory,
    CascadeResult,
    CascadeModelTier,
    CLAUDE_CASCADE,
    GPT_CASCADE,
    GEMINI_CASCADE,
    GROK_CASCADE,
)
from app.services.llm.legal_prompts_v4_3 import (
    LegalDomain,
    LLMModel,
    TaskComplexity,
    assemble_expert_prompt,
    detect_domains,
)
from app.services.llm.openrouter_client import OpenRouterClient
from app.services.learning import (
    LearningManager,
    get_learning_manager,
    OutcomeStatus,
)
from app.services.memory import (
    MemoryManager,
    MemoryType,
    get_memory_manager,
)
from app.services.rag.hybrid_orchestrator import (
    HybridRAGOrchestrator,
    HybridSearchResult,
    RAGConfig,
)

logger = logging.getLogger(__name__)


class ChatProvider(str, Enum):
    """채팅에 사용 가능한 LLM 제공자."""
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    GROK = "grok"


@dataclass
class ChatConfig:
    """개인 채팅 설정."""
    # CascadeFlow 설정
    enable_cascade: bool = True
    min_drafter_confidence: float = 0.6

    # RAG 설정
    enable_rag: bool = True
    max_legal_docs: int = 5
    max_experiences: int = 3

    # Memory 설정
    enable_memory: bool = True
    memory_ttl: int = 3600  # 1시간

    # Learning 설정
    enable_learning: bool = True
    auto_record_outcomes: bool = True

    # 응답 설정
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class ChatMessage:
    """채팅 메시지."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    model: str
    provider: ChatProvider
    timestamp: datetime
    tokens_used: int = 0
    cost_usd: float = 0.0
    cascade_tier: Optional[str] = None  # "drafter" or "verifier"
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatSession:
    """채팅 세션."""
    session_id: str
    user_id: str
    provider: ChatProvider
    domain: LegalDomain
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_tokens: int = 0
    total_cost: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """채팅 응답."""
    message: ChatMessage
    session: ChatSession
    rag_context: Optional[str] = None
    memory_context: Optional[dict] = None
    learning_recorded: bool = False


class IndividualChatService:
    """
    개인 채팅 서비스.

    위원회 시스템과 동일한 인프라(CascadeFlow, RAG, Memory, Learning)를 사용하여
    개별 LLM과 1:1 법률 상담을 제공합니다.

    주요 기능:
    1. 모델 선택 - Claude, GPT, Gemini, Grok 중 선택
    2. CascadeFlow - 비용 최적화된 응답 생성
    3. RAG 통합 - 법률 문서 및 경험 기반 컨텍스트
    4. Memory 통합 - 세션/단기/장기 메모리
    5. Learning 통합 - 응답 품질 학습 및 개선
    """

    # 제공자별 CascadePair 매핑
    PROVIDER_CASCADE_MAP = {
        ChatProvider.CLAUDE: CLAUDE_CASCADE,
        ChatProvider.GPT: GPT_CASCADE,
        ChatProvider.GEMINI: GEMINI_CASCADE,
        ChatProvider.GROK: GROK_CASCADE,
    }

    # 제공자별 LLMModel 매핑 (프롬프트용)
    PROVIDER_MODEL_MAP = {
        ChatProvider.CLAUDE: LLMModel.CLAUDE_SONNET,
        ChatProvider.GPT: LLMModel.GPT_51,
        ChatProvider.GEMINI: LLMModel.GEMINI_3_PRO,
        ChatProvider.GROK: LLMModel.GROK_41,
    }

    def __init__(
        self,
        db: AsyncSession,
        config: Optional[ChatConfig] = None,
    ):
        """
        초기화.

        Args:
            db: SQLAlchemy AsyncSession
            config: 채팅 설정
        """
        self.db = db
        self.config = config or ChatConfig()

        # 시스템 컴포넌트 (지연 초기화)
        self._cascade_services: dict[ChatProvider, CascadeService] = {}
        self._learning_manager: Optional[LearningManager] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._rag_orchestrator: Optional[HybridRAGOrchestrator] = None

        # 세션 캐시
        self._sessions: dict[str, ChatSession] = {}

        # 통계
        self.stats = {
            "total_messages": 0,
            "drafter_success": 0,
            "verifier_calls": 0,
            "rag_hits": 0,
            "learning_records": 0,
        }

    async def _init_cascade_service(self, provider: ChatProvider) -> CascadeService:
        """CascadeService 초기화 (제공자별)."""
        if provider not in self._cascade_services:
            cascade_pair = self.PROVIDER_CASCADE_MAP[provider]
            self._cascade_services[provider] = CascadeServiceFactory.create_from_pair(
                cascade_pair
            )
        return self._cascade_services[provider]

    async def _init_learning_manager(self) -> LearningManager:
        """LearningManager 초기화."""
        if self._learning_manager is None:
            self._learning_manager = await get_learning_manager()
        return self._learning_manager

    async def _init_memory_manager(self) -> MemoryManager:
        """MemoryManager 초기화."""
        if self._memory_manager is None:
            self._memory_manager = await get_memory_manager()
        return self._memory_manager

    async def _init_rag_orchestrator(self) -> HybridRAGOrchestrator:
        """HybridRAGOrchestrator 초기화."""
        if self._rag_orchestrator is None:
            rag_config = RAGConfig(
                max_legal_docs=self.config.max_legal_docs,
                max_experiences=self.config.max_experiences,
                legal_weight=0.7,
                experience_weight=0.3,
            )
            self._rag_orchestrator = HybridRAGOrchestrator(
                db=self.db,
                config=rag_config,
            )
        return self._rag_orchestrator

    def create_session(
        self,
        user_id: str,
        provider: ChatProvider = ChatProvider.CLAUDE,
        domain: LegalDomain = LegalDomain.GENERAL_CIVIL,
    ) -> ChatSession:
        """새 채팅 세션 생성."""
        session = ChatSession(
            session_id=str(uuid4()),
            user_id=user_id,
            provider=provider,
            domain=domain,
        )
        self._sessions[session.session_id] = session
        logger.info(f"Created chat session: {session.session_id} with {provider.value}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 조회."""
        return self._sessions.get(session_id)

    def update_session_provider(
        self,
        session_id: str,
        provider: ChatProvider,
    ) -> Optional[ChatSession]:
        """세션의 LLM 제공자 변경."""
        session = self._sessions.get(session_id)
        if session:
            session.provider = provider
            logger.info(f"Updated session {session_id} provider to {provider.value}")
        return session

    async def _get_memory_context(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, str]:
        """메모리 컨텍스트 조회."""
        if not self.config.enable_memory:
            return {}

        try:
            memory_manager = await self._init_memory_manager()

            # 세션 메모리
            session_memory = await memory_manager.get_memory(
                key=f"chat_session:{session_id}",
                memory_type=MemoryType.SESSION,
            )

            # 단기 메모리 (최근 대화)
            short_term = await memory_manager.get_memory(
                key=f"chat_recent:{user_id}",
                memory_type=MemoryType.SHORT_TERM,
            )

            # 장기 메모리 (사용자 선호도 등)
            long_term = await memory_manager.get_memory(
                key=f"chat_user:{user_id}",
                memory_type=MemoryType.LONG_TERM,
            )

            return {
                "session": session_memory or "",
                "short_term": short_term or "",
                "long_term": long_term or "",
            }
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            return {}

    async def _get_rag_context(
        self,
        query: str,
        domain: LegalDomain,
        user_id: str,
    ) -> str:
        """RAG 컨텍스트 조회."""
        if not self.config.enable_rag:
            return ""

        try:
            rag = await self._init_rag_orchestrator()

            result: HybridSearchResult = await rag.search(
                query=query,
                domain=domain,
                user_id=user_id,
            )

            if result and result.formatted_context:
                self.stats["rag_hits"] += 1
                return result.formatted_context

            return ""
        except Exception as e:
            logger.warning(f"Failed to get RAG context: {e}")
            return ""

    async def _build_system_prompt(
        self,
        provider: ChatProvider,
        domain: LegalDomain,
        memory_context: dict[str, str],
        rag_context: str,
    ) -> str:
        """시스템 프롬프트 생성."""
        llm_model = self.PROVIDER_MODEL_MAP[provider]

        return assemble_expert_prompt(
            domain=domain,
            model=llm_model,
            session=memory_context.get("session", ""),
            recent=memory_context.get("short_term", ""),
            history=memory_context.get("long_term", ""),
            rag=rag_context,
        )

    async def _record_outcome(
        self,
        session: ChatSession,
        message: ChatMessage,
        cascade_result: CascadeResult,
    ) -> None:
        """학습 시스템에 결과 기록."""
        if not self.config.enable_learning or not self.config.auto_record_outcomes:
            return

        try:
            learning = await self._init_learning_manager()

            await learning.record_outcome(
                agent_id=f"chat-{message.provider.value}",
                model_id=message.model,
                tier=message.cascade_tier or "drafter",
                consultation_id=session.session_id,
                user_id=session.user_id,
                category=session.domain.value,
                query_complexity="moderate",
                status=OutcomeStatus.SUCCESS,
                response_quality=cascade_result.quality_score,
                response_time_ms=int(cascade_result.latency_ms),
                tokens_used=message.tokens_used,
                cost_usd=message.cost_usd,
                metadata={
                    "escalated": cascade_result.escalated,
                    "drafter_sufficient": cascade_result.drafter_sufficient,
                },
            )

            self.stats["learning_records"] += 1
            logger.debug(f"Recorded learning outcome for message {message.id}")
        except Exception as e:
            logger.warning(f"Failed to record outcome: {e}")

    async def _update_memory(
        self,
        session: ChatSession,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """메모리 업데이트."""
        if not self.config.enable_memory:
            return

        try:
            memory = await self._init_memory_manager()

            # 세션 메모리 업데이트
            session_context = f"[{session.domain.value}] Q: {user_message[:200]}... A: {assistant_message[:300]}..."
            await memory.set_memory(
                key=f"chat_session:{session.session_id}",
                value=session_context,
                memory_type=MemoryType.SESSION,
                ttl=self.config.memory_ttl,
            )

            # 단기 메모리에 추가
            recent_key = f"chat_recent:{session.user_id}"
            existing = await memory.get_memory(
                key=recent_key,
                memory_type=MemoryType.SHORT_TERM,
            )
            new_entry = f"\n---\n{session_context}"
            updated = (existing or "") + new_entry

            # 최대 길이 제한 (최근 5000자만 유지)
            if len(updated) > 5000:
                updated = updated[-5000:]

            await memory.set_memory(
                key=recent_key,
                value=updated,
                memory_type=MemoryType.SHORT_TERM,
                ttl=self.config.memory_ttl * 2,
            )
        except Exception as e:
            logger.warning(f"Failed to update memory: {e}")

    async def send_message(
        self,
        session_id: str,
        content: str,
        provider: Optional[ChatProvider] = None,
    ) -> ChatResponse:
        """
        메시지 전송 및 응답 생성.

        Args:
            session_id: 세션 ID
            content: 사용자 메시지
            provider: LLM 제공자 (None이면 세션 기본값)

        Returns:
            ChatResponse 객체
        """
        start_time = time.time()

        # 세션 조회
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # 제공자 결정
        active_provider = provider or session.provider
        if provider and provider != session.provider:
            session.provider = provider

        # 도메인 자동 감지
        detected_domains = detect_domains(content)
        if detected_domains:
            session.domain = detected_domains[0]

        # 사용자 메시지 기록
        user_msg = ChatMessage(
            id=str(uuid4()),
            role="user",
            content=content,
            model="user",
            provider=active_provider,
            timestamp=datetime.now(timezone.utc),
        )
        session.messages.append(user_msg)

        # 컨텍스트 수집
        memory_context = await self._get_memory_context(session_id, session.user_id)
        rag_context = await self._get_rag_context(content, session.domain, session.user_id)

        # 시스템 프롬프트 생성
        system_prompt = await self._build_system_prompt(
            provider=active_provider,
            domain=session.domain,
            memory_context=memory_context,
            rag_context=rag_context,
        )

        # 대화 히스토리 구성 (최근 10개 메시지)
        history_messages = session.messages[-10:]
        history_text = ""
        for msg in history_messages[:-1]:  # 마지막(현재) 메시지 제외
            role_label = "사용자" if msg.role == "user" else "법률 전문가"
            history_text += f"\n{role_label}: {msg.content}\n"

        user_prompt = content
        if history_text:
            user_prompt = f"[이전 대화]\n{history_text}\n\n[현재 질문]\n{content}"

        # CascadeFlow로 응답 생성
        cascade_service = await self._init_cascade_service(active_provider)

        cascade_result: CascadeResult = await cascade_service.execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # 응답 메시지 생성
        assistant_msg = ChatMessage(
            id=str(uuid4()),
            role="assistant",
            content=cascade_result.response,
            model=cascade_result.model_used,
            provider=active_provider,
            timestamp=datetime.now(timezone.utc),
            tokens_used=cascade_result.tokens_used,
            cost_usd=cascade_result.cost_usd,
            cascade_tier="verifier" if cascade_result.escalated else "drafter",
            metadata={
                "drafter_sufficient": cascade_result.drafter_sufficient,
                "quality_score": cascade_result.quality_score,
                "latency_ms": cascade_result.latency_ms,
            },
        )
        session.messages.append(assistant_msg)

        # 세션 통계 업데이트
        session.total_tokens += cascade_result.tokens_used
        session.total_cost += cascade_result.cost_usd
        session.last_activity = datetime.now(timezone.utc)

        # 서비스 통계 업데이트
        self.stats["total_messages"] += 1
        if cascade_result.drafter_sufficient:
            self.stats["drafter_success"] += 1
        if cascade_result.escalated:
            self.stats["verifier_calls"] += 1

        # 학습 기록
        await self._record_outcome(session, assistant_msg, cascade_result)

        # 메모리 업데이트
        await self._update_memory(session, content, cascade_result.response)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Chat response generated in {elapsed_ms:.0f}ms "
            f"[{active_provider.value}:{cascade_result.model_used}] "
            f"escalated={cascade_result.escalated}"
        )

        return ChatResponse(
            message=assistant_msg,
            session=session,
            rag_context=rag_context if rag_context else None,
            memory_context=memory_context if memory_context else None,
            learning_recorded=self.config.enable_learning,
        )

    async def send_message_stream(
        self,
        session_id: str,
        content: str,
        provider: Optional[ChatProvider] = None,
    ) -> AsyncIterator[str]:
        """
        스트리밍 메시지 전송.

        Args:
            session_id: 세션 ID
            content: 사용자 메시지
            provider: LLM 제공자

        Yields:
            응답 텍스트 청크
        """
        # 세션 조회
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        active_provider = provider or session.provider

        # 컨텍스트 수집
        memory_context = await self._get_memory_context(session_id, session.user_id)
        rag_context = await self._get_rag_context(content, session.domain, session.user_id)

        # 시스템 프롬프트
        system_prompt = await self._build_system_prompt(
            provider=active_provider,
            domain=session.domain,
            memory_context=memory_context,
            rag_context=rag_context,
        )

        # OpenRouter 클라이언트로 스트리밍
        cascade_pair = self.PROVIDER_CASCADE_MAP[active_provider]

        # 드래프터 모델로 스트리밍 (비용 최적화)
        client = OpenRouterClient(
            model_name=cascade_pair.drafter.openrouter_id,
            domain=session.domain,
        )
        await client._initialize_client()

        full_response = ""
        async for chunk in client._generate_stream(
            system_prompt=system_prompt,
            user_prompt=content,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            full_response += chunk
            yield chunk

        # 응답 완료 후 메시지 기록
        user_msg = ChatMessage(
            id=str(uuid4()),
            role="user",
            content=content,
            model="user",
            provider=active_provider,
            timestamp=datetime.now(timezone.utc),
        )
        session.messages.append(user_msg)

        assistant_msg = ChatMessage(
            id=str(uuid4()),
            role="assistant",
            content=full_response,
            model=cascade_pair.drafter.openrouter_id,
            provider=active_provider,
            timestamp=datetime.now(timezone.utc),
            cascade_tier="drafter",
        )
        session.messages.append(assistant_msg)

        # 메모리 업데이트
        await self._update_memory(session, content, full_response)

    def get_available_providers(self) -> list[dict]:
        """사용 가능한 LLM 제공자 목록."""
        return [
            {
                "id": ChatProvider.CLAUDE.value,
                "name": "Claude (Anthropic)",
                "description": "정교한 법률 분석과 논리적 추론에 강점",
                "models": {
                    "drafter": CLAUDE_CASCADE.drafter.openrouter_id,
                    "verifier": CLAUDE_CASCADE.verifier.openrouter_id,
                },
            },
            {
                "id": ChatProvider.GPT.value,
                "name": "GPT (OpenAI)",
                "description": "다양한 법률 지식과 유연한 응답 생성",
                "models": {
                    "drafter": GPT_CASCADE.drafter.openrouter_id,
                    "verifier": GPT_CASCADE.verifier.openrouter_id,
                },
            },
            {
                "id": ChatProvider.GEMINI.value,
                "name": "Gemini (Google)",
                "description": "최신 정보 통합과 빠른 응답 속도",
                "models": {
                    "drafter": GEMINI_CASCADE.drafter.openrouter_id,
                    "verifier": GEMINI_CASCADE.verifier.openrouter_id,
                },
            },
            {
                "id": ChatProvider.GROK.value,
                "name": "Grok (xAI)",
                "description": "실시간 정보 접근과 실용적 조언",
                "models": {
                    "drafter": GROK_CASCADE.drafter.openrouter_id,
                    "verifier": GROK_CASCADE.verifier.openrouter_id,
                },
            },
        ]

    def get_session_history(self, session_id: str) -> list[dict]:
        """세션 대화 히스토리."""
        session = self.get_session(session_id)
        if not session:
            return []

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "model": msg.model,
                "provider": msg.provider.value,
                "timestamp": msg.timestamp.isoformat(),
                "tokens_used": msg.tokens_used,
                "cascade_tier": msg.cascade_tier,
            }
            for msg in session.messages
        ]

    def get_stats(self) -> dict:
        """서비스 통계."""
        drafter_rate = (
            self.stats["drafter_success"] / self.stats["total_messages"] * 100
            if self.stats["total_messages"] > 0
            else 0
        )

        return {
            **self.stats,
            "drafter_success_rate": f"{drafter_rate:.1f}%",
            "active_sessions": len(self._sessions),
        }


# Factory function
def get_chat_service(db: AsyncSession) -> IndividualChatService:
    """개인 채팅 서비스 인스턴스 생성."""
    return IndividualChatService(db=db)
