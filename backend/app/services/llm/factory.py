"""
LLM Client Factory v4.3.1

LLM 클라이언트 인스턴스 생성 팩토리.
v4.3.1 특징:
- 분야별 모듈형 프롬프트 (~560 토큰 vs 전체 2,000+)
- LLM별 최적화된 addon 자동 적용
- 키워드 기반 복합 사안 감지 (비용 0)
- OpenRouter 통합: 단일 API를 통한 모든 LLM 제공자 접근
"""

import logging
from typing import Literal, Optional

from app.core.config import settings
from app.services.llm.base import BaseLLMClient, ModelRole, StreamingLLMClient
from app.services.llm.legal_prompts_v4_3 import (
    TaskComplexity,
    get_model_parameters,
    LLMModel,
    LegalDomain,
    detect_domains,
)

logger = logging.getLogger(__name__)


LLMProvider = Literal["openai", "anthropic", "google", "xai"]


class LLMClientFactory:
    """
    Factory for creating LLM client instances with v4.3.1 and OpenRouter support.

    v4.3.1 분야별 모듈형 프롬프트 시스템:
    - 선택된 분야만 로드 (~560 토큰)
    - LLM별 최적화된 addon 자동 적용
    - 키워드 기반 복합 사안 감지
    """

    @staticmethod
    def create_council_member(
        provider: LLMProvider,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,
    ) -> StreamingLLMClient:
        """
        Create a council member client (Stage 1) with v4.3.1 support.

        Args:
            provider: LLM provider name
            complexity: Task complexity for parameter optimization
            domain: 법률 분야 (v4.3.1 모듈형 프롬프트)

        Returns:
            Configured LLM client with v4.3.1 parameters
        """
        # OpenRouter 사용 시
        if settings.USE_OPENROUTER:
            return LLMClientFactory._create_openrouter_council_member(
                provider, complexity, domain
            )

        # Direct API 사용 시
        return LLMClientFactory._create_direct_council_member(provider, complexity)

    @staticmethod
    def _create_openrouter_council_member(
        provider: LLMProvider,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,
    ) -> StreamingLLMClient:
        """OpenRouter를 통한 위원회 멤버 생성 (v4.3.1)."""
        from app.services.llm.openrouter_client import OpenRouterCouncilMember

        # 제공자별 모델 ID 매핑
        provider_model_mapping = {
            "openai": settings.OPENROUTER_COUNCIL_MODELS[0] if settings.OPENROUTER_COUNCIL_MODELS else "openai/gpt-4o",
            "anthropic": settings.OPENROUTER_COUNCIL_MODELS[1] if len(settings.OPENROUTER_COUNCIL_MODELS) > 1 else "anthropic/claude-sonnet-4",
            "google": settings.OPENROUTER_COUNCIL_MODELS[2] if len(settings.OPENROUTER_COUNCIL_MODELS) > 2 else "google/gemini-2.5-pro-preview",
            "xai": settings.OPENROUTER_COUNCIL_MODELS[3] if len(settings.OPENROUTER_COUNCIL_MODELS) > 3 else "x-ai/grok-2",
        }

        model_name = provider_model_mapping.get(provider, "anthropic/claude-sonnet-4")

        return OpenRouterCouncilMember(
            model_name=model_name,
            provider=provider,
            complexity=complexity,
            domain=domain,  # v4.3.1
        )

    @staticmethod
    def _create_direct_council_member(
        provider: LLMProvider,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ) -> StreamingLLMClient:
        """Direct API를 통한 위원회 멤버 생성 (레거시)."""
        # Get v4.1 optimized parameters based on complexity
        model_key = _get_model_key_for_provider(provider, "council_member")
        v41_params = get_model_parameters(model_key, complexity) if model_key else {}

        if provider == "openai":
            from app.services.llm.openai_client import OpenAIClient
            return OpenAIClient(
                role=ModelRole.COUNCIL_MEMBER,
                reasoning_effort=v41_params.get("reasoning_effort"),
            )

        elif provider == "anthropic":
            from app.services.llm.anthropic_client import ClaudeSonnetClient
            return ClaudeSonnetClient(role=ModelRole.COUNCIL_MEMBER)

        elif provider == "google":
            from app.services.llm.google_client import GeminiClient
            return GeminiClient(
                role=ModelRole.COUNCIL_MEMBER,
                thinking_level=v41_params.get("thinking_level"),
            )

        elif provider == "xai":
            from app.services.llm.xai_client import GrokClient
            return GrokClient(role=ModelRole.COUNCIL_MEMBER)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def create_peer_reviewer(
        domain: LegalDomain | None = None,
    ) -> StreamingLLMClient:
        """
        Create peer reviewer client (Stage 2) with v4.3.1 support.

        Uses Claude Sonnet for blind peer review.

        Args:
            domain: 법률 분야 (v4.3.1)
        """
        if settings.USE_OPENROUTER:
            from app.services.llm.openrouter_client import OpenRouterPeerReviewer
            return OpenRouterPeerReviewer(
                model_name=settings.OPENROUTER_REVIEWER_MODEL,
                domain=domain,  # v4.3.1
            )

        from app.services.llm.anthropic_client import ClaudePeerReviewer
        return ClaudePeerReviewer()

    @staticmethod
    def create_chairman(
        thinking_budget: int | None = None,
        complexity: TaskComplexity = TaskComplexity.COMPLEX,
        domain: LegalDomain | None = None,
    ) -> StreamingLLMClient:
        """
        Create chairman client (Stage 3) with v4.3.1 support.

        Uses Claude Opus with extended thinking.
        v4.3.1: LLM별 최적화된 의장 프롬프트 적용.

        Args:
            thinking_budget: Optional thinking token budget
            complexity: Task complexity for effort optimization
            domain: 법률 분야 (v4.3.1)
        """
        if settings.USE_OPENROUTER:
            from app.services.llm.openrouter_client import OpenRouterChairman
            return OpenRouterChairman(
                model_name=settings.OPENROUTER_CHAIRMAN_MODEL,
                complexity=complexity,
                domain=domain,  # v4.3.1
            )

        from app.services.llm.anthropic_client import ClaudeChairman

        # Get v4.3.1 optimized parameters
        v43_params = get_model_parameters(
            LLMModel.CLAUDE_OPUS_45,
            complexity,
        )

        return ClaudeChairman(
            thinking_budget=thinking_budget,
        )

    @staticmethod
    def create_all_council_members(
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,
    ) -> list[StreamingLLMClient]:
        """
        Create all four council member clients with v4.3.1 support.

        Args:
            complexity: Task complexity for parameter optimization
            domain: 법률 분야 (v4.3.1 모듈형 프롬프트)

        Returns:
            List of all council member clients with v4.3.1 parameters
        """
        if settings.USE_OPENROUTER:
            return LLMClientFactory._create_all_openrouter_council_members(
                complexity, domain
            )

        providers: list[LLMProvider] = ["openai", "anthropic", "google", "xai"]
        return [
            LLMClientFactory.create_council_member(provider, complexity, domain)
            for provider in providers
        ]

    @staticmethod
    def _create_all_openrouter_council_members(
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,
    ) -> list[StreamingLLMClient]:
        """OpenRouter를 통한 모든 위원회 멤버 생성 (v4.3.1)."""
        from app.services.llm.openrouter_client import OpenRouterCouncilMember

        clients = []
        provider_mapping = {
            0: "openai",
            1: "anthropic",
            2: "google",
            3: "xai",
        }

        for i, model_name in enumerate(settings.OPENROUTER_COUNCIL_MODELS):
            provider = provider_mapping.get(i, "anthropic")
            clients.append(
                OpenRouterCouncilMember(
                    model_name=model_name,
                    provider=provider,
                    complexity=complexity,
                    domain=domain,  # v4.3.1
                )
            )

        return clients

    @staticmethod
    def create_client(
        provider: LLMProvider,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,
        **kwargs,
    ) -> StreamingLLMClient:
        """
        Create a generic LLM client with v4.3.1 support.

        Args:
            provider: LLM provider name
            role: Model role in the council
            complexity: Task complexity for v4.3.1 parameter optimization
            domain: 법률 분야 (v4.3.1)
            **kwargs: Additional client parameters

        Returns:
            Configured LLM client with v4.3.1 parameters
        """
        if role == ModelRole.PEER_REVIEWER:
            return LLMClientFactory.create_peer_reviewer(domain=domain)

        if role == ModelRole.CHAIRMAN:
            return LLMClientFactory.create_chairman(
                thinking_budget=kwargs.get("thinking_budget"),
                complexity=complexity,
                domain=domain,
            )

        return LLMClientFactory.create_council_member(provider, complexity, domain)


def _get_model_key_for_provider(
    provider: LLMProvider,
    role: str,
) -> Optional[str]:
    """
    Provider와 역할에 해당하는 LLMModel 키 반환.

    Args:
        provider: LLM provider name
        role: council_member, peer_reviewer, chairman

    Returns:
        LLMModel enum value or None
    """
    model_mapping = {
        "openai": {
            "council_member": LLMModel.GPT_51.value,
            "chairman": LLMModel.GPT_51.value,
        },
        "anthropic": {
            "council_member": LLMModel.CLAUDE_SONNET_45.value,
            "peer_reviewer": LLMModel.CLAUDE_SONNET_45.value,
            "chairman": LLMModel.CLAUDE_OPUS_45.value,
        },
        "google": {
            "council_member": LLMModel.GEMINI_3_PRO.value,
            "chairman": LLMModel.GEMINI_3_PRO.value,
        },
        "xai": {
            "council_member": LLMModel.GROK_4.value,
            "chairman": LLMModel.GROK_4.value,
        },
    }
    return model_mapping.get(provider, {}).get(role)


# ============================================================
# Convenience Functions v4.3.1
# ============================================================


async def get_council_members(
    complexity: TaskComplexity = TaskComplexity.MEDIUM,
    domain: LegalDomain | None = None,
    query: str | None = None,
) -> list[StreamingLLMClient]:
    """
    위원회 멤버 클라이언트들 획득 및 초기화 (v4.3.1).

    Args:
        complexity: 작업 복잡도
        domain: 법률 분야 (직접 지정)
        query: 질문 (키워드 기반 분야 자동 감지용)

    Returns:
        초기화된 위원회 멤버 클라이언트 목록
    """
    # v4.3.1: 키워드 기반 분야 자동 감지
    if domain is None and query:
        detected = detect_domains(query)
        domain = detected[0] if detected else LegalDomain.GENERAL_CIVIL

    clients = LLMClientFactory.create_all_council_members(complexity, domain)

    # 모든 클라이언트 초기화
    for client in clients:
        await client._initialize_client()

    return clients


async def get_chairman(
    complexity: TaskComplexity = TaskComplexity.COMPLEX,
    domain: LegalDomain | None = None,
    query: str | None = None,
) -> StreamingLLMClient:
    """
    의장 클라이언트 획득 및 초기화 (v4.3.1).

    Args:
        complexity: 작업 복잡도
        domain: 법률 분야 (직접 지정)
        query: 질문 (키워드 기반 분야 자동 감지용)

    Returns:
        초기화된 의장 클라이언트
    """
    # v4.3.1: 키워드 기반 분야 자동 감지
    if domain is None and query:
        detected = detect_domains(query)
        domain = detected[0] if detected else LegalDomain.GENERAL_CIVIL

    chairman = LLMClientFactory.create_chairman(
        complexity=complexity,
        domain=domain,
    )
    await chairman._initialize_client()
    return chairman


async def get_peer_reviewer(
    domain: LegalDomain | None = None,
    query: str | None = None,
) -> StreamingLLMClient:
    """
    동료 평가자 클라이언트 획득 및 초기화 (v4.3.1).

    Args:
        domain: 법률 분야 (직접 지정)
        query: 질문 (키워드 기반 분야 자동 감지용)

    Returns:
        초기화된 동료 평가자 클라이언트
    """
    # v4.3.1: 키워드 기반 분야 자동 감지
    if domain is None and query:
        detected = detect_domains(query)
        domain = detected[0] if detected else LegalDomain.GENERAL_CIVIL

    reviewer = LLMClientFactory.create_peer_reviewer(domain=domain)
    await reviewer._initialize_client()
    return reviewer


def detect_legal_domains(query: str) -> list[LegalDomain]:
    """
    질문에서 법률 분야를 키워드 기반으로 감지 (비용 0).

    Args:
        query: 법률 질문

    Returns:
        감지된 법률 분야 목록 (복합 사안 지원)
    """
    return detect_domains(query)
