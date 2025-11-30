"""
OpenRouter Unified LLM Client v4.3.1

OpenRouter API를 통한 통합 LLM 클라이언트.
모든 LLM 제공자(OpenAI, Anthropic, Google, xAI)를 단일 API로 호출.

v4.3.1 특징:
- 분야별 모듈형 프롬프트 (선택된 분야만 로드 ~560 토큰)
- LLM별 최적화된 addon
- 키워드 기반 복합 사안 감지 (비용 0)
"""

import logging
from typing import AsyncIterator, Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import (
    LegalContext,
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.legal_prompts_v4_3 import (
    LegalDomain,
    LLMModel,
    TaskComplexity,
    assemble_expert_prompt,
    assemble_multi_domain_prompt,
    get_chairman_prompt,
    get_stage2_prompt,
    get_model_parameters,
    detect_domains,
    get_llm_model_from_openrouter_id,
)

logger = logging.getLogger(__name__)


# OpenRouter 모델 ID 매핑
OPENROUTER_MODELS = {
    # OpenAI Models
    "gpt-5.1": "openai/gpt-4o",  # GPT-5.1은 아직 없으므로 gpt-4o 사용
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    "o1-preview": "openai/o1-preview",

    # Anthropic Models
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",

    # Google Models
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "gemini-pro": "google/gemini-pro",

    # xAI Models
    "grok-2": "x-ai/grok-2",
    "grok-2-vision": "x-ai/grok-2-vision",
    "grok-beta": "x-ai/grok-beta",
}


class OpenRouterClient(StreamingLLMClient):
    """
    OpenRouter 통합 LLM 클라이언트 v4.3.1

    OpenRouter API를 통해 모든 LLM 제공자를 단일 인터페이스로 호출.
    OpenAI SDK 호환 API를 사용.

    v4.3.1 특징:
    - 분야별 모듈형 프롬프트 시스템
    - LLM별 최적화된 addon 자동 적용
    - 키워드 기반 복합 사안 감지
    """

    def __init__(
        self,
        model_name: str,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        provider: str = "anthropic",  # 기본 제공자
        domain: LegalDomain | None = None,  # v4.3.1: 법률 분야
    ):
        """
        OpenRouter 클라이언트 초기화.

        Args:
            model_name: OpenRouter 모델 ID 또는 별칭
            role: 위원회에서의 역할
            complexity: 작업 복잡도 (파라미터 최적화용)
            provider: 원본 제공자 (프롬프트 선택용)
            domain: 법률 분야 (v4.3.1)
        """
        # 모델 ID 정규화
        self.openrouter_model = self._normalize_model_id(model_name)
        super().__init__(model_name=self.openrouter_model, role=role)

        self.complexity = complexity
        self.provider = provider
        self.domain = domain or LegalDomain.GENERAL_CIVIL  # v4.3.1
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL

        # v4.3.1: LLMModel enum으로 변환
        self._llm_model = get_llm_model_from_openrouter_id(self.openrouter_model)

        # v4.3.1 파라미터 로드
        self._model_params = self._get_model_params()

    def _normalize_model_id(self, model_name: str) -> str:
        """모델 이름을 OpenRouter 모델 ID로 변환."""
        # 이미 OpenRouter 형식이면 그대로 반환
        if "/" in model_name:
            return model_name

        # 별칭 매핑
        return OPENROUTER_MODELS.get(model_name, model_name)

    def _get_model_params(self) -> dict:
        """v4.3.1 모델 파라미터 획득."""
        return get_model_parameters(self._llm_model, self.complexity)

    @property
    def display_name(self) -> str:
        """표시용 모델 이름."""
        role_suffix = {
            ModelRole.COUNCIL_MEMBER: "위원",
            ModelRole.PEER_REVIEWER: "평가자",
            ModelRole.CHAIRMAN: "의장",
        }
        return f"{self.openrouter_model} ({role_suffix.get(self.role, '')})"

    def _get_council_member_prompt(
        self,
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        rag_results: str = "",
    ) -> str:
        """
        v4.3.1 모듈형 법률 전문가 프롬프트 반환.

        선택된 분야만 로드하여 ~560 토큰으로 효율적 구성.
        UNIVERSAL_CORE + DOMAIN_MODULE + MODEL_ADDON 구조.
        """
        return assemble_expert_prompt(
            domain=self.domain,
            model=self._llm_model,
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_results=rag_results,
        )

    def _get_chairman_prompt(
        self,
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        rag_results: str = "",
        domain: LegalDomain | None = None,
        advisory_year: int = 2025,
        advisory_number: int = 1,
        advisory_date: str = "",
    ) -> str:
        """
        v4.3.1 의장 프롬프트 반환.

        LLM별 최적화된 의장 프롬프트 적용.
        """
        # 의장은 항상 claude-opus 또는 지정된 모델 사용
        chairman_model = self._llm_model
        if "opus" not in self.openrouter_model.lower():
            chairman_model = LLMModel.CLAUDE_OPUS_45

        return get_chairman_prompt(
            model=chairman_model,
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_results=rag_results,
            domain=domain or self.domain,
            advisory_year=advisory_year,
            advisory_number=advisory_number,
            advisory_date=advisory_date,
        )

    async def _initialize_client(self) -> None:
        """OpenRouter 클라이언트 초기화 (OpenAI SDK 사용)."""
        if not self.api_key:
            raise ValueError(
                "OpenRouter API 키가 설정되지 않았습니다. "
                "OPENROUTER_API_KEY 환경변수를 설정하세요."
            )

        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_SITE_URL or "http://localhost:3000",
                    "X-Title": settings.OPENROUTER_APP_NAME or "Legal Advisory Council",
                },
            )
            logger.info(f"OpenRouter client initialized: {self.openrouter_model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """OpenRouter API를 통해 응답 생성."""
        try:
            # 모델별 파라미터 조정
            params = {
                "model": self.openrouter_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # OpenRouter 특정 설정
            extra_body = {}

            # Anthropic 모델의 경우 확장 사고 지원
            if "claude" in self.openrouter_model.lower() and self.role == ModelRole.CHAIRMAN:
                # Claude 모델에 대한 추가 설정
                pass  # OpenRouter에서 자동 처리

            # OpenAI o1/o3 모델의 경우 reasoning 파라미터
            if "o1" in self.openrouter_model.lower() or "o3" in self.openrouter_model.lower():
                # reasoning 모델은 temperature 사용 불가
                params.pop("temperature", None)

            if extra_body:
                params["extra_body"] = extra_body

            response = await self._client.chat.completions.create(**params)

            message = response.choices[0].message
            usage = response.usage

            return LLMResponse(
                content=message.content or "",
                model=self.openrouter_model,
                role=self.role,
                tokens_used=usage.total_tokens if usage else 0,
                raw_response=response,
                metadata={
                    "provider": self.provider,
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                },
            )

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

    async def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성."""
        try:
            params = {
                "model": self.openrouter_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }

            # o1/o3 모델은 temperature 제거
            if "o1" in self.openrouter_model.lower() or "o3" in self.openrouter_model.lower():
                params.pop("temperature", None)

            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise

    async def generate_peer_review(
        self,
        original_query: str,
        opinion_to_review: str,
        category: str = "general",
        rag_results: str = "",
        domain: LegalDomain | None = None,
    ) -> LLMResponse:
        """
        v4.3.1 블라인드 동료 평가 생성 (Stage 2).

        Args:
            original_query: 원본 법률 질문
            opinion_to_review: 평가할 익명화된 의견
            category: 법률 분야
            rag_results: RAG 검색 결과 (인용 검증용)
            domain: 법률 분야 enum (v4.3.1)

        Returns:
            평가 결과가 담긴 LLMResponse
        """
        if not self._client:
            await self._initialize_client()

        # v4.3.1 Stage 2 평가 프롬프트 (LLM별 최적화)
        system_prompt = get_stage2_prompt(
            model=self._llm_model,
            original_question=original_query,
            domain=domain or self.domain,
            anonymized_opinions=opinion_to_review,
            rag_results=rag_results,
        )

        user_prompt = f"""## 법률 분야: {category}

위 익명화된 의견을 v4.3.1 평가 기준에 따라 블라인드 교차 평가해 주세요.
5가지 기준(정확성, 완전성, 메모리 통합, 유용성, 명확성) 각 10점 만점으로 평가하세요.
RAG 인용 검증 결과도 함께 제공하세요."""

        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=3000,
        )

    async def generate_chairman_synthesis(
        self,
        query: str,
        category: str,
        opinions: list[dict],
        reviews: list[dict],
        rag_context: str = "",
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        domain: LegalDomain | None = None,
        advisory_year: int = 2025,
        advisory_number: int = 1,
        advisory_date: str = "",
    ) -> LLMResponse:
        """
        v4.3.1 의장 최종 합성 생성 (Stage 3).

        Args:
            query: 원본 법률 질문
            category: 법률 분야
            opinions: 위원 의견 목록
            reviews: 동료 평가 목록
            rag_context: RAG 검색 결과
            session_memory: 세션 메모리
            short_term_memory: 단기 메모리
            long_term_memory: 장기 메모리
            domain: 법률 분야 enum (v4.3.1)
            advisory_year: 자문 연도
            advisory_number: 자문 번호
            advisory_date: 자문 날짜

        Returns:
            최종 합성이 담긴 LLMResponse
        """
        if not self._client:
            await self._initialize_client()

        # v4.3.1 의장 프롬프트 (LLM별 최적화 + 메모리 시스템)
        system_prompt = self._get_chairman_prompt(
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_results=rag_context,
            domain=domain or self.domain,
            advisory_year=advisory_year,
            advisory_number=advisory_number,
            advisory_date=advisory_date,
        )

        # 종합 프롬프트 구성
        parts = [f"## 원본 질문\n{query}\n\n## 법률 분야\n{category}"]

        if rag_context:
            parts.append(f"\n## 참고 문헌 (RAG 검색 결과)\n{rag_context}")

        parts.append("\n## Stage 1: 위원 의견")
        for i, opinion in enumerate(opinions, 1):
            model = opinion.get("model", f"위원 {i}")
            content = opinion.get("content", "")
            parts.append(f"\n### {model}\n{content}")

        parts.append("\n## Stage 2: 블라인드 교차 평가 결과")
        for i, review in enumerate(reviews, 1):
            reviewer = review.get("reviewer", f"평가자 {i}")
            reviewed = review.get("reviewed", f"의견 {i}")
            content = review.get("content", "")
            scores = review.get("scores", {})
            parts.append(f"\n### {reviewer}의 {reviewed} 평가")
            if scores:
                parts.append(f"점수: {scores}")
            parts.append(f"\n{content}")

        user_prompt = "\n".join(parts)
        user_prompt += """

## 의장 지시사항

v4.3.1 6단계 합성 프로토콜에 따라 최종 자문을 작성해 주세요:
1. 인용 검증 및 권위 평가
2. 합의 및 분기 지점 분석
3. 충돌 해결
4. 일관성 검증 (장기 기억 참조)
5. 공백 보완
6. 최종 합성

모든 법적 진술에 검증된 RAG 인용을 포함하세요.
결론부에는 "결론이 아직 작성되지 않았습니다"라고 자리 표시하지 말고 실제 결론을 반드시 작성하세요."""

        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=16000,
        )


# ============================================================
# 편의 클래스들 (역할별 사전 구성) - v4.3.1
# ============================================================


class OpenRouterCouncilMember(OpenRouterClient):
    """
    OpenRouter 위원회 멤버 클라이언트 v4.3.1

    분야별 모듈형 프롬프트 시스템 지원.
    """

    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-4",
        provider: str = "anthropic",
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        domain: LegalDomain | None = None,  # v4.3.1
    ):
        super().__init__(
            model_name=model_name,
            role=ModelRole.COUNCIL_MEMBER,
            complexity=complexity,
            provider=provider,
            domain=domain,
        )


class OpenRouterPeerReviewer(OpenRouterClient):
    """
    OpenRouter 동료 평가자 클라이언트 v4.3.1

    LLM별 최적화된 평가 프롬프트 적용.
    """

    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-4",
        domain: LegalDomain | None = None,  # v4.3.1
    ):
        super().__init__(
            model_name=model_name,
            role=ModelRole.PEER_REVIEWER,
            complexity=TaskComplexity.MEDIUM,
            provider="anthropic",
            domain=domain,
        )

    @property
    def display_name(self) -> str:
        return f"{self.openrouter_model} (Peer Reviewer)"


class OpenRouterChairman(OpenRouterClient):
    """
    OpenRouter 의장 클라이언트 v4.3.1

    LLM별 최적화된 의장 프롬프트 적용.
    """

    def __init__(
        self,
        model_name: str = "anthropic/claude-opus-4",
        complexity: TaskComplexity = TaskComplexity.COMPLEX,
        domain: LegalDomain | None = None,  # v4.3.1
    ):
        super().__init__(
            model_name=model_name,
            role=ModelRole.CHAIRMAN,
            complexity=complexity,
            provider="anthropic",
            domain=domain,
        )

    @property
    def display_name(self) -> str:
        return f"{self.openrouter_model} (Chairman)"
