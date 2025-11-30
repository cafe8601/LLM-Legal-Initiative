"""
Anthropic Claude Client

Stage 1: Claude Sonnet 4.5 (위원회 멤버)
Stage 2: Claude Sonnet 4.5 (블라인드 교차 검토)
Stage 3: Claude Opus 4.5 (의장)

v4.1 한글 프롬프트 적용:
- Claude Opus: XML 구조, minimalism_directive, 6단계 합성 프로토콜
- Claude Sonnet: 간소화된 프롬프트, Stage 2 평가 기준
"""

import logging
from typing import AsyncIterator

from app.core.config import settings
from app.services.llm.base import (
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.legal_prompts_v4_1 import (
    get_legal_expert_prompt_kr,
    get_chairman_prompt_kr,
    get_stage2_review_prompt_kr,
    get_model_parameters,
    LLMModel,
    TaskComplexity,
    CLAUDE_OPUS_EFFORT_PARAMS,
)

logger = logging.getLogger(__name__)


class AnthropicClient(StreamingLLMClient):
    """
    Anthropic Claude client for legal consultation.

    Supports:
    - Claude Sonnet 4.5 for council member and peer review
    - Claude Opus 4.5 for chairman synthesis with extended thinking

    v4.1 Features:
    - XML 구조 프롬프트 (Opus)
    - minimalism_directive (과도한 확장 방지)
    - 6단계 합성 프로토콜 (의장)
    - 5가지 평가 기준 (Stage 2)
    """

    def __init__(
        self,
        model_name: str | None = None,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        thinking_budget: int = 10000,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ):
        # Default model based on role
        if model_name is None:
            if role == ModelRole.CHAIRMAN:
                model_name = settings.CHAIRMAN_MODEL
            else:
                model_name = settings.CLAUDE_SONNET_MODEL

        super().__init__(model_name=model_name, role=role)
        self.thinking_budget = thinking_budget
        self.api_key = settings.ANTHROPIC_API_KEY
        self.complexity = complexity

        # v4.1 파라미터 로드
        if self.is_opus:
            self._model_params = get_model_parameters(
                LLMModel.CLAUDE_OPUS_45.value, complexity
            )
        else:
            self._model_params = get_model_parameters(
                LLMModel.CLAUDE_SONNET_45.value, complexity
            )

    @property
    def display_name(self) -> str:
        if "opus" in self.model_name.lower():
            return "Claude Opus 4.5 (Chairman)"
        return "Claude Sonnet 4.5"

    @property
    def is_opus(self) -> bool:
        """Check if using Opus model (chairman)."""
        return "opus" in self.model_name.lower()

    def _get_council_member_prompt(self) -> str:
        """v4.1 한글 법률 전문가 프롬프트 반환"""
        model = LLMModel.CLAUDE_OPUS_45.value if self.is_opus else LLMModel.CLAUDE_SONNET_45.value
        return get_legal_expert_prompt_kr(
            model=model,
            session_memory="",
            short_term_memory="",
            long_term_memory="",
            rag_results="",
        )

    def _get_chairman_prompt(self) -> str:
        """v4.1 의장 프롬프트 반환 (6단계 합성 프로토콜)"""
        return get_chairman_prompt_kr(
            model=LLMModel.CLAUDE_OPUS_45.value,
            session_memory="",
            short_term_memory="",
            long_term_memory="",
            rag_results="",
        )

    async def _initialize_client(self) -> None:
        """Initialize Anthropic async client."""
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized: {self.model_name}")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate response using Claude."""
        try:
            # Prepare request params
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            # Enable extended thinking for Opus (chairman)
            if self.is_opus and self.role == ModelRole.CHAIRMAN:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
                # Note: temperature must be 1 when thinking is enabled
            else:
                params["temperature"] = temperature

            response = await self._client.messages.create(**params)

            # Extract content and thinking
            content = ""
            thinking_content = ""

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "thinking":
                    thinking_content = block.thinking

            return LLMResponse(
                content=content,
                model=self.model_name,
                role=self.role,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                raw_response=response,
                metadata={
                    "thinking": thinking_content if thinking_content else None,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        try:
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            if self.is_opus and self.role == ModelRole.CHAIRMAN:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            else:
                params["temperature"] = temperature

            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    async def generate_peer_review(
        self,
        original_query: str,
        opinion_to_review: str,
        category: str = "general",
        rag_results: str = "",
    ) -> LLMResponse:
        """
        Generate blind peer review of another model's opinion.

        v4.1: 5가지 평가 기준 (정확성/완전성/메모리/유용성/명확성), RAG 검증

        Args:
            original_query: The original legal question
            opinion_to_review: The opinion text to review (anonymized)
            category: Legal category
            rag_results: RAG search results for citation verification

        Returns:
            LLMResponse with peer review
        """
        if not self._client:
            await self._initialize_client()

        # v4.1 Stage 2 평가 프롬프트 사용
        system_prompt = get_stage2_review_prompt_kr(
            model=LLMModel.CLAUDE_SONNET_45.value,
            original_question=original_query,
            anonymized_opinions=opinion_to_review,
            rag_results=rag_results,
        )

        user_prompt = f"""## 법률 분야: {category}

위 익명화된 의견을 v4.1 평가 기준에 따라 블라인드 교차 평가해 주세요.
5가지 기준(정확성, 완전성, 메모리 통합, 유용성, 명확성) 각 10점 만점으로 평가하세요."""

        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,  # Lower temperature for more consistent reviews
            max_tokens=3000,  # v4.1: 상세 평가를 위해 증가
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
    ) -> LLMResponse:
        """
        Generate chairman synthesis of all opinions and reviews.

        v4.1: 6단계 합성 프로토콜, 인용 검증, 일관성 검토

        Args:
            query: Original legal question
            category: Legal category
            opinions: List of council member opinions
            reviews: List of peer reviews
            rag_context: RAG search results
            session_memory: Current session context
            short_term_memory: Recent consultations (7 days)
            long_term_memory: Client history

        Returns:
            LLMResponse with final synthesis
        """
        if not self._client:
            await self._initialize_client()

        # v4.1 의장 프롬프트 (메모리 시스템 주입)
        system_prompt = get_chairman_prompt_kr(
            model=LLMModel.CLAUDE_OPUS_45.value,
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_results=rag_context,
        )

        # Build comprehensive prompt
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

v4.1 6단계 합성 프로토콜에 따라 최종 자문을 작성해 주세요:
1. 인용 검증 및 권위 평가
2. 합의 및 분기 지점 분석
3. 충돌 해결
4. 일관성 검증 (장기 기억 참조)
5. 공백 보완
6. 최종 합성

모든 법적 진술에 검증된 RAG 인용을 포함하세요."""

        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=16000,  # v4.1: 64K 출력 활용 (Opus)
        )


class ClaudeSonnetClient(AnthropicClient):
    """Claude Sonnet 4.5 client - Council Member."""

    def __init__(self, role: ModelRole = ModelRole.COUNCIL_MEMBER):
        super().__init__(
            model_name=settings.CLAUDE_SONNET_MODEL,
            role=role,
        )


class ClaudePeerReviewer(AnthropicClient):
    """Claude Sonnet 4.5 client - Peer Reviewer (Stage 2)."""

    def __init__(self):
        super().__init__(
            model_name=settings.CLAUDE_SONNET_MODEL,
            role=ModelRole.PEER_REVIEWER,
        )

    @property
    def display_name(self) -> str:
        return "Claude Sonnet 4.5 (Peer Reviewer)"


class ClaudeChairman(AnthropicClient):
    """Claude Opus 4.5 client - Chairman (Stage 3)."""

    def __init__(self, thinking_budget: int | None = None):
        # Map effort to thinking budget
        effort_budgets = {
            "low": 5000,
            "medium": 10000,
            "high": 20000,
        }
        budget = thinking_budget or effort_budgets.get(
            settings.CHAIRMAN_EFFORT, 10000
        )

        super().__init__(
            model_name=settings.CHAIRMAN_MODEL,
            role=ModelRole.CHAIRMAN,
            thinking_budget=budget,
        )

    @property
    def display_name(self) -> str:
        return f"Claude Opus 4.5 (Chairman, thinking: {self.thinking_budget})"
