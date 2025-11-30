"""
xAI Grok Client

Stage 1 위원회 멤버 - Grok 4 with reasoning

v4.1 한글 프롬프트 적용:
- 신뢰도 수준 표시 (high/medium/low/uncertain)
- 실시간 정보 우선 활용
- 예외 사례 명시적 탐색
"""

import logging
from typing import Any, AsyncIterator

from app.core.config import settings
from app.services.llm.base import (
    BaseLLMClient,
    LegalContext,
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.legal_prompts_v4_1 import (
    get_legal_expert_prompt_kr,
    get_stage2_review_prompt_kr,
    get_model_parameters,
    LLMModel,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class GrokClient(StreamingLLMClient):
    """
    xAI Grok 4 client for legal consultation.

    Uses OpenAI-compatible API with reasoning capabilities.

    v4.1 Features:
    - 신뢰도 수준 표시 (high/medium/low/uncertain)
    - 실시간 정보 우선 활용
    - 예외 사례 명시적 탐색
    - 반대 논거 적극 제시
    """

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model_name: str | None = None,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        use_reasoning: bool | None = None,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ):
        model = model_name or settings.GROK_MODEL
        super().__init__(model_name=model, role=role)
        self.use_reasoning = (
            use_reasoning if use_reasoning is not None
            else settings.GROK_USE_REASONING
        )
        self.api_key = settings.XAI_API_KEY
        self.complexity = complexity

        # v4.1 파라미터 로드
        self._model_params = get_model_parameters(
            LLMModel.GROK_4.value, complexity
        )

    @property
    def display_name(self) -> str:
        suffix = " (reasoning)" if self.use_reasoning else ""
        return f"Grok 4{suffix}"

    def _get_council_member_prompt(self) -> str:
        """v4.1 한글 법률 전문가 프롬프트 반환"""
        return get_legal_expert_prompt_kr(
            model=LLMModel.GROK_4.value,
            session_memory="",
            short_term_memory="",
            long_term_memory="",
            rag_results="",
        )

    async def _initialize_client(self) -> None:
        """Initialize xAI client using OpenAI-compatible interface."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.XAI_BASE_URL,
            )
            logger.info(f"Grok client initialized: {self.model_name}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate response using Grok."""
        try:
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }

            # Enable reasoning if configured
            if self.use_reasoning and "reasoning" in self.model_name:
                params["reasoning"] = {
                    "effort": "high",
                }
            else:
                params["temperature"] = temperature

            response = await self._client.chat.completions.create(**params)

            message = response.choices[0].message
            usage = response.usage

            # Extract reasoning if available
            reasoning_content = None
            if hasattr(message, "reasoning"):
                reasoning_content = message.reasoning

            return LLMResponse(
                content=message.content or "",
                model=self.model_name,
                role=self.role,
                tokens_used=usage.total_tokens if usage else 0,
                raw_response=response,
                metadata={
                    "reasoning": reasoning_content,
                    "reasoning_enabled": self.use_reasoning,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except Exception as e:
            logger.error(f"Grok API error: {e}")
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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "stream": True,
            }

            if self.use_reasoning and "reasoning" in self.model_name:
                params["reasoning"] = {"effort": "high"}
            else:
                params["temperature"] = temperature

            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Grok streaming error: {e}")
            raise

    async def generate_with_live_search(
        self,
        context: LegalContext,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Generate response with Grok's live web search capability.

        Note: This uses Grok's built-in X/Twitter and web search
        for real-time information (if supported by the model version).
        """
        if not self._client:
            await self._initialize_client()

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Add instruction for using web search
        enhanced_system = f"""{system_prompt}

추가 지시사항:
- 필요한 경우 실시간 정보를 검색하여 최신 법률 동향을 반영하세요.
- 최근 판례나 법률 개정 사항이 있다면 언급해 주세요."""

        user_prompt = self._build_user_prompt(context)

        try:
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 4096,
            }

            # Enable search if model supports it
            if "live" in self.model_name.lower():
                params["search"] = {"enabled": True}

            if self.use_reasoning:
                params["reasoning"] = {"effort": "high"}
            else:
                params["temperature"] = 0.7

            response = await self._client.chat.completions.create(**params)

            message = response.choices[0].message

            # Extract search results if available
            search_results = []
            if hasattr(message, "search_results"):
                search_results = message.search_results

            return LLMResponse(
                content=message.content or "",
                model=self.model_name,
                role=self.role,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                raw_response=response,
                metadata={
                    "search_enabled": True,
                    "search_results": search_results,
                    "reasoning_enabled": self.use_reasoning,
                },
            )

        except Exception as e:
            logger.error(f"Grok live search error: {e}")
            # Fallback to standard generation
            return await self.generate(context, system_prompt)
