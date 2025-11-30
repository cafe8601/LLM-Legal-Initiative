"""
OpenAI GPT-5.1 Client

Stage 1 위원회 멤버 - OpenAI GPT-5.1 reasoning model
v4.1 한글 프롬프트 적용 (planning_instruction, persistence 지시 포함)
"""

import logging
from typing import AsyncIterator

from app.core.config import settings
from app.services.llm.base import (
    LegalContext,
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.legal_prompts_v4_1 import (
    get_legal_expert_prompt_kr,
    get_model_parameters,
    LLMModel,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class OpenAIClient(StreamingLLMClient):
    """
    OpenAI GPT-5.1 client for legal consultation.

    Uses the reasoning model with configurable effort level.
    v4.1: planning_instruction, persistence 지시, 메모리 시스템 통합
    """

    def __init__(
        self,
        model_name: str | None = None,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        reasoning_effort: str | None = None,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ):
        model = model_name or settings.GPT_MODEL
        super().__init__(model_name=model, role=role)
        self.reasoning_effort = reasoning_effort or settings.GPT_REASONING_EFFORT
        self.api_key = settings.OPENAI_API_KEY
        self.complexity = complexity

        # v4.1 파라미터 로드
        self._model_params = get_model_parameters(
            LLMModel.GPT_51.value, complexity
        )

    @property
    def display_name(self) -> str:
        return f"GPT-5.1 ({self.reasoning_effort})"

    def _get_council_member_prompt(self) -> str:
        """v4.1 한글 법률 전문가 프롬프트 반환"""
        return get_legal_expert_prompt_kr(
            model=LLMModel.GPT_51.value,
            session_memory="",  # 실제 사용 시 context에서 주입
            short_term_memory="",
            long_term_memory="",
            rag_results="",
        )

    async def _initialize_client(self) -> None:
        """Initialize OpenAI async client."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized: {self.model_name}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate response using GPT-5.1 reasoning model."""
        try:
            # GPT-5.1 uses reasoning parameter instead of system message
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": "auto",
                },
            )

            message = response.choices[0].message
            usage = response.usage

            # Extract structured content if available
            content = message.content or ""

            return LLMResponse(
                content=content,
                model=self.model_name,
                role=self.role,
                tokens_used=usage.total_tokens if usage else 0,
                raw_response=response,
                metadata={
                    "reasoning_effort": self.reasoning_effort,
                    "reasoning_tokens": getattr(usage, "reasoning_tokens", 0) if usage else 0,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": "auto",
                },
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def generate_with_tools(
        self,
        context: LegalContext,
        tools: list[dict],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Generate response with function calling tools.

        Useful for structured output extraction.
        """
        if not self._client:
            await self._initialize_client()

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_user_prompt(context)

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice="auto",
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": "auto",
                },
            )

            message = response.choices[0].message

            # Handle tool calls
            tool_calls = message.tool_calls or []
            tool_results = {}
            for tool_call in tool_calls:
                tool_results[tool_call.function.name] = tool_call.function.arguments

            return LLMResponse(
                content=message.content or "",
                model=self.model_name,
                role=self.role,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                raw_response=response,
                metadata={
                    "tool_calls": tool_results,
                    "reasoning_effort": self.reasoning_effort,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI tools API error: {e}")
            raise
