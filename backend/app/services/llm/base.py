"""
Base LLM Client Interface

모든 LLM 클라이언트의 기본 인터페이스 정의
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    """Model roles in the council."""

    COUNCIL_MEMBER = "council_member"  # Stage 1: Opinion providers
    PEER_REVIEWER = "peer_reviewer"    # Stage 2: Blind review
    CHAIRMAN = "chairman"              # Stage 3: Final synthesis


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    model: str
    role: ModelRole
    tokens_used: int = 0
    latency_ms: int = 0
    raw_response: Any = None
    metadata: dict = field(default_factory=dict)

    # For structured responses
    legal_basis: str | None = None
    risk_assessment: str | None = None
    recommendations: str | None = None
    confidence_level: str | None = None

    # For peer review
    reviewed_model: str | None = None
    review_scores: dict | None = None

    # Error handling
    error: str | None = None
    is_error: bool = False


@dataclass
class LegalContext:
    """Legal consultation context for LLM prompts."""

    query: str
    category: str = "general"
    conversation_history: list[dict] = field(default_factory=list)
    rag_context: list[dict] = field(default_factory=list)
    document_texts: list[str] = field(default_factory=list)
    user_tier: str = "basic"

    def to_context_string(self) -> str:
        """Convert RAG context to string for prompts."""
        if not self.rag_context:
            return ""

        context_parts = []
        for i, ctx in enumerate(self.rag_context, 1):
            source = ctx.get("source", "Unknown")
            content = ctx.get("content", "")
            doc_type = ctx.get("doc_type", "")

            context_parts.append(
                f"[참고문헌 {i}] ({doc_type}) {source}\n{content}"
            )

        return "\n\n".join(context_parts)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model_name: str,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.role = role
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable model name."""
        pass

    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the API client."""
        pass

    @abstractmethod
    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    async def generate(
        self,
        context: LegalContext,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a legal opinion/response.

        Args:
            context: Legal consultation context
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with the generated content
        """
        if not self._client:
            await self._initialize_client()

        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Build user prompt with context
        user_prompt = self._build_user_prompt(context)

        start_time = time.time()

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self._generate_response(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=self.timeout,
                )

                response.latency_ms = int((time.time() - start_time) * 1000)
                return response

            except asyncio.TimeoutError:
                last_error = f"Request timed out after {self.timeout}s"
                logger.warning(f"{self.display_name}: {last_error} (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"{self.display_name}: Error - {last_error} (attempt {attempt + 1})")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # Return error response after all retries failed
        return LLMResponse(
            content="",
            model=self.model_name,
            role=self.role,
            latency_ms=int((time.time() - start_time) * 1000),
            error=last_error,
            is_error=True,
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on role."""
        if self.role == ModelRole.COUNCIL_MEMBER:
            return self._get_council_member_prompt()
        elif self.role == ModelRole.PEER_REVIEWER:
            return self._get_peer_reviewer_prompt()
        elif self.role == ModelRole.CHAIRMAN:
            return self._get_chairman_prompt()
        return ""

    def _get_council_member_prompt(self) -> str:
        """System prompt for council members (Stage 1)."""
        return """당신은 AI 법률 자문 위원회의 위원입니다.
주어진 법률 질문에 대해 전문적이고 상세한 법률 의견을 제시해야 합니다.

## 응답 형식

다음 형식으로 응답하세요:

### 법률 의견
[질문에 대한 상세한 법률 분석 및 의견]

### 법적 근거
[관련 법률, 판례, 조문 인용]

### 리스크 평가
[잠재적 법적 리스크 및 주의사항]

### 권고사항
[구체적인 행동 권고]

### 신뢰도
[높음/중간/낮음] - 근거의 명확성과 적용 가능성을 기준으로

## 주의사항
1. 모든 의견은 한국 법률 체계를 기준으로 합니다.
2. 불확실한 부분은 명시적으로 언급하세요.
3. 개인정보나 민감한 정보는 다루지 마세요.
4. 이 의견은 참고용이며 실제 법률 자문을 대체할 수 없음을 인지하세요."""

    def _get_peer_reviewer_prompt(self) -> str:
        """System prompt for peer reviewer (Stage 2)."""
        return """당신은 AI 법률 자문 위원회의 교차 검토 담당입니다.
다른 AI의 법률 의견을 블라인드 평가하여 정확성, 완전성, 논리성을 검증합니다.

## 평가 기준

1. **정확성 (1-10)**: 법적 근거의 정확성
2. **완전성 (1-10)**: 주요 법적 쟁점 포괄 여부
3. **논리성 (1-10)**: 논증의 논리적 일관성
4. **실용성 (1-10)**: 권고사항의 실행 가능성

## 응답 형식

### 평가 요약
[전반적인 의견 품질 평가]

### 점수
- 정확성: X/10
- 완전성: X/10
- 논리성: X/10
- 실용성: X/10
- **종합: X/10**

### 강점
[의견의 강점]

### 개선점
[보완이 필요한 부분]

### 추가 고려사항
[검토 의견에서 누락된 중요 사항]

## 주의사항
- 객관적이고 건설적인 피드백을 제공하세요.
- 어떤 모델의 의견인지 알 수 없으므로 내용만으로 평가하세요."""

    def _get_chairman_prompt(self) -> str:
        """System prompt for chairman (Stage 3)."""
        return """당신은 AI 법률 자문 위원회의 의장입니다.
4명의 위원 의견과 블라인드 교차 평가 결과를 종합하여 최종 자문을 작성합니다.

## 역할

1. 모든 위원 의견 분석 및 종합
2. 교차 평가 점수 반영
3. 의견 간 상충점 조율
4. 최종 통합 자문 작성

## 응답 형식

### 의장 종합 의견

#### 핵심 결론
[가장 중요한 법률적 결론]

#### 종합 분석
[모든 위원 의견을 종합한 상세 분석]

#### 법적 근거 (종합)
[가장 신뢰할 수 있는 법적 근거들]

#### 리스크 종합 평가
[모든 의견에서 제기된 리스크 통합]

#### 최종 권고사항
[실행 가능한 구체적 권고]

### 위원 의견 요약
[각 위원 의견의 핵심 요약]

### 교차 평가 반영
[교차 평가에서 지적된 주요 사항과 반영 내용]

### 주의사항
[이 자문의 한계 및 추가 조치 필요 사항]

---
*본 자문은 AI 법률 자문 위원회의 종합 의견이며, 실제 법률 자문을 대체하지 않습니다.
중요한 법률 문제의 경우 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*"""

    def _build_user_prompt(self, context: LegalContext) -> str:
        """Build user prompt with context."""
        parts = []

        # Category
        category_names = {
            "general": "일반 법률",
            "contract": "계약법",
            "intellectual_property": "지식재산권",
            "labor": "노동법",
            "criminal": "형사법",
            "family": "가족법",
            "real_estate": "부동산법",
            "corporate": "기업법",
            "tax": "세법",
            "other": "기타",
        }
        parts.append(f"## 법률 분야: {category_names.get(context.category, context.category)}")

        # RAG context if available
        rag_context = context.to_context_string()
        if rag_context:
            parts.append(f"\n## 참고 자료\n{rag_context}")

        # Attached documents
        if context.document_texts:
            parts.append("\n## 첨부 문서")
            for i, doc_text in enumerate(context.document_texts, 1):
                # Truncate if too long
                truncated = doc_text[:3000] + "..." if len(doc_text) > 3000 else doc_text
                parts.append(f"\n### 문서 {i}\n{truncated}")

        # Conversation history
        if context.conversation_history:
            parts.append("\n## 이전 대화")
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                role = "사용자" if msg.get("role") == "user" else "자문위원회"
                parts.append(f"**{role}**: {msg.get('content', '')}")

        # Current query
        parts.append(f"\n## 질문\n{context.query}")

        return "\n".join(parts)

    async def close(self) -> None:
        """Close the client connection."""
        self._client = None

    async def __aenter__(self):
        await self._initialize_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class StreamingLLMClient(BaseLLMClient):
    """Base class for LLM clients with streaming support."""

    @abstractmethod
    async def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        pass

    async def generate_stream(
        self,
        context: LegalContext,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Generate streaming legal opinion/response.

        Args:
            context: Legal consultation context
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Chunks of generated text
        """
        if not self._client:
            await self._initialize_client()

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_user_prompt(context)

        try:
            async for chunk in self._generate_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"{self.display_name}: Streaming error - {e}")
            yield f"\n\n[오류 발생: {str(e)}]"
