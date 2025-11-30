"""
CascadeFlow 통합 서비스 v1.0

CascadeFlow 라이브러리 개념을 적용한 지능형 모델 캐스케이딩.
난이도 기반으로 저렴한 모델(Drafter)을 먼저 시도하고,
품질 검증 실패 시에만 고성능 모델(Verifier)로 에스컬레이션.

비용 절감 목표: 40-65%
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 모델 캐스케이드 설정
# =============================================================================


class CascadeModelTier(str, Enum):
    """모델 티어 (비용 기준)"""
    DRAFTER = "drafter"      # 저렴한 모델 (1차 시도)
    VERIFIER = "verifier"    # 고성능 모델 (에스컬레이션)


@dataclass
class CascadeModel:
    """캐스케이드 모델 설정"""
    openrouter_id: str           # OpenRouter 모델 ID
    display_name: str            # 표시 이름
    tier: CascadeModelTier       # 모델 티어
    cost_per_1m_tokens: float    # 1M 토큰당 비용 (USD)
    max_tokens: int = 4096       # 최대 출력 토큰
    temperature: float = 0.7     # 온도
    supports_reasoning: bool = False  # 추론 모델 여부


@dataclass
class CascadePair:
    """Drafter-Verifier 모델 쌍"""
    drafter: CascadeModel
    verifier: CascadeModel

    @property
    def cost_ratio(self) -> float:
        """비용 비율 (Verifier / Drafter)"""
        if self.drafter.cost_per_1m_tokens == 0:
            return float('inf')
        return self.verifier.cost_per_1m_tokens / self.drafter.cost_per_1m_tokens

    @property
    def potential_savings(self) -> float:
        """잠재적 비용 절감률 (60% 드래프터 사용 가정)"""
        drafter_ratio = 0.6  # 60%는 드래프터로 처리
        verifier_ratio = 0.4  # 40%는 에스컬레이션

        # 기존 비용 (100% Verifier) vs 캐스케이드 비용
        original_cost = self.verifier.cost_per_1m_tokens
        cascade_cost = (
            drafter_ratio * self.drafter.cost_per_1m_tokens +
            verifier_ratio * self.verifier.cost_per_1m_tokens
        )

        return (original_cost - cascade_cost) / original_cost


# =============================================================================
# 사전 정의된 모델 캐스케이드 쌍 (OpenRouter 통합)
# =============================================================================


# Claude 모델 쌍: Haiku 4.5 (드래프터) → Sonnet 4.5 (검증자)
CLAUDE_CASCADE = CascadePair(
    drafter=CascadeModel(
        openrouter_id="anthropic/claude-haiku-4",
        display_name="Claude Haiku 4.5",
        tier=CascadeModelTier.DRAFTER,
        cost_per_1m_tokens=0.80,  # $0.80/1M input tokens (추정)
        max_tokens=4096,
        temperature=0.7,
    ),
    verifier=CascadeModel(
        openrouter_id="anthropic/claude-sonnet-4",
        display_name="Claude Sonnet 4.5",
        tier=CascadeModelTier.VERIFIER,
        cost_per_1m_tokens=3.00,  # $3/1M input tokens (추정)
        max_tokens=8192,
        temperature=0.7,
    ),
)

# GPT 모델 쌍: GPT-4o-mini (드래프터) → GPT-5.1 (검증자)
GPT_CASCADE = CascadePair(
    drafter=CascadeModel(
        openrouter_id="openai/gpt-4o-mini",
        display_name="GPT-4o Mini",
        tier=CascadeModelTier.DRAFTER,
        cost_per_1m_tokens=0.15,  # $0.15/1M input tokens (추정)
        max_tokens=4096,
        temperature=0.7,
    ),
    verifier=CascadeModel(
        openrouter_id="openai/gpt-5.1",
        display_name="GPT-5.1",
        tier=CascadeModelTier.VERIFIER,
        cost_per_1m_tokens=2.50,  # $2.50/1M input tokens (추정)
        max_tokens=8192,
        temperature=0.7,
    ),
)

# Gemini 모델 쌍: Flash Latest (드래프터) → Pro 3 Preview (검증자)
GEMINI_CASCADE = CascadePair(
    drafter=CascadeModel(
        openrouter_id="google/gemini-flash-latest",
        display_name="Gemini Flash Latest",
        tier=CascadeModelTier.DRAFTER,
        cost_per_1m_tokens=0.075,  # $0.075/1M input tokens (추정)
        max_tokens=4096,
        temperature=0.7,
    ),
    verifier=CascadeModel(
        openrouter_id="google/gemini-3-pro-preview",
        display_name="Gemini 3 Pro Preview",
        tier=CascadeModelTier.VERIFIER,
        cost_per_1m_tokens=1.25,  # $1.25/1M input tokens (추정)
        max_tokens=8192,
        temperature=0.7,
    ),
)

# Grok 모델 쌍 (xAI): Grok 4 Fast (드래프터) → Grok 4.1 (검증자)
GROK_CASCADE = CascadePair(
    drafter=CascadeModel(
        openrouter_id="x-ai/grok-4-fast",
        display_name="Grok 4 Fast",
        tier=CascadeModelTier.DRAFTER,
        cost_per_1m_tokens=0.10,  # $0.10/1M input tokens (추정)
        max_tokens=4096,
        temperature=0.7,
    ),
    verifier=CascadeModel(
        openrouter_id="x-ai/grok-4.1",
        display_name="Grok 4.1",
        tier=CascadeModelTier.VERIFIER,
        cost_per_1m_tokens=2.00,  # $2/1M input tokens (추정)
        max_tokens=8192,
        temperature=0.7,
    ),
)

# Chairman용 특별 캐스케이드: Haiku 4.5 (드래프터) → Opus 4.5 (검증자)
CHAIRMAN_CASCADE = CascadePair(
    drafter=CascadeModel(
        openrouter_id="anthropic/claude-haiku-4",
        display_name="Claude Haiku 4.5",
        tier=CascadeModelTier.DRAFTER,
        cost_per_1m_tokens=0.80,  # $0.80/1M input tokens (추정)
        max_tokens=8192,
        temperature=0.7,
    ),
    verifier=CascadeModel(
        openrouter_id="anthropic/claude-opus-4",
        display_name="Claude Opus 4.5",
        tier=CascadeModelTier.VERIFIER,
        cost_per_1m_tokens=15.00,  # $15/1M input tokens (추정)
        max_tokens=16000,
        temperature=0.7,
    ),
)


# 제공자별 캐스케이드 매핑
CASCADE_BY_PROVIDER: dict[str, CascadePair] = {
    "anthropic": CLAUDE_CASCADE,
    "openai": GPT_CASCADE,
    "google": GEMINI_CASCADE,
    "x-ai": GROK_CASCADE,
}


# =============================================================================
# 품질 검증 엔진
# =============================================================================


class QueryComplexity(str, Enum):
    """쿼리 복잡도"""
    SIMPLE = "simple"      # 단순 질문 → 항상 Drafter
    MODERATE = "moderate"  # 중간 복잡도 → Drafter 시도 후 검증
    COMPLEX = "complex"    # 복잡한 질문 → 바로 Verifier


@dataclass
class QualityCheckResult:
    """품질 검증 결과"""
    passed: bool
    confidence: float  # 0.0 ~ 1.0
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeResult:
    """캐스케이드 실행 결과"""
    content: str
    model_used: str
    tier_used: CascadeModelTier
    escalated: bool  # 에스컬레이션 발생 여부
    drafter_attempted: bool
    quality_check: QualityCheckResult | None
    latency_ms: int
    tokens_used: int
    estimated_cost: float
    cost_saved: float  # 절감된 비용 (Verifier 직접 호출 대비)
    metadata: dict[str, Any] = field(default_factory=dict)


class QualityValidator:
    """
    응답 품질 검증기.

    CascadeFlow의 품질 검증 엔진을 법률 도메인에 맞게 구현.
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 10000,
        confidence_threshold: float = 0.7,
        require_legal_basis: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.require_legal_basis = require_legal_basis

        # 법률 도메인 품질 패턴
        self.legal_patterns = [
            r"제\d+조",  # 법조문 인용
            r"대법원|법원|판례",  # 법원/판례 언급
            r"법률|법령|시행령",  # 법률 용어
            r"원고|피고|청구인",  # 소송 당사자
        ]

        # 저품질 응답 패턴
        self.low_quality_patterns = [
            r"잘 모르겠습니다",
            r"확실하지 않습니다",
            r"더 많은 정보가 필요합니다",
            r"일반적인 정보만",
            r"법률 전문가.*상담",  # 과도한 면책
        ]

    def check_response_quality(
        self,
        query: str,
        response: str,
        domain: str = "general",
    ) -> QualityCheckResult:
        """
        응답 품질 검증.

        검증 기준:
        1. 길이 검증 (너무 짧거나 길지 않은지)
        2. 완전성 검증 (질문에 대한 답변인지)
        3. 법률 도메인 검증 (법적 근거 포함 여부)
        4. 신뢰도 검증 (불확실한 표현 여부)
        """
        reasons = []
        metrics = {}

        # 1. 길이 검증
        response_length = len(response)
        metrics["length"] = response_length

        if response_length < self.min_length:
            reasons.append(f"응답이 너무 짧습니다 ({response_length} < {self.min_length})")
        elif response_length > self.max_length:
            reasons.append(f"응답이 너무 깁니다 ({response_length} > {self.max_length})")

        length_score = 1.0 if self.min_length <= response_length <= self.max_length else 0.5

        # 2. 법률 도메인 검증
        legal_pattern_count = sum(
            1 for pattern in self.legal_patterns
            if re.search(pattern, response)
        )
        metrics["legal_patterns"] = legal_pattern_count

        legal_score = min(legal_pattern_count / 2, 1.0)  # 2개 이상이면 만점

        if self.require_legal_basis and legal_pattern_count == 0:
            reasons.append("법적 근거(법조문, 판례 등)가 포함되지 않았습니다")

        # 3. 저품질 패턴 검증
        low_quality_count = sum(
            1 for pattern in self.low_quality_patterns
            if re.search(pattern, response)
        )
        metrics["low_quality_patterns"] = low_quality_count

        quality_score = 1.0 - (low_quality_count * 0.2)  # 패턴당 0.2점 감점

        if low_quality_count > 0:
            reasons.append(f"불확실한 표현이 {low_quality_count}개 감지되었습니다")

        # 4. 쿼리-응답 연관성 (간단한 키워드 기반)
        query_keywords = set(re.findall(r'\b\w{2,}\b', query.lower()))
        response_keywords = set(re.findall(r'\b\w{2,}\b', response.lower()))

        if query_keywords:
            overlap = len(query_keywords & response_keywords) / len(query_keywords)
            metrics["keyword_overlap"] = overlap
            relevance_score = min(overlap * 2, 1.0)  # 50% 이상 겹치면 만점
        else:
            relevance_score = 0.5
            metrics["keyword_overlap"] = 0

        # 종합 신뢰도 계산
        confidence = (
            length_score * 0.2 +
            legal_score * 0.3 +
            quality_score * 0.3 +
            relevance_score * 0.2
        )
        metrics["confidence"] = confidence

        passed = confidence >= self.confidence_threshold and len(reasons) == 0

        return QualityCheckResult(
            passed=passed,
            confidence=confidence,
            reasons=reasons,
            metrics=metrics,
        )

    def classify_query_complexity(self, query: str) -> QueryComplexity:
        """
        쿼리 복잡도 분류.

        복잡도 기준:
        - SIMPLE: 단순 사실 확인, 정의 질문
        - MODERATE: 일반적인 법률 상담
        - COMPLEX: 복잡한 사안, 다중 쟁점, 분쟁 해결
        """
        query_lower = query.lower()
        query_length = len(query)

        # 복잡한 쿼리 패턴
        complex_patterns = [
            r"분쟁|소송|재판",
            r"손해배상.*청구",
            r"계약.*해지.*위약금",
            r"가압류|가처분|강제집행",
            r"형사|고소|고발",
            r"상속.*분쟁|유류분",
            r"다음.*경우.*어떻게",  # 복합 조건
        ]

        # 단순 쿼리 패턴
        simple_patterns = [
            r"^.{0,50}(무엇|뭐|정의|뜻)",  # 짧은 정의 질문
            r"기간.*며칠|몇 일",
            r"비용.*얼마",
            r"어디.*신청",
            r"서류.*뭐",
        ]

        # 복잡 패턴 체크
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                return QueryComplexity.COMPLEX

        # 단순 패턴 체크
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return QueryComplexity.SIMPLE

        # 길이 기반 분류
        if query_length < 50:
            return QueryComplexity.SIMPLE
        elif query_length > 300:
            return QueryComplexity.COMPLEX

        return QueryComplexity.MODERATE


# =============================================================================
# 캐스케이드 서비스
# =============================================================================


class CascadeService:
    """
    CascadeFlow 기반 지능형 모델 캐스케이딩 서비스.

    핵심 기능:
    1. 쿼리 복잡도에 따른 모델 라우팅
    2. 저렴한 모델 우선 실행 (Speculative Execution)
    3. 품질 검증 후 필요시 에스컬레이션
    4. 비용 추적 및 절감 리포팅
    """

    def __init__(
        self,
        cascade_pair: CascadePair,
        quality_validator: QualityValidator | None = None,
        always_verify: bool = False,  # True면 항상 검증자 사용
        skip_drafter_on_complex: bool = True,  # 복잡한 쿼리는 바로 검증자
    ):
        self.cascade_pair = cascade_pair
        self.quality_validator = quality_validator or QualityValidator()
        self.always_verify = always_verify
        self.skip_drafter_on_complex = skip_drafter_on_complex

        # OpenRouter 클라이언트
        self._client: AsyncOpenAI | None = None

        # 통계
        self.stats = {
            "total_requests": 0,
            "drafter_success": 0,
            "escalations": 0,
            "total_cost": 0.0,
            "cost_saved": 0.0,
        }

    async def _ensure_client(self) -> None:
        """OpenRouter 클라이언트 초기화"""
        if self._client is None:
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다")

            self._client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_SITE_URL or "http://localhost:3000",
                    "X-Title": settings.OPENROUTER_APP_NAME or "Legal Advisory Council",
                },
            )

    async def _call_model(
        self,
        model: CascadeModel,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """
        모델 호출.

        Returns:
            (응답 내용, 사용 토큰, 지연시간 ms)
        """
        await self._ensure_client()

        start_time = time.time()

        params = {
            "model": model.openrouter_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
        }

        # 추론 모델은 temperature 제거
        if model.supports_reasoning or "o1" in model.openrouter_id.lower():
            params.pop("temperature", None)

        response = await self._client.chat.completions.create(**params)

        latency_ms = int((time.time() - start_time) * 1000)
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        return content, tokens_used, latency_ms

    def _estimate_cost(self, model: CascadeModel, tokens: int) -> float:
        """비용 추정 (USD)"""
        return (tokens / 1_000_000) * model.cost_per_1m_tokens

    async def execute(
        self,
        system_prompt: str,
        user_prompt: str,
        query_for_complexity: str | None = None,
        force_verifier: bool = False,
    ) -> CascadeResult:
        """
        캐스케이드 실행.

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            query_for_complexity: 복잡도 분석용 쿼리 (없으면 user_prompt 사용)
            force_verifier: True면 검증자 직접 사용

        Returns:
            CascadeResult with response and metrics
        """
        self.stats["total_requests"] += 1

        query = query_for_complexity or user_prompt
        complexity = self.quality_validator.classify_query_complexity(query)

        # 검증자 직접 사용 조건
        use_verifier_directly = (
            force_verifier or
            self.always_verify or
            (self.skip_drafter_on_complex and complexity == QueryComplexity.COMPLEX)
        )

        total_latency_ms = 0
        total_tokens = 0
        drafter_attempted = False
        escalated = False
        quality_check = None

        if use_verifier_directly:
            # 검증자 직접 호출
            logger.info(f"[Cascade] 복잡도={complexity.value}, 검증자 직접 호출: {self.cascade_pair.verifier.display_name}")

            content, tokens_used, latency_ms = await self._call_model(
                self.cascade_pair.verifier,
                system_prompt,
                user_prompt,
            )

            total_latency_ms = latency_ms
            total_tokens = tokens_used
            model_used = self.cascade_pair.verifier.display_name
            tier_used = CascadeModelTier.VERIFIER
            estimated_cost = self._estimate_cost(self.cascade_pair.verifier, tokens_used)
            cost_saved = 0.0

        else:
            # 드래프터 먼저 시도
            logger.info(f"[Cascade] 복잡도={complexity.value}, 드래프터 시도: {self.cascade_pair.drafter.display_name}")
            drafter_attempted = True

            drafter_content, drafter_tokens, drafter_latency = await self._call_model(
                self.cascade_pair.drafter,
                system_prompt,
                user_prompt,
            )

            total_latency_ms += drafter_latency
            total_tokens += drafter_tokens

            # 품질 검증
            quality_check = self.quality_validator.check_response_quality(
                query=query,
                response=drafter_content,
            )

            if quality_check.passed:
                # 드래프터 응답 사용
                logger.info(f"[Cascade] 드래프터 응답 품질 통과 (confidence={quality_check.confidence:.2f})")
                self.stats["drafter_success"] += 1

                content = drafter_content
                model_used = self.cascade_pair.drafter.display_name
                tier_used = CascadeModelTier.DRAFTER

                # 비용 계산 (검증자 사용했을 경우 대비 절감액)
                drafter_cost = self._estimate_cost(self.cascade_pair.drafter, drafter_tokens)
                verifier_would_cost = self._estimate_cost(self.cascade_pair.verifier, drafter_tokens)
                estimated_cost = drafter_cost
                cost_saved = verifier_would_cost - drafter_cost

            else:
                # 에스컬레이션
                logger.info(f"[Cascade] 품질 검증 실패, 에스컬레이션: {quality_check.reasons}")
                escalated = True
                self.stats["escalations"] += 1

                verifier_content, verifier_tokens, verifier_latency = await self._call_model(
                    self.cascade_pair.verifier,
                    system_prompt,
                    user_prompt,
                )

                total_latency_ms += verifier_latency
                total_tokens += verifier_tokens

                content = verifier_content
                model_used = self.cascade_pair.verifier.display_name
                tier_used = CascadeModelTier.VERIFIER

                # 에스컬레이션 비용 (드래프터 + 검증자)
                drafter_cost = self._estimate_cost(self.cascade_pair.drafter, drafter_tokens)
                verifier_cost = self._estimate_cost(self.cascade_pair.verifier, verifier_tokens)
                estimated_cost = drafter_cost + verifier_cost
                cost_saved = 0.0  # 에스컬레이션 시 절감 없음

        # 통계 업데이트
        self.stats["total_cost"] += estimated_cost
        self.stats["cost_saved"] += cost_saved

        return CascadeResult(
            content=content,
            model_used=model_used,
            tier_used=tier_used,
            escalated=escalated,
            drafter_attempted=drafter_attempted,
            quality_check=quality_check,
            latency_ms=total_latency_ms,
            tokens_used=total_tokens,
            estimated_cost=estimated_cost,
            cost_saved=cost_saved,
            metadata={
                "complexity": complexity.value,
                "cascade_pair": f"{self.cascade_pair.drafter.display_name} → {self.cascade_pair.verifier.display_name}",
            },
        )

    def get_stats_summary(self) -> dict:
        """통계 요약"""
        total = self.stats["total_requests"]
        if total == 0:
            return {"message": "아직 요청이 없습니다"}

        drafter_rate = self.stats["drafter_success"] / total * 100
        escalation_rate = self.stats["escalations"] / total * 100

        return {
            "total_requests": total,
            "drafter_success_rate": f"{drafter_rate:.1f}%",
            "escalation_rate": f"{escalation_rate:.1f}%",
            "total_cost_usd": f"${self.stats['total_cost']:.6f}",
            "cost_saved_usd": f"${self.stats['cost_saved']:.6f}",
            "savings_percentage": f"{(self.stats['cost_saved'] / (self.stats['total_cost'] + self.stats['cost_saved']) * 100):.1f}%" if (self.stats['total_cost'] + self.stats['cost_saved']) > 0 else "0%",
        }

    async def close(self) -> None:
        """클라이언트 정리"""
        if self._client:
            await self._client.close()
            self._client = None


# =============================================================================
# 캐스케이드 서비스 팩토리
# =============================================================================


class CascadeServiceFactory:
    """캐스케이드 서비스 팩토리"""

    @staticmethod
    def create_for_provider(provider: str) -> CascadeService:
        """제공자별 캐스케이드 서비스 생성"""
        cascade_pair = CASCADE_BY_PROVIDER.get(provider)
        if not cascade_pair:
            raise ValueError(f"지원하지 않는 제공자: {provider}")

        return CascadeService(cascade_pair=cascade_pair)

    @staticmethod
    def create_claude_cascade() -> CascadeService:
        """Claude 캐스케이드 (Haiku → Sonnet)"""
        return CascadeService(cascade_pair=CLAUDE_CASCADE)

    @staticmethod
    def create_gpt_cascade() -> CascadeService:
        """GPT 캐스케이드 (4o-mini → 4o)"""
        return CascadeService(cascade_pair=GPT_CASCADE)

    @staticmethod
    def create_gemini_cascade() -> CascadeService:
        """Gemini 캐스케이드 (Flash → Pro)"""
        return CascadeService(cascade_pair=GEMINI_CASCADE)

    @staticmethod
    def create_chairman_cascade() -> CascadeService:
        """의장 캐스케이드 (Sonnet → Opus)"""
        return CascadeService(
            cascade_pair=CHAIRMAN_CASCADE,
            skip_drafter_on_complex=False,  # 의장은 항상 캐스케이드 시도
        )

    @staticmethod
    def create_all_council_cascades() -> dict[str, CascadeService]:
        """모든 위원회 멤버용 캐스케이드 서비스"""
        return {
            "claude": CascadeServiceFactory.create_claude_cascade(),
            "gpt": CascadeServiceFactory.create_gpt_cascade(),
            "gemini": CascadeServiceFactory.create_gemini_cascade(),
            "grok": CascadeService(cascade_pair=GROK_CASCADE),
        }
