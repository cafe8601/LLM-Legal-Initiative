"""
CascadeFlow 통합 법률 자문 위원회 v1.0

기존 CouncilOrchestrator에 CascadeFlow를 통합하여
비용 최적화된 3단계 법률 자문을 제공합니다.

핵심 변경:
- Stage 1: 4개 LLM 병렬 호출 → 캐스케이드 적용 (Drafter → Verifier)
- Stage 2: 교차 평가 → 캐스케이드 적용
- Stage 3: 의장 종합 → 복잡도에 따라 캐스케이드 적용

예상 비용 절감: 40-65%
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.core.config import settings
from app.services.llm.cascade_service import (
    CHAIRMAN_CASCADE,
    CLAUDE_CASCADE,
    GEMINI_CASCADE,
    GPT_CASCADE,
    GROK_CASCADE,
    CascadeModel,
    CascadeModelTier,
    CascadePair,
    CascadeResult,
    CascadeService,
    QualityValidator,
)
from app.services.llm.legal_prompts_v4_3 import (
    LegalDomain,
    LLMModel,
    assemble_expert_prompt,
    detect_domains,
    get_chairman_prompt,
    get_stage2_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================


@dataclass
class CascadeOpinion:
    """캐스케이드 적용된 위원 의견"""
    model: str  # 실제 사용된 모델
    display_name: str
    content: str
    cascade_result: CascadeResult | None = None  # 캐스케이드 메타데이터

    # 법률 분석 필드
    legal_basis: str = ""
    risk_assessment: str = ""
    recommendations: str = ""
    confidence_level: str = ""

    # 메트릭
    latency_ms: int = 0
    tokens_used: int = 0
    error: str | None = None

    # 캐스케이드 정보
    drafter_used: bool = False  # 드래프터로 처리됨
    escalated: bool = False  # 에스컬레이션됨
    cost_saved: float = 0.0


@dataclass
class CascadeReview:
    """캐스케이드 적용된 교차 평가"""
    reviewer_model: str
    reviewed_model: str
    content: str
    scores: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    cascade_result: CascadeResult | None = None


@dataclass
class CascadeSynthesis:
    """캐스케이드 적용된 의장 종합"""
    content: str
    model: str  # 실제 사용된 모델
    thinking_content: str | None = None
    latency_ms: int = 0
    tokens_used: int = 0
    cascade_result: CascadeResult | None = None


@dataclass
class CascadeCouncilResult:
    """캐스케이드 위원회 결과"""
    opinions: list[CascadeOpinion]
    reviews: list[CascadeReview]
    synthesis: CascadeSynthesis | None

    # 전체 메트릭
    total_latency_ms: int = 0
    total_tokens_used: int = 0
    stage_timings: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # 캐스케이드 비용 정보
    total_estimated_cost: float = 0.0
    total_cost_saved: float = 0.0
    drafter_success_count: int = 0
    escalation_count: int = 0

    @property
    def savings_percentage(self) -> float:
        """비용 절감률"""
        total = self.total_estimated_cost + self.total_cost_saved
        if total == 0:
            return 0.0
        return (self.total_cost_saved / total) * 100


# =============================================================================
# 캐스케이드 통합 위원회 오케스트레이터
# =============================================================================


class CascadeCouncilOrchestrator:
    """
    CascadeFlow 통합 3단계 법률 자문 위원회.

    Stage 1: 병렬 의견 수집 (4개 LLM, 캐스케이드 적용)
    Stage 2: 블라인드 교차 평가 (캐스케이드 적용)
    Stage 3: 의장 최종 종합 (복잡도 기반 캐스케이드)

    비용 절감 목표: 40-65%
    """

    def __init__(
        self,
        enable_peer_review: bool = True,
        enable_cascade: bool = True,  # 캐스케이드 활성화
        max_concurrent: int = 4,
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        domain: LegalDomain | None = None,
    ):
        self.enable_peer_review = enable_peer_review
        self.enable_cascade = enable_cascade
        self.max_concurrent = max_concurrent

        # 메모리 시스템
        self.session_memory = session_memory
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

        # 법률 분야
        self.domain = domain

        # 캐스케이드 서비스 (지연 초기화)
        self._cascade_services: dict[str, CascadeService] | None = None
        self._chairman_cascade: CascadeService | None = None

        # 품질 검증기 (공유)
        self._quality_validator = QualityValidator(
            min_length=200,  # 법률 응답은 더 길어야 함
            max_length=15000,
            confidence_threshold=0.65,  # 약간 관대하게
            require_legal_basis=True,
        )

    def _get_cascade_services(self) -> dict[str, CascadeService]:
        """캐스케이드 서비스 초기화 및 반환"""
        if self._cascade_services is None:
            self._cascade_services = {
                "claude": CascadeService(
                    cascade_pair=CLAUDE_CASCADE,
                    quality_validator=self._quality_validator,
                ),
                "gpt": CascadeService(
                    cascade_pair=GPT_CASCADE,
                    quality_validator=self._quality_validator,
                ),
                "gemini": CascadeService(
                    cascade_pair=GEMINI_CASCADE,
                    quality_validator=self._quality_validator,
                ),
                "grok": CascadeService(
                    cascade_pair=GROK_CASCADE,
                    quality_validator=self._quality_validator,
                ),
            }
        return self._cascade_services

    def _get_chairman_cascade(self) -> CascadeService:
        """의장 캐스케이드 서비스"""
        if self._chairman_cascade is None:
            self._chairman_cascade = CascadeService(
                cascade_pair=CHAIRMAN_CASCADE,
                quality_validator=QualityValidator(
                    min_length=500,  # 의장 응답은 더 상세해야 함
                    max_length=20000,
                    confidence_threshold=0.75,  # 더 엄격하게
                    require_legal_basis=True,
                ),
                skip_drafter_on_complex=False,  # 의장은 항상 캐스케이드 시도
            )
        return self._chairman_cascade

    def _detect_domain_from_query(self, query: str) -> LegalDomain:
        """키워드 기반 법률 분야 자동 감지"""
        detected = detect_domains(query)
        return detected[0] if detected else LegalDomain.GENERAL_CIVIL

    def _get_provider_key(self, model_id: str) -> str:
        """OpenRouter 모델 ID에서 제공자 키 추출"""
        if "claude" in model_id.lower() or "anthropic" in model_id.lower():
            return "claude"
        elif "gpt" in model_id.lower() or "openai" in model_id.lower():
            return "gpt"
        elif "gemini" in model_id.lower() or "google" in model_id.lower():
            return "gemini"
        elif "grok" in model_id.lower() or "x-ai" in model_id.lower():
            return "grok"
        return "claude"  # 기본값

    async def consult(
        self,
        query: str,
        category: str = "general",
        rag_context: str = "",
        progress_callback: Any | None = None,
        domain: LegalDomain | None = None,
    ) -> CascadeCouncilResult:
        """
        캐스케이드 적용 위원회 자문 실행.

        Args:
            query: 법률 질문
            category: 법률 분야
            rag_context: RAG 검색 결과
            progress_callback: 진행 콜백
            domain: 법률 분야 (자동 감지 가능)

        Returns:
            CascadeCouncilResult with all opinions, reviews, and synthesis
        """
        start_time = time.time()
        stage_timings = {}
        errors = []

        # 총 비용/절감 추적
        total_cost = 0.0
        total_saved = 0.0
        drafter_success = 0
        escalations = 0

        # 분야 자동 감지
        effective_domain = domain or self.domain
        if effective_domain is None:
            effective_domain = self._detect_domain_from_query(query)
            logger.info(f"[CascadeCouncil] 자동 감지된 법률 분야: {effective_domain.value}")

        async def report_progress(stage: str, progress: float, message: str):
            if progress_callback:
                try:
                    await progress_callback(stage, progress, message)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # =====================================================================
        # Stage 1: 병렬 의견 수집 (캐스케이드 적용)
        # =====================================================================
        await report_progress("stage1", 0.0, f"위원회 의견 수집 시작... (분야: {effective_domain.value})")

        stage1_start = time.time()
        opinions = await self._collect_cascade_opinions(
            query=query,
            category=category,
            rag_context=rag_context,
            domain=effective_domain,
            report_progress=report_progress,
        )
        stage_timings["stage1"] = int((time.time() - stage1_start) * 1000)

        # 통계 집계
        successful_opinions = [o for o in opinions if not o.error]
        for opinion in opinions:
            if opinion.cascade_result:
                total_cost += opinion.cascade_result.estimated_cost
                total_saved += opinion.cascade_result.cost_saved
                if opinion.drafter_used:
                    drafter_success += 1
                if opinion.escalated:
                    escalations += 1
            if opinion.error:
                errors.append(f"{opinion.display_name}: {opinion.error}")

        if len(successful_opinions) < 2:
            return CascadeCouncilResult(
                opinions=opinions,
                reviews=[],
                synthesis=None,
                total_latency_ms=int((time.time() - start_time) * 1000),
                stage_timings=stage_timings,
                errors=errors + ["최소 2개 이상의 의견이 필요합니다."],
                total_estimated_cost=total_cost,
                total_cost_saved=total_saved,
                drafter_success_count=drafter_success,
                escalation_count=escalations,
            )

        await report_progress("stage1", 1.0, f"{len(successful_opinions)}개 의견 수집 완료")

        # =====================================================================
        # Stage 2: 블라인드 교차 평가 (캐스케이드 적용)
        # =====================================================================
        reviews = []
        if self.enable_peer_review and len(successful_opinions) >= 2:
            await report_progress("stage2", 0.0, "블라인드 교차 평가 시작...")

            stage2_start = time.time()
            reviews = await self._conduct_cascade_reviews(
                query=query,
                category=category,
                opinions=successful_opinions,
                domain=effective_domain,
                report_progress=report_progress,
            )
            stage_timings["stage2"] = int((time.time() - stage2_start) * 1000)

            # 리뷰 비용 집계
            for review in reviews:
                if review.cascade_result:
                    total_cost += review.cascade_result.estimated_cost
                    total_saved += review.cascade_result.cost_saved
                    if not review.cascade_result.escalated:
                        drafter_success += 1
                    else:
                        escalations += 1

            await report_progress("stage2", 1.0, f"{len(reviews)}개 교차 평가 완료")

        # =====================================================================
        # Stage 3: 의장 종합 (캐스케이드 적용)
        # =====================================================================
        await report_progress("stage3", 0.0, "의장 종합 의견 작성 중...")

        stage3_start = time.time()
        synthesis = await self._cascade_synthesis(
            query=query,
            category=category,
            opinions=successful_opinions,
            reviews=reviews,
            rag_context=rag_context,
            domain=effective_domain,
        )
        stage_timings["stage3"] = int((time.time() - stage3_start) * 1000)

        if synthesis and synthesis.cascade_result:
            total_cost += synthesis.cascade_result.estimated_cost
            total_saved += synthesis.cascade_result.cost_saved
            if not synthesis.cascade_result.escalated:
                drafter_success += 1
            else:
                escalations += 1

        await report_progress("stage3", 1.0, "의장 종합 의견 완료")

        total_tokens = sum(o.tokens_used for o in opinions)
        if synthesis:
            total_tokens += synthesis.tokens_used

        return CascadeCouncilResult(
            opinions=opinions,
            reviews=reviews,
            synthesis=synthesis,
            total_latency_ms=int((time.time() - start_time) * 1000),
            total_tokens_used=total_tokens,
            stage_timings=stage_timings,
            errors=errors,
            total_estimated_cost=total_cost,
            total_cost_saved=total_saved,
            drafter_success_count=drafter_success,
            escalation_count=escalations,
        )

    async def _collect_cascade_opinions(
        self,
        query: str,
        category: str,
        rag_context: str,
        domain: LegalDomain,
        report_progress: Any,
    ) -> list[CascadeOpinion]:
        """Stage 1: 캐스케이드 적용 병렬 의견 수집"""
        cascade_services = self._get_cascade_services()

        # 위원별 설정
        council_config = [
            ("claude", "anthropic/claude-sonnet-4", "Claude Sonnet 4 위원"),
            ("gpt", "openai/gpt-4o", "GPT-4o 위원"),
            ("gemini", "google/gemini-2.5-pro-preview", "Gemini 2.5 Pro 위원"),
            ("grok", "x-ai/grok-2", "Grok 2 위원"),
        ]

        async def get_opinion(
            provider_key: str,
            model_id: str,
            display_name: str,
            index: int,
        ) -> CascadeOpinion:
            start_time = time.time()

            try:
                cascade_service = cascade_services[provider_key]

                # 시스템 프롬프트 생성
                system_prompt = assemble_expert_prompt(
                    domain=domain,
                    model=self._get_llm_model(model_id),
                    session_memory=self.session_memory,
                    short_term_memory=self.short_term_memory,
                    long_term_memory=self.long_term_memory,
                    rag_results=rag_context,
                )

                user_prompt = f"""## 법률 질문
{query}

## 법률 분야
{category}

위의 질문에 대해 법률 전문가로서 상세한 의견을 제시해 주세요.
법적 근거, 위험 평가, 권장 사항을 포함해 주세요."""

                # 캐스케이드 실행
                if self.enable_cascade:
                    cascade_result = await cascade_service.execute(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        query_for_complexity=query,
                    )

                    await report_progress(
                        "stage1",
                        (index + 1) / len(council_config),
                        f"{display_name} 의견 수집 완료 "
                        f"({'드래프터' if cascade_result.tier_used == CascadeModelTier.DRAFTER else '검증자'})",
                    )

                    return CascadeOpinion(
                        model=cascade_result.model_used,
                        display_name=display_name,
                        content=cascade_result.content,
                        cascade_result=cascade_result,
                        latency_ms=cascade_result.latency_ms,
                        tokens_used=cascade_result.tokens_used,
                        drafter_used=cascade_result.tier_used == CascadeModelTier.DRAFTER,
                        escalated=cascade_result.escalated,
                        cost_saved=cascade_result.cost_saved,
                    )
                else:
                    # 캐스케이드 비활성화 시 검증자 직접 사용
                    cascade_result = await cascade_service.execute(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        force_verifier=True,
                    )

                    return CascadeOpinion(
                        model=cascade_result.model_used,
                        display_name=display_name,
                        content=cascade_result.content,
                        cascade_result=cascade_result,
                        latency_ms=cascade_result.latency_ms,
                        tokens_used=cascade_result.tokens_used,
                    )

            except Exception as e:
                logger.error(f"Opinion collection error ({display_name}): {e}")
                return CascadeOpinion(
                    model=model_id,
                    display_name=display_name,
                    content="",
                    error=str(e),
                    latency_ms=int((time.time() - start_time) * 1000),
                )

        # 병렬 실행
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_get_opinion(
            provider_key: str,
            model_id: str,
            display_name: str,
            index: int,
        ):
            async with semaphore:
                return await get_opinion(provider_key, model_id, display_name, index)

        tasks = [
            limited_get_opinion(provider_key, model_id, display_name, i)
            for i, (provider_key, model_id, display_name) in enumerate(council_config)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        opinions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _, model_id, display_name = council_config[i]
                logger.error(f"Unexpected error for {display_name}: {result}")
                opinions.append(CascadeOpinion(
                    model=model_id,
                    display_name=display_name,
                    content="",
                    error=f"예상치 못한 오류: {str(result)}",
                ))
            else:
                opinions.append(result)

        return opinions

    async def _conduct_cascade_reviews(
        self,
        query: str,
        category: str,
        opinions: list[CascadeOpinion],
        domain: LegalDomain,
        report_progress: Any,
    ) -> list[CascadeReview]:
        """Stage 2: 캐스케이드 적용 블라인드 교차 평가"""
        cascade_services = self._get_cascade_services()
        reviews = []

        # 모든 위원이 다른 모든 위원을 평가 (n * (n-1))
        total_reviews = len(opinions) * (len(opinions) - 1)
        completed_reviews = 0

        review_tasks = []
        for reviewer_idx, reviewer in enumerate(opinions):
            if reviewer.error:
                continue

            for target_idx, target in enumerate(opinions):
                if reviewer_idx == target_idx or target.error:
                    continue

                review_tasks.append({
                    "reviewer": reviewer,
                    "target": target,
                    "reviewer_idx": reviewer_idx,
                })

        async def perform_review(task: dict) -> CascadeReview:
            nonlocal completed_reviews
            reviewer = task["reviewer"]
            target = task["target"]

            try:
                # 리뷰어의 제공자 키로 캐스케이드 서비스 선택
                provider_key = self._get_provider_key(reviewer.model)
                cascade_service = cascade_services.get(provider_key, cascade_services["claude"])

                system_prompt = get_stage2_prompt(
                    model=self._get_llm_model(reviewer.model),
                    original_question=query,
                    domain=domain,
                    anonymized_opinions=target.content,
                    rag_results="",
                )

                user_prompt = f"""## 법률 분야: {category}

위 익명화된 의견을 평가 기준에 따라 블라인드 교차 평가해 주세요.
5가지 기준(정확성, 완전성, 메모리 통합, 유용성, 명확성) 각 10점 만점으로 평가하세요."""

                cascade_result = await cascade_service.execute(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    query_for_complexity=query,
                )

                completed_reviews += 1
                await report_progress(
                    "stage2",
                    completed_reviews / max(total_reviews, 1),
                    f"{completed_reviews}/{total_reviews} 교차 평가 완료",
                )

                return CascadeReview(
                    reviewer_model=reviewer.display_name,
                    reviewed_model=target.display_name,
                    content=cascade_result.content,
                    scores=self._parse_review_scores(cascade_result.content),
                    latency_ms=cascade_result.latency_ms,
                    cascade_result=cascade_result,
                )

            except Exception as e:
                logger.error(f"Review error: {reviewer.display_name} → {target.display_name}: {e}")
                completed_reviews += 1
                return CascadeReview(
                    reviewer_model=reviewer.display_name,
                    reviewed_model=target.display_name,
                    content=f"평가 중 오류 발생: {str(e)}",
                    scores={},
                )

        # 병렬 실행 (리뷰는 더 많은 동시성 허용)
        semaphore = asyncio.Semaphore(self.max_concurrent * 2)

        async def limited_review(task: dict) -> CascadeReview:
            async with semaphore:
                return await perform_review(task)

        review_results = await asyncio.gather(
            *[limited_review(task) for task in review_tasks],
            return_exceptions=True,
        )

        for result in review_results:
            if isinstance(result, Exception):
                logger.error(f"Review task failed: {result}")
            elif isinstance(result, CascadeReview):
                reviews.append(result)

        return reviews

    async def _cascade_synthesis(
        self,
        query: str,
        category: str,
        opinions: list[CascadeOpinion],
        reviews: list[CascadeReview],
        rag_context: str,
        domain: LegalDomain,
    ) -> CascadeSynthesis | None:
        """Stage 3: 캐스케이드 적용 의장 종합"""
        try:
            chairman_cascade = self._get_chairman_cascade()

            # 의장 프롬프트
            system_prompt = get_chairman_prompt(
                model=LLMModel.CLAUDE_OPUS_45,
                session_memory=self.session_memory,
                short_term_memory=self.short_term_memory,
                long_term_memory=self.long_term_memory,
                rag_results=rag_context,
                domain=domain,
            )

            # 종합 프롬프트 구성
            parts = [f"## 원본 질문\n{query}\n\n## 법률 분야\n{category}"]

            if rag_context:
                parts.append(f"\n## 참고 문헌 (RAG 검색 결과)\n{rag_context}")

            parts.append("\n## Stage 1: 위원 의견")
            for opinion in opinions:
                parts.append(f"\n### {opinion.display_name}\n{opinion.content}")

            parts.append("\n## Stage 2: 블라인드 교차 평가 결과")
            for review in reviews:
                parts.append(f"\n### {review.reviewer_model}의 {review.reviewed_model} 평가")
                if review.scores:
                    parts.append(f"점수: {review.scores}")
                parts.append(f"\n{review.content}")

            user_prompt = "\n".join(parts)
            user_prompt += """

## 의장 지시사항

6단계 합성 프로토콜에 따라 최종 자문을 작성해 주세요:
1. 인용 검증 및 권위 평가
2. 합의 및 분기 지점 분석
3. 충돌 해결
4. 일관성 검증 (장기 기억 참조)
5. 공백 보완
6. 최종 합성

모든 법적 진술에 검증된 RAG 인용을 포함하세요."""

            cascade_result = await chairman_cascade.execute(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                query_for_complexity=query,
            )

            return CascadeSynthesis(
                content=cascade_result.content,
                model=cascade_result.model_used,
                latency_ms=cascade_result.latency_ms,
                tokens_used=cascade_result.tokens_used,
                cascade_result=cascade_result,
            )

        except Exception as e:
            logger.error(f"Chairman synthesis error: {e}")
            return None

    def _get_llm_model(self, model_id: str) -> LLMModel:
        """OpenRouter 모델 ID를 LLMModel enum으로 변환"""
        model_lower = model_id.lower()

        if "opus" in model_lower:
            return LLMModel.CLAUDE_OPUS_45
        elif "sonnet" in model_lower:
            return LLMModel.CLAUDE_SONNET_45
        elif "haiku" in model_lower:
            return LLMModel.CLAUDE_SONNET_45  # Haiku는 Sonnet 프롬프트 사용
        elif "gpt-4o" in model_lower:
            return LLMModel.GPT_51
        elif "gemini" in model_lower:
            return LLMModel.GEMINI_3_PRO
        elif "grok" in model_lower:
            return LLMModel.GROK_4
        else:
            return LLMModel.CLAUDE_SONNET_45

    def _parse_review_scores(self, review_content: str) -> dict:
        """리뷰에서 점수 파싱"""
        import re
        scores = {}

        score_patterns = {
            "정확성": r"정확성[:\s]*(\d+)",
            "완전성": r"완전성[:\s]*(\d+)",
            "메모리_통합": r"메모리\s*통합[:\s]*(\d+)",
            "유용성": r"유용성[:\s]*(\d+)",
            "명확성": r"명확성[:\s]*(\d+)",
            "종합": r"종합[:\s]*(\d+)",
        }

        for name, pattern in score_patterns.items():
            match = re.search(pattern, review_content)
            if match:
                try:
                    scores[name] = int(match.group(1))
                except ValueError:
                    pass

        return scores

    def get_cascade_stats(self) -> dict:
        """캐스케이드 통계 반환"""
        cascade_services = self._get_cascade_services()
        stats = {
            "services": {},
            "total": {
                "requests": 0,
                "drafter_success": 0,
                "escalations": 0,
                "cost": 0.0,
                "saved": 0.0,
            },
        }

        for name, service in cascade_services.items():
            service_stats = service.get_stats_summary()
            stats["services"][name] = service_stats
            stats["total"]["requests"] += service.stats["total_requests"]
            stats["total"]["drafter_success"] += service.stats["drafter_success"]
            stats["total"]["escalations"] += service.stats["escalations"]
            stats["total"]["cost"] += service.stats["total_cost"]
            stats["total"]["saved"] += service.stats["cost_saved"]

        if self._chairman_cascade:
            chairman_stats = self._chairman_cascade.get_stats_summary()
            stats["services"]["chairman"] = chairman_stats
            stats["total"]["requests"] += self._chairman_cascade.stats["total_requests"]
            stats["total"]["drafter_success"] += self._chairman_cascade.stats["drafter_success"]
            stats["total"]["escalations"] += self._chairman_cascade.stats["escalations"]
            stats["total"]["cost"] += self._chairman_cascade.stats["total_cost"]
            stats["total"]["saved"] += self._chairman_cascade.stats["cost_saved"]

        total_with_saved = stats["total"]["cost"] + stats["total"]["saved"]
        if total_with_saved > 0:
            stats["total"]["savings_percentage"] = f"{(stats['total']['saved'] / total_with_saved * 100):.1f}%"
        else:
            stats["total"]["savings_percentage"] = "0%"

        return stats

    async def close(self):
        """모든 캐스케이드 서비스 정리"""
        if self._cascade_services:
            for service in self._cascade_services.values():
                await service.close()
            self._cascade_services = None

        if self._chairman_cascade:
            await self._chairman_cascade.close()
            self._chairman_cascade = None
