"""
Council Orchestrator v4.3.2

4-Model Council 오케스트레이션:
- Stage 1: 4개 LLM 병렬 의견 수집
- Stage 2: 모든 위원이 서로 블라인드 교차 평가 (n×(n-1) = 12개 리뷰)
- Stage 3: Claude Opus 4.5 의장 종합

v4.3.2 변경:
- Stage 2 개선: 단일 리뷰어 → 모든 위원이 서로 평가하는 진정한 블라인드 피어리뷰
- 4명 위원 × 3개 평가 = 12개 교차 평가로 다양한 관점 확보
- 병렬 처리로 속도 최적화

v4.3.1 특징:
- 분야별 모듈형 프롬프트 (~560 토큰 vs 전체 2,000+)
- LLM별 최적화된 addon 자동 적용
- 키워드 기반 복합 사안 감지 (비용 0)
- 5가지 평가 기준 (정확성/완전성/메모리 통합/유용성/명확성)
- 6단계 합성 프로토콜
- 메모리 시스템 통합 (세션/단기/장기)
- RAG 인용 검증
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from uuid import UUID

from app.services.llm.base import LegalContext, LLMResponse, ModelRole
from app.services.llm.factory import LLMClientFactory
from app.services.llm.legal_prompts_v4_3 import (
    LegalDomain,
    detect_domains,
    STAGE2_EVALUATION_CRITERIA_KR,
)

logger = logging.getLogger(__name__)


@dataclass
class CouncilOpinion:
    """Individual council member opinion."""

    model: str
    display_name: str
    content: str
    legal_basis: str | None = None
    risk_assessment: str | None = None
    recommendations: str | None = None
    confidence_level: str | None = None
    latency_ms: int = 0
    tokens_used: int = 0
    error: str | None = None


@dataclass
class PeerReview:
    """
    Peer review of a council opinion.

    v4.1 평가 기준:
    - 정확성 (Accuracy): 법적 정확성, 인용 정확성 (10점)
    - 완전성 (Completeness): 쟁점 포괄, 예외 고려 (10점)
    - 메모리 통합 (Memory Integration): 맥락 활용도 (10점)
    - 유용성 (Usefulness): 실무 적용 가능성 (10점)
    - 명확성 (Clarity): 구조, 이해 용이성 (10점)
    """

    reviewer_model: str
    reviewed_model: str
    content: str
    scores: dict = field(default_factory=dict)  # v4.1: 5가지 평가 기준
    latency_ms: int = 0


@dataclass
class ChairmanSynthesis:
    """Chairman's final synthesis."""

    content: str
    model: str
    thinking_content: str | None = None
    latency_ms: int = 0
    tokens_used: int = 0


@dataclass
class CouncilResult:
    """Complete council consultation result."""

    opinions: list[CouncilOpinion]
    reviews: list[PeerReview]
    synthesis: ChairmanSynthesis | None
    total_latency_ms: int = 0
    total_tokens_used: int = 0
    stage_timings: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class CouncilOrchestrator:
    """
    Orchestrates the 3-stage legal consultation council v4.3.1.

    Stage 1: Parallel opinion collection from 4 LLMs
    Stage 2: All council members cross-review each other (n×(n-1) = 12 reviews)
    Stage 3: Final synthesis by Claude Opus 4.5 (Chairman)

    v4.3.2 Features:
    - 분야별 모듈형 프롬프트 (~560 토큰)
    - LLM별 최적화된 addon 자동 적용
    - 키워드 기반 복합 사안 감지 (비용 0)
    - 메모리 시스템 통합 (세션/단기/장기)
    - 모든 위원 상호 블라인드 평가 (5가지 기준)
    - 6단계 의장 합성 프로토콜
    - RAG 인용 검증
    """

    def __init__(
        self,
        enable_peer_review: bool = True,
        max_concurrent: int = 4,
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        domain: LegalDomain | None = None,  # v4.3.1: 법률 분야
    ):
        self.enable_peer_review = enable_peer_review
        self.max_concurrent = max_concurrent

        # v4.1 메모리 시스템
        self.session_memory = session_memory
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

        # v4.3.1: 법률 분야
        self.domain = domain

        # Initialize clients lazily
        # v4.3.2: _peer_reviewer 제거 (모든 위원이 리뷰어 역할 수행)
        self._council_members = None
        self._chairman = None

    def _detect_domain_from_query(self, query: str) -> LegalDomain:
        """
        v4.3.1: 키워드 기반 분야 자동 감지 (비용 0).

        Args:
            query: 법률 질문

        Returns:
            감지된 첫 번째 법률 분야
        """
        detected = detect_domains(query)
        return detected[0] if detected else LegalDomain.GENERAL_CIVIL

    async def _get_council_members(self, domain: LegalDomain | None = None):
        """Get or create council member clients with v4.3.1 domain support."""
        effective_domain = domain or self.domain
        if self._council_members is None:
            self._council_members = LLMClientFactory.create_all_council_members(
                domain=effective_domain
            )
        return self._council_members

    # v4.3.2: _get_peer_reviewer 제거 - 모든 council_members가 피어리뷰 수행

    async def _get_chairman(self, domain: LegalDomain | None = None):
        """Get or create chairman client with v4.3.1 domain support."""
        effective_domain = domain or self.domain
        if self._chairman is None:
            self._chairman = LLMClientFactory.create_chairman(
                domain=effective_domain
            )
        return self._chairman

    async def consult(
        self,
        context: LegalContext,
        progress_callback: Any | None = None,
        domain: LegalDomain | None = None,
    ) -> CouncilResult:
        """
        Run full council consultation with v4.3.1 support.

        Args:
            context: Legal consultation context
            progress_callback: Optional async callback for progress updates
                              Signature: async callback(stage: str, progress: float, message: str)
            domain: 법률 분야 (None이면 키워드 기반 자동 감지)

        Returns:
            CouncilResult with all opinions, reviews, and synthesis
        """
        start_time = time.time()
        stage_timings = {}
        errors = []
        total_tokens = 0

        # v4.3.1: 분야 자동 감지 (비용 0)
        effective_domain = domain or self.domain
        if effective_domain is None:
            effective_domain = self._detect_domain_from_query(context.query)
            logger.info(f"v4.3.1 자동 감지된 법률 분야: {effective_domain.value}")

        async def report_progress(stage: str, progress: float, message: str):
            if progress_callback:
                try:
                    await progress_callback(stage, progress, message)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # =====================================================================
        # Stage 1: Parallel Opinion Collection (v4.3.1 분야 적용)
        # =====================================================================
        await report_progress("stage1", 0.0, f"위원회 의견 수집 시작... (분야: {effective_domain.value})")

        stage1_start = time.time()
        opinions = await self._collect_opinions(context, report_progress, effective_domain)
        stage_timings["stage1"] = int((time.time() - stage1_start) * 1000)

        # Count successful opinions
        successful_opinions = [o for o in opinions if not o.error]
        total_tokens += sum(o.tokens_used for o in successful_opinions)

        for opinion in opinions:
            if opinion.error:
                errors.append(f"{opinion.display_name}: {opinion.error}")

        if len(successful_opinions) < 2:
            # Not enough opinions to proceed
            return CouncilResult(
                opinions=opinions,
                reviews=[],
                synthesis=None,
                total_latency_ms=int((time.time() - start_time) * 1000),
                total_tokens_used=total_tokens,
                stage_timings=stage_timings,
                errors=errors + ["최소 2개 이상의 의견이 필요합니다."],
            )

        await report_progress("stage1", 1.0, f"{len(successful_opinions)}개 의견 수집 완료")

        # =====================================================================
        # Stage 2: Peer Review (if enabled) - v4.3.1 분야 적용
        # =====================================================================
        reviews = []
        if self.enable_peer_review and len(successful_opinions) >= 2:
            await report_progress("stage2", 0.0, "블라인드 교차 평가 시작...")

            stage2_start = time.time()
            reviews = await self._conduct_peer_reviews(
                context, successful_opinions, report_progress, effective_domain
            )
            stage_timings["stage2"] = int((time.time() - stage2_start) * 1000)

            total_tokens += sum(r.latency_ms for r in reviews)  # Approximate

            await report_progress("stage2", 1.0, f"{len(reviews)}개 교차 평가 완료")

        # =====================================================================
        # Stage 3: Chairman Synthesis - v4.3.1 분야 적용
        # =====================================================================
        await report_progress("stage3", 0.0, "의장 종합 의견 작성 중...")

        stage3_start = time.time()
        synthesis = await self._synthesize(
            context, successful_opinions, reviews, report_progress, effective_domain
        )
        stage_timings["stage3"] = int((time.time() - stage3_start) * 1000)

        if synthesis:
            total_tokens += synthesis.tokens_used

        await report_progress("stage3", 1.0, "의장 종합 의견 완료")

        return CouncilResult(
            opinions=opinions,
            reviews=reviews,
            synthesis=synthesis,
            total_latency_ms=int((time.time() - start_time) * 1000),
            total_tokens_used=total_tokens,
            stage_timings=stage_timings,
            errors=errors,
        )

    async def _collect_opinions(
        self,
        context: LegalContext,
        report_progress: Any,
        domain: LegalDomain | None = None,
    ) -> list[CouncilOpinion]:
        """
        Stage 1: Collect opinions from all council members in parallel (v4.3.1).

        OpenRouter 통합으로 모든 LLM이 단일 API를 통해 호출됩니다.
        v4.3.1: 분야별 모듈형 프롬프트로 ~560 토큰만 사용.
        """
        council_members = await self._get_council_members(domain)

        # 클라이언트 초기화 (병렬)
        init_tasks = []
        for client in council_members:
            if client._client is None:
                init_tasks.append(client._initialize_client())

        if init_tasks:
            init_results = await asyncio.gather(*init_tasks, return_exceptions=True)
            for i, result in enumerate(init_results):
                if isinstance(result, Exception):
                    logger.warning(f"Client initialization warning: {result}")

        async def get_opinion(client, index: int) -> CouncilOpinion:
            start_time = time.time()
            try:
                # 클라이언트가 초기화되지 않은 경우 재시도
                if client._client is None:
                    try:
                        await client._initialize_client()
                    except Exception as init_error:
                        logger.error(f"Failed to initialize {client.display_name}: {init_error}")
                        return CouncilOpinion(
                            model=getattr(client, 'model_name', 'unknown'),
                            display_name=getattr(client, 'display_name', 'Unknown Model'),
                            content="",
                            error=f"클라이언트 초기화 실패: {str(init_error)}",
                            latency_ms=int((time.time() - start_time) * 1000),
                        )

                response = await client.generate(context)

                await report_progress(
                    "stage1",
                    (index + 1) / len(council_members),
                    f"{client.display_name} 의견 수집 완료",
                )

                if response.is_error:
                    return CouncilOpinion(
                        model=client.model_name,
                        display_name=client.display_name,
                        content="",
                        error=response.error,
                        latency_ms=response.latency_ms,
                    )

                return CouncilOpinion(
                    model=client.model_name,
                    display_name=client.display_name,
                    content=response.content,
                    legal_basis=response.legal_basis,
                    risk_assessment=response.risk_assessment,
                    recommendations=response.recommendations,
                    confidence_level=response.confidence_level,
                    latency_ms=response.latency_ms,
                    tokens_used=response.tokens_used,
                )

            except asyncio.TimeoutError:
                logger.error(f"Opinion collection timeout ({client.display_name})")
                return CouncilOpinion(
                    model=getattr(client, 'model_name', 'unknown'),
                    display_name=getattr(client, 'display_name', 'Unknown Model'),
                    content="",
                    error="요청 시간 초과",
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            except Exception as e:
                logger.error(f"Opinion collection error ({client.display_name}): {e}")
                return CouncilOpinion(
                    model=getattr(client, 'model_name', 'unknown'),
                    display_name=getattr(client, 'display_name', 'Unknown Model'),
                    content="",
                    error=str(e),
                    latency_ms=int((time.time() - start_time) * 1000),
                )

        # Run all in parallel with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_get_opinion(client, index):
            async with semaphore:
                return await get_opinion(client, index)

        tasks = [
            limited_get_opinion(client, i)
            for i, client in enumerate(council_members)
        ]

        # return_exceptions=True로 개별 실패가 전체를 중단시키지 않도록 함
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opinions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 예외가 발생한 경우 에러 OpinIon으로 변환
                client = council_members[i]
                logger.error(f"Unexpected error for {client.display_name}: {result}")
                opinions.append(CouncilOpinion(
                    model=getattr(client, 'model_name', 'unknown'),
                    display_name=getattr(client, 'display_name', 'Unknown Model'),
                    content="",
                    error=f"예상치 못한 오류: {str(result)}",
                ))
            else:
                opinions.append(result)

        return opinions

    async def _conduct_peer_reviews(
        self,
        context: LegalContext,
        opinions: list[CouncilOpinion],
        report_progress: Any,
        domain: LegalDomain | None = None,
    ) -> list[PeerReview]:
        """
        Stage 2: Conduct blind peer reviews (v4.3.1).

        모든 위원이 자신을 제외한 다른 위원들의 의견을 블라인드 평가합니다.
        - 4명 위원 × 3개 의견 평가 = 총 12개 피어리뷰
        - 병렬 처리로 속도 최적화
        """
        council_members = await self._get_council_members(domain)
        reviews = []

        # 총 리뷰 수: 각 위원이 자신 제외 다른 모든 위원 평가
        # n명 위원 → n * (n-1) 개 리뷰
        total_reviews = len(opinions) * (len(opinions) - 1)
        completed_reviews = 0

        # 리뷰 태스크 생성: 각 위원이 다른 모든 위원을 평가
        review_tasks = []
        for reviewer_idx, reviewer_client in enumerate(council_members):
            reviewer_opinion = opinions[reviewer_idx] if reviewer_idx < len(opinions) else None
            if reviewer_opinion is None or reviewer_opinion.error:
                continue  # 의견 생성 실패한 위원은 리뷰어에서 제외

            for target_idx, target_opinion in enumerate(opinions):
                # 자기 자신은 평가하지 않음
                if reviewer_idx == target_idx:
                    continue
                if target_opinion.error:
                    continue  # 오류가 있는 의견은 평가 대상에서 제외

                review_tasks.append({
                    "reviewer_client": reviewer_client,
                    "reviewer_name": reviewer_opinion.display_name,
                    "target_opinion": target_opinion,
                })

        # 병렬로 피어리뷰 수행
        async def perform_review(task: dict) -> PeerReview:
            reviewer_client = task["reviewer_client"]
            reviewer_name = task["reviewer_name"]
            target_opinion = task["target_opinion"]

            try:
                # 클라이언트 초기화 (아직 안된 경우)
                if reviewer_client._client is None:
                    await reviewer_client._initialize_client()

                # 블라인드 리뷰 생성 (의견 작성자 숨김)
                response = await reviewer_client.generate_peer_review(
                    original_query=context.query,
                    opinion_to_review=target_opinion.content,
                    category=context.category,
                    domain=domain,
                )

                scores = self._parse_review_scores(response.content)

                return PeerReview(
                    reviewer_model=reviewer_name,
                    reviewed_model=target_opinion.display_name,
                    content=response.content,
                    scores=scores,
                    latency_ms=response.latency_ms,
                )

            except Exception as e:
                logger.error(f"Peer review error: {reviewer_name} → {target_opinion.display_name}: {e}")
                return PeerReview(
                    reviewer_model=reviewer_name,
                    reviewed_model=target_opinion.display_name,
                    content=f"평가 중 오류 발생: {str(e)}",
                    scores={},
                )

        # 병렬 실행 (최대 동시성 제한)
        semaphore = asyncio.Semaphore(self.max_concurrent * 2)  # 리뷰는 더 많은 동시성 허용

        async def limited_review(task: dict) -> PeerReview:
            nonlocal completed_reviews
            async with semaphore:
                result = await perform_review(task)
                completed_reviews += 1
                await report_progress(
                    "stage2",
                    completed_reviews / max(total_reviews, 1),
                    f"{completed_reviews}/{total_reviews} 교차 평가 완료",
                )
                return result

        # 모든 리뷰 병렬 실행
        review_results = await asyncio.gather(
            *[limited_review(task) for task in review_tasks],
            return_exceptions=True
        )

        # 결과 수집 (예외 처리)
        for result in review_results:
            if isinstance(result, Exception):
                logger.error(f"Peer review task failed: {result}")
            elif isinstance(result, PeerReview):
                reviews.append(result)

        return reviews

    def _parse_review_scores(self, review_content: str) -> dict:
        """
        Parse numeric scores from review content.

        v4.1 평가 기준 (각 10점 만점):
        - 정확성: 법적 정확성, 인용 정확성
        - 완전성: 쟁점 포괄, 예외 고려
        - 메모리 통합: 맥락 활용도
        - 유용성: 실무 적용 가능성
        - 명확성: 구조, 이해 용이성
        """
        scores = {}
        # v4.1 5가지 평가 기준
        score_patterns = {
            "정확성": r"정확성[:\s]*(\d+)",
            "완전성": r"완전성[:\s]*(\d+)",
            "메모리_통합": r"메모리\s*통합[:\s]*(\d+)",
            "유용성": r"유용성[:\s]*(\d+)",
            "명확성": r"명확성[:\s]*(\d+)",
            "종합": r"종합[:\s]*(\d+)",
        }

        import re
        for name, pattern in score_patterns.items():
            match = re.search(pattern, review_content)
            if match:
                try:
                    scores[name] = int(match.group(1))
                except ValueError:
                    pass

        # 종합 점수 계산 (5가지 기준의 평균)
        if len(scores) >= 5 and "종합" not in scores:
            base_scores = [v for k, v in scores.items() if k != "종합"]
            scores["종합"] = sum(base_scores) // len(base_scores)

        return scores

    async def _synthesize(
        self,
        context: LegalContext,
        opinions: list[CouncilOpinion],
        reviews: list[PeerReview],
        report_progress: Any,
        domain: LegalDomain | None = None,
    ) -> ChairmanSynthesis | None:
        """
        Stage 3: Chairman synthesis (v4.3.1).

        v4.3.1 6단계 합성 프로토콜:
        1. 인용 검증 및 권위 평가
        2. 합의 및 분기 지점 분석
        3. 충돌 해결
        4. 일관성 검증 (장기 기억 참조)
        5. 공백 보완
        6. 최종 합성 (LLM별 최적화된 의장 프롬프트 적용)
        """
        chairman = await self._get_chairman(domain)

        try:
            # Prepare data for synthesis
            opinion_data = [
                {
                    "model": o.display_name,
                    "content": o.content,
                }
                for o in opinions
            ]

            review_data = [
                {
                    "reviewer": r.reviewer_model,
                    "reviewed": r.reviewed_model,
                    "content": r.content,
                    "scores": r.scores,
                }
                for r in reviews
            ]

            rag_context = context.to_context_string()

            # v4.3.1: 메모리 시스템 + 분야 전달
            response = await chairman.generate_chairman_synthesis(
                query=context.query,
                category=context.category,
                opinions=opinion_data,
                reviews=review_data,
                rag_context=rag_context,
                session_memory=self.session_memory,
                short_term_memory=self.short_term_memory,
                long_term_memory=self.long_term_memory,
                domain=domain,  # v4.3.1
            )

            await report_progress("stage3", 0.8, "의장 종합 의견 작성 완료")

            return ChairmanSynthesis(
                content=response.content,
                model=chairman.model_name,
                thinking_content=response.metadata.get("thinking"),
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.error(f"Chairman synthesis error: {e}")
            return None

    async def consult_stream(
        self,
        context: LegalContext,
        domain: LegalDomain | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream council consultation with progress events (v4.3.1).

        Args:
            context: Legal consultation context
            domain: 법률 분야 (None이면 키워드 기반 자동 감지)

        Yields:
            Progress events with type, stage, progress, and data
        """
        # v4.3.1: 분야 자동 감지 (비용 0)
        effective_domain = domain or self.domain
        if effective_domain is None:
            effective_domain = self._detect_domain_from_query(context.query)
            logger.info(f"v4.3.1 스트리밍 - 자동 감지된 법률 분야: {effective_domain.value}")

        async def progress_callback(stage: str, progress: float, message: str):
            # This will be called but we handle streaming separately
            pass

        # Stage 1: Stream opinion collection progress
        yield {
            "type": "stage_start",
            "stage": "stage1",
            "message": f"위원회 의견 수집 시작... (분야: {effective_domain.value})",
            "domain": effective_domain.value,  # v4.3.1
        }

        council_members = await self._get_council_members(effective_domain)
        opinions = []

        for i, client in enumerate(council_members):
            yield {
                "type": "opinion_start",
                "stage": "stage1",
                "model": client.display_name,
            }

            try:
                response = await client.generate(context)

                if not response.is_error:
                    opinions.append(CouncilOpinion(
                        model=client.model_name,
                        display_name=client.display_name,
                        content=response.content,
                        latency_ms=response.latency_ms,
                        tokens_used=response.tokens_used,
                    ))

                    yield {
                        "type": "opinion_complete",
                        "stage": "stage1",
                        "model": client.display_name,
                        "content": response.content[:500] + "..." if len(response.content) > 500 else response.content,
                        "progress": (i + 1) / len(council_members),
                    }
                else:
                    yield {
                        "type": "opinion_error",
                        "stage": "stage1",
                        "model": client.display_name,
                        "error": response.error,
                    }

            except Exception as e:
                yield {
                    "type": "opinion_error",
                    "stage": "stage1",
                    "model": client.display_name,
                    "error": str(e),
                }

        if len(opinions) < 2:
            yield {
                "type": "error",
                "message": "최소 2개 이상의 의견이 필요합니다.",
            }
            return

        # Stage 2: Peer Review (v4.3.1) - 모든 위원이 서로 평가
        if self.enable_peer_review:
            yield {
                "type": "stage_start",
                "stage": "stage2",
                "message": "블라인드 교차 평가 시작 (모든 위원이 서로 평가)...",
            }

            reviews = []
            # 총 리뷰 수: n명 위원 → n * (n-1) 개 리뷰
            total_reviews = len(opinions) * (len(opinions) - 1)
            completed_reviews = 0

            # 각 위원이 다른 모든 위원의 의견을 평가
            for reviewer_idx, reviewer_client in enumerate(council_members):
                reviewer_opinion = opinions[reviewer_idx] if reviewer_idx < len(opinions) else None
                if reviewer_opinion is None:
                    continue

                for target_idx, target_opinion in enumerate(opinions):
                    # 자기 자신은 평가하지 않음
                    if reviewer_idx == target_idx:
                        continue

                    try:
                        # 클라이언트 초기화 확인
                        if reviewer_client._client is None:
                            await reviewer_client._initialize_client()

                        response = await reviewer_client.generate_peer_review(
                            original_query=context.query,
                            opinion_to_review=target_opinion.content,
                            category=context.category,
                            domain=effective_domain,
                        )

                        reviews.append(PeerReview(
                            reviewer_model=reviewer_opinion.display_name,
                            reviewed_model=target_opinion.display_name,
                            content=response.content,
                            scores=self._parse_review_scores(response.content),
                            latency_ms=response.latency_ms,
                        ))

                        completed_reviews += 1
                        yield {
                            "type": "review_complete",
                            "stage": "stage2",
                            "reviewer_model": reviewer_opinion.display_name,
                            "reviewed_model": target_opinion.display_name,
                            "progress": completed_reviews / max(total_reviews, 1),
                        }

                    except Exception as e:
                        yield {
                            "type": "review_error",
                            "stage": "stage2",
                            "reviewer_model": reviewer_opinion.display_name,
                            "reviewed_model": target_opinion.display_name,
                            "error": str(e),
                        }
                        reviews.append(PeerReview(
                            reviewer_model=reviewer_opinion.display_name,
                            reviewed_model=target_opinion.display_name,
                            content="",
                            scores={},
                        ))
                        completed_reviews += 1
        else:
            reviews = []

        # Stage 3: Chairman Synthesis with streaming (v4.3.1)
        yield {
            "type": "stage_start",
            "stage": "stage3",
            "message": "의장 종합 의견 작성 중...",
        }

        chairman = await self._get_chairman(effective_domain)

        try:
            opinion_data = [{"model": o.display_name, "content": o.content} for o in opinions]
            review_data = [
                {"reviewer": r.reviewer_model, "reviewed": r.reviewed_model, "content": r.content}
                for r in reviews
            ]

            # v4.3.1: 메모리 시스템 + 분야 전달
            response = await chairman.generate_chairman_synthesis(
                query=context.query,
                category=context.category,
                opinions=opinion_data,
                reviews=review_data,
                rag_context=context.to_context_string(),
                session_memory=self.session_memory,
                short_term_memory=self.short_term_memory,
                long_term_memory=self.long_term_memory,
                domain=effective_domain,  # v4.3.1
            )

            yield {
                "type": "synthesis_complete",
                "stage": "stage3",
                "content": response.content,
                "thinking": response.metadata.get("thinking"),
            }

        except Exception as e:
            yield {
                "type": "synthesis_error",
                "stage": "stage3",
                "error": str(e),
            }

        yield {
            "type": "complete",
            "message": "위원회 자문 완료",
        }

    async def close(self):
        """Close all client connections."""
        if self._council_members:
            for client in self._council_members:
                await client.close()
            self._council_members = None

        # v4.3.2: _peer_reviewer 제거 - council_members가 피어리뷰 수행

        if self._chairman:
            await self._chairman.close()
            self._chairman = None
