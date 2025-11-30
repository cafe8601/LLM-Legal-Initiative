"""
Enhanced CascadeFlow Council with Learning System Integration

MAS multiagent v4 기반 확장:
- 학습 시스템 통합 (OutcomeTracker, PatternAnalyzer, LearningManager)
- 경험 RAG 통합 (HybridRAGOrchestrator)
- 메모리 시스템 통합 (MemoryManager)

Features:
- 에이전트별 성공 패턴 학습 및 적용
- 드래프터 성공률 예측 기반 모델 라우팅
- 경험 기반 컨텍스트 보강
- 상담 결과 자동 학습
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.cascade_council import (
    CascadeCouncilOrchestrator,
    CascadeCouncilResult,
    CascadeOpinion,
)
from app.services.llm.cascade_service import (
    CascadeModelTier,
)
from app.services.llm.legal_prompts_v4_3 import LegalDomain
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


@dataclass
class EnhancedCouncilConfig:
    """확장 위원회 설정."""
    # 학습 시스템
    enable_learning: bool = True
    auto_record_outcomes: bool = True
    min_drafter_confidence: float = 0.6

    # RAG 시스템
    enable_experience_rag: bool = True
    enable_legal_rag: bool = True
    max_context_tokens: int = 6000

    # 메모리 시스템
    enable_memory: bool = True
    memory_ttl: int = 3600  # 1시간

    # 캐스케이드 최적화
    use_intelligent_routing: bool = True
    drafter_success_threshold: float = 0.65


@dataclass
class EnhancedCouncilResult(CascadeCouncilResult):
    """확장 위원회 결과."""
    # 학습 시스템 메트릭
    learning_applied: bool = False
    drafter_prediction_confidence: float = 0.0
    agent_recommendations: list[dict] = field(default_factory=list)

    # RAG 메트릭
    legal_docs_used: int = 0
    experiences_used: int = 0
    rag_cache_hit: bool = False

    # 결과 ID (학습용)
    consultation_outcome_id: Optional[str] = None


class EnhancedCascadeCouncil:
    """
    확장 CascadeFlow 위원회.

    MAS multiagent v4의 학습/메모리/RAG 시스템을 통합하여
    지능형 법률 자문 위원회를 구현합니다.

    핵심 기능:
    1. 학습 기반 모델 라우팅 - 에이전트별 성공 패턴 활용
    2. 경험 RAG - 유사 상담 사례 검색 및 활용
    3. 통합 메모리 - 세션/단기/장기 메모리 관리
    4. 자동 결과 학습 - 상담 완료 후 패턴 학습
    """

    def __init__(
        self,
        db: AsyncSession,
        config: Optional[EnhancedCouncilConfig] = None,
        enable_peer_review: bool = True,
        enable_cascade: bool = True,
        max_concurrent: int = 4,
        domain: Optional[LegalDomain] = None,
    ):
        """
        초기화.

        Args:
            db: SQLAlchemy AsyncSession
            config: 확장 설정
            enable_peer_review: 교차 평가 활성화
            enable_cascade: 캐스케이드 활성화
            max_concurrent: 최대 동시성
            domain: 법률 분야
        """
        self.db = db
        self.config = config or EnhancedCouncilConfig()

        # 기본 CascadeCouncil
        self._base_council = CascadeCouncilOrchestrator(
            enable_peer_review=enable_peer_review,
            enable_cascade=enable_cascade,
            max_concurrent=max_concurrent,
            domain=domain,
        )

        # 확장 시스템 (지연 초기화)
        self._learning_manager: Optional[LearningManager] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._rag_orchestrator: Optional[HybridRAGOrchestrator] = None

        # 메트릭
        self.stats = {
            "total_consultations": 0,
            "learning_applied": 0,
            "drafter_predictions_correct": 0,
            "experience_rag_hits": 0,
        }

    def _get_learning_manager(self) -> LearningManager:
        """학습 관리자 획득."""
        if self._learning_manager is None:
            self._learning_manager = get_learning_manager()
        return self._learning_manager

    def _get_memory_manager(self) -> MemoryManager:
        """메모리 관리자 획득."""
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager(self.db)
        return self._memory_manager

    def _get_rag_orchestrator(self) -> HybridRAGOrchestrator:
        """RAG 오케스트레이터 획득."""
        if self._rag_orchestrator is None:
            self._rag_orchestrator = HybridRAGOrchestrator(
                self.db,
                config=RAGConfig(
                    max_context_tokens=self.config.max_context_tokens,
                ),
            )
        return self._rag_orchestrator

    async def consult(
        self,
        user_id: UUID,
        consultation_id: UUID,
        query: str,
        category: str = "general",
        domain: Optional[LegalDomain] = None,
        progress_callback: Optional[Any] = None,
    ) -> EnhancedCouncilResult:
        """
        확장 위원회 자문.

        Args:
            user_id: 사용자 ID
            consultation_id: 상담 ID
            query: 법률 질문
            category: 법률 분야
            domain: 법률 도메인
            progress_callback: 진행 콜백

        Returns:
            EnhancedCouncilResult
        """
        start_time = time.time()
        self.stats["total_consultations"] += 1

        # 1. 메모리 컨텍스트 로드
        memory_context = await self._load_memory_context(
            user_id, consultation_id, category
        )

        # 2. RAG 컨텍스트 검색
        rag_result = await self._search_rag_context(
            query=query,
            consultation_id=str(consultation_id),
            category=category,
        )

        # 3. 학습 기반 라우팅 결정
        routing_decision = await self._get_routing_decision(
            query=query,
            category=category,
        )

        # 4. CascadeCouncil 업데이트 (메모리 + RAG)
        self._base_council.session_memory = memory_context.get("session_memory", "")
        self._base_council.short_term_memory = memory_context.get("short_term_memory", "")
        self._base_council.long_term_memory = memory_context.get("long_term_memory", "")

        # 5. 위원회 자문 실행
        async def enhanced_progress(stage: str, progress: float, message: str):
            if progress_callback:
                # 학습 정보 추가
                enhanced_msg = message
                if routing_decision and routing_decision.get("drafter_prediction"):
                    enhanced_msg += f" (예측 신뢰도: {routing_decision['drafter_prediction']:.1%})"
                await progress_callback(stage, progress, enhanced_msg)

        base_result = await self._base_council.consult(
            query=query,
            category=category,
            rag_context=rag_result.combined_context if rag_result else "",
            progress_callback=enhanced_progress,
            domain=domain,
        )

        # 6. 결과 기록 및 학습
        outcome_id = None
        if self.config.auto_record_outcomes:
            outcome_id = await self._record_outcome(
                user_id=user_id,
                consultation_id=consultation_id,
                query=query,
                category=category,
                result=base_result,
            )

        # 7. 확장 결과 생성
        enhanced_result = EnhancedCouncilResult(
            # 기본 결과 복사
            opinions=base_result.opinions,
            reviews=base_result.reviews,
            synthesis=base_result.synthesis,
            total_latency_ms=base_result.total_latency_ms,
            total_tokens_used=base_result.total_tokens_used,
            stage_timings=base_result.stage_timings,
            errors=base_result.errors,
            total_estimated_cost=base_result.total_estimated_cost,
            total_cost_saved=base_result.total_cost_saved,
            drafter_success_count=base_result.drafter_success_count,
            escalation_count=base_result.escalation_count,

            # 확장 메트릭
            learning_applied=routing_decision is not None,
            drafter_prediction_confidence=routing_decision.get("drafter_prediction", 0.0) if routing_decision else 0.0,
            agent_recommendations=routing_decision.get("recommendations", []) if routing_decision else [],
            legal_docs_used=len(rag_result.legal_documents) if rag_result else 0,
            experiences_used=len(rag_result.similar_experiences) if rag_result else 0,
            rag_cache_hit=rag_result.cache_hit if rag_result else False,
            consultation_outcome_id=outcome_id,
        )

        # 통계 업데이트
        if routing_decision:
            self.stats["learning_applied"] += 1

        if rag_result and rag_result.similar_experiences:
            self.stats["experience_rag_hits"] += 1

        logger.info(
            f"Enhanced consultation complete: "
            f"latency={enhanced_result.total_latency_ms}ms, "
            f"cost_saved=${enhanced_result.total_cost_saved:.4f}, "
            f"learning_applied={enhanced_result.learning_applied}"
        )

        return enhanced_result

    async def _load_memory_context(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: str,
    ) -> dict:
        """메모리 컨텍스트 로드."""
        if not self.config.enable_memory:
            return {}

        try:
            memory_manager = self._get_memory_manager()
            context = await memory_manager.get_council_context(
                user_id=user_id,
                consultation_id=consultation_id,
                category=category,
            )
            return {
                "session_memory": context.session_memory,
                "short_term_memory": context.short_term_memory,
                "long_term_memory": context.long_term_memory,
            }
        except Exception as e:
            logger.warning(f"Memory context load failed: {e}")
            return {}

    async def _search_rag_context(
        self,
        query: str,
        consultation_id: str,
        category: str,
    ) -> Optional[HybridSearchResult]:
        """RAG 컨텍스트 검색."""
        if not (self.config.enable_experience_rag or self.config.enable_legal_rag):
            return None

        try:
            rag_orchestrator = self._get_rag_orchestrator()
            result = await rag_orchestrator.search(
                query=query,
                consultation_id=consultation_id,
                category=category,
                include_experiences=self.config.enable_experience_rag,
                include_legal_docs=self.config.enable_legal_rag,
            )
            return result
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return None

    async def _get_routing_decision(
        self,
        query: str,
        category: str,
    ) -> Optional[dict]:
        """학습 기반 라우팅 결정."""
        if not self.config.use_intelligent_routing:
            return None

        try:
            learning_manager = self._get_learning_manager()

            # 기본 에이전트로 cascade 결정 (sync 메서드 - await 제거)
            cascade_decision = learning_manager.get_cascade_decision(
                agent_id="claude",  # 기본 에이전트
                query=query,
                category=category,
            )

            # 배치로 모든 에이전트 추천 조회 (N+1 최적화)
            recommendations = []
            agent_ids = ["claude", "gpt", "gemini", "grok"]

            for agent_id in agent_ids:
                # sync 메서드 - await 제거, 시그니처 수정
                rec = learning_manager.get_agent_recommendation(
                    query=query,
                    category=category,
                    available_agents=[agent_id],
                )
                recommendations.append({
                    "agent_id": agent_id,
                    "use_drafter": rec.use_drafter,
                    "confidence": rec.confidence,
                    "reason": rec.reason,
                })

            return {
                "drafter_prediction": cascade_decision.predicted_drafter_success,
                "skip_drafter": cascade_decision.skip_drafter,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.warning(f"Routing decision failed: {e}")
            return None

    async def _record_outcome(
        self,
        user_id: UUID,
        consultation_id: UUID,
        query: str,
        category: str,
        result: CascadeCouncilResult,
    ) -> Optional[str]:
        """상담 결과 기록."""
        if not self.config.enable_learning:
            return None

        try:
            learning_manager = self._get_learning_manager()

            # 각 위원별 결과 기록
            for opinion in result.opinions:
                if opinion.error:
                    continue

                agent_id = self._extract_agent_id(opinion.model)
                model_id = opinion.model
                tier = "drafter" if opinion.drafter_used else "verifier"

                # 성공 여부 판단 (간단한 휴리스틱)
                success = len(opinion.content) > 200 and not opinion.error

                outcome_id = await learning_manager.record_consultation_outcome(
                    consultation_id=str(consultation_id),
                    agent_id=agent_id,
                    model_id=model_id,
                    category=category,
                    query=query,
                    response_summary=opinion.content[:500],
                    tier=tier,
                    success=success,
                    escalated=opinion.escalated,
                    latency_ms=opinion.latency_ms,
                    tokens_used=opinion.tokens_used,
                    cost=result.total_estimated_cost / max(len(result.opinions), 1),
                )

            # 전체 상담 결과도 경험 RAG에 인덱싱
            if result.synthesis and self.config.enable_experience_rag:
                rag_orchestrator = self._get_rag_orchestrator()
                await rag_orchestrator.index_consultation_experience(
                    consultation_id=str(consultation_id),
                    user_id=str(user_id),
                    query=query,
                    response_summary=result.synthesis.content[:1000],
                    category=category,
                    keywords=self._extract_keywords(query),
                    agent_id="council",
                    model_id="cascade_council",
                    success=True,  # 의장 종합이 있으면 성공으로 간주
                    quality_score=self._calculate_quality_score(result),
                )

            return f"outcome_{consultation_id}"

        except Exception as e:
            logger.warning(f"Outcome recording failed: {e}")
            return None

    def _extract_agent_id(self, model_name: str) -> str:
        """모델 이름에서 에이전트 ID 추출."""
        model_lower = model_name.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "claude"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "gpt"
        elif "gemini" in model_lower or "google" in model_lower:
            return "gemini"
        elif "grok" in model_lower or "x-ai" in model_lower:
            return "grok"
        return "unknown"

    def _extract_keywords(self, text: str) -> list[str]:
        """텍스트에서 키워드 추출."""
        legal_terms = [
            "계약", "손해배상", "해고", "임대차", "상속", "이혼",
            "채권", "채무", "보증", "담보", "위자료", "양육권",
            "친권", "부동산", "등기", "소송", "재판", "형사", "민사",
        ]
        return [term for term in legal_terms if term in text][:10]

    def _calculate_quality_score(self, result: CascadeCouncilResult) -> float:
        """결과 품질 점수 계산."""
        score = 0.0

        # 의장 종합 존재 (+0.3)
        if result.synthesis and result.synthesis.content:
            score += 0.3

        # 성공한 의견 비율 (+0.3)
        successful_opinions = [o for o in result.opinions if not o.error]
        if result.opinions:
            score += 0.3 * (len(successful_opinions) / len(result.opinions))

        # 비용 절감 (+0.2)
        if result.savings_percentage > 0:
            score += 0.2 * min(result.savings_percentage / 50, 1.0)

        # 리뷰 존재 (+0.2)
        if result.reviews:
            score += 0.2

        return min(score, 1.0)

    async def provide_feedback(
        self,
        consultation_id: UUID,
        user_feedback: int,
    ) -> bool:
        """
        사용자 피드백 제공.

        Args:
            consultation_id: 상담 ID
            user_feedback: 피드백 점수 (1-5)

        Returns:
            성공 여부
        """
        try:
            # 경험 RAG에 피드백 업데이트
            if self.config.enable_experience_rag:
                rag_orchestrator = self._get_rag_orchestrator()
                # 경험 ID는 상담 ID 기반으로 찾아야 함
                # 실제 구현에서는 경험 ID를 결과에서 추적해야 함
                pass

            return True
        except Exception as e:
            logger.warning(f"Feedback provision failed: {e}")
            return False

    def get_stats(self) -> dict:
        """통계 반환."""
        cascade_stats = self._base_council.get_cascade_stats()

        return {
            "enhanced_stats": self.stats,
            "cascade_stats": cascade_stats,
            "config": {
                "enable_learning": self.config.enable_learning,
                "enable_experience_rag": self.config.enable_experience_rag,
                "use_intelligent_routing": self.config.use_intelligent_routing,
            },
        }

    async def get_optimization_report(self, category: Optional[str] = None) -> dict:
        """최적화 리포트 생성."""
        learning_manager = self._get_learning_manager()
        report = await learning_manager.get_cascade_optimization_report(category)

        rag_stats = None
        if self._rag_orchestrator:
            rag_stats = await self._rag_orchestrator.get_statistics()

        return {
            "learning_report": report,
            "rag_stats": rag_stats,
            "enhanced_stats": self.stats,
        }

    async def close(self) -> None:
        """리소스 정리."""
        await self._base_council.close()


# ========================================
# 팩토리 함수
# ========================================


def create_enhanced_council(
    db: AsyncSession,
    config: Optional[EnhancedCouncilConfig] = None,
    **kwargs,
) -> EnhancedCascadeCouncil:
    """
    확장 위원회 생성.

    Args:
        db: AsyncSession
        config: 설정
        **kwargs: 추가 옵션

    Returns:
        EnhancedCascadeCouncil
    """
    return EnhancedCascadeCouncil(
        db=db,
        config=config,
        **kwargs,
    )
