"""
Learning Manager

학습 시스템의 중앙 관리자.
MAS multiagent v4의 LearningManager를 법률 자문 시스템에 맞게 확장.

Features:
- OutcomeTracker + PatternAnalyzer 통합 관리
- CascadeFlow와 연동하여 지능형 라우팅
- 실시간 학습 및 추천
- 위원회 전체 성능 최적화
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.services.learning.outcome_tracker import (
    OutcomeTracker,
    ConsultationOutcome,
    ConsultationTier,
    OutcomeStatus,
)
from app.services.learning.pattern_analyzer import PatternAnalyzer, AgentPerformance

logger = logging.getLogger(__name__)


@dataclass
class AgentRecommendation:
    """에이전트 추천 결과."""
    recommended_agent: str
    confidence: float
    reason: str
    use_drafter: bool  # CascadeFlow 드래프터 사용 추천
    drafter_success_probability: float
    similar_cases: list[dict]


@dataclass
class CascadeDecision:
    """CascadeFlow 결정."""
    use_cascade: bool
    skip_drafter: bool  # True면 검증자 직접 사용
    predicted_drafter_success: float
    decision_reason: str


class LearningManager:
    """
    학습 관리자.

    위원회의 성능을 추적하고 최적의 위원 선택 및 CascadeFlow 전략을 추천합니다.
    """

    # 기본 설정
    DEFAULT_DRAFTER_SUCCESS_THRESHOLD = 0.6  # 드래프터 사용 최소 성공률
    MIN_DATA_FOR_PREDICTION = 5  # 예측을 위한 최소 데이터

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        drafter_threshold: float = DEFAULT_DRAFTER_SUCCESS_THRESHOLD,
    ):
        """
        초기화.

        Args:
            storage_path: 학습 데이터 저장 경로
            drafter_threshold: 드래프터 사용 결정 임계값
        """
        if storage_path:
            storage_path = Path(storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)

        self.tracker = OutcomeTracker(storage_path=storage_path)
        self.analyzer = PatternAnalyzer(self.tracker)
        self.drafter_threshold = drafter_threshold

        logger.info(f"LearningManager initialized (storage: {storage_path})")

    # ========================================
    # 결과 기록
    # ========================================

    def record_consultation_outcome(
        self,
        agent_id: str,
        model_id: str,
        tier: ConsultationTier,
        consultation_id: str,
        user_id: str,
        category: str,
        query: str,
        success: bool,
        response_quality: float,
        response_time_ms: int,
        tokens_used: int,
        cost_usd: float,
        escalated: bool = False,
        escalation_reason: Optional[str] = None,
        error_reason: Optional[str] = None,
    ) -> ConsultationOutcome:
        """
        상담 결과 기록 (통합 인터페이스).

        Args:
            success: 성공 여부
            escalated: 에스컬레이션 여부 (드래프터 실패 시)

        Returns:
            기록된 ConsultationOutcome
        """
        # 쿼리 복잡도 추론
        query_complexity = self.analyzer._infer_complexity(query)

        metadata = {
            "query": query[:500],  # 쿼리 저장 (분석용)
        }

        if escalated:
            return self.tracker.record_escalation(
                agent_id=agent_id,
                model_id=model_id,
                consultation_id=consultation_id,
                user_id=user_id,
                category=category,
                query_complexity=query_complexity,
                escalation_reason=escalation_reason or "quality_threshold",
                drafter_quality=response_quality,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                metadata=metadata,
            )
        elif success:
            return self.tracker.record_success(
                agent_id=agent_id,
                model_id=model_id,
                tier=tier,
                consultation_id=consultation_id,
                user_id=user_id,
                category=category,
                query_complexity=query_complexity,
                response_quality=response_quality,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                metadata=metadata,
            )
        else:
            return self.tracker.record_failure(
                agent_id=agent_id,
                model_id=model_id,
                tier=tier,
                consultation_id=consultation_id,
                user_id=user_id,
                category=category,
                query_complexity=query_complexity,
                error_reason=error_reason or "unknown",
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                metadata=metadata,
            )

    def record_user_feedback(
        self,
        outcome_id: str,
        feedback_score: int,
    ) -> bool:
        """사용자 피드백 기록."""
        return self.tracker.update_user_feedback(outcome_id, feedback_score)

    # ========================================
    # 추천
    # ========================================

    def get_agent_recommendation(
        self,
        query: str,
        category: Optional[str] = None,
        available_agents: Optional[list[str]] = None,
    ) -> AgentRecommendation:
        """
        쿼리에 최적인 에이전트 추천.

        Args:
            query: 사용자 쿼리
            category: 법률 분야 (None이면 자동 추론)
            available_agents: 사용 가능한 에이전트 목록

        Returns:
            AgentRecommendation 객체
        """
        # 분야 추론
        if not category:
            category = self.analyzer._infer_category(query) or "general"

        # 복잡도 추론
        complexity = self.analyzer._infer_complexity(query)

        # 최적 에이전트 찾기
        best_agent, confidence, reason = self.analyzer.get_best_agent_for_query(query)

        # 가용 에이전트 필터
        if available_agents and best_agent not in available_agents:
            # 가용 에이전트 중 최적 선택
            best_perf = None
            for agent in available_agents:
                perf = self.analyzer.analyze_agent_performance(agent)
                cat_perf = perf.performance_by_category.get(category, {})
                agent_rate = cat_perf.get("success_rate", perf.success_rate)

                if best_perf is None or agent_rate > best_perf[1]:
                    best_perf = (agent, agent_rate)

            if best_perf:
                best_agent, confidence = best_perf
                reason = f"가용 에이전트 중 최고 성능"

        # 드래프터 성공 확률
        drafter_prob = self.analyzer.predict_drafter_success(
            best_agent, category, complexity
        )

        # 유사 케이스 찾기
        similar = self.analyzer._find_similar_queries(query, limit=3)
        similar_cases = [
            {
                "category": o.category,
                "status": o.status,
                "quality": o.response_quality,
                "agent": o.agent_id,
            }
            for o in similar
        ]

        return AgentRecommendation(
            recommended_agent=best_agent,
            confidence=confidence,
            reason=reason,
            use_drafter=drafter_prob >= self.drafter_threshold,
            drafter_success_probability=drafter_prob,
            similar_cases=similar_cases,
        )

    def get_cascade_decision(
        self,
        agent_id: str,
        query: str,
        category: Optional[str] = None,
    ) -> CascadeDecision:
        """
        CascadeFlow 사용 결정.

        드래프터를 사용할지, 검증자를 직접 사용할지 결정합니다.

        Args:
            agent_id: 에이전트 ID
            query: 사용자 쿼리
            category: 법률 분야

        Returns:
            CascadeDecision 객체
        """
        if not category:
            category = self.analyzer._infer_category(query) or "general"

        complexity = self.analyzer._infer_complexity(query)

        # 드래프터 성공 확률 예측
        drafter_prob = self.analyzer.predict_drafter_success(
            agent_id, category, complexity
        )

        # 복잡한 쿼리는 검증자 직접 사용 권장
        if complexity == "complex":
            return CascadeDecision(
                use_cascade=True,
                skip_drafter=True,
                predicted_drafter_success=drafter_prob,
                decision_reason=f"복잡한 쿼리 - 검증자 직접 사용 (예측 성공률: {drafter_prob*100:.1f}%)",
            )

        # 드래프터 성공률이 높으면 드래프터 사용
        if drafter_prob >= self.drafter_threshold:
            return CascadeDecision(
                use_cascade=True,
                skip_drafter=False,
                predicted_drafter_success=drafter_prob,
                decision_reason=f"드래프터 사용 권장 (예측 성공률: {drafter_prob*100:.1f}%)",
            )

        # 드래프터 성공률이 낮으면 검증자 직접 사용
        return CascadeDecision(
            use_cascade=True,
            skip_drafter=True,
            predicted_drafter_success=drafter_prob,
            decision_reason=f"드래프터 성공률 낮음 - 검증자 직접 사용 (예측: {drafter_prob*100:.1f}%)",
        )

    def get_optimal_routing(
        self,
        query: str,
        available_agents: list[str],
        category: Optional[str] = None,
    ) -> dict[str, CascadeDecision]:
        """
        모든 에이전트에 대한 최적 라우팅 결정.

        Args:
            query: 사용자 쿼리
            available_agents: 사용 가능한 에이전트 목록
            category: 법률 분야

        Returns:
            {agent_id: CascadeDecision} 딕셔너리
        """
        return {
            agent: self.get_cascade_decision(agent, query, category)
            for agent in available_agents
        }

    # ========================================
    # 통계 및 분석
    # ========================================

    def get_learning_stats(self) -> dict:
        """학습 통계."""
        tracker_stats = self.tracker.get_statistics()
        analysis = self.analyzer.get_analysis_summary()

        return {
            "outcomes": tracker_stats,
            "analysis": analysis,
            "config": {
                "drafter_threshold": self.drafter_threshold,
                "min_data_for_prediction": self.MIN_DATA_FOR_PREDICTION,
            },
        }

    def get_agent_performance(self, agent_id: str) -> AgentPerformance:
        """특정 에이전트 성능."""
        return self.analyzer.analyze_agent_performance(agent_id)

    def get_all_agent_performances(self) -> dict[str, AgentPerformance]:
        """모든 에이전트 성능."""
        return self.analyzer.analyze_all_agents()

    def get_category_insights(self) -> list:
        """분야별 인사이트."""
        return self.analyzer.get_category_insights()

    def get_recommendations(self) -> list[str]:
        """개선 추천 사항."""
        analysis = self.analyzer.get_analysis_summary()
        return analysis.get("recommendations", [])

    # ========================================
    # CascadeFlow 최적화
    # ========================================

    def get_cascade_optimization_report(self) -> dict:
        """
        CascadeFlow 최적화 보고서.

        드래프터 성공률, 비용 절감, 개선점 등을 분석합니다.
        """
        outcomes = self.tracker.get_recent_outcomes(limit=500)

        drafter_outcomes = [o for o in outcomes if o.tier == ConsultationTier.DRAFTER.value]
        verifier_outcomes = [o for o in outcomes if o.tier == ConsultationTier.VERIFIER.value]

        if not drafter_outcomes:
            return {"message": "드래프터 데이터 부족"}

        # 드래프터 성공률
        drafter_successes = sum(1 for o in drafter_outcomes if o.status == OutcomeStatus.SUCCESS.value)
        drafter_success_rate = drafter_successes / len(drafter_outcomes)

        # 에스컬레이션률
        escalations = sum(1 for o in drafter_outcomes if o.status == OutcomeStatus.ESCALATED.value)
        escalation_rate = escalations / len(drafter_outcomes)

        # 비용 분석
        drafter_cost = sum(o.cost_usd for o in drafter_outcomes)
        verifier_cost = sum(o.cost_usd for o in verifier_outcomes)
        total_cost = drafter_cost + verifier_cost

        # 에이전트별 드래프터 성능
        agent_drafter_performance = {}
        for agent in set(o.agent_id for o in drafter_outcomes):
            agent_drafters = [o for o in drafter_outcomes if o.agent_id == agent]
            agent_successes = sum(1 for o in agent_drafters if o.status == OutcomeStatus.SUCCESS.value)
            agent_drafter_performance[agent] = {
                "count": len(agent_drafters),
                "success_rate": agent_successes / len(agent_drafters) if agent_drafters else 0,
            }

        # 분야별 드래프터 성능
        category_drafter_performance = {}
        for cat in set(o.category for o in drafter_outcomes):
            cat_drafters = [o for o in drafter_outcomes if o.category == cat]
            cat_successes = sum(1 for o in cat_drafters if o.status == OutcomeStatus.SUCCESS.value)
            category_drafter_performance[cat] = {
                "count": len(cat_drafters),
                "success_rate": cat_successes / len(cat_drafters) if cat_drafters else 0,
            }

        # 최적화 추천
        optimizations = []

        # 드래프터 성공률 낮은 에이전트
        for agent, perf in agent_drafter_performance.items():
            if perf["success_rate"] < 0.5 and perf["count"] >= 5:
                optimizations.append(
                    f"{agent}: 드래프터 성공률 {perf['success_rate']*100:.1f}% - "
                    f"프롬프트 개선 또는 검증자 직접 사용 고려"
                )

        # 드래프터 성공률 낮은 분야
        for cat, perf in category_drafter_performance.items():
            if perf["success_rate"] < 0.5 and perf["count"] >= 5:
                optimizations.append(
                    f"{cat} 분야: 드래프터 성공률 {perf['success_rate']*100:.1f}% - "
                    f"복잡한 분야로 검증자 직접 사용 권장"
                )

        return {
            "summary": {
                "total_drafter_calls": len(drafter_outcomes),
                "total_verifier_calls": len(verifier_outcomes),
                "drafter_success_rate": drafter_success_rate,
                "escalation_rate": escalation_rate,
            },
            "cost_analysis": {
                "drafter_cost_usd": drafter_cost,
                "verifier_cost_usd": verifier_cost,
                "total_cost_usd": total_cost,
                "drafter_cost_ratio": drafter_cost / total_cost if total_cost > 0 else 0,
            },
            "by_agent": agent_drafter_performance,
            "by_category": category_drafter_performance,
            "optimizations": optimizations,
            "estimated_savings": {
                "current_cascade_savings_pct": drafter_success_rate * 0.6 * 100,  # 대략적 추정
                "potential_improvement": "드래프터 성공률 10% 향상 시 약 6% 추가 절감",
            },
        }

    def suggest_drafter_improvements(self, agent_id: str) -> list[str]:
        """
        특정 에이전트의 드래프터 개선 제안.

        Args:
            agent_id: 에이전트 ID

        Returns:
            개선 제안 목록
        """
        perf = self.analyzer.analyze_agent_performance(agent_id)
        suggestions = []

        if perf.escalation_rate > 0.4:
            suggestions.append(
                f"에스컬레이션률 {perf.escalation_rate*100:.1f}% - "
                f"드래프터 프롬프트에 더 구체적인 지시 추가 필요"
            )

        # 성능 낮은 분야 식별
        weak_categories = [
            cat for cat, stats in perf.performance_by_category.items()
            if stats.get("success_rate", 1) < 0.6
        ]
        if weak_categories:
            suggestions.append(
                f"{', '.join(weak_categories)} 분야 성능 저조 - "
                f"해당 분야 특화 프롬프트 추가 고려"
            )

        # 복잡도별 분석
        complex_perf = perf.performance_by_complexity.get("complex", {})
        if complex_perf.get("success_rate", 1) < 0.5:
            suggestions.append(
                f"복잡한 쿼리 성공률 {complex_perf.get('success_rate', 0)*100:.1f}% - "
                f"복잡한 쿼리는 검증자 직접 사용 권장"
            )

        if not suggestions:
            suggestions.append(f"{agent_id} 드래프터 성능 양호 (에스컬레이션률 {perf.escalation_rate*100:.1f}%)")

        return suggestions


# 싱글톤 인스턴스
_learning_manager: Optional[LearningManager] = None


def get_learning_manager(
    storage_path: Optional[Path] = None,
) -> LearningManager:
    """
    LearningManager 싱글톤 인스턴스 획득.

    Args:
        storage_path: 저장 경로 (첫 호출 시만 적용)

    Returns:
        LearningManager 인스턴스
    """
    global _learning_manager

    if _learning_manager is None:
        default_path = Path("data/learning") if storage_path is None else storage_path
        _learning_manager = LearningManager(storage_path=default_path)

    return _learning_manager


def reset_learning_manager() -> None:
    """싱글톤 리셋 (테스트용)."""
    global _learning_manager
    _learning_manager = None
