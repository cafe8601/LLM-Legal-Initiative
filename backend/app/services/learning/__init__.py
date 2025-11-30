"""
Learning System

MAS multiagent v4 기반 학습 시스템 통합.
법률 자문 위원들의 성능을 추적하고 최적의 위원을 추천합니다.

Components:
- OutcomeTracker: 상담 결과 추적
- PatternAnalyzer: 위원별/분야별 성능 분석
- LearningManager: 통합 학습 관리 및 추천

Usage:
    from app.services.learning import LearningManager, get_learning_manager

    # 싱글톤 인스턴스 사용
    learning_manager = get_learning_manager()

    # 상담 결과 기록
    await learning_manager.record_consultation_outcome(...)

    # 에이전트 추천 받기
    recommendation = await learning_manager.get_agent_recommendation(...)
"""

from app.services.learning.outcome_tracker import (
    OutcomeTracker,
    ConsultationOutcome,
    OutcomeStatus,
    ConsultationTier,
)
from app.services.learning.pattern_analyzer import (
    PatternAnalyzer,
    AgentPerformance,
    CategoryInsight,
)
from app.services.learning.learning_manager import (
    LearningManager,
    AgentRecommendation,
    CascadeDecision,
    get_learning_manager,
)

__all__ = [
    # OutcomeTracker
    "OutcomeTracker",
    "ConsultationOutcome",
    "OutcomeStatus",
    "ConsultationTier",
    # PatternAnalyzer
    "PatternAnalyzer",
    "AgentPerformance",
    "CategoryInsight",
    # LearningManager
    "LearningManager",
    "AgentRecommendation",
    "CascadeDecision",
    "get_learning_manager",
]
