"""
Pattern Analyzer

법률 자문 위원들의 성능 패턴을 분석합니다.
MAS multiagent v4의 PatternAnalyzer를 법률 도메인에 맞게 확장.

Features:
- 위원별 성능 분석
- 법률 분야별 최적 위원 식별
- 쿼리 복잡도별 성능 분석
- CascadeFlow 드래프터 성공률 예측
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from app.services.learning.outcome_tracker import (
    OutcomeTracker,
    ConsultationOutcome,
    OutcomeStatus,
    ConsultationTier,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """에이전트 성능 데이터."""
    agent_id: str
    total_consultations: int
    successes: int
    failures: int
    escalations: int
    success_rate: float
    escalation_rate: float
    average_quality: float
    average_response_time_ms: float
    total_cost_usd: float

    # 분야별 성능
    performance_by_category: dict[str, dict] = field(default_factory=dict)

    # 복잡도별 성능
    performance_by_complexity: dict[str, dict] = field(default_factory=dict)


@dataclass
class CategoryInsight:
    """법률 분야별 인사이트."""
    category: str
    total_consultations: int
    best_agent: str
    best_agent_success_rate: float
    average_quality: float
    common_keywords: list[str]


class PatternAnalyzer:
    """
    패턴 분석기.

    위원들의 성능 패턴을 분석하여 최적의 위원 선택을 지원합니다.
    """

    # 법률 키워드 (한국어)
    LEGAL_KEYWORDS = {
        "민사": ["계약", "손해배상", "채권", "채무", "임대차", "매매", "보증", "이혼", "상속", "유류분"],
        "형사": ["고소", "고발", "구속", "기소", "형사", "처벌", "범죄", "형법", "피해자", "피고인"],
        "행정": ["허가", "인가", "신고", "행정처분", "취소", "무효", "행정소송", "국가배상"],
        "노동": ["해고", "임금", "퇴직금", "근로계약", "부당해고", "근로기준법", "산재"],
        "가족": ["이혼", "양육권", "친권", "재산분할", "위자료", "상속", "유언"],
        "부동산": ["등기", "소유권", "전세", "임대차", "분양", "재개발", "토지"],
        "회사": ["주주", "이사", "설립", "해산", "합병", "정관", "상법"],
        "지식재산": ["특허", "상표", "저작권", "영업비밀", "침해"],
    }

    # 복잡도 판단 키워드
    COMPLEXITY_INDICATORS = {
        "complex": ["소송", "재판", "가압류", "가처분", "항소", "상고", "유류분", "분쟁"],
        "moderate": ["계약", "해지", "해제", "손해배상", "청구", "절차"],
        "simple": ["뜻", "의미", "비용", "기간", "방법", "신청"],
    }

    # 불용어
    STOP_WORDS = {
        "의", "가", "이", "은", "는", "을", "를", "에", "와", "과", "로", "으로",
        "에서", "까지", "부터", "하다", "되다", "있다", "없다", "수", "것", "등",
        "및", "또는", "그", "이", "저", "무엇", "어떻게", "언제", "어디",
    }

    def __init__(self, outcome_tracker: OutcomeTracker):
        """
        초기화.

        Args:
            outcome_tracker: 결과 추적기
        """
        self.tracker = outcome_tracker

    def analyze_agent_performance(self, agent_id: str) -> AgentPerformance:
        """
        특정 에이전트의 성능 분석.

        Args:
            agent_id: 에이전트 ID

        Returns:
            AgentPerformance 객체
        """
        outcomes = self.tracker.get_outcomes_for_agent(agent_id)

        if not outcomes:
            return AgentPerformance(
                agent_id=agent_id,
                total_consultations=0,
                successes=0,
                failures=0,
                escalations=0,
                success_rate=0.0,
                escalation_rate=0.0,
                average_quality=0.0,
                average_response_time_ms=0.0,
                total_cost_usd=0.0,
            )

        successes = sum(1 for o in outcomes if o.status == OutcomeStatus.SUCCESS.value)
        failures = sum(1 for o in outcomes if o.status == OutcomeStatus.FAILURE.value)
        escalations = sum(1 for o in outcomes if o.status == OutcomeStatus.ESCALATED.value)
        drafter_outcomes = [o for o in outcomes if o.tier == ConsultationTier.DRAFTER.value]

        # 분야별 성능
        by_category = {}
        for cat in set(o.category for o in outcomes):
            cat_outcomes = [o for o in outcomes if o.category == cat]
            cat_successes = sum(1 for o in cat_outcomes if o.status == OutcomeStatus.SUCCESS.value)
            by_category[cat] = {
                "count": len(cat_outcomes),
                "success_rate": cat_successes / len(cat_outcomes) if cat_outcomes else 0,
                "average_quality": sum(o.response_quality for o in cat_outcomes) / len(cat_outcomes),
            }

        # 복잡도별 성능
        by_complexity = {}
        for comp in set(o.query_complexity for o in outcomes):
            comp_outcomes = [o for o in outcomes if o.query_complexity == comp]
            comp_successes = sum(1 for o in comp_outcomes if o.status == OutcomeStatus.SUCCESS.value)
            by_complexity[comp] = {
                "count": len(comp_outcomes),
                "success_rate": comp_successes / len(comp_outcomes) if comp_outcomes else 0,
                "average_quality": sum(o.response_quality for o in comp_outcomes) / len(comp_outcomes),
            }

        return AgentPerformance(
            agent_id=agent_id,
            total_consultations=len(outcomes),
            successes=successes,
            failures=failures,
            escalations=escalations,
            success_rate=successes / len(outcomes),
            escalation_rate=escalations / len(drafter_outcomes) if drafter_outcomes else 0,
            average_quality=sum(o.response_quality for o in outcomes) / len(outcomes),
            average_response_time_ms=sum(o.response_time_ms for o in outcomes) / len(outcomes),
            total_cost_usd=sum(o.cost_usd for o in outcomes),
            performance_by_category=by_category,
            performance_by_complexity=by_complexity,
        )

    def analyze_all_agents(self) -> dict[str, AgentPerformance]:
        """모든 에이전트 성능 분석."""
        outcomes = self.tracker.get_recent_outcomes(limit=1000)
        agent_ids = set(o.agent_id for o in outcomes)

        return {
            agent_id: self.analyze_agent_performance(agent_id)
            for agent_id in agent_ids
        }

    def get_best_agent_for_category(
        self,
        category: str,
        min_consultations: int = 5,
    ) -> tuple[str, float]:
        """
        특정 법률 분야에서 최적의 에이전트 찾기.

        Args:
            category: 법률 분야
            min_consultations: 최소 상담 수 (신뢰도 확보)

        Returns:
            (에이전트 ID, 성공률) 튜플
        """
        outcomes = self.tracker.get_outcomes_for_category(category)

        if not outcomes:
            return ("", 0.0)

        # 에이전트별 성공률 계산
        agent_stats = {}
        for outcome in outcomes:
            if outcome.agent_id not in agent_stats:
                agent_stats[outcome.agent_id] = {"success": 0, "total": 0}
            agent_stats[outcome.agent_id]["total"] += 1
            if outcome.status == OutcomeStatus.SUCCESS.value:
                agent_stats[outcome.agent_id]["success"] += 1

        # 최소 상담 수 이상인 에이전트 필터
        valid_agents = {
            agent: stats for agent, stats in agent_stats.items()
            if stats["total"] >= min_consultations
        }

        if not valid_agents:
            # 최소 상담 수 미달 시 전체에서 선택
            valid_agents = agent_stats

        # 최고 성공률 에이전트
        best_agent = max(
            valid_agents.keys(),
            key=lambda a: valid_agents[a]["success"] / valid_agents[a]["total"],
        )
        best_rate = valid_agents[best_agent]["success"] / valid_agents[best_agent]["total"]

        return (best_agent, best_rate)

    def get_best_agent_for_query(
        self,
        query: str,
        min_consultations: int = 3,
    ) -> tuple[str, float, str]:
        """
        쿼리에 최적인 에이전트 찾기.

        Args:
            query: 사용자 쿼리
            min_consultations: 최소 상담 수

        Returns:
            (에이전트 ID, 신뢰도, 추천 이유) 튜플
        """
        # 쿼리에서 법률 분야 추출
        category = self._infer_category(query)

        if category:
            agent, rate = self.get_best_agent_for_category(category, min_consultations)
            if agent:
                return (
                    agent,
                    rate,
                    f"{category} 분야에서 {rate*100:.1f}% 성공률"
                )

        # 분야 추론 실패 시 유사 쿼리 기반 추천
        similar_outcomes = self._find_similar_queries(query)
        if similar_outcomes:
            agent_counts = Counter(o.agent_id for o in similar_outcomes if o.status == OutcomeStatus.SUCCESS.value)
            if agent_counts:
                best_agent = agent_counts.most_common(1)[0][0]
                confidence = agent_counts[best_agent] / len(similar_outcomes)
                return (
                    best_agent,
                    confidence,
                    f"유사 쿼리에서 {agent_counts[best_agent]}회 성공"
                )

        # 전체 성공률 기반 추천
        all_performance = self.analyze_all_agents()
        if all_performance:
            best = max(all_performance.values(), key=lambda p: p.success_rate)
            return (
                best.agent_id,
                best.success_rate,
                f"전체 성공률 {best.success_rate*100:.1f}%"
            )

        return ("", 0.0, "데이터 부족")

    def predict_drafter_success(
        self,
        agent_id: str,
        category: str,
        query_complexity: str,
    ) -> float:
        """
        드래프터 성공 확률 예측.

        CascadeFlow에서 드래프터 사용 여부 결정에 활용.

        Args:
            agent_id: 에이전트 ID
            category: 법률 분야
            query_complexity: 쿼리 복잡도

        Returns:
            예측 성공률 (0.0 ~ 1.0)
        """
        outcomes = self.tracker.get_outcomes_for_agent(agent_id)
        drafter_outcomes = [o for o in outcomes if o.tier == ConsultationTier.DRAFTER.value]

        if not drafter_outcomes:
            return 0.6  # 기본값 (데이터 없음)

        # 동일 분야 + 복잡도 필터
        matching = [
            o for o in drafter_outcomes
            if o.category == category and o.query_complexity == query_complexity
        ]

        if len(matching) >= 3:
            successes = sum(1 for o in matching if o.status == OutcomeStatus.SUCCESS.value)
            return successes / len(matching)

        # 동일 분야만
        category_matching = [o for o in drafter_outcomes if o.category == category]
        if len(category_matching) >= 3:
            successes = sum(1 for o in category_matching if o.status == OutcomeStatus.SUCCESS.value)
            return successes / len(category_matching)

        # 전체 드래프터 성공률
        successes = sum(1 for o in drafter_outcomes if o.status == OutcomeStatus.SUCCESS.value)
        return successes / len(drafter_outcomes)

    def get_category_insights(self) -> list[CategoryInsight]:
        """모든 법률 분야에 대한 인사이트."""
        outcomes = self.tracker.get_recent_outcomes(limit=1000)
        categories = set(o.category for o in outcomes)

        insights = []
        for cat in categories:
            cat_outcomes = [o for o in outcomes if o.category == cat]
            best_agent, best_rate = self.get_best_agent_for_category(cat)

            # 키워드 추출
            all_text = " ".join(o.metadata.get("query", "") for o in cat_outcomes if o.metadata)
            keywords = self._extract_keywords(all_text)

            insights.append(CategoryInsight(
                category=cat,
                total_consultations=len(cat_outcomes),
                best_agent=best_agent,
                best_agent_success_rate=best_rate,
                average_quality=sum(o.response_quality for o in cat_outcomes) / len(cat_outcomes),
                common_keywords=keywords[:5],
            ))

        return sorted(insights, key=lambda x: x.total_consultations, reverse=True)

    def _infer_category(self, query: str) -> Optional[str]:
        """쿼리에서 법률 분야 추론."""
        query_lower = query.lower()

        scores = {}
        for category, keywords in self.LEGAL_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])

        return None

    def _infer_complexity(self, query: str) -> str:
        """쿼리 복잡도 추론."""
        query_lower = query.lower()

        for complexity, keywords in self.COMPLEXITY_INDICATORS.items():
            if any(kw in query_lower for kw in keywords):
                return complexity

        # 길이 기반 추론
        if len(query) > 200:
            return "complex"
        elif len(query) > 50:
            return "moderate"

        return "simple"

    def _find_similar_queries(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ConsultationOutcome]:
        """유사 쿼리 찾기."""
        keywords = set(self._extract_keywords(query))
        if not keywords:
            return []

        outcomes = self.tracker.get_recent_outcomes(limit=500)

        # 키워드 오버랩 기반 유사도
        similar = []
        for outcome in outcomes:
            stored_query = outcome.metadata.get("query", "")
            if not stored_query:
                continue

            stored_keywords = set(self._extract_keywords(stored_query))
            overlap = len(keywords & stored_keywords)
            if overlap >= 2:
                similar.append((outcome, overlap))

        # 오버랩 높은 순 정렬
        similar.sort(key=lambda x: x[1], reverse=True)
        return [o for o, _ in similar[:limit]]

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """텍스트에서 키워드 추출."""
        # 한국어 단어 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', text)

        # 불용어 제거
        filtered = [w for w in words if w not in self.STOP_WORDS]

        # 빈도 기반 상위 키워드
        counter = Counter(filtered)
        return [w for w, _ in counter.most_common(max_keywords)]

    def get_analysis_summary(self) -> dict:
        """전체 분석 요약."""
        all_agents = self.analyze_all_agents()
        category_insights = self.get_category_insights()

        return {
            "total_agents": len(all_agents),
            "agents": {
                agent_id: {
                    "success_rate": perf.success_rate,
                    "average_quality": perf.average_quality,
                    "total_consultations": perf.total_consultations,
                    "best_categories": sorted(
                        perf.performance_by_category.items(),
                        key=lambda x: x[1].get("success_rate", 0),
                        reverse=True,
                    )[:3],
                }
                for agent_id, perf in all_agents.items()
            },
            "categories": {
                insight.category: {
                    "best_agent": insight.best_agent,
                    "success_rate": insight.best_agent_success_rate,
                    "total_consultations": insight.total_consultations,
                }
                for insight in category_insights
            },
            "recommendations": self._generate_recommendations(all_agents, category_insights),
        }

    def _generate_recommendations(
        self,
        agents: dict[str, AgentPerformance],
        insights: list[CategoryInsight],
    ) -> list[str]:
        """분석 기반 추천 생성."""
        recommendations = []

        # 에이전트별 추천
        for agent_id, perf in agents.items():
            if perf.escalation_rate > 0.5:
                recommendations.append(
                    f"{agent_id}: 에스컬레이션률 {perf.escalation_rate*100:.1f}% - "
                    f"드래프터 품질 개선 필요"
                )

            if perf.success_rate < 0.7:
                weak_categories = [
                    cat for cat, stats in perf.performance_by_category.items()
                    if stats.get("success_rate", 1) < 0.7
                ]
                if weak_categories:
                    recommendations.append(
                        f"{agent_id}: {', '.join(weak_categories)} 분야 성능 저조"
                    )

        # 분야별 추천
        for insight in insights:
            if insight.best_agent_success_rate < 0.6:
                recommendations.append(
                    f"{insight.category} 분야: 전반적 성공률 낮음 - "
                    f"프롬프트 개선 검토"
                )

        return recommendations[:10]  # 상위 10개
