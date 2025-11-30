"""
Outcome Tracker

법률 자문 위원들의 상담 결과를 추적하고 기록합니다.
MAS multiagent v4의 OutcomeTracker를 법률 도메인에 맞게 확장.

Features:
- 위원별 성공/실패 기록
- 법률 분야별 결과 추적
- CascadeFlow 드래프터/검증자 결과 분리 추적
- 사용자 피드백 연동
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class OutcomeStatus(str, Enum):
    """상담 결과 상태."""
    SUCCESS = "success"
    FAILURE = "failure"
    ESCALATED = "escalated"  # 드래프터 → 검증자 에스컬레이션
    PARTIAL = "partial"  # 부분적 성공


class ConsultationTier(str, Enum):
    """CascadeFlow 티어."""
    DRAFTER = "drafter"
    VERIFIER = "verifier"
    DIRECT = "direct"  # CascadeFlow 미사용


@dataclass
class ConsultationOutcome:
    """상담 결과 데이터."""
    outcome_id: str
    timestamp: str

    # 위원 정보
    agent_id: str  # e.g., "claude", "gpt", "gemini", "grok"
    model_id: str  # e.g., "anthropic/claude-sonnet-4"
    tier: str  # drafter, verifier, direct

    # 상담 정보
    consultation_id: str
    user_id: str
    category: str  # 법률 분야
    query_complexity: str  # simple, moderate, complex

    # 결과
    status: str  # success, failure, escalated, partial
    response_quality: float  # 0.0 ~ 1.0 (품질 점수)

    # 메트릭
    response_time_ms: int
    tokens_used: int
    cost_usd: float

    # 피드백
    user_feedback: Optional[int] = None  # 1-5 점
    escalation_reason: Optional[str] = None

    # 메타데이터
    metadata: dict = field(default_factory=dict)


class OutcomeTracker:
    """
    상담 결과 추적기.

    법률 자문 위원들의 성능을 기록하고 분석을 위한 데이터를 제공합니다.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_outcomes_in_memory: int = 1000,
    ):
        """
        초기화.

        Args:
            storage_path: 결과 저장 경로 (None이면 메모리만 사용)
            max_outcomes_in_memory: 메모리에 유지할 최대 결과 수
        """
        self.storage_path = storage_path
        self.max_outcomes = max_outcomes_in_memory
        self._outcomes: list[ConsultationOutcome] = []
        self._index: dict[str, int] = {}  # outcome_id → index

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_recent_outcomes()

    def _load_recent_outcomes(self) -> None:
        """저장된 최근 결과 로드."""
        if not self.storage_path:
            return

        index_file = self.storage_path / "index.json"
        if not index_file.exists():
            return

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            # 최근 결과만 로드
            recent_ids = index_data.get("recent", [])[-self.max_outcomes:]

            for outcome_id in recent_ids:
                outcome_file = self.storage_path / f"{outcome_id}.json"
                if outcome_file.exists():
                    with open(outcome_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    outcome = ConsultationOutcome(**data)
                    self._outcomes.append(outcome)
                    self._index[outcome_id] = len(self._outcomes) - 1

            logger.info(f"Loaded {len(self._outcomes)} outcomes from storage")
        except Exception as e:
            logger.error(f"Failed to load outcomes: {e}")

    def _save_outcome(self, outcome: ConsultationOutcome) -> None:
        """결과를 파일에 저장."""
        if not self.storage_path:
            return

        try:
            # 개별 결과 저장
            outcome_file = self.storage_path / f"{outcome.outcome_id}.json"
            with open(outcome_file, "w", encoding="utf-8") as f:
                json.dump(asdict(outcome), f, ensure_ascii=False, indent=2)

            # 인덱스 업데이트
            index_file = self.storage_path / "index.json"
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            else:
                index_data = {"recent": [], "by_agent": {}, "by_category": {}}

            index_data["recent"].append(outcome.outcome_id)

            # 에이전트별 인덱스
            if outcome.agent_id not in index_data["by_agent"]:
                index_data["by_agent"][outcome.agent_id] = []
            index_data["by_agent"][outcome.agent_id].append(outcome.outcome_id)

            # 카테고리별 인덱스
            if outcome.category not in index_data["by_category"]:
                index_data["by_category"][outcome.category] = []
            index_data["by_category"][outcome.category].append(outcome.outcome_id)

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save outcome: {e}")

    def record_success(
        self,
        agent_id: str,
        model_id: str,
        tier: ConsultationTier,
        consultation_id: str | UUID,
        user_id: str | UUID,
        category: str,
        query_complexity: str,
        response_quality: float,
        response_time_ms: int,
        tokens_used: int,
        cost_usd: float,
        metadata: Optional[dict] = None,
    ) -> ConsultationOutcome:
        """
        성공한 상담 결과 기록.

        Returns:
            생성된 ConsultationOutcome
        """
        outcome = ConsultationOutcome(
            outcome_id=f"success_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            model_id=model_id,
            tier=tier.value,
            consultation_id=str(consultation_id),
            user_id=str(user_id),
            category=category,
            query_complexity=query_complexity,
            status=OutcomeStatus.SUCCESS.value,
            response_quality=response_quality,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        self._add_outcome(outcome)
        logger.info(f"Recorded success: agent={agent_id}, category={category}, quality={response_quality:.2f}")
        return outcome

    def record_failure(
        self,
        agent_id: str,
        model_id: str,
        tier: ConsultationTier,
        consultation_id: str | UUID,
        user_id: str | UUID,
        category: str,
        query_complexity: str,
        error_reason: str,
        response_time_ms: int,
        tokens_used: int,
        cost_usd: float,
        metadata: Optional[dict] = None,
    ) -> ConsultationOutcome:
        """
        실패한 상담 결과 기록.

        Returns:
            생성된 ConsultationOutcome
        """
        outcome = ConsultationOutcome(
            outcome_id=f"failure_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            model_id=model_id,
            tier=tier.value,
            consultation_id=str(consultation_id),
            user_id=str(user_id),
            category=category,
            query_complexity=query_complexity,
            status=OutcomeStatus.FAILURE.value,
            response_quality=0.0,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            escalation_reason=error_reason,
            metadata=metadata or {},
        )

        self._add_outcome(outcome)
        logger.info(f"Recorded failure: agent={agent_id}, category={category}, reason={error_reason[:50]}")
        return outcome

    def record_escalation(
        self,
        agent_id: str,
        model_id: str,
        consultation_id: str | UUID,
        user_id: str | UUID,
        category: str,
        query_complexity: str,
        escalation_reason: str,
        drafter_quality: float,
        response_time_ms: int,
        tokens_used: int,
        cost_usd: float,
        metadata: Optional[dict] = None,
    ) -> ConsultationOutcome:
        """
        에스컬레이션된 상담 결과 기록 (드래프터 → 검증자).

        Returns:
            생성된 ConsultationOutcome
        """
        outcome = ConsultationOutcome(
            outcome_id=f"escalated_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            model_id=model_id,
            tier=ConsultationTier.DRAFTER.value,
            consultation_id=str(consultation_id),
            user_id=str(user_id),
            category=category,
            query_complexity=query_complexity,
            status=OutcomeStatus.ESCALATED.value,
            response_quality=drafter_quality,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            escalation_reason=escalation_reason,
            metadata=metadata or {},
        )

        self._add_outcome(outcome)
        logger.info(f"Recorded escalation: agent={agent_id}, category={category}, quality={drafter_quality:.2f}")
        return outcome

    def update_user_feedback(
        self,
        outcome_id: str,
        feedback_score: int,
    ) -> bool:
        """
        사용자 피드백 업데이트.

        Args:
            outcome_id: 결과 ID
            feedback_score: 1-5 점수

        Returns:
            업데이트 성공 여부
        """
        if outcome_id not in self._index:
            return False

        idx = self._index[outcome_id]
        self._outcomes[idx].user_feedback = feedback_score
        self._save_outcome(self._outcomes[idx])
        return True

    def _add_outcome(self, outcome: ConsultationOutcome) -> None:
        """결과 추가."""
        # 메모리 관리
        if len(self._outcomes) >= self.max_outcomes:
            # 가장 오래된 결과 제거
            old_outcome = self._outcomes.pop(0)
            del self._index[old_outcome.outcome_id]
            # 인덱스 재조정
            self._index = {oid: i for i, o in enumerate(self._outcomes) for oid in [o.outcome_id]}

        self._outcomes.append(outcome)
        self._index[outcome.outcome_id] = len(self._outcomes) - 1
        self._save_outcome(outcome)

    def get_outcomes_for_agent(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> list[ConsultationOutcome]:
        """에이전트별 결과 조회."""
        return [o for o in self._outcomes if o.agent_id == agent_id][-limit:]

    def get_outcomes_for_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[ConsultationOutcome]:
        """카테고리별 결과 조회."""
        return [o for o in self._outcomes if o.category == category][-limit:]

    def get_outcomes_for_tier(
        self,
        tier: ConsultationTier,
        limit: int = 100,
    ) -> list[ConsultationOutcome]:
        """티어별 결과 조회."""
        return [o for o in self._outcomes if o.tier == tier.value][-limit:]

    def get_recent_outcomes(self, limit: int = 50) -> list[ConsultationOutcome]:
        """최근 결과 조회."""
        return self._outcomes[-limit:]

    def get_success_rate(
        self,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        tier: Optional[ConsultationTier] = None,
    ) -> float:
        """
        성공률 계산.

        Args:
            agent_id: 에이전트 필터
            category: 카테고리 필터
            tier: 티어 필터

        Returns:
            성공률 (0.0 ~ 1.0)
        """
        filtered = self._outcomes

        if agent_id:
            filtered = [o for o in filtered if o.agent_id == agent_id]
        if category:
            filtered = [o for o in filtered if o.category == category]
        if tier:
            filtered = [o for o in filtered if o.tier == tier.value]

        if not filtered:
            return 0.0

        successes = sum(1 for o in filtered if o.status == OutcomeStatus.SUCCESS.value)
        return successes / len(filtered)

    def get_escalation_rate(
        self,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> float:
        """
        에스컬레이션률 계산 (드래프터 기준).

        Returns:
            에스컬레이션률 (0.0 ~ 1.0)
        """
        filtered = [o for o in self._outcomes if o.tier == ConsultationTier.DRAFTER.value]

        if agent_id:
            filtered = [o for o in filtered if o.agent_id == agent_id]
        if category:
            filtered = [o for o in filtered if o.category == category]

        if not filtered:
            return 0.0

        escalations = sum(1 for o in filtered if o.status == OutcomeStatus.ESCALATED.value)
        return escalations / len(filtered)

    def get_average_quality(
        self,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        tier: Optional[ConsultationTier] = None,
    ) -> float:
        """평균 품질 점수."""
        filtered = self._outcomes

        if agent_id:
            filtered = [o for o in filtered if o.agent_id == agent_id]
        if category:
            filtered = [o for o in filtered if o.category == category]
        if tier:
            filtered = [o for o in filtered if o.tier == tier.value]

        if not filtered:
            return 0.0

        return sum(o.response_quality for o in filtered) / len(filtered)

    def get_statistics(self) -> dict:
        """전체 통계."""
        return {
            "total_outcomes": len(self._outcomes),
            "success_rate": self.get_success_rate(),
            "escalation_rate": self.get_escalation_rate(),
            "average_quality": self.get_average_quality(),
            "by_agent": {
                agent: {
                    "count": len([o for o in self._outcomes if o.agent_id == agent]),
                    "success_rate": self.get_success_rate(agent_id=agent),
                    "average_quality": self.get_average_quality(agent_id=agent),
                }
                for agent in set(o.agent_id for o in self._outcomes)
            },
            "by_category": {
                cat: {
                    "count": len([o for o in self._outcomes if o.category == cat]),
                    "success_rate": self.get_success_rate(category=cat),
                }
                for cat in set(o.category for o in self._outcomes)
            },
        }
