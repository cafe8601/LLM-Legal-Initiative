"""
Metrics Collection System

Prometheus 호환 메트릭 수집 및 관리.
MAS multiagent v4의 메트릭 시스템을 법률 자문 시스템에 맞게 확장.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """개별 메트릭 값."""
    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


@dataclass
class HistogramBucket:
    """히스토그램 버킷."""
    le: float  # less than or equal
    count: int = 0


class MetricsCollector:
    """
    메트릭 수집기.

    Prometheus 형식의 메트릭을 수집하고 관리합니다.
    스레드 안전합니다.
    """

    def __init__(self, prefix: str = "legal_council"):
        """
        초기화.

        Args:
            prefix: 메트릭 이름 접두사
        """
        self._prefix = prefix
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[HistogramBucket]] = {}
        self._histogram_sums: dict[str, float] = {}
        self._histogram_counts: dict[str, int] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()

        # 기본 버킷 (응답 시간용)
        self._default_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def _full_name(self, name: str, labels: Optional[dict[str, str]] = None) -> str:
        """전체 메트릭 이름 생성."""
        full = f"{self._prefix}_{name}"
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            full = f"{full}{{{label_str}}}"
        return full

    # ========================================
    # Counter 메트릭
    # ========================================

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """카운터 증가."""
        key = self._full_name(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def get_counter(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
    ) -> float:
        """카운터 값 조회."""
        key = self._full_name(name, labels)
        with self._lock:
            return self._counters.get(key, 0)

    # ========================================
    # Gauge 메트릭
    # ========================================

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """게이지 설정."""
        key = self._full_name(name, labels)
        with self._lock:
            self._gauges[key] = value

    def get_gauge(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
    ) -> Optional[float]:
        """게이지 값 조회."""
        key = self._full_name(name, labels)
        with self._lock:
            return self._gauges.get(key)

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """게이지 증가."""
        key = self._full_name(name, labels)
        with self._lock:
            self._gauges[key] = self._gauges.get(key, 0) + value

    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """게이지 감소."""
        key = self._full_name(name, labels)
        with self._lock:
            self._gauges[key] = self._gauges.get(key, 0) - value

    # ========================================
    # Histogram 메트릭
    # ========================================

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        buckets: Optional[list[float]] = None,
    ) -> None:
        """히스토그램에 값 기록."""
        key = self._full_name(name, labels)
        buckets = buckets or self._default_buckets

        with self._lock:
            # 버킷 초기화
            if key not in self._histograms:
                self._histograms[key] = [
                    HistogramBucket(le=b) for b in buckets
                ]
                self._histograms[key].append(HistogramBucket(le=float('inf')))
                self._histogram_sums[key] = 0.0
                self._histogram_counts[key] = 0

            # 버킷 업데이트
            for bucket in self._histograms[key]:
                if value <= bucket.le:
                    bucket.count += 1

            # 합계 및 카운트 업데이트
            self._histogram_sums[key] += value
            self._histogram_counts[key] += 1

    def get_histogram(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        """히스토그램 통계 조회."""
        key = self._full_name(name, labels)
        with self._lock:
            if key not in self._histograms:
                return None

            return {
                "buckets": [
                    {"le": b.le, "count": b.count}
                    for b in self._histograms[key]
                ],
                "sum": self._histogram_sums[key],
                "count": self._histogram_counts[key],
            }

    # ========================================
    # 법률 자문 시스템 전용 메트릭
    # ========================================

    def record_consultation_request(
        self,
        user_tier: str,
        success: bool = True,
    ) -> None:
        """상담 요청 기록."""
        labels = {"tier": user_tier, "status": "success" if success else "failure"}
        self.increment("consultation_requests_total", labels=labels)

    def record_cascade_result(
        self,
        provider: str,
        escalated: bool,
        response_time: float,
    ) -> None:
        """캐스케이드 결과 기록."""
        labels = {"provider": provider}

        # 에스컬레이션 카운터
        if escalated:
            self.increment("cascade_escalations_total", labels=labels)
        else:
            self.increment("cascade_drafter_success_total", labels=labels)

        # 응답 시간 히스토그램
        self.observe("cascade_response_seconds", response_time, labels=labels)

    def record_rag_search(
        self,
        source: str,
        result_count: int,
        latency: float,
    ) -> None:
        """RAG 검색 기록."""
        labels = {"source": source}
        self.increment("rag_searches_total", labels=labels)
        self.observe("rag_search_seconds", latency, labels=labels)
        self.observe("rag_result_count", result_count, labels=labels)

    def record_memory_operation(
        self,
        operation: str,
        session_id: str,
        success: bool = True,
    ) -> None:
        """메모리 작업 기록."""
        labels = {"operation": operation, "status": "success" if success else "failure"}
        self.increment("memory_operations_total", labels=labels)

    def update_active_sessions(self, count: int) -> None:
        """활성 세션 수 업데이트."""
        self.set_gauge("active_sessions", count)

    def update_cache_stats(
        self,
        total_entries: int,
        total_sessions: int,
        hit_rate: float,
    ) -> None:
        """캐시 통계 업데이트."""
        self.set_gauge("cache_entries_total", total_entries)
        self.set_gauge("cache_sessions_total", total_sessions)
        self.set_gauge("cache_hit_rate", hit_rate)

    # ========================================
    # 내보내기
    # ========================================

    def get_all_metrics(self) -> dict:
        """모든 메트릭 조회."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        "buckets": [{"le": b.le, "count": b.count} for b in buckets],
                        "sum": self._histogram_sums[name],
                        "count": self._histogram_counts[name],
                    }
                    for name, buckets in self._histograms.items()
                },
                "uptime_seconds": time.time() - self._start_time,
            }

    def to_prometheus_format(self) -> str:
        """Prometheus 텍스트 형식으로 내보내기."""
        lines = []

        with self._lock:
            # Counters
            for name, value in self._counters.items():
                lines.append(f"{name} {value}")

            # Gauges
            for name, value in self._gauges.items():
                lines.append(f"{name} {value}")

            # Histograms
            for name, buckets in self._histograms.items():
                for bucket in buckets:
                    le_str = "+Inf" if bucket.le == float('inf') else str(bucket.le)
                    lines.append(f'{name}_bucket{{le="{le_str}"}} {bucket.count}')
                lines.append(f"{name}_sum {self._histogram_sums[name]}")
                lines.append(f"{name}_count {self._histogram_counts[name]}")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """메트릭 요약."""
        with self._lock:
            return {
                "total_counters": len(self._counters),
                "total_gauges": len(self._gauges),
                "total_histograms": len(self._histograms),
                "uptime_seconds": time.time() - self._start_time,
                "prefix": self._prefix,
            }

    def reset(self) -> None:
        """모든 메트릭 초기화."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._histogram_sums.clear()
            self._histogram_counts.clear()
            self._start_time = time.time()
            logger.info("Metrics collector reset")


# 싱글톤 인스턴스
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """MetricsCollector 싱글톤 인스턴스 획득."""
    global _metrics_collector

    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(
            prefix=settings.METRICS_PREFIX,
        )

    return _metrics_collector
