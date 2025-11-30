"""
Monitoring System Tests

메트릭 수집 및 헬스체크 시스템 테스트
"""

import asyncio
import pytest
from datetime import datetime, timezone

from app.services.monitoring.metrics import (
    MetricsCollector,
    MetricType,
    get_metrics_collector,
)
from app.services.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    get_health_checker,
    check_cache_health,
    check_llm_service_health,
)


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """메트릭 수집기 테스트"""

    def setup_method(self):
        """각 테스트 전 새 메트릭 수집기 생성"""
        self.metrics = MetricsCollector(prefix="test")

    def test_counter_increment(self):
        """카운터 증가 테스트"""
        self.metrics.increment("requests_total")
        self.metrics.increment("requests_total")
        self.metrics.increment("requests_total", value=3)

        assert self.metrics.get_counter("requests_total") == 5

    def test_counter_with_labels(self):
        """라벨이 있는 카운터 테스트"""
        self.metrics.increment("requests_total", labels={"method": "GET"})
        self.metrics.increment("requests_total", labels={"method": "POST"})
        self.metrics.increment("requests_total", labels={"method": "GET"})

        assert self.metrics.get_counter("requests_total", labels={"method": "GET"}) == 2
        assert self.metrics.get_counter("requests_total", labels={"method": "POST"}) == 1

    def test_gauge_set_and_get(self):
        """게이지 설정 및 조회 테스트"""
        self.metrics.set_gauge("active_sessions", 10)
        assert self.metrics.get_gauge("active_sessions") == 10

        self.metrics.set_gauge("active_sessions", 15)
        assert self.metrics.get_gauge("active_sessions") == 15

    def test_gauge_increment_decrement(self):
        """게이지 증감 테스트"""
        self.metrics.set_gauge("connections", 10)
        self.metrics.inc_gauge("connections", 5)
        assert self.metrics.get_gauge("connections") == 15

        self.metrics.dec_gauge("connections", 3)
        assert self.metrics.get_gauge("connections") == 12

    def test_histogram_observe(self):
        """히스토그램 관측 테스트"""
        # 여러 값 기록
        for value in [0.1, 0.2, 0.5, 1.0, 2.0]:
            self.metrics.observe("response_seconds", value)

        result = self.metrics.get_histogram("response_seconds")
        assert result is not None
        assert result["count"] == 5
        assert result["sum"] == 3.8  # 0.1 + 0.2 + 0.5 + 1.0 + 2.0

    def test_histogram_buckets(self):
        """히스토그램 버킷 테스트"""
        self.metrics.observe("latency", 0.05)  # <= 0.05
        self.metrics.observe("latency", 0.15)  # <= 0.25
        self.metrics.observe("latency", 0.8)   # <= 1.0

        result = self.metrics.get_histogram("latency")
        assert result is not None

        # 각 버킷의 카운트 확인
        buckets = result["buckets"]
        # 0.05 값은 0.05 버킷 이상에 포함
        assert any(b["le"] >= 0.05 and b["count"] >= 1 for b in buckets)

    def test_cascade_result_recording(self):
        """캐스케이드 결과 기록 테스트"""
        self.metrics.record_cascade_result(
            provider="claude",
            escalated=False,
            response_time=0.5,
        )
        self.metrics.record_cascade_result(
            provider="claude",
            escalated=True,
            response_time=1.2,
        )

        assert self.metrics.get_counter(
            "cascade_drafter_success_total",
            labels={"provider": "claude"}
        ) == 1
        assert self.metrics.get_counter(
            "cascade_escalations_total",
            labels={"provider": "claude"}
        ) == 1

    def test_consultation_request_recording(self):
        """상담 요청 기록 테스트"""
        self.metrics.record_consultation_request(user_tier="basic", success=True)
        self.metrics.record_consultation_request(user_tier="basic", success=False)
        self.metrics.record_consultation_request(user_tier="pro", success=True)

        assert self.metrics.get_counter(
            "consultation_requests_total",
            labels={"tier": "basic", "status": "success"}
        ) == 1
        assert self.metrics.get_counter(
            "consultation_requests_total",
            labels={"tier": "basic", "status": "failure"}
        ) == 1

    def test_get_all_metrics(self):
        """모든 메트릭 조회 테스트"""
        self.metrics.increment("test_counter")
        self.metrics.set_gauge("test_gauge", 42)
        self.metrics.observe("test_histogram", 0.5)

        all_metrics = self.metrics.get_all_metrics()

        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "uptime_seconds" in all_metrics

    def test_prometheus_format(self):
        """Prometheus 형식 내보내기 테스트"""
        self.metrics.increment("http_requests_total")
        self.metrics.set_gauge("active_users", 100)

        output = self.metrics.to_prometheus_format()

        assert "test_http_requests_total" in output
        assert "test_active_users" in output

    def test_reset(self):
        """메트릭 초기화 테스트"""
        self.metrics.increment("counter", 10)
        self.metrics.set_gauge("gauge", 50)

        self.metrics.reset()

        assert self.metrics.get_counter("counter") == 0
        assert self.metrics.get_gauge("gauge") is None


class TestMetricsCollectorSingleton:
    """메트릭 수집기 싱글톤 테스트"""

    def test_singleton_instance(self):
        """싱글톤 인스턴스 테스트"""
        metrics1 = get_metrics_collector()
        metrics2 = get_metrics_collector()

        assert metrics1 is metrics2


# =============================================================================
# HealthChecker Tests
# =============================================================================


class TestHealthChecker:
    """헬스체커 테스트"""

    def setup_method(self):
        """각 테스트 전 새 헬스체커 생성"""
        self.checker = HealthChecker(timeout=5, include_details=True)

    @pytest.mark.asyncio
    async def test_healthy_component(self):
        """정상 컴포넌트 체크 테스트"""
        async def healthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        self.checker.register_check("test", healthy_check)
        result = await self.checker.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 1
        assert result.components[0].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_component(self):
        """비정상 컴포넌트 체크 테스트"""
        async def unhealthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="test",
                status=HealthStatus.UNHEALTHY,
                message="Component failed",
            )

        self.checker.register_check("test", unhealthy_check)
        result = await self.checker.check_all()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_degraded_component(self):
        """저하된 컴포넌트 체크 테스트"""
        async def degraded_check() -> ComponentHealth:
            return ComponentHealth(
                name="test",
                status=HealthStatus.DEGRADED,
                message="Component degraded",
            )

        self.checker.register_check("test", degraded_check)
        result = await self.checker.check_all()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_multiple_components(self):
        """다중 컴포넌트 체크 테스트"""
        async def healthy_check() -> ComponentHealth:
            return ComponentHealth(
                name="db",
                status=HealthStatus.HEALTHY,
                message="Database OK",
            )

        async def degraded_check() -> ComponentHealth:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message="Cache slow",
            )

        self.checker.register_check("db", healthy_check)
        self.checker.register_check("cache", degraded_check)

        result = await self.checker.check_all()

        # 하나라도 DEGRADED이면 전체 DEGRADED
        assert result.status == HealthStatus.DEGRADED
        assert len(result.components) == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        async def slow_check() -> ComponentHealth:
            await asyncio.sleep(10)  # 타임아웃보다 길게
            return ComponentHealth(
                name="slow",
                status=HealthStatus.HEALTHY,
                message="Never reached",
            )

        checker = HealthChecker(timeout=1, include_details=True)
        result = await checker.check_component("slow", slow_check)

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """예외 처리 테스트"""
        async def failing_check() -> ComponentHealth:
            raise RuntimeError("Check failed!")

        result = await self.checker.check_component("failing", failing_check)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed!" in result.message

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            environment="test",
            uptime_seconds=100.5,
            components=[
                ComponentHealth(
                    name="db",
                    status=HealthStatus.HEALTHY,
                    latency_ms=5.2,
                    message="OK",
                )
            ],
        )

        result = self.checker.to_dict(health)

        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"
        assert result["environment"] == "test"
        assert "components" in result
        assert len(result["components"]) == 1


class TestHealthCheckFunctions:
    """헬스체크 함수 테스트"""

    @pytest.mark.asyncio
    async def test_check_cache_health(self):
        """캐시 헬스체크 테스트"""
        from app.services.memory import get_consultation_cache

        cache = get_consultation_cache()
        result = await check_cache_health(cache)

        assert result.name == "cache"
        assert result.status == HealthStatus.HEALTHY
        assert "total_sessions" in result.details

    @pytest.mark.asyncio
    async def test_check_llm_service_health_with_key(self):
        """LLM 서비스 헬스체크 (API 키 있음) 테스트"""
        result = await check_llm_service_health(openrouter_available=True)

        # API 키 설정에 따라 결과 다름
        assert result.name == "llm_service"
        # API 키가 없으면 DEGRADED, 있으면 HEALTHY
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_check_llm_service_health_unavailable(self):
        """LLM 서비스 헬스체크 (사용 불가) 테스트"""
        result = await check_llm_service_health(openrouter_available=False)

        # API 키 설정 여부에 따라 UNHEALTHY 또는 DEGRADED
        assert result.name == "llm_service"


class TestHealthCheckerSingleton:
    """헬스체커 싱글톤 테스트"""

    def test_singleton_instance(self):
        """싱글톤 인스턴스 테스트"""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2


# =============================================================================
# Integration Tests
# =============================================================================


class TestMonitoringIntegration:
    """모니터링 통합 테스트"""

    @pytest.mark.asyncio
    async def test_metrics_and_health_together(self):
        """메트릭과 헬스체크 통합 테스트"""
        metrics = MetricsCollector(prefix="integration_test")
        checker = HealthChecker(timeout=5, include_details=True)

        # 메트릭 기록
        metrics.record_consultation_request("basic", success=True)
        metrics.record_cascade_result("claude", escalated=False, response_time=0.5)

        # 헬스체크 등록 및 실행
        async def metrics_check() -> ComponentHealth:
            summary = metrics.get_summary()
            return ComponentHealth(
                name="metrics",
                status=HealthStatus.HEALTHY,
                message="Metrics operational",
                details=summary,
            )

        checker.register_check("metrics", metrics_check)
        health = await checker.check_all()

        assert health.status == HealthStatus.HEALTHY
        assert any(c.name == "metrics" for c in health.components)

    def test_full_metrics_workflow(self):
        """전체 메트릭 워크플로우 테스트"""
        metrics = MetricsCollector(prefix="workflow_test")

        # 시뮬레이션: 여러 상담 요청 처리
        for i in range(10):
            tier = "basic" if i < 7 else "pro"
            success = i < 9  # 마지막 하나만 실패
            metrics.record_consultation_request(tier, success)

            if success:
                escalated = i % 3 == 0  # 3번마다 에스컬레이션
                response_time = 0.5 + (0.1 * i)
                metrics.record_cascade_result("claude", escalated, response_time)

        # 통계 확인
        all_metrics = metrics.get_all_metrics()
        assert all_metrics["uptime_seconds"] > 0
        assert len(all_metrics["counters"]) > 0
        assert len(all_metrics["histograms"]) > 0

        # Prometheus 형식 출력 확인
        prometheus_output = metrics.to_prometheus_format()
        assert "consultation_requests_total" in prometheus_output
        assert "cascade" in prometheus_output
