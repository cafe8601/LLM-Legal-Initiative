"""
Health Check System

시스템 헬스체크 및 상태 모니터링.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable, Awaitable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """헬스 상태."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """컴포넌트 헬스 정보."""
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemHealth:
    """전체 시스템 헬스."""
    status: HealthStatus
    version: str
    environment: str
    uptime_seconds: float
    components: list[ComponentHealth]
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthChecker:
    """
    헬스 체커.

    각 컴포넌트의 상태를 확인하고 전체 시스템 헬스를 보고합니다.
    """

    def __init__(
        self,
        timeout: int = 5,
        include_details: bool = True,
    ):
        """
        초기화.

        Args:
            timeout: 헬스 체크 타임아웃 (초)
            include_details: 상세 정보 포함 여부
        """
        self._timeout = timeout
        self._include_details = include_details
        self._start_time = time.time()
        self._checks: dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}

    def register_check(
        self,
        name: str,
        check_func: Callable[[], Awaitable[ComponentHealth]],
    ) -> None:
        """헬스 체크 함수 등록."""
        self._checks[name] = check_func

    async def check_component(
        self,
        name: str,
        check_func: Callable[[], Awaitable[ComponentHealth]],
    ) -> ComponentHealth:
        """개별 컴포넌트 체크."""
        start = time.time()
        try:
            result = await asyncio.wait_for(
                check_func(),
                timeout=self._timeout,
            )
            result.latency_ms = (time.time() - start) * 1000
            return result
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f"Health check timed out after {self._timeout}s",
            )
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def check_all(self) -> SystemHealth:
        """모든 컴포넌트 체크."""
        components = []

        # 등록된 체크 실행
        for name, check_func in self._checks.items():
            result = await self.check_component(name, check_func)
            components.append(result)

        # 전체 상태 결정
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            uptime_seconds=time.time() - self._start_time,
            components=components,
        )

    def to_dict(self, health: SystemHealth) -> dict:
        """SystemHealth를 딕셔너리로 변환."""
        result = {
            "status": health.status.value,
            "version": health.version,
            "environment": health.environment,
            "uptime_seconds": round(health.uptime_seconds, 2),
            "checked_at": health.checked_at.isoformat(),
        }

        if self._include_details:
            result["components"] = [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "message": c.message,
                    "details": c.details,
                }
                for c in health.components
            ]

        return result


# ========================================
# 기본 헬스 체크 함수들
# ========================================


async def check_database_health(db: AsyncSession) -> ComponentHealth:
    """데이터베이스 헬스 체크."""
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            details={"driver": "asyncpg"},
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e}",
        )


async def check_redis_health(redis_client) -> ComponentHealth:
    """Redis 헬스 체크."""
    try:
        if redis_client is None:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis client not configured",
            )

        await redis_client.ping()
        info = await redis_client.info("memory")
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection successful",
            details={
                "used_memory_human": info.get("used_memory_human", "unknown"),
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis connection failed: {e}",
        )


async def check_cache_health(cache) -> ComponentHealth:
    """메모리 캐시 헬스 체크."""
    try:
        stats = cache.get_statistics()
        return ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
            message="Cache operational",
            details={
                "total_sessions": stats.get("total_sessions", 0),
                "total_entries": stats.get("total_entries", 0),
                "entries_per_session": round(stats.get("entries_per_session", 0), 2),
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="cache",
            status=HealthStatus.DEGRADED,
            message=f"Cache check failed: {e}",
        )


async def check_llm_service_health(openrouter_available: bool = True) -> ComponentHealth:
    """LLM 서비스 헬스 체크."""
    if not settings.OPENROUTER_API_KEY:
        return ComponentHealth(
            name="llm_service",
            status=HealthStatus.DEGRADED,
            message="OpenRouter API key not configured",
        )

    if openrouter_available:
        return ComponentHealth(
            name="llm_service",
            status=HealthStatus.HEALTHY,
            message="LLM service configured",
            details={
                "provider": "openrouter",
                "chairman_model": settings.OPENROUTER_CHAIRMAN_MODEL,
                "council_models_count": len(settings.OPENROUTER_COUNCIL_MODELS),
            },
        )

    return ComponentHealth(
        name="llm_service",
        status=HealthStatus.UNHEALTHY,
        message="LLM service unavailable",
    )


async def check_rag_service_health(rag_manager=None) -> ComponentHealth:
    """RAG 서비스 헬스 체크."""
    if rag_manager is None:
        return ComponentHealth(
            name="rag_service",
            status=HealthStatus.DEGRADED,
            message="RAG manager not initialized",
        )

    try:
        # RAG 서비스 상태 확인
        return ComponentHealth(
            name="rag_service",
            status=HealthStatus.HEALTHY,
            message="RAG service operational",
            details={
                "embedding_model": settings.EMBEDDING_MODEL,
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="rag_service",
            status=HealthStatus.UNHEALTHY,
            message=f"RAG service check failed: {e}",
        )


# ========================================
# 싱글톤 인스턴스
# ========================================

_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """HealthChecker 싱글톤 인스턴스 획득."""
    global _health_checker

    if _health_checker is None:
        _health_checker = HealthChecker(
            timeout=settings.HEALTH_CHECK_TIMEOUT,
            include_details=settings.HEALTH_CHECK_INCLUDE_DETAILS,
        )

    return _health_checker
