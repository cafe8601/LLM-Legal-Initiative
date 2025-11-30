"""
LLM Legal Advisory Council - FastAPI Application Entry Point

AI 기반 법률 자문 위원회 백엔드 서버
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import AppException
from app.api.v1.router import api_router
from app.utils.logger import get_logger
from app.services.monitoring import (
    get_metrics_collector,
    get_health_checker,
    HealthStatus,
    ComponentHealth,
    check_cache_health,
    check_llm_service_health,
)
from app.services.memory import get_consultation_cache

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info(
        "Starting LLM Legal Advisory Council API",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # Initialize database connection pool
    # await init_db()

    # Initialize Redis connection
    # await init_redis()

    # Initialize RAG service (Google File Search)
    # await init_rag_service()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Close database connections
    # await close_db()

    # Close Redis connections
    # await close_redis()

    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        description="""
## LLM Legal Advisory Council API

AI 기반 법률 자문 위원회 서비스의 백엔드 API입니다.

### 주요 기능
- **법률 상담**: 4개 LLM(GPT-5.1, Claude Sonnet 4.5, Gemini 3 Pro, Grok 4)의 병렬 의견 수집
- **교차 평가**: Claude Sonnet 4.5의 블라인드 피어 리뷰
- **종합 의견**: Claude Opus 4.5 의장의 최종 종합
- **RAG 검색**: Google File Search 기반 법률 문서 검색
- **사용자 인증**: JWT 기반 인증 시스템
- **티어 관리**: Basic, Pro, Enterprise 구독 관리

### API 버전
- v1: `/api/v1/`
        """,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        """Handle custom application exceptions."""
        logger.warning(
            "Application exception",
            error_code=exc.error_code,
            detail=exc.detail,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected error",
            error=str(exc),
            path=str(request.url),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )

    # Include API routers
    app.include_router(api_router, prefix="/api/v1")

    # Health Check Endpoints
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict:
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/health/ready", tags=["Health"])
    async def readiness_check() -> dict:
        """
        Readiness check for Kubernetes/load balancers.
        Checks all critical components.
        """
        health_checker = get_health_checker()
        cache = get_consultation_cache()

        # 캐시 헬스 체크 등록
        health_checker.register_check(
            "cache",
            lambda: check_cache_health(cache),
        )

        # LLM 서비스 체크 등록
        health_checker.register_check(
            "llm_service",
            lambda: check_llm_service_health(bool(settings.OPENROUTER_API_KEY)),
        )

        # 전체 헬스 체크 실행
        system_health = await health_checker.check_all()

        # HTTP 상태 코드 결정
        if system_health.status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_checker.to_dict(system_health),
            )

        return health_checker.to_dict(system_health)

    @app.get("/health/live", tags=["Health"])
    async def liveness_check() -> dict:
        """Liveness check for Kubernetes."""
        return {"status": "alive"}

    # Metrics Endpoints
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics() -> dict:
        """Get all metrics in JSON format."""
        metrics = get_metrics_collector()
        return metrics.get_all_metrics()

    @app.get("/metrics/prometheus", tags=["Monitoring"])
    async def get_prometheus_metrics() -> str:
        """Get metrics in Prometheus text format."""
        from fastapi.responses import PlainTextResponse
        metrics = get_metrics_collector()
        return PlainTextResponse(
            content=metrics.to_prometheus_format(),
            media_type="text/plain",
        )

    @app.get("/metrics/summary", tags=["Monitoring"])
    async def get_metrics_summary() -> dict:
        """Get metrics summary."""
        metrics = get_metrics_collector()
        return metrics.get_summary()

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
