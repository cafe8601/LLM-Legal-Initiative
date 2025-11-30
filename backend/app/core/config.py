"""
Application Configuration

환경 변수 기반 설정 관리
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application
    # =========================================================================
    APP_NAME: str = "LLM Legal Advisory Council"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # =========================================================================
    # Database
    # =========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://legal_user:legal_password@localhost:5432/legal_council"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # =========================================================================
    # Redis
    # =========================================================================
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: str | None = None

    # =========================================================================
    # Security
    # =========================================================================
    SECRET_KEY: str = Field(
        default="",
        description="JWT 서명용 비밀 키 (프로덕션에서 반드시 설정 필요)"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("SECRET_KEY", mode="after")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """프로덕션 환경에서 SECRET_KEY 설정 검증."""
        import os
        import warnings

        if not v or v == "change-this-in-production":
            env = os.getenv("ENVIRONMENT", "development")
            if env == "production":
                raise ValueError(
                    "프로덕션 환경에서는 SECRET_KEY를 반드시 설정해야 합니다. "
                    "안전한 랜덤 키 생성: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
            # 개발 환경에서는 경고와 함께 기본값 사용
            warnings.warn(
                "SECRET_KEY가 설정되지 않았습니다. 개발 환경에서만 기본값을 사용합니다.",
                UserWarning,
                stacklevel=2,
            )
            return "dev-only-secret-key-not-for-production"
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # =========================================================================
    # OpenRouter Configuration (Primary LLM Gateway)
    # =========================================================================
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = "http://localhost:3000"  # 사이트 URL (리퍼러)
    OPENROUTER_APP_NAME: str = "AI Legal Advisory Council"  # 앱 이름

    # OpenRouter 사용 여부 (True면 모든 LLM 호출을 OpenRouter를 통해 수행)
    USE_OPENROUTER: bool = True

    # =========================================================================
    # Direct LLM API Keys (USE_OPENROUTER=False 일 때만 사용 - 레거시)
    # 권장: OpenRouter API Key 하나로 모든 LLM 및 임베딩 통합
    # =========================================================================
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""  # 레거시 Google File Search용 (선택)
    XAI_API_KEY: str = ""

    # =========================================================================
    # RAG Configuration (pgvector 기반 - OpenRouter 임베딩 사용)
    # =========================================================================
    # 임베딩 모델: openai-small (1536d), openai-large (3072d), google (768d)
    EMBEDDING_MODEL: str = "openai-small"

    # 레거시: Google File Search (google-generativeai 패키지 필요)
    # 새 프로젝트는 pgvector 사용 권장 (GOOGLE_API_KEY 불필요)
    GOOGLE_PROJECT_ID: str = ""
    GOOGLE_LEGAL_CORPUS_NAME: str = "legal-documents-corpus"
    USE_GOOGLE_FILE_SEARCH: bool = False  # True면 레거시 Google File Search 사용

    # =========================================================================
    # LLM Model Configuration
    # =========================================================================

    # OpenRouter 모델 ID (USE_OPENROUTER=True 일 때 사용)
    # 참고: https://openrouter.ai/models
    OPENROUTER_CHAIRMAN_MODEL: str = "anthropic/claude-opus-4"
    OPENROUTER_COUNCIL_MODELS: list[str] = Field(
        default=[
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4",
            "google/gemini-2.5-pro-preview",
            "x-ai/grok-2",
        ]
    )
    OPENROUTER_REVIEWER_MODEL: str = "anthropic/claude-sonnet-4"

    # Chairman (Claude Opus 4.5)
    CHAIRMAN_MODEL: str = "claude-opus-4-5-20251101"
    CHAIRMAN_EFFORT: Literal["low", "medium", "high"] = "high"

    # Stage 1 Council Members (Direct API용)
    GPT_MODEL: str = "gpt-5.1"
    GPT_REASONING_EFFORT: Literal["low", "medium", "high"] = "high"

    CLAUDE_SONNET_MODEL: str = "claude-sonnet-4-5-20250929"

    GEMINI_MODEL: str = "gemini-3-pro-preview"
    GEMINI_TEMPERATURE: float = 1.0

    GROK_MODEL: str = "grok-4-1-fast-reasoning"
    GROK_USE_REASONING: bool = True

    @field_validator("OPENROUTER_COUNCIL_MODELS", mode="before")
    @classmethod
    def parse_council_models(cls, v: str | list[str]) -> list[str]:
        """Parse council models from comma-separated string or list."""
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return v

    # =========================================================================
    # Storage (AWS S3)
    # =========================================================================
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET: str = "legal-council-documents"

    # =========================================================================
    # Email (SendGrid)
    # =========================================================================
    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@legalcouncil.ai"
    EMAIL_FROM_NAME: str = "AI 법률 자문 위원회"
    SUPPORT_EMAIL: str = "support@legalcouncil.ai"
    FRONTEND_URL: str = "http://localhost:3000"

    # =========================================================================
    # Payment (Stripe)
    # =========================================================================
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_PRO_MONTHLY: str = ""
    STRIPE_PRICE_ENTERPRISE_MONTHLY: str = ""

    # =========================================================================
    # Rate Limiting (requests per minute)
    # =========================================================================
    RATE_LIMIT_BASIC_RPM: int = 10
    RATE_LIMIT_PRO_RPM: int = 60
    RATE_LIMIT_ENTERPRISE_RPM: int = 200

    # Monthly consultation limits
    CONSULTATION_LIMIT_BASIC: int = 3
    CONSULTATION_LIMIT_PRO: int = -1  # Unlimited
    CONSULTATION_LIMIT_ENTERPRISE: int = -1  # Unlimited

    # =========================================================================
    # Logging (Component-level configuration)
    # =========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    # 컴포넌트별 로그 레벨 (기본값: LOG_LEVEL 사용)
    LOG_LEVEL_DATABASE: str = ""  # 빈 문자열이면 LOG_LEVEL 사용
    LOG_LEVEL_LLM: str = ""       # LLM 서비스 로깅
    LOG_LEVEL_CACHE: str = ""     # 캐시 로깅
    LOG_LEVEL_RAG: str = ""       # RAG 서비스 로깅
    LOG_LEVEL_AUTH: str = ""      # 인증 로깅

    # =========================================================================
    # Monitoring
    # =========================================================================
    SENTRY_DSN: str | None = None
    METRICS_ENABLED: bool = True
    METRICS_PREFIX: str = "legal_council"

    # =========================================================================
    # Cache Configuration (Dynamic TTL)
    # =========================================================================
    CACHE_DEFAULT_TTL: int = 3600  # 1시간
    CACHE_SESSION_TTL: int = 7200  # 2시간
    CACHE_RAG_TTL: int = 300  # 5분
    CACHE_MEMORY_TTL: int = 1800  # 30분
    CACHE_MAX_ENTRIES_PER_SESSION: int = 100

    # =========================================================================
    # Health Check
    # =========================================================================
    HEALTH_CHECK_TIMEOUT: int = 5  # seconds
    HEALTH_CHECK_INCLUDE_DETAILS: bool = True

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
