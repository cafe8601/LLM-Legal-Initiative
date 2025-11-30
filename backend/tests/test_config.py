"""
Configuration Tests

설정 모듈 테스트 - 환경 변수 로딩 및 기본값 검증
"""

import os
from unittest.mock import patch

import pytest

from app.core.config import Settings, settings


class TestSettings:
    """Settings 클래스 테스트"""

    def test_default_settings(self):
        """기본 설정값 테스트"""
        assert settings.APP_NAME is not None
        assert len(settings.APP_NAME) > 0
        assert settings.ALGORITHM == "HS256"

    def test_environment_setting(self):
        """환경 설정 기본값 테스트"""
        assert settings.ENVIRONMENT in ["development", "staging", "production"]
        assert settings.DEBUG in [True, False]

    def test_database_url_format(self):
        """데이터베이스 URL 형식 테스트"""
        db_url = settings.DATABASE_URL
        # PostgreSQL URL 형식 확인
        assert "postgresql" in db_url or "sqlite" in db_url

    def test_cache_ttl_settings(self):
        """캐시 TTL 설정 테스트"""
        assert settings.CACHE_DEFAULT_TTL > 0
        assert settings.CACHE_SESSION_TTL > 0
        assert settings.CACHE_RAG_TTL > 0
        assert settings.CACHE_MEMORY_TTL > 0
        assert settings.CACHE_MAX_ENTRIES_PER_SESSION > 0

    def test_health_check_settings(self):
        """헬스체크 설정 테스트"""
        assert settings.HEALTH_CHECK_TIMEOUT > 0
        assert isinstance(settings.HEALTH_CHECK_INCLUDE_DETAILS, bool)

    def test_metrics_settings(self):
        """메트릭 설정 테스트"""
        assert isinstance(settings.METRICS_ENABLED, bool)
        assert settings.METRICS_PREFIX != ""

    def test_log_level_settings(self):
        """로그 레벨 설정 테스트"""
        valid_levels = ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.LOG_LEVEL in valid_levels or settings.LOG_LEVEL.upper() in valid_levels
        # 컴포넌트별 로그 레벨은 빈 문자열이거나 유효한 레벨이어야 함
        for level in [
            settings.LOG_LEVEL_DATABASE,
            settings.LOG_LEVEL_LLM,
            settings.LOG_LEVEL_CACHE,
            settings.LOG_LEVEL_RAG,
            settings.LOG_LEVEL_AUTH,
        ]:
            assert level == "" or level.upper() in valid_levels

    def test_token_expire_settings(self):
        """토큰 만료 설정 테스트"""
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0
        assert settings.REFRESH_TOKEN_EXPIRE_DAYS > 0
        # Refresh 토큰이 Access 토큰보다 더 오래 유지되어야 함
        refresh_minutes = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60
        assert refresh_minutes > settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def test_secret_key_exists(self):
        """시크릿 키 존재 테스트"""
        assert settings.SECRET_KEY is not None
        assert len(settings.SECRET_KEY) > 10  # 최소 길이 체크


class TestSettingsEnvironmentOverride:
    """환경 변수 오버라이드 테스트"""

    def test_cache_ttl_override(self):
        """캐시 TTL 환경 변수 오버라이드 테스트"""
        with patch.dict(os.environ, {"CACHE_DEFAULT_TTL": "1800"}):
            test_settings = Settings()
            assert test_settings.CACHE_DEFAULT_TTL == 1800

    def test_metrics_enabled_override(self):
        """메트릭 활성화 환경 변수 오버라이드 테스트"""
        with patch.dict(os.environ, {"METRICS_ENABLED": "false"}):
            test_settings = Settings()
            assert test_settings.METRICS_ENABLED is False

    def test_health_check_timeout_override(self):
        """헬스체크 타임아웃 환경 변수 오버라이드 테스트"""
        with patch.dict(os.environ, {"HEALTH_CHECK_TIMEOUT": "10"}):
            test_settings = Settings()
            assert test_settings.HEALTH_CHECK_TIMEOUT == 10

    def test_log_format_override(self):
        """로그 포맷 환경 변수 오버라이드 테스트"""
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            test_settings = Settings()
            assert test_settings.LOG_FORMAT == "json"


class TestSettingsValidation:
    """설정 유효성 검증 테스트"""

    def test_tier_settings(self):
        """사용자 티어 설정 테스트"""
        # 티어 관련 설정이 있다면 검증
        # 기본 설정에서 티어 제한이 합리적인지 확인
        pass  # 티어 설정이 Settings에 있다면 추가

    def test_cors_origins(self):
        """CORS 오리진 설정 테스트"""
        # CORS 설정이 있다면 검증
        if hasattr(settings, "CORS_ORIGINS"):
            assert isinstance(settings.CORS_ORIGINS, list)
