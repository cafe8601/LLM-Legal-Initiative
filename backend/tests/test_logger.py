"""
Logger Tests

로깅 모듈 테스트 - structlog 설정 및 컴포넌트별 로그 레벨 검증
"""

import logging
from unittest.mock import patch
import os

import pytest
import structlog

from app.utils.logger import (
    get_log_level,
    get_logger,
    log_context,
    clear_log_context,
    configure_logging,
)


class TestGetLogLevel:
    """get_log_level 함수 테스트"""

    def test_default_log_level(self):
        """기본 로그 레벨 테스트"""
        level = get_log_level()
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_component_log_level_database(self):
        """데이터베이스 컴포넌트 로그 레벨 테스트"""
        level = get_log_level("database")
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_component_log_level_llm(self):
        """LLM 컴포넌트 로그 레벨 테스트"""
        level = get_log_level("llm")
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_component_log_level_cache(self):
        """캐시 컴포넌트 로그 레벨 테스트"""
        level = get_log_level("cache")
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_component_log_level_rag(self):
        """RAG 컴포넌트 로그 레벨 테스트"""
        level = get_log_level("rag")
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_component_log_level_auth(self):
        """인증 컴포넌트 로그 레벨 테스트"""
        level = get_log_level("auth")
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_unknown_component_uses_default(self):
        """알 수 없는 컴포넌트는 기본 레벨 사용 테스트"""
        default_level = get_log_level()
        unknown_level = get_log_level("unknown_component")
        # 알 수 없는 컴포넌트는 기본 레벨을 반환해야 함
        assert unknown_level == default_level

    def test_case_insensitive_component(self):
        """컴포넌트 이름 대소문자 구분 없음 테스트"""
        level_lower = get_log_level("database")
        level_upper = get_log_level("DATABASE")
        level_mixed = get_log_level("DataBase")
        # 모두 같은 레벨을 반환해야 함
        assert level_lower == level_upper == level_mixed


class TestGetLogger:
    """get_logger 함수 테스트"""

    def test_get_logger_returns_bound_logger(self):
        """get_logger가 BoundLogger 반환 테스트"""
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_with_name(self):
        """이름이 있는 로거 테스트"""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_without_name(self):
        """이름이 없는 로거 테스트"""
        logger = get_logger()
        assert logger is not None

    def test_logger_can_log(self):
        """로거가 로그 메시지를 출력할 수 있는지 테스트"""
        logger = get_logger("test")
        # 예외 없이 로그가 출력되어야 함
        try:
            logger.info("Test message")
            logger.debug("Debug message")
            logger.warning("Warning message")
            logger.error("Error message")
        except Exception as e:
            pytest.fail(f"Logger failed to log: {e}")


class TestLogContext:
    """log_context 함수 테스트"""

    def test_log_context_adds_variables(self):
        """log_context가 컨텍스트 변수를 추가하는지 테스트"""
        clear_log_context()  # 초기화
        log_context(request_id="test-123", user_id="user-456")
        # structlog 컨텍스트에서 변수 확인
        ctx_vars = structlog.contextvars.get_contextvars()
        assert ctx_vars.get("request_id") == "test-123"
        assert ctx_vars.get("user_id") == "user-456"
        clear_log_context()  # 정리

    def test_log_context_multiple_calls(self):
        """log_context 여러 번 호출 테스트"""
        clear_log_context()
        log_context(var1="value1")
        log_context(var2="value2")
        ctx_vars = structlog.contextvars.get_contextvars()
        assert ctx_vars.get("var1") == "value1"
        assert ctx_vars.get("var2") == "value2"
        clear_log_context()


class TestClearLogContext:
    """clear_log_context 함수 테스트"""

    def test_clear_log_context_removes_all(self):
        """clear_log_context가 모든 변수를 제거하는지 테스트"""
        log_context(test_var="test_value")
        clear_log_context()
        ctx_vars = structlog.contextvars.get_contextvars()
        assert ctx_vars.get("test_var") is None


class TestConfigureLogging:
    """configure_logging 함수 테스트"""

    def test_configure_logging_runs_without_error(self):
        """configure_logging이 에러 없이 실행되는지 테스트"""
        try:
            configure_logging()
        except Exception as e:
            pytest.fail(f"configure_logging failed: {e}")

    def test_configure_logging_with_json_format(self):
        """JSON 포맷 설정 테스트"""
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            try:
                configure_logging()
            except Exception as e:
                pytest.fail(f"configure_logging with JSON format failed: {e}")

    def test_configure_logging_with_console_format(self):
        """콘솔 포맷 설정 테스트"""
        with patch.dict(os.environ, {"LOG_FORMAT": "console"}):
            try:
                configure_logging()
            except Exception as e:
                pytest.fail(f"configure_logging with console format failed: {e}")


class TestLogLevelMapping:
    """로그 레벨 매핑 테스트"""

    def test_debug_level_value(self):
        """DEBUG 레벨 값 테스트"""
        assert logging.DEBUG == 10

    def test_info_level_value(self):
        """INFO 레벨 값 테스트"""
        assert logging.INFO == 20

    def test_warning_level_value(self):
        """WARNING 레벨 값 테스트"""
        assert logging.WARNING == 30

    def test_error_level_value(self):
        """ERROR 레벨 값 테스트"""
        assert logging.ERROR == 40

    def test_critical_level_value(self):
        """CRITICAL 레벨 값 테스트"""
        assert logging.CRITICAL == 50
