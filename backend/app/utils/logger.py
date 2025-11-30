"""
Logging Configuration

structlog 기반 구조화 로깅 설정
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from app.core.config import settings


def get_log_level(component: str | None = None) -> int:
    """
    Get log level from settings.

    Args:
        component: Optional component name for component-specific log level.
                   Supported: 'database', 'llm', 'cache', 'rag', 'auth'

    Returns:
        Log level integer
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # 컴포넌트별 로그 레벨 체크
    if component:
        component_level_map = {
            "database": settings.LOG_LEVEL_DATABASE,
            "llm": settings.LOG_LEVEL_LLM,
            "cache": settings.LOG_LEVEL_CACHE,
            "rag": settings.LOG_LEVEL_RAG,
            "auth": settings.LOG_LEVEL_AUTH,
        }
        component_level = component_level_map.get(component.lower(), "")
        if component_level:
            return level_map.get(component_level.upper(), logging.INFO)

    return level_map.get(settings.LOG_LEVEL.upper(), logging.INFO)


def configure_logging() -> None:
    """Configure structlog for the application."""

    # Common processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.LOG_FORMAT == "json":
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(get_log_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=get_log_level(),
    )


# Configure logging on module import
configure_logging()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog BoundLogger
    """
    return structlog.get_logger(name)


def log_context(**kwargs: Any) -> None:
    """
    Add context variables to all subsequent log messages.

    Usage:
        log_context(request_id="abc123", user_id="user_456")
        logger.info("Processing request")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_log_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()
