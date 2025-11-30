"""
Custom Application Exceptions

애플리케이션 전용 예외 클래스 정의
"""

from typing import Any

from fastapi import status


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        detail: str,
        error_code: str = "APP_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        self.headers = headers
        super().__init__(detail)


# =============================================================================
# Authentication Exceptions
# =============================================================================


class AuthenticationError(AppException):
    """Base authentication error."""

    def __init__(
        self,
        detail: str = "Authentication failed",
        error_code: str = "AUTH_ERROR",
    ) -> None:
        super().__init__(
            detail=detail,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )


class InvalidCredentialsError(AuthenticationError):
    """Invalid login credentials."""

    def __init__(self, detail: str = "Invalid email or password") -> None:
        super().__init__(detail=detail, error_code="INVALID_CREDENTIALS")


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""

    def __init__(self, detail: str = "Token has expired") -> None:
        super().__init__(detail=detail, error_code="TOKEN_EXPIRED")


class InvalidTokenError(AuthenticationError):
    """JWT token is invalid."""

    def __init__(self, detail: str = "Invalid token") -> None:
        super().__init__(detail=detail, error_code="INVALID_TOKEN")


class UserNotVerifiedError(AuthenticationError):
    """User email not verified."""

    def __init__(self, detail: str = "Please verify your email address") -> None:
        super().__init__(detail=detail, error_code="USER_NOT_VERIFIED")


# =============================================================================
# Authorization Exceptions
# =============================================================================


class AuthorizationError(AppException):
    """Base authorization error."""

    def __init__(
        self,
        detail: str = "Access denied",
        error_code: str = "FORBIDDEN",
    ) -> None:
        super().__init__(
            detail=detail,
            error_code=error_code,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class InsufficientTierError(AuthorizationError):
    """User tier is insufficient for this operation."""

    def __init__(self, required_tier: str, detail: str | None = None) -> None:
        super().__init__(
            detail=detail or f"This feature requires {required_tier} tier or higher",
            error_code="INSUFFICIENT_TIER",
        )


class RateLimitExceededError(AuthorizationError):
    """Rate limit exceeded."""

    def __init__(self, detail: str = "Rate limit exceeded. Please try again later.") -> None:
        super().__init__(
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
        )
        self.status_code = status.HTTP_429_TOO_MANY_REQUESTS


class ConsultationLimitExceededError(AuthorizationError):
    """Monthly consultation limit exceeded."""

    def __init__(
        self, limit: int, detail: str | None = None
    ) -> None:
        super().__init__(
            detail=detail or f"Monthly consultation limit ({limit}) reached. Upgrade to continue.",
            error_code="CONSULTATION_LIMIT_EXCEEDED",
        )


# =============================================================================
# Resource Exceptions
# =============================================================================


class NotFoundError(AppException):
    """Resource not found."""

    def __init__(
        self,
        resource: str = "Resource",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            detail=detail or f"{resource} not found",
            error_code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class UserNotFoundError(NotFoundError):
    """User not found."""

    def __init__(self, detail: str = "User not found") -> None:
        super().__init__(resource="User", detail=detail)


class ConsultationNotFoundError(NotFoundError):
    """Consultation not found."""

    def __init__(self, detail: str = "Consultation not found") -> None:
        super().__init__(resource="Consultation", detail=detail)


class DocumentNotFoundError(NotFoundError):
    """Document not found."""

    def __init__(self, detail: str = "Document not found") -> None:
        super().__init__(resource="Document", detail=detail)


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(AppException):
    """Validation error."""

    def __init__(
        self,
        detail: str = "Validation failed",
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            detail=detail,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        self.errors = errors or []


class UserAlreadyExistsError(ValidationError):
    """User with this email already exists."""

    def __init__(self, detail: str = "User with this email already exists") -> None:
        super().__init__(detail=detail)
        self.error_code = "USER_EXISTS"
        self.status_code = status.HTTP_409_CONFLICT


# =============================================================================
# LLM Service Exceptions
# =============================================================================


class LLMServiceError(AppException):
    """Base LLM service error."""

    def __init__(
        self,
        detail: str = "LLM service error",
        error_code: str = "LLM_ERROR",
        model: str | None = None,
    ) -> None:
        super().__init__(
            detail=detail,
            error_code=error_code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.model = model


class LLMRateLimitError(LLMServiceError):
    """LLM API rate limit exceeded."""

    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            detail=detail or f"{model} API rate limit exceeded",
            error_code="LLM_RATE_LIMIT",
            model=model,
        )


class LLMTimeoutError(LLMServiceError):
    """LLM API timeout."""

    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            detail=detail or f"{model} API request timed out",
            error_code="LLM_TIMEOUT",
            model=model,
        )


class LLMUnavailableError(LLMServiceError):
    """LLM service unavailable."""

    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            detail=detail or f"{model} service is currently unavailable",
            error_code="LLM_UNAVAILABLE",
            model=model,
        )


# =============================================================================
# RAG Service Exceptions
# =============================================================================


class RAGServiceError(AppException):
    """Base RAG service error."""

    def __init__(
        self,
        detail: str = "RAG service error",
        error_code: str = "RAG_ERROR",
    ) -> None:
        super().__init__(
            detail=detail,
            error_code=error_code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class RAGSearchError(RAGServiceError):
    """RAG search failed."""

    def __init__(self, detail: str = "Legal document search failed") -> None:
        super().__init__(detail=detail, error_code="RAG_SEARCH_ERROR")


class RAGIngestionError(RAGServiceError):
    """RAG document ingestion failed."""

    def __init__(self, detail: str = "Document ingestion failed") -> None:
        super().__init__(detail=detail, error_code="RAG_INGESTION_ERROR")


# =============================================================================
# External Service Exceptions
# =============================================================================


class ExternalServiceError(AppException):
    """External service error."""

    def __init__(
        self,
        service: str,
        detail: str | None = None,
    ) -> None:
        super().__init__(
            detail=detail or f"{service} service error",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.service = service


class StorageServiceError(ExternalServiceError):
    """Storage (S3) service error."""

    def __init__(self, detail: str = "Storage service error") -> None:
        super().__init__(service="Storage", detail=detail)


class EmailServiceError(ExternalServiceError):
    """Email service error."""

    def __init__(self, detail: str = "Email service error") -> None:
        super().__init__(service="Email", detail=detail)


class PaymentServiceError(ExternalServiceError):
    """Payment (Stripe) service error."""

    def __init__(self, detail: str = "Payment service error") -> None:
        super().__init__(service="Payment", detail=detail)


# =============================================================================
# Document Exceptions
# =============================================================================


class DocumentProcessingError(AppException):
    """Document processing failed."""

    def __init__(self, detail: str = "Document processing failed") -> None:
        super().__init__(
            detail=detail,
            error_code="DOCUMENT_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class InvalidFileTypeError(ValidationError):
    """Invalid file type uploaded."""

    def __init__(self, detail: str = "Invalid file type") -> None:
        super().__init__(detail=detail)
        self.error_code = "INVALID_FILE_TYPE"


class FileSizeExceededError(ValidationError):
    """File size exceeds limit."""

    def __init__(self, detail: str = "File size exceeds limit") -> None:
        super().__init__(detail=detail)
        self.error_code = "FILE_SIZE_EXCEEDED"
        self.status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class StorageQuotaExceededError(AuthorizationError):
    """Storage quota exceeded."""

    def __init__(self, detail: str = "Storage quota exceeded") -> None:
        super().__init__(
            detail=detail,
            error_code="STORAGE_QUOTA_EXCEEDED",
        )


# =============================================================================
# General Exceptions
# =============================================================================


class UnauthorizedError(AuthorizationError):
    """Unauthorized access."""

    def __init__(self, detail: str = "Unauthorized access") -> None:
        super().__init__(detail=detail, error_code="UNAUTHORIZED")
