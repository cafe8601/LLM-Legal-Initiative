"""
Rate Limiting Module

티어별 요청 제한 및 상담 제한 관리
"""

import logging
from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


# =============================================================================
# Tier Limits Configuration
# =============================================================================


class TierLimits:
    """Rate limit configuration by tier."""

    LIMITS = {
        "basic": {
            "requests_per_minute": settings.RATE_LIMIT_BASIC_RPM,
            "requests_per_hour": settings.RATE_LIMIT_BASIC_RPM * 30,
            "consultations_per_month": settings.CONSULTATION_LIMIT_BASIC,
            "documents_per_month": 10,
            "max_document_size_mb": 5,
        },
        "pro": {
            "requests_per_minute": settings.RATE_LIMIT_PRO_RPM,
            "requests_per_hour": settings.RATE_LIMIT_PRO_RPM * 30,
            "consultations_per_month": settings.CONSULTATION_LIMIT_PRO,  # -1 = unlimited
            "documents_per_month": 100,
            "max_document_size_mb": 20,
        },
        "enterprise": {
            "requests_per_minute": settings.RATE_LIMIT_ENTERPRISE_RPM,
            "requests_per_hour": settings.RATE_LIMIT_ENTERPRISE_RPM * 30,
            "consultations_per_month": settings.CONSULTATION_LIMIT_ENTERPRISE,  # -1 = unlimited
            "documents_per_month": -1,  # unlimited
            "max_document_size_mb": 50,
        },
    }

    @classmethod
    def get_limit(cls, tier: str, limit_type: str) -> int:
        """Get specific limit for a tier."""
        tier_limits = cls.LIMITS.get(tier, cls.LIMITS["basic"])
        return tier_limits.get(limit_type, 0)

    @classmethod
    def is_unlimited(cls, limit: int) -> bool:
        """Check if limit is unlimited (-1)."""
        return limit < 0


# =============================================================================
# Rate Limiter Implementation
# =============================================================================


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        user_id: UUID | str,
        tier: str,
        endpoint: str = "default",
        window_seconds: int = 60,
    ) -> tuple[bool, int, int]:
        """
        Check if user is within rate limit.

        Args:
            user_id: User identifier
            tier: User's subscription tier
            endpoint: API endpoint being accessed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, current_count, limit)

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        limit = TierLimits.get_limit(tier, "requests_per_minute")

        if TierLimits.is_unlimited(limit):
            return True, 0, -1

        key = f"rate_limit:{user_id}:{endpoint}"

        # Increment counter
        current = await self.redis.incr(key)

        # Set expiry on first request
        if current == 1:
            await self.redis.expire(key, window_seconds)

        # Check limit
        if current > limit:
            ttl = await self.redis.ttl(key)
            raise RateLimitExceededError(
                f"요청 한도 초과 (분당 {limit}회). {ttl}초 후 다시 시도해주세요."
            )

        return True, current, limit

    async def check_hourly_limit(
        self,
        user_id: UUID | str,
        tier: str,
    ) -> tuple[bool, int, int]:
        """Check hourly rate limit."""
        limit = TierLimits.get_limit(tier, "requests_per_hour")

        if TierLimits.is_unlimited(limit):
            return True, 0, -1

        key = f"rate_limit_hour:{user_id}"
        current = await self.redis.incr(key)

        if current == 1:
            await self.redis.expire(key, 3600)  # 1 hour

        if current > limit:
            ttl = await self.redis.ttl(key)
            raise RateLimitExceededError(
                f"시간당 요청 한도 초과. {ttl // 60}분 후 다시 시도해주세요."
            )

        return True, current, limit

    async def get_remaining_requests(
        self,
        user_id: UUID | str,
        tier: str,
        endpoint: str = "default",
    ) -> dict:
        """Get remaining requests info."""
        minute_limit = TierLimits.get_limit(tier, "requests_per_minute")
        hour_limit = TierLimits.get_limit(tier, "requests_per_hour")

        minute_key = f"rate_limit:{user_id}:{endpoint}"
        hour_key = f"rate_limit_hour:{user_id}"

        minute_used = int(await self.redis.get(minute_key) or 0)
        hour_used = int(await self.redis.get(hour_key) or 0)

        minute_ttl = await self.redis.ttl(minute_key)
        hour_ttl = await self.redis.ttl(hour_key)

        return {
            "minute": {
                "limit": minute_limit,
                "used": minute_used,
                "remaining": max(0, minute_limit - minute_used) if minute_limit > 0 else -1,
                "reset_in_seconds": max(0, minute_ttl),
            },
            "hour": {
                "limit": hour_limit,
                "used": hour_used,
                "remaining": max(0, hour_limit - hour_used) if hour_limit > 0 else -1,
                "reset_in_seconds": max(0, hour_ttl),
            },
        }

    async def reset_user_limits(self, user_id: UUID | str) -> None:
        """Reset all rate limits for a user (admin function)."""
        pattern = f"rate_limit*:{user_id}*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)


# =============================================================================
# Consultation Limiter
# =============================================================================


class ConsultationLimiter:
    """
    Monthly consultation limit checker.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    def _get_month_key(self) -> str:
        """Get current month key."""
        now = datetime.now(timezone.utc)
        return f"{now.year}:{now.month:02d}"

    async def check_consultation_limit(
        self,
        user_id: UUID | str,
        tier: str,
    ) -> tuple[bool, int, int]:
        """
        Check if user can create new consultation.

        Returns:
            Tuple of (is_allowed, current_count, limit)
        """
        limit = TierLimits.get_limit(tier, "consultations_per_month")

        if TierLimits.is_unlimited(limit):
            return True, 0, -1

        month_key = self._get_month_key()
        key = f"consultations:{user_id}:{month_key}"

        current = int(await self.redis.get(key) or 0)

        if current >= limit:
            return False, current, limit

        return True, current, limit

    async def increment_consultation_count(
        self,
        user_id: UUID | str,
    ) -> int:
        """Increment consultation count for current month."""
        month_key = self._get_month_key()
        key = f"consultations:{user_id}:{month_key}"

        count = await self.redis.incr(key)

        # Set expiry to end of month + 1 day buffer
        if count == 1:
            now = datetime.now(timezone.utc)
            # Days until end of month + 1
            import calendar

            days_in_month = calendar.monthrange(now.year, now.month)[1]
            days_remaining = days_in_month - now.day + 1
            await self.redis.expire(key, days_remaining * 24 * 60 * 60)

        return count

    async def get_consultation_stats(
        self,
        user_id: UUID | str,
        tier: str,
    ) -> dict:
        """Get consultation statistics for user."""
        limit = TierLimits.get_limit(tier, "consultations_per_month")
        month_key = self._get_month_key()
        key = f"consultations:{user_id}:{month_key}"

        current = int(await self.redis.get(key) or 0)

        return {
            "month": month_key,
            "limit": limit,
            "used": current,
            "remaining": max(0, limit - current) if limit > 0 else -1,
            "is_unlimited": TierLimits.is_unlimited(limit),
        }


# =============================================================================
# Rate Limit Middleware
# =============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global rate limiting.

    Applied to all requests, extracts user from JWT if present.
    """

    # Endpoints to skip rate limiting
    SKIP_PATHS = {
        "/health",
        "/health/ready",
        "/health/live",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip certain paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Check for Authorization header
        auth_header = request.headers.get("Authorization")

        if auth_header and auth_header.startswith("Bearer "):
            # Try to extract user info from token
            try:
                from app.core.security import decode_access_token

                token = auth_header.split(" ")[1]
                payload = decode_access_token(token)

                if payload:
                    # Get Redis from app state
                    redis = getattr(request.app.state, "redis", None)

                    if redis:
                        rate_limiter = RateLimiter(redis)
                        tier = payload.tier or "basic"

                        # Check rate limit
                        try:
                            await rate_limiter.check_rate_limit(
                                user_id=payload.sub,
                                tier=tier,
                                endpoint=request.url.path,
                            )
                        except RateLimitExceededError as e:
                            return self._rate_limit_response(str(e))

            except Exception as e:
                logger.warning(f"Rate limit middleware error: {e}")

        # Continue with request
        response = await call_next(request)

        return response

    def _rate_limit_response(self, detail: str):
        """Create rate limit exceeded response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": detail,
                "error": "rate_limit_exceeded",
            },
            headers={
                "Retry-After": "60",
            },
        )


# =============================================================================
# Utility Functions
# =============================================================================


def get_rate_limit_headers(
    limit: int,
    remaining: int,
    reset_seconds: int,
) -> dict[str, str]:
    """Get standard rate limit headers for response."""
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(reset_seconds),
    }
