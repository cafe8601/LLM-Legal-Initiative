"""
API Dependencies

의존성 주입 함수
"""

from typing import Annotated, AsyncGenerator
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.exceptions import (
    AuthenticationError,
    InsufficientTierError,
    InvalidTokenError,
    RateLimitExceededError,
    ConsultationLimitExceededError,
)
from app.db.session import async_session_factory
from app.models.user import User
from app.repositories.user import UserRepository

# Security scheme
security = HTTPBearer(auto_error=False)
security_required = HTTPBearer(auto_error=True)


# =============================================================================
# Database Dependencies
# =============================================================================


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session.

    Yields:
        AsyncSession: SQLAlchemy async session
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# Redis Dependencies
# =============================================================================

# Global Redis client (initialized on first use)
_redis_client = None


async def get_redis():
    """
    Get Redis client instance.

    Returns:
        Redis: Redis async client (singleton)
    """
    global _redis_client

    if _redis_client is None:
        from redis.asyncio import Redis

        _redis_client = Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )

    return _redis_client


async def get_redis_dependency():
    """
    Get Redis client as a dependency (for FastAPI Depends).

    Yields:
        Redis: Redis async client
    """
    redis = await get_redis()
    yield redis


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# =============================================================================
# Authentication Dependencies
# =============================================================================


async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_required)],
) -> UUID:
    """
    Extract and validate user ID from JWT token.

    Returns:
        UUID: Current user's ID

    Raises:
        InvalidTokenError: If token is invalid or expired
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    if not payload:
        raise InvalidTokenError("유효하지 않거나 만료된 토큰입니다")

    try:
        return UUID(payload.sub)
    except ValueError:
        raise InvalidTokenError("잘못된 토큰 형식입니다")


async def get_current_user(
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from database.

    Returns:
        User: Current user model

    Raises:
        AuthenticationError: If user not found or inactive
    """
    user_repo = UserRepository(db)
    user = await user_repo.get(user_id)

    if not user:
        raise AuthenticationError("사용자를 찾을 수 없습니다")

    if not user.is_active:
        raise AuthenticationError("비활성화된 계정입니다")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Ensure current user is active and verified.

    Returns:
        User: Current active user

    Raises:
        AuthenticationError: If user is inactive
    """
    if not current_user.is_active:
        raise AuthenticationError("비활성화된 계정입니다")

    if not current_user.is_verified:
        raise AuthenticationError("이메일 인증이 필요합니다")

    return current_user


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Get current user if authenticated, otherwise return None.

    Returns:
        User | None: Current user or None
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = decode_access_token(token)

    if not payload:
        return None

    try:
        user_id = UUID(payload.sub)
    except ValueError:
        return None

    user_repo = UserRepository(db)
    user = await user_repo.get(user_id)

    if not user or not user.is_active:
        return None

    return user


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """
    Ensure current user is an admin.

    Returns:
        User: Current admin user

    Raises:
        AuthenticationError: If user is not admin
    """
    if not current_user.is_admin:
        raise AuthenticationError("관리자 권한이 필요합니다")
    return current_user


# =============================================================================
# Tier-based Authorization Dependencies
# =============================================================================


def require_tier(allowed_tiers: list[str]):
    """
    Create a dependency that requires specific subscription tiers.

    Args:
        allowed_tiers: List of allowed tier names (e.g., ["pro", "enterprise"])

    Returns:
        Dependency function that checks user tier

    Usage:
        @router.get("/premium-feature")
        async def premium_feature(user = Depends(require_tier(["pro", "enterprise"]))):
            ...
    """

    async def tier_checker(
        current_user: Annotated[User, Depends(get_current_active_user)],
    ) -> User:
        if current_user.tier not in allowed_tiers:
            raise InsufficientTierError(
                required_tier=", ".join(allowed_tiers),
            )
        return current_user

    return tier_checker


# Convenience dependencies for common tier requirements
def require_pro():
    """Require Pro or Enterprise tier."""
    return require_tier(["pro", "enterprise"])


def require_enterprise():
    """Require Enterprise tier."""
    return require_tier(["enterprise"])


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================


# Tier-based rate limits (requests per minute)
TIER_RATE_LIMITS = {
    "basic": settings.RATE_LIMIT_BASIC_RPM,
    "pro": settings.RATE_LIMIT_PRO_RPM,
    "enterprise": settings.RATE_LIMIT_ENTERPRISE_RPM,
}

# Tier-based consultation limits (per month)
TIER_CONSULTATION_LIMITS = {
    "basic": settings.CONSULTATION_LIMIT_BASIC,
    "pro": settings.CONSULTATION_LIMIT_PRO,  # -1 = unlimited
    "enterprise": settings.CONSULTATION_LIMIT_ENTERPRISE,  # -1 = unlimited
}


async def check_rate_limit(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> User:
    """
    Check if user has exceeded rate limit.

    Returns:
        User: Current user if within rate limit

    Raises:
        RateLimitExceededError: If rate limit exceeded
    """
    # Get endpoint for rate limiting key
    endpoint = request.url.path

    # Get user's rate limit based on tier
    limit = TIER_RATE_LIMITS.get(current_user.tier, settings.RATE_LIMIT_BASIC_RPM)

    # Create rate limit key
    key = f"rate_limit:{current_user.id}:{endpoint}"

    # Increment counter
    current = await redis.incr(key)

    # Set expiry on first request
    if current == 1:
        await redis.expire(key, 60)  # 1 minute window

    # Check if limit exceeded
    if current > limit:
        raise RateLimitExceededError(
            f"요청 한도 초과 (분당 {limit}회). 잠시 후 다시 시도해주세요."
        )

    return current_user


async def check_consultation_limit(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Check if user has exceeded monthly consultation limit.

    Returns:
        User: Current user if within limit

    Raises:
        ConsultationLimitExceededError: If limit exceeded
    """
    from app.repositories.consultation import ConsultationRepository

    # Get user's monthly limit
    limit = TIER_CONSULTATION_LIMITS.get(current_user.tier, settings.CONSULTATION_LIMIT_BASIC)

    # Unlimited tiers
    if limit < 0:
        return current_user

    # Check current month's consultation count
    consultation_repo = ConsultationRepository(db)
    current_count = await consultation_repo.count_user_consultations_this_month(current_user.id)

    if current_count >= limit:
        raise ConsultationLimitExceededError(
            f"이번 달 상담 한도({limit}건)에 도달했습니다. 업그레이드하여 무제한으로 이용하세요."
        )

    return current_user


# =============================================================================
# Type Aliases for cleaner endpoint signatures
# =============================================================================

# Usage: async def endpoint(user: CurrentUser):
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]
CurrentAdminUser = Annotated[User, Depends(get_current_admin_user)]
CurrentUserOptional = Annotated[User | None, Depends(get_current_user_optional)]
CurrentUserId = Annotated[UUID, Depends(get_current_user_id)]

# Rate-limited user
RateLimitedUser = Annotated[User, Depends(check_rate_limit)]

# Consultation-limited user
ConsultationLimitedUser = Annotated[User, Depends(check_consultation_limit)]

# Database session
DBSession = Annotated[AsyncSession, Depends(get_db)]
