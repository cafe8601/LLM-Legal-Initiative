"""
Security Module

JWT 인증 및 비밀번호 해싱 유틸리티
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings

# Password hashing context
# bcrypt 4.x 사용 - 72바이트 제한이 있으나 대부분의 실제 비밀번호에 충분
# 참고: bcrypt 5.x는 passlib과 호환성 문제가 있어 4.x 사용
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    """JWT token payload schema."""

    sub: str  # user_id
    exp: datetime
    type: str  # "access" or "refresh"
    tier: str | None = None  # user tier for quick permission checks


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# =============================================================================
# Password Utilities
# =============================================================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


# =============================================================================
# JWT Token Utilities
# =============================================================================


def create_access_token(
    user_id: UUID,
    tier: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User's UUID
        tier: User's subscription tier for quick permission checks
        expires_delta: Custom expiration time

    Returns:
        Encoded JWT access token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode: dict[str, Any] = {
        "sub": str(user_id),
        "exp": expire,
        "type": "access",
    }

    if tier:
        to_encode["tier"] = tier

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    user_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User's UUID
        expires_delta: Custom expiration time

    Returns:
        Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode: dict[str, Any] = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_token_pair(user_id: UUID, tier: str | None = None) -> TokenPair:
    """
    Create both access and refresh tokens.

    Args:
        user_id: User's UUID
        tier: User's subscription tier

    Returns:
        TokenPair with access_token and refresh_token
    """
    access_token = create_access_token(user_id, tier)
    refresh_token = create_refresh_token(user_id)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


def decode_token(token: str) -> TokenPayload | None:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenPayload if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return TokenPayload(**payload)
    except JWTError:
        return None


def decode_access_token(token: str) -> TokenPayload | None:
    """
    Decode and validate an access token.

    Args:
        token: JWT access token string

    Returns:
        TokenPayload if valid access token, None otherwise
    """
    payload = decode_token(token)
    if payload and payload.type == "access":
        return payload
    return None


def decode_refresh_token(token: str) -> TokenPayload | None:
    """
    Decode and validate a refresh token.

    Args:
        token: JWT refresh token string

    Returns:
        TokenPayload if valid refresh token, None otherwise
    """
    payload = decode_token(token)
    if payload and payload.type == "refresh":
        return payload
    return None


# =============================================================================
# Utility Tokens (Email verification, Password reset)
# =============================================================================


def create_verification_token(user_id: UUID, expires_hours: int = 24) -> str:
    """Create an email verification token."""
    expire = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "type": "email_verification",
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_password_reset_token(user_id: UUID, expires_hours: int = 1) -> str:
    """Create a password reset token."""
    expire = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "type": "password_reset",
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_verification_token(token: str) -> str | None:
    """
    Decode an email verification token.

    Returns:
        User ID string if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        if payload.get("type") == "email_verification":
            return payload.get("sub")
    except JWTError:
        pass
    return None


def decode_password_reset_token(token: str) -> str | None:
    """
    Decode a password reset token.

    Returns:
        User ID string if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        if payload.get("type") == "password_reset":
            return payload.get("sub")
    except JWTError:
        pass
    return None
