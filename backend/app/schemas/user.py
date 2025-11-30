"""
User Schemas

사용자 관련 요청/응답 스키마
"""

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator
import re

from app.schemas.common import BaseSchema, TimestampSchema


class UserTier(str, Enum):
    """User subscription tiers."""

    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# ============================================================================
# User Schemas
# ============================================================================


class UserBase(BaseSchema):
    """Base user schema."""

    email: EmailStr
    full_name: str = Field(min_length=2, max_length=100)
    phone: str | None = Field(default=None, max_length=20)
    company: str | None = Field(default=None, max_length=100)


class UserCreate(UserBase):
    """User registration schema."""

    password: str = Field(min_length=8, max_length=100)
    confirm_password: str = Field(min_length=8, max_length=100)
    terms_accepted: bool = Field(default=False)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("비밀번호에 대문자가 포함되어야 합니다")
        if not re.search(r"[a-z]", v):
            raise ValueError("비밀번호에 소문자가 포함되어야 합니다")
        if not re.search(r"\d", v):
            raise ValueError("비밀번호에 숫자가 포함되어야 합니다")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("비밀번호에 특수문자가 포함되어야 합니다")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Ensure passwords match."""
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("비밀번호가 일치하지 않습니다")
        return v

    @field_validator("terms_accepted")
    @classmethod
    def must_accept_terms(cls, v: bool) -> bool:
        """Ensure terms are accepted."""
        if not v:
            raise ValueError("이용약관에 동의해야 합니다")
        return v


class UserUpdate(BaseSchema):
    """User profile update schema."""

    full_name: str | None = Field(default=None, min_length=2, max_length=100)
    phone: str | None = Field(default=None, max_length=20)
    company: str | None = Field(default=None, max_length=100)
    preferred_language: str | None = Field(default=None, max_length=10)
    notification_email: bool | None = None


class UserResponse(TimestampSchema):
    """User response schema."""

    id: UUID
    email: EmailStr
    full_name: str
    phone: str | None
    company: str | None
    tier: UserTier
    is_active: bool
    is_verified: bool
    consultation_count_this_month: int
    preferred_language: str
    notification_email: bool


class UserInDB(UserResponse):
    """User schema with sensitive fields (internal use only)."""

    password_hash: str
    is_admin: bool
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    last_consultation_reset: datetime | None


class UserStats(BaseSchema):
    """User statistics schema."""

    total_consultations: int
    consultations_this_month: int
    monthly_limit: int
    remaining_consultations: int
    documents_uploaded: int
    tier: UserTier


# ============================================================================
# Authentication Schemas
# ============================================================================


class LoginRequest(BaseModel):
    """Login request schema."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """JWT token payload schema."""

    sub: str  # user_id
    type: str  # access, refresh, etc.
    tier: str | None = None
    exp: datetime
    iat: datetime


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset request schema."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    token: str
    new_password: str = Field(min_length=8, max_length=100)
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("비밀번호에 대문자가 포함되어야 합니다")
        if not re.search(r"[a-z]", v):
            raise ValueError("비밀번호에 소문자가 포함되어야 합니다")
        if not re.search(r"\d", v):
            raise ValueError("비밀번호에 숫자가 포함되어야 합니다")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("비밀번호에 특수문자가 포함되어야 합니다")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Ensure passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("비밀번호가 일치하지 않습니다")
        return v


class ChangePassword(BaseModel):
    """Change password schema."""

    current_password: str
    new_password: str = Field(min_length=8, max_length=100)
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("비밀번호에 대문자가 포함되어야 합니다")
        if not re.search(r"[a-z]", v):
            raise ValueError("비밀번호에 소문자가 포함되어야 합니다")
        if not re.search(r"\d", v):
            raise ValueError("비밀번호에 숫자가 포함되어야 합니다")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("비밀번호에 특수문자가 포함되어야 합니다")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Ensure passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("비밀번호가 일치하지 않습니다")
        return v


class VerifyEmailRequest(BaseModel):
    """Email verification request schema."""

    token: str


# ============================================================================
# Contact Schemas
# ============================================================================


class ContactCreate(BaseModel):
    """Contact form submission schema."""

    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    phone: str | None = Field(default=None, max_length=20)
    company: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=10, max_length=5000)


class ContactResponse(TimestampSchema):
    """Contact submission response schema."""

    id: UUID
    name: str
    email: str
    phone: str | None
    company: str | None
    message: str
    is_read: bool
    is_replied: bool
    replied_at: datetime | None
