"""
User Model

사용자 정보 및 인증 데이터
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, Enum as SQLEnum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.consultation import Consultation
    from app.models.document import Document
    from app.models.memory import UserMemory


class UserTier(str, Enum):
    """User subscription tiers."""

    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    """User account model."""

    __tablename__ = "users"

    # Authentication
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    company: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Subscription
    tier: Mapped[str] = mapped_column(
        String(20),
        default=UserTier.BASIC,
        nullable=False,
        index=True,
    )

    # Usage tracking
    consultation_count_this_month: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    last_consultation_reset: Mapped[datetime | None] = mapped_column(nullable=True)

    # Stripe integration
    stripe_customer_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Preferences
    preferred_language: Mapped[str] = mapped_column(
        String(10),
        default="ko",
        nullable=False,
    )
    notification_email: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    consultations: Mapped[list["Consultation"]] = relationship(
        "Consultation",
        back_populates="user",
        lazy="dynamic",
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="user",
        lazy="dynamic",
    )
    memories: Mapped[list["UserMemory"]] = relationship(
        "UserMemory",
        back_populates="user",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


class RefreshToken(Base):
    """Refresh token storage for token rotation."""

    __tablename__ = "refresh_tokens"

    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Device/session info for security
    device_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)


class ContactSubmission(Base):
    """Contact form submissions."""

    __tablename__ = "contact_submissions"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    company: Mapped[str | None] = mapped_column(String(100), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    is_replied: Mapped[bool] = mapped_column(Boolean, default=False)
    replied_at: Mapped[datetime | None] = mapped_column(nullable=True)
    replied_by: Mapped[UUID | None] = mapped_column(nullable=True)
