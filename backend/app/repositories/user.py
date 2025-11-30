"""
User Repository

사용자 데이터 액세스 레이어
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, RefreshToken, ContactSubmission, UserTier
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for User model."""

    def __init__(self, db: AsyncSession):
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_stripe_customer_id(self, customer_id: str) -> User | None:
        """Get user by Stripe customer ID."""
        result = await self.db.execute(
            select(User).where(User.stripe_customer_id == customer_id)
        )
        return result.scalar_one_or_none()

    async def get_active_users(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        tier: str | None = None,
    ) -> list[User]:
        """Get active users with optional tier filter."""
        query = select(User).where(User.is_active == True)

        if tier:
            query = query.where(User.tier == tier)

        query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count_by_tier(self) -> dict[str, int]:
        """Count users by tier."""
        result = await self.db.execute(
            select(User.tier, func.count(User.id)).group_by(User.tier)
        )
        return {row[0]: row[1] for row in result.all()}

    async def search_users(
        self,
        query: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> list[User]:
        """Search users by email or name."""
        search_pattern = f"%{query}%"
        result = await self.db.execute(
            select(User)
            .where(
                or_(
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern),
                )
            )
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def increment_consultation_count(self, user_id: UUID) -> User | None:
        """Increment user's monthly consultation count."""
        user = await self.get(user_id)
        if user is None:
            return None

        user.consultation_count_this_month += 1
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def reset_monthly_consultation_counts(self) -> int:
        """Reset all users' monthly consultation counts."""
        from sqlalchemy import update

        result = await self.db.execute(
            update(User).values(
                consultation_count_this_month=0,
                last_consultation_reset=datetime.now(timezone.utc),
            )
        )
        return result.rowcount

    async def update_tier(self, user_id: UUID, tier: str) -> User | None:
        """Update user's subscription tier."""
        return await self.update(user_id, tier=tier)

    async def verify_user(self, user_id: UUID) -> User | None:
        """Mark user as verified."""
        return await self.update(user_id, is_verified=True)

    async def deactivate_user(self, user_id: UUID) -> User | None:
        """Deactivate a user account."""
        return await self.update(user_id, is_active=False)


class RefreshTokenRepository(BaseRepository[RefreshToken]):
    """Repository for RefreshToken model."""

    def __init__(self, db: AsyncSession):
        super().__init__(RefreshToken, db)

    async def get_by_token_hash(self, token_hash: str) -> RefreshToken | None:
        """Get refresh token by hash."""
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.token_hash == token_hash,
                    RefreshToken.is_revoked == False,
                    RefreshToken.expires_at > datetime.now(timezone.utc),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_user_tokens(self, user_id: UUID) -> list[RefreshToken]:
        """Get all active tokens for a user."""
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.user_id == user_id,
                    RefreshToken.is_revoked == False,
                    RefreshToken.expires_at > datetime.now(timezone.utc),
                )
            )
        )
        return list(result.scalars().all())

    async def revoke_token(self, token_id: UUID) -> bool:
        """Revoke a specific token."""
        result = await self.update(token_id, is_revoked=True)
        return result is not None

    async def revoke_all_user_tokens(self, user_id: UUID) -> int:
        """Revoke all tokens for a user."""
        from sqlalchemy import update

        result = await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id)
            .values(is_revoked=True)
        )
        return result.rowcount

    async def cleanup_expired_tokens(self) -> int:
        """Delete expired tokens."""
        from sqlalchemy import delete

        result = await self.db.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < datetime.now(timezone.utc))
        )
        return result.rowcount


class ContactSubmissionRepository(BaseRepository[ContactSubmission]):
    """Repository for ContactSubmission model."""

    def __init__(self, db: AsyncSession):
        super().__init__(ContactSubmission, db)

    async def get_unread(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ContactSubmission]:
        """Get unread contact submissions."""
        result = await self.db.execute(
            select(ContactSubmission)
            .where(ContactSubmission.is_read == False)
            .order_by(ContactSubmission.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_as_read(self, submission_id: UUID) -> ContactSubmission | None:
        """Mark a submission as read."""
        return await self.update(submission_id, is_read=True)

    async def mark_as_replied(
        self,
        submission_id: UUID,
        replied_by: UUID,
    ) -> ContactSubmission | None:
        """Mark a submission as replied."""
        return await self.update(
            submission_id,
            is_replied=True,
            replied_at=datetime.now(timezone.utc),
            replied_by=replied_by,
        )

    async def count_unread(self) -> int:
        """Count unread submissions."""
        result = await self.db.execute(
            select(func.count()).select_from(ContactSubmission).where(ContactSubmission.is_read == False)
        )
        return result.scalar() or 0
