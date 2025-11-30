"""
User Service

사용자 관련 비즈니스 로직
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    UserNotFoundError,
    InsufficientTierError,
    ConsultationLimitExceededError,
)
from app.models.user import User, UserTier
from app.repositories.user import UserRepository
from app.schemas.user import UserUpdate, UserStats


class UserService:
    """User service handling user operations."""

    # Monthly consultation limits by tier
    TIER_LIMITS = {
        UserTier.BASIC: 3,
        UserTier.PRO: -1,  # Unlimited
        UserTier.ENTERPRISE: -1,  # Unlimited
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)

    async def get_user(self, user_id: UUID) -> User:
        """
        Get user by ID.

        Args:
            user_id: User UUID

        Returns:
            User object

        Raises:
            UserNotFoundError: If user not found
        """
        user = await self.user_repo.get(user_id)
        if not user:
            raise UserNotFoundError("사용자를 찾을 수 없습니다")
        return user

    async def get_user_by_email(self, email: str) -> User | None:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User object or None
        """
        return await self.user_repo.get_by_email(email)

    async def update_user(self, user_id: UUID, data: UserUpdate) -> User:
        """
        Update user profile.

        Args:
            user_id: User UUID
            data: Update data

        Returns:
            Updated user

        Raises:
            UserNotFoundError: If user not found
        """
        user = await self.user_repo.get(user_id)
        if not user:
            raise UserNotFoundError("사용자를 찾을 수 없습니다")

        update_data = data.model_dump(exclude_unset=True)
        if update_data:
            user = await self.user_repo.update(user_id, **update_data)

        return user

    async def get_user_stats(self, user_id: UUID) -> UserStats:
        """
        Get user statistics.

        Args:
            user_id: User UUID

        Returns:
            User stats
        """
        user = await self.get_user(user_id)

        # Get consultation repository for counting
        from app.repositories.consultation import ConsultationRepository

        consultation_repo = ConsultationRepository(self.db)

        # Get document repository for counting
        from app.repositories.document import DocumentRepository

        document_repo = DocumentRepository(self.db)

        total_consultations = await consultation_repo.count_user_consultations(user_id)
        consultations_this_month = await consultation_repo.count_user_consultations_this_month(user_id)
        documents_uploaded = await document_repo.count_user_documents(user_id)

        monthly_limit = self.TIER_LIMITS.get(user.tier, 3)
        remaining = monthly_limit - consultations_this_month if monthly_limit > 0 else -1

        return UserStats(
            total_consultations=total_consultations,
            consultations_this_month=consultations_this_month,
            monthly_limit=monthly_limit,
            remaining_consultations=max(0, remaining) if remaining >= 0 else -1,
            documents_uploaded=documents_uploaded,
            tier=user.tier,
        )

    async def check_consultation_limit(self, user_id: UUID) -> bool:
        """
        Check if user can create new consultation.

        Args:
            user_id: User UUID

        Returns:
            True if user can create consultation

        Raises:
            ConsultationLimitExceededError: If limit reached
        """
        user = await self.get_user(user_id)
        limit = self.TIER_LIMITS.get(user.tier, 3)

        # Unlimited for pro/enterprise
        if limit < 0:
            return True

        # Check this month's count
        from app.repositories.consultation import ConsultationRepository

        consultation_repo = ConsultationRepository(self.db)
        current_count = await consultation_repo.count_user_consultations_this_month(user_id)

        if current_count >= limit:
            raise ConsultationLimitExceededError(
                f"이번 달 상담 한도({limit}건)에 도달했습니다. 업그레이드하여 무제한으로 이용하세요."
            )

        return True

    async def increment_consultation_count(self, user_id: UUID) -> User:
        """
        Increment user's monthly consultation count.

        Args:
            user_id: User UUID

        Returns:
            Updated user
        """
        return await self.user_repo.increment_consultation_count(user_id)

    async def check_feature_access(self, user_id: UUID, feature: str) -> bool:
        """
        Check if user has access to a feature based on tier.

        Args:
            user_id: User UUID
            feature: Feature name

        Returns:
            True if user has access

        Raises:
            InsufficientTierError: If tier is insufficient
        """
        user = await self.get_user(user_id)

        # Feature-tier mapping
        feature_tiers = {
            "priority_processing": [UserTier.PRO, UserTier.ENTERPRISE],
            "document_analysis": [UserTier.PRO, UserTier.ENTERPRISE],
            "export_report": [UserTier.PRO, UserTier.ENTERPRISE],
            "api_access": [UserTier.ENTERPRISE],
            "custom_prompts": [UserTier.ENTERPRISE],
            "dedicated_support": [UserTier.ENTERPRISE],
        }

        allowed_tiers = feature_tiers.get(feature, [])

        if allowed_tiers and user.tier not in allowed_tiers:
            tier_names = ", ".join(t.value for t in allowed_tiers)
            raise InsufficientTierError(
                f"이 기능은 {tier_names} 등급에서 사용 가능합니다."
            )

        return True

    async def update_tier(
        self,
        user_id: UUID,
        new_tier: str,
        stripe_subscription_id: str | None = None,
    ) -> User:
        """
        Update user's subscription tier.

        Args:
            user_id: User UUID
            new_tier: New tier value
            stripe_subscription_id: Stripe subscription ID

        Returns:
            Updated user
        """
        update_data = {"tier": new_tier}
        if stripe_subscription_id:
            update_data["stripe_subscription_id"] = stripe_subscription_id

        return await self.user_repo.update(user_id, **update_data)

    async def deactivate_user(self, user_id: UUID) -> User:
        """
        Deactivate user account.

        Args:
            user_id: User UUID

        Returns:
            Updated user
        """
        return await self.user_repo.deactivate_user(user_id)

    async def reset_monthly_counts(self) -> int:
        """
        Reset all users' monthly consultation counts.
        Should be called on the first day of each month.

        Returns:
            Number of users updated
        """
        return await self.user_repo.reset_monthly_consultation_counts()
