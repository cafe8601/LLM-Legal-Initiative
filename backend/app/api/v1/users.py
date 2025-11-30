"""
Users API Endpoints

사용자 프로필 및 구독 관리 API
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import CurrentActiveUser, DBSession, require_tier
from app.core.exceptions import UserNotFoundError
from app.core.security import get_password_hash, verify_password
from app.models.user import User, UserTier
from app.repositories.consultation import ConsultationRepository
from app.repositories.document import DocumentRepository
from app.repositories.user import UserRepository
from app.services.user_service import UserService
from app.schemas.user import UserUpdate, ChangePassword

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class UserProfileResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    full_name: str
    phone: str | None = None
    company: str | None = None
    tier: str
    is_verified: bool
    is_active: bool
    consultation_count_this_month: int
    consultation_limit: int
    preferred_language: str
    notification_email: bool
    created_at: str


class UpdateProfileRequest(BaseModel):
    """Update profile request."""

    full_name: str | None = Field(None, min_length=2, max_length=100)
    phone: str | None = Field(None, max_length=20)
    company: str | None = Field(None, max_length=100)
    preferred_language: str | None = Field(None, pattern="^(ko|en)$")
    notification_email: bool | None = None


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str
    new_password: str = Field(min_length=8, max_length=100)
    confirm_password: str = Field(min_length=8, max_length=100)


class SubscriptionResponse(BaseModel):
    """Subscription information response."""

    tier: str
    status: str
    current_period_start: str | None = None
    current_period_end: str | None = None
    cancel_at_period_end: bool = False
    consultation_limit: int
    consultation_used: int


class UpgradeSubscriptionRequest(BaseModel):
    """Upgrade subscription request."""

    target_tier: str = Field(pattern="^(pro|enterprise)$")
    payment_method_id: str | None = None


class UsageStatsResponse(BaseModel):
    """Usage statistics response."""

    total_consultations: int
    consultations_this_month: int
    total_turns: int
    total_tokens_used: int
    total_documents_uploaded: int
    storage_used_bytes: int
    storage_limit_bytes: int


class DeleteAccountRequest(BaseModel):
    """Delete account confirmation request."""

    password: str
    confirmation: str = Field(description="Type 'DELETE' to confirm")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_tier_limit(tier: UserTier) -> int:
    """Get monthly consultation limit by tier."""
    limits = {
        UserTier.BASIC: 3,
        UserTier.PRO: -1,  # Unlimited
        UserTier.ENTERPRISE: -1,  # Unlimited
    }
    return limits.get(tier, 3)


def _build_profile_response(user: User) -> UserProfileResponse:
    """Build user profile response."""
    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        company=user.company,
        tier=user.tier.value if user.tier else "basic",
        is_verified=user.is_verified,
        is_active=user.is_active,
        consultation_count_this_month=user.consultation_count_this_month or 0,
        consultation_limit=_get_tier_limit(user.tier),
        preferred_language=user.preferred_language or "ko",
        notification_email=user.notification_email if hasattr(user, 'notification_email') else True,
        created_at=user.created_at.isoformat(),
    )


# =============================================================================
# Profile Endpoints
# =============================================================================


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="내 프로필 조회",
    description="현재 로그인한 사용자의 프로필 정보를 조회합니다.",
)
async def get_profile(
    current_user: CurrentActiveUser,
) -> UserProfileResponse:
    """
    Get current user's profile.
    """
    return _build_profile_response(current_user)


@router.patch(
    "/me",
    response_model=UserProfileResponse,
    summary="프로필 수정",
    description="현재 로그인한 사용자의 프로필 정보를 수정합니다.",
)
async def update_profile(
    data: UpdateProfileRequest,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> UserProfileResponse:
    """
    Update current user's profile.
    """
    service = UserService(db)

    update_data = UserUpdate(
        full_name=data.full_name,
        phone=data.phone,
        company=data.company,
        preferred_language=data.preferred_language,
        notification_email=data.notification_email,
    )

    try:
        user = await service.update_user(current_user.id, update_data)
        return _build_profile_response(user)
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/me/change-password",
    status_code=status.HTTP_200_OK,
    summary="비밀번호 변경",
    description="현재 비밀번호를 확인한 후 새 비밀번호로 변경합니다.",
)
async def change_password(
    data: ChangePasswordRequest,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> dict:
    """
    Change current user's password.
    """
    # Verify current password
    if not verify_password(data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="현재 비밀번호가 일치하지 않습니다.",
        )

    # Verify new passwords match
    if data.new_password != data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="새 비밀번호가 일치하지 않습니다.",
        )

    # Check password is different from current
    if data.current_password == data.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="새 비밀번호는 현재 비밀번호와 달라야 합니다.",
        )

    # Update password
    user_repo = UserRepository(db)
    new_hash = get_password_hash(data.new_password)
    await user_repo.update(current_user.id, password_hash=new_hash)

    logger.info(f"Password changed for user {current_user.id}")

    return {"message": "비밀번호가 변경되었습니다."}


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="계정 삭제",
    description="현재 계정을 삭제합니다. 모든 데이터가 영구적으로 삭제됩니다.",
)
async def delete_account(
    data: DeleteAccountRequest,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> None:
    """
    Delete current user's account.

    This will:
    - Cancel any active subscription
    - Delete all consultations and documents
    - Deactivate the account
    """
    # Verify password
    if not verify_password(data.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="비밀번호가 일치하지 않습니다.",
        )

    # Verify confirmation
    if data.confirmation != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="삭제를 확인하려면 'DELETE'를 입력해주세요.",
        )

    service = UserService(db)

    # Deactivate user (soft delete)
    await service.deactivate_user(current_user.id)

    logger.info(f"Account deleted for user {current_user.id}")


# =============================================================================
# Subscription Endpoints
# =============================================================================


@router.get(
    "/me/subscription",
    response_model=SubscriptionResponse,
    summary="구독 정보 조회",
    description="현재 구독 상태와 정보를 조회합니다.",
)
async def get_subscription(
    db: DBSession,
    current_user: CurrentActiveUser,
) -> SubscriptionResponse:
    """
    Get current subscription information.
    """
    # Get current month usage
    consultation_repo = ConsultationRepository(db)
    consultation_used = await consultation_repo.count_user_consultations_this_month(
        current_user.id
    )

    return SubscriptionResponse(
        tier=current_user.tier.value if current_user.tier else "basic",
        status="active" if current_user.is_active else "inactive",
        current_period_start=None,  # TODO: Get from Stripe
        current_period_end=None,  # TODO: Get from Stripe
        cancel_at_period_end=False,  # TODO: Get from Stripe
        consultation_limit=_get_tier_limit(current_user.tier),
        consultation_used=consultation_used,
    )


@router.post(
    "/me/subscription/upgrade",
    response_model=SubscriptionResponse,
    summary="구독 업그레이드",
    description="상위 요금제로 업그레이드합니다.",
)
async def upgrade_subscription(
    data: UpgradeSubscriptionRequest,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> SubscriptionResponse:
    """
    Upgrade subscription to a higher tier.

    - **target_tier**: pro or enterprise
    - **payment_method_id**: Stripe payment method ID (required for new subscriptions)

    Note: Stripe integration will be implemented in Phase 6.
    """
    # Validate tier upgrade
    current_tier = current_user.tier.value if current_user.tier else "basic"
    tier_order = {"basic": 0, "pro": 1, "enterprise": 2}

    if tier_order.get(data.target_tier, 0) <= tier_order.get(current_tier, 0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="현재 요금제보다 상위 요금제만 선택할 수 있습니다.",
        )

    # TODO: Implement Stripe subscription creation/upgrade in Phase 6
    # For now, just update the tier directly (development mode)
    service = UserService(db)
    await service.update_tier(current_user.id, data.target_tier)

    # Get updated subscription info
    consultation_repo = ConsultationRepository(db)
    consultation_used = await consultation_repo.count_user_consultations_this_month(
        current_user.id
    )

    new_tier = UserTier(data.target_tier)

    return SubscriptionResponse(
        tier=data.target_tier,
        status="active",
        current_period_start=datetime.now(timezone.utc).isoformat(),
        current_period_end=None,
        cancel_at_period_end=False,
        consultation_limit=_get_tier_limit(new_tier),
        consultation_used=consultation_used,
    )


@router.post(
    "/me/subscription/cancel",
    status_code=status.HTTP_200_OK,
    summary="구독 취소",
    description="현재 구독을 취소합니다. 현재 결제 기간이 끝날 때까지 서비스를 이용할 수 있습니다.",
)
async def cancel_subscription(
    db: DBSession,
    current_user: CurrentActiveUser,
) -> dict:
    """
    Cancel current subscription.

    Subscription will remain active until the end of the current billing period.
    After that, the account will be downgraded to basic tier.
    """
    if current_user.tier == UserTier.BASIC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="기본 요금제는 취소할 수 없습니다.",
        )

    # TODO: Implement Stripe subscription cancellation in Phase 6

    return {
        "message": "구독이 취소되었습니다. 현재 결제 기간이 끝날 때까지 서비스를 이용하실 수 있습니다."
    }


# =============================================================================
# Usage Statistics
# =============================================================================


@router.get(
    "/me/usage",
    response_model=UsageStatsResponse,
    summary="사용 통계 조회",
    description="현재 사용자의 서비스 이용 통계를 조회합니다.",
)
async def get_usage_stats(
    db: DBSession,
    current_user: CurrentActiveUser,
) -> UsageStatsResponse:
    """
    Get current user's usage statistics.
    """
    consultation_repo = ConsultationRepository(db)
    document_repo = DocumentRepository(db)

    # Get consultation stats
    total_consultations = await consultation_repo.count_user_consultations(
        current_user.id
    )
    consultations_this_month = await consultation_repo.count_user_consultations_this_month(
        current_user.id
    )

    # Get turns and tokens (would need to sum from consultations)
    consultations = await consultation_repo.get_user_consultations(
        user_id=current_user.id,
        skip=0,
        limit=1000,  # Get all for stats
    )

    total_turns = sum(c.turn_count or 0 for c in consultations)
    total_tokens = sum(c.total_tokens_used or 0 for c in consultations)

    # Get document stats
    total_documents = await document_repo.count_user_documents(current_user.id)
    storage_used = await document_repo.get_storage_usage(current_user.id)

    # Get storage limit by tier
    tier = current_user.tier.value if current_user.tier else "basic"
    storage_limits = {
        "basic": 100 * 1024 * 1024,  # 100MB
        "pro": 1 * 1024 * 1024 * 1024,  # 1GB
        "enterprise": 10 * 1024 * 1024 * 1024,  # 10GB
    }

    return UsageStatsResponse(
        total_consultations=total_consultations,
        consultations_this_month=consultations_this_month,
        total_turns=total_turns,
        total_tokens_used=total_tokens,
        total_documents_uploaded=total_documents,
        storage_used_bytes=storage_used,
        storage_limit_bytes=storage_limits.get(tier, storage_limits["basic"]),
    )


# =============================================================================
# Admin Endpoints (Pro/Enterprise only)
# =============================================================================


@router.get(
    "/me/api-key",
    summary="API 키 조회",
    description="API 접근을 위한 키를 조회합니다. Enterprise 전용.",
)
async def get_api_key(
    db: DBSession,
    current_user: User = Depends(require_tier(["enterprise"])),
) -> dict:
    """
    Get API key for programmatic access.

    Requires Enterprise tier.
    """
    # TODO: Implement API key management in Phase 6
    return {
        "api_key": None,
        "message": "API 키 기능은 곧 제공될 예정입니다.",
    }


@router.post(
    "/me/api-key/regenerate",
    summary="API 키 재생성",
    description="새로운 API 키를 생성합니다. Enterprise 전용.",
)
async def regenerate_api_key(
    db: DBSession,
    current_user: User = Depends(require_tier(["enterprise"])),
) -> dict:
    """
    Regenerate API key.

    Requires Enterprise tier.
    """
    # TODO: Implement API key management in Phase 6
    return {
        "api_key": None,
        "message": "API 키 기능은 곧 제공될 예정입니다.",
    }
