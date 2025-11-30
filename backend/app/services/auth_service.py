"""
Authentication Service

인증 관련 비즈니스 로직
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    InvalidCredentialsError,
    TokenExpiredError,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserNotVerifiedError,
)
from app.core.security import (
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    create_verification_token,
    decode_password_reset_token,
    decode_refresh_token,
    decode_verification_token,
    get_password_hash,
    verify_password,
)
from app.models.user import User, RefreshToken
from app.repositories.user import UserRepository, RefreshTokenRepository
from app.schemas.user import TokenResponse, UserCreate
from app.services.email_service import EmailService


class AuthService:
    """Authentication service handling user auth operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)
        self.token_repo = RefreshTokenRepository(db)
        self.email_service = EmailService()

    async def register(self, data: UserCreate) -> User:
        """
        Register a new user.

        Args:
            data: User registration data

        Returns:
            Created user

        Raises:
            UserAlreadyExistsError: If email is already registered
        """
        # Check if email already exists
        existing_user = await self.user_repo.get_by_email(data.email)
        if existing_user:
            raise UserAlreadyExistsError(f"Email {data.email} is already registered")

        # Create user
        user = await self.user_repo.create(
            email=data.email,
            password_hash=get_password_hash(data.password),
            full_name=data.full_name,
            phone=data.phone,
            company=data.company,
            is_verified=False,
            is_active=True,
        )

        # Send verification email
        verification_token = create_verification_token(user.id)
        await self.email_service.send_verification_email(
            to_email=user.email,
            user_name=user.full_name,
            token=verification_token,
        )

        return user

    async def login(self, email: str, password: str) -> TokenResponse:
        """
        Authenticate user and return tokens.

        Args:
            email: User email
            password: Plain text password

        Returns:
            Token response with access and refresh tokens

        Raises:
            InvalidCredentialsError: If credentials are invalid
            UserNotVerifiedError: If user email is not verified
        """
        # Get user by email
        user = await self.user_repo.get_by_email(email)

        if not user:
            raise InvalidCredentialsError("잘못된 이메일 또는 비밀번호입니다")

        # Verify password
        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError("잘못된 이메일 또는 비밀번호입니다")

        # Check if user is verified
        if not user.is_verified:
            raise UserNotVerifiedError("이메일 인증이 필요합니다")

        # Check if user is active
        if not user.is_active:
            raise InvalidCredentialsError("비활성화된 계정입니다")

        # Create tokens
        access_token = create_access_token(user.id, user.tier)
        refresh_token = create_refresh_token(user.id)

        # Store refresh token
        await self._store_refresh_token(user.id, refresh_token)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: JWT refresh token

        Returns:
            New token pair

        Raises:
            TokenExpiredError: If refresh token is invalid or expired
        """
        # Decode refresh token
        payload = decode_refresh_token(refresh_token)
        if not payload:
            raise TokenExpiredError("유효하지 않은 리프레시 토큰입니다")

        user_id = UUID(payload.sub)

        # Verify refresh token exists in database
        is_valid = await self._verify_refresh_token(user_id, refresh_token)
        if not is_valid:
            raise TokenExpiredError("리프레시 토큰이 취소되었거나 만료되었습니다")

        # Get user
        user = await self.user_repo.get(user_id)
        if not user or not user.is_active:
            raise TokenExpiredError("사용자를 찾을 수 없습니다")

        # Revoke old refresh token (token rotation)
        await self._revoke_refresh_token(user_id, refresh_token)

        # Create new tokens
        new_access_token = create_access_token(user.id, user.tier)
        new_refresh_token = create_refresh_token(user.id)

        # Store new refresh token
        await self._store_refresh_token(user.id, new_refresh_token)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def logout(self, user_id: UUID, refresh_token: str | None = None) -> None:
        """
        Logout user by revoking refresh tokens.

        Args:
            user_id: User ID
            refresh_token: Optional specific refresh token to revoke
        """
        if refresh_token:
            await self._revoke_refresh_token(user_id, refresh_token)
        else:
            # Revoke all user's refresh tokens
            await self.token_repo.revoke_all_user_tokens(user_id)

    async def verify_email(self, token: str) -> User:
        """
        Verify user email.

        Args:
            token: Email verification token

        Returns:
            Verified user

        Raises:
            TokenExpiredError: If token is invalid or expired
            UserNotFoundError: If user not found
        """
        user_id_str = decode_verification_token(token)
        if not user_id_str:
            raise TokenExpiredError("유효하지 않거나 만료된 인증 토큰입니다")

        user = await self.user_repo.get(UUID(user_id_str))
        if not user:
            raise UserNotFoundError("사용자를 찾을 수 없습니다")

        if user.is_verified:
            return user  # Already verified

        # Mark user as verified
        user = await self.user_repo.verify_user(user.id)
        return user

    async def request_password_reset(self, email: str) -> None:
        """
        Request password reset email.

        Args:
            email: User email

        Note:
            Always returns success to prevent email enumeration
        """
        user = await self.user_repo.get_by_email(email)

        if user:
            # Create reset token
            reset_token = create_password_reset_token(user.id)

            # Send reset email
            await self.email_service.send_password_reset_email(
                to_email=user.email,
                user_name=user.full_name,
                token=reset_token,
            )

        # Always return success to prevent email enumeration

    async def reset_password(self, token: str, new_password: str) -> User:
        """
        Reset user password.

        Args:
            token: Password reset token
            new_password: New password

        Returns:
            Updated user

        Raises:
            TokenExpiredError: If token is invalid or expired
            UserNotFoundError: If user not found
        """
        user_id_str = decode_password_reset_token(token)
        if not user_id_str:
            raise TokenExpiredError("유효하지 않거나 만료된 재설정 토큰입니다")

        user = await self.user_repo.get(UUID(user_id_str))
        if not user:
            raise UserNotFoundError("사용자를 찾을 수 없습니다")

        # Update password
        user = await self.user_repo.update(
            user.id,
            password_hash=get_password_hash(new_password),
        )

        # Revoke all refresh tokens for security
        await self.token_repo.revoke_all_user_tokens(user.id)

        return user

    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> User:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            Updated user

        Raises:
            InvalidCredentialsError: If current password is incorrect
        """
        user = await self.user_repo.get(user_id)
        if not user:
            raise UserNotFoundError("사용자를 찾을 수 없습니다")

        # Verify current password
        if not verify_password(current_password, user.password_hash):
            raise InvalidCredentialsError("현재 비밀번호가 올바르지 않습니다")

        # Update password
        user = await self.user_repo.update(
            user_id,
            password_hash=get_password_hash(new_password),
        )

        # Revoke all refresh tokens except current session
        await self.token_repo.revoke_all_user_tokens(user_id)

        return user

    async def resend_verification_email(self, email: str) -> None:
        """
        Resend email verification.

        Args:
            email: User email
        """
        user = await self.user_repo.get_by_email(email)

        if user and not user.is_verified:
            verification_token = create_verification_token(user.id)
            await self.email_service.send_verification_email(
                to_email=user.email,
                user_name=user.full_name,
                token=verification_token,
            )

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _store_refresh_token(self, user_id: UUID, token: str) -> None:
        """Store refresh token hash in database."""
        token_hash = get_password_hash(token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

        await self.token_repo.create(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )

    async def _verify_refresh_token(self, user_id: UUID, token: str) -> bool:
        """Verify refresh token exists and is valid."""
        tokens = await self.token_repo.get_user_tokens(user_id)

        for stored_token in tokens:
            if verify_password(token, stored_token.token_hash):
                return True

        return False

    async def _revoke_refresh_token(self, user_id: UUID, token: str) -> None:
        """Revoke a specific refresh token."""
        tokens = await self.token_repo.get_user_tokens(user_id)

        for stored_token in tokens:
            if verify_password(token, stored_token.token_hash):
                await self.token_repo.revoke_token(stored_token.id)
                break
