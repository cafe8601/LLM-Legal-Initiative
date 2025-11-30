"""
Authentication API Endpoints

인증 관련 API
"""

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession, CurrentActiveUser, get_db
from app.core.exceptions import (
    InvalidCredentialsError,
    TokenExpiredError,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserNotVerifiedError,
)
from app.schemas.user import (
    ChangePassword,
    ContactCreate,
    ContactResponse,
    LoginRequest,
    PasswordReset,
    PasswordResetConfirm,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    VerifyEmailRequest,
)
from app.schemas.common import MessageResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Registration & Login
# =============================================================================


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="회원가입",
    description="새 사용자 계정을 생성합니다. 이메일 인증이 필요합니다.",
)
async def register(
    data: UserCreate,
    db: DBSession,
) -> UserResponse:
    """
    Register a new user account.

    - **email**: Valid email address
    - **password**: Min 8 chars, must include uppercase, lowercase, digit, special char
    - **full_name**: User's full name (2-100 characters)
    - **terms_accepted**: Must be true

    A verification email will be sent to the provided email address.
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.register(data)
        return UserResponse.model_validate(user)
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="로그인",
    description="이메일과 비밀번호로 인증하고 JWT 토큰을 발급받습니다.",
)
async def login(
    data: LoginRequest,
    response: Response,
    db: DBSession,
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.

    Returns:
    - **access_token**: Valid for 30 minutes
    - **refresh_token**: Valid for 7 days

    The refresh_token is also set as an HttpOnly cookie for security.
    """
    auth_service = AuthService(db)

    try:
        tokens = await auth_service.login(data.email, data.password)

        # Set refresh token as HttpOnly cookie
        response.set_cookie(
            key="refresh_token",
            value=tokens.refresh_token,
            httponly=True,
            secure=True,  # Only send over HTTPS
            samesite="lax",
            max_age=7 * 24 * 60 * 60,  # 7 days
            path="/api/v1/auth",
        )

        return tokens

    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except UserNotVerifiedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="토큰 갱신",
    description="리프레시 토큰을 사용하여 새 액세스 토큰을 발급받습니다.",
)
async def refresh_tokens(
    data: RefreshTokenRequest | None = None,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    The refresh token can be provided either:
    - In the request body
    - As an HttpOnly cookie (automatically sent by browser)

    Token rotation: The old refresh token is revoked and a new one is issued.
    """
    # Get refresh token from body or cookie
    refresh_token = None
    if data and data.refresh_token:
        refresh_token = data.refresh_token
    elif refresh_token_cookie:
        refresh_token = refresh_token_cookie

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="리프레시 토큰이 필요합니다",
        )

    auth_service = AuthService(db)

    try:
        return await auth_service.refresh_tokens(refresh_token)
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="로그아웃",
    description="현재 세션의 리프레시 토큰을 취소합니다.",
)
async def logout(
    response: Response,
    current_user: CurrentActiveUser,
    refresh_token_cookie: str | None = Cookie(default=None, alias="refresh_token"),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Logout user and revoke refresh token.

    Clears the refresh_token cookie and revokes the token in the database.
    """
    auth_service = AuthService(db)
    await auth_service.logout(current_user.id, refresh_token_cookie)

    # Clear refresh token cookie
    response.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth",
    )


# =============================================================================
# Email Verification
# =============================================================================


@router.post(
    "/verify-email",
    response_model=MessageResponse,
    summary="이메일 인증",
    description="이메일 인증 토큰을 검증하고 계정을 활성화합니다.",
)
async def verify_email(
    data: VerifyEmailRequest,
    db: DBSession,
) -> MessageResponse:
    """
    Verify user's email address.

    The token is sent via email during registration.
    Token is valid for 24 hours.
    """
    auth_service = AuthService(db)

    try:
        await auth_service.verify_email(data.token)
        return MessageResponse(message="이메일이 성공적으로 인증되었습니다")
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    summary="인증 메일 재발송",
    description="이메일 인증 메일을 재발송합니다.",
)
async def resend_verification_email(
    data: PasswordReset,  # Reuse schema - just needs email
    db: DBSession,
) -> MessageResponse:
    """
    Resend email verification link.

    A new verification email will be sent if the email exists
    and is not already verified.
    """
    auth_service = AuthService(db)
    await auth_service.resend_verification_email(data.email)

    # Always return success to prevent email enumeration
    return MessageResponse(message="인증 메일이 발송되었습니다")


# =============================================================================
# Password Reset
# =============================================================================


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="비밀번호 찾기",
    description="비밀번호 재설정 이메일을 발송합니다.",
)
async def forgot_password(
    data: PasswordReset,
    db: DBSession,
) -> MessageResponse:
    """
    Request password reset email.

    If the email exists, a password reset link will be sent.
    The link is valid for 1 hour.

    Always returns success to prevent email enumeration.
    """
    auth_service = AuthService(db)
    await auth_service.request_password_reset(data.email)

    # Always return success to prevent email enumeration
    return MessageResponse(message="비밀번호 재설정 메일이 발송되었습니다")


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="비밀번호 재설정",
    description="토큰을 사용하여 비밀번호를 재설정합니다.",
)
async def reset_password(
    data: PasswordResetConfirm,
    db: DBSession,
) -> MessageResponse:
    """
    Reset password with token.

    The token is received via email from the forgot-password endpoint.
    All active sessions will be logged out after password reset.
    """
    auth_service = AuthService(db)

    try:
        await auth_service.reset_password(data.token, data.new_password)
        return MessageResponse(message="비밀번호가 성공적으로 재설정되었습니다")
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="비밀번호 변경",
    description="현재 비밀번호를 확인하고 새 비밀번호로 변경합니다.",
)
async def change_password(
    data: ChangePassword,
    current_user: CurrentActiveUser,
    db: DBSession,
) -> MessageResponse:
    """
    Change password for authenticated user.

    Requires current password verification.
    All other sessions will be logged out.
    """
    auth_service = AuthService(db)

    try:
        await auth_service.change_password(
            current_user.id,
            data.current_password,
            data.new_password,
        )
        return MessageResponse(message="비밀번호가 성공적으로 변경되었습니다")
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# =============================================================================
# Current User
# =============================================================================


@router.get(
    "/me",
    response_model=UserResponse,
    summary="내 정보 조회",
    description="현재 로그인한 사용자의 정보를 조회합니다.",
)
async def get_me(
    current_user: CurrentActiveUser,
) -> UserResponse:
    """
    Get current authenticated user's information.

    Returns the full user profile including tier and verification status.
    """
    return UserResponse.model_validate(current_user)
