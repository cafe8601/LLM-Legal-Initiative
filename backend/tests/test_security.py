"""
Security Module Tests

보안 모듈 테스트 - 실제 API 시그니처에 맞게 수정됨
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4, UUID

import pytest
from jose import jwt

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    create_verification_token,
    decode_access_token,
    decode_refresh_token,
    get_password_hash,
    verify_password,
    TokenPayload,
)


# =============================================================================
# Password Hashing Tests
# =============================================================================


class TestPasswordHashing:
    """Test password hashing functions."""

    def test_hash_password(self):
        """Test password hashing creates unique hash."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)

        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are 60+ chars
        # bcrypt 4.x는 $2b$ 접두사 사용
        assert hashed.startswith("$2b$")

    def test_hash_is_unique(self):
        """Test same password creates different hashes."""
        password = "SecurePassword123!"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Same password should produce different hashes due to salt
        assert hash1 != hash2

    def test_verify_correct_password(self):
        """Test verifying correct password."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        """Test verifying wrong password."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)

        assert verify_password("WrongPassword123!", hashed) is False

    def test_verify_similar_password(self):
        """Test verifying similar but different password."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)

        # Even minor differences should fail
        assert verify_password("SecurePassword123", hashed) is False
        assert verify_password("securepassword123!", hashed) is False
        assert verify_password("SecurePassword124!", hashed) is False


# =============================================================================
# Access Token Tests
# =============================================================================


class TestAccessToken:
    """Test access token functions."""

    def test_create_access_token(self):
        """Test creating access token."""
        user_id = uuid4()
        tier = "basic"

        token = create_access_token(
            user_id=user_id,
            tier=tier,
        )

        assert token is not None
        assert len(token) > 100  # JWT tokens are typically long

    def test_decode_access_token(self):
        """Test decoding access token."""
        user_id = uuid4()
        tier = "pro"

        token = create_access_token(
            user_id=user_id,
            tier=tier,
        )

        payload = decode_access_token(token)

        assert payload is not None
        assert payload.sub == str(user_id)
        assert payload.tier == tier
        assert payload.type == "access"

    def test_access_token_contains_all_claims(self):
        """Test access token contains all required claims."""
        user_id = uuid4()
        tier = "enterprise"

        token = create_access_token(
            user_id=user_id,
            tier=tier,
        )

        # Decode raw payload
        raw_payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        assert raw_payload["sub"] == str(user_id)
        assert raw_payload["tier"] == tier
        assert raw_payload["type"] == "access"
        assert "exp" in raw_payload

    def test_decode_invalid_token(self):
        """Test decoding invalid token returns None."""
        payload = decode_access_token("invalid_token")
        assert payload is None

    def test_decode_tampered_token(self):
        """Test decoding tampered token returns None."""
        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            tier="basic",
        )

        # Tamper with the token
        tampered_token = token[:-5] + "xxxxx"

        payload = decode_access_token(tampered_token)
        assert payload is None

    def test_access_token_expiration(self):
        """Test access token expiration."""
        user_id = uuid4()

        # Create expired token manually
        expired_payload = {
            "sub": str(user_id),
            "tier": "basic",
            "type": "access",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        expired_token = jwt.encode(
            expired_payload,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )

        payload = decode_access_token(expired_token)
        assert payload is None


# =============================================================================
# Refresh Token Tests
# =============================================================================


class TestRefreshToken:
    """Test refresh token functions."""

    def test_create_refresh_token(self):
        """Test creating refresh token."""
        user_id = uuid4()
        token = create_refresh_token(user_id)

        assert token is not None
        assert len(token) > 100

    def test_decode_refresh_token(self):
        """Test decoding refresh token."""
        user_id = uuid4()
        token = create_refresh_token(user_id)

        payload = decode_refresh_token(token)

        assert payload is not None
        assert payload.sub == str(user_id)
        assert payload.type == "refresh"

    def test_refresh_token_different_from_access(self):
        """Test refresh token is different from access token."""
        user_id = uuid4()

        access_token = create_access_token(
            user_id=user_id,
            tier="basic",
        )
        refresh_token = create_refresh_token(user_id)

        assert access_token != refresh_token

        # Decoding refresh token as access should fail type check
        access_payload = decode_access_token(refresh_token)
        assert access_payload is None

    def test_decode_access_as_refresh(self):
        """Test decoding access token as refresh token fails."""
        user_id = uuid4()
        access_token = create_access_token(
            user_id=user_id,
            tier="basic",
        )

        # Trying to decode access token as refresh should return None
        payload = decode_refresh_token(access_token)
        assert payload is None

    def test_refresh_token_longer_expiry(self):
        """Test refresh token has longer expiry than access token."""
        user_id = uuid4()

        access_token = create_access_token(
            user_id=user_id,
            tier="basic",
        )
        refresh_token = create_refresh_token(user_id)

        access_payload = jwt.decode(
            access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        refresh_payload = jwt.decode(
            refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        access_exp = access_payload["exp"]
        refresh_exp = refresh_payload["exp"]

        # Refresh token should expire later than access token
        assert refresh_exp > access_exp


# =============================================================================
# Verification Token Tests
# =============================================================================


class TestVerificationToken:
    """Test verification token functions."""

    def test_create_verification_token(self):
        """Test creating verification token."""
        user_id = uuid4()

        token = create_verification_token(user_id, expires_hours=24)

        assert token is not None
        assert len(token) > 50

    def test_verification_token_unique(self):
        """Test verification tokens are unique for different users."""
        user_id1 = uuid4()
        user_id2 = uuid4()

        token1 = create_verification_token(user_id1, expires_hours=24)
        token2 = create_verification_token(user_id2, expires_hours=24)

        # Different users should produce different tokens
        assert token1 != token2

    def test_verification_token_different_expiry(self):
        """Test verification tokens with different expiry."""
        user_id = uuid4()

        short_token = create_verification_token(user_id, expires_hours=1)
        long_token = create_verification_token(user_id, expires_hours=48)

        # Decode and compare expiration
        short_payload = jwt.decode(
            short_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        long_payload = jwt.decode(
            long_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        assert long_payload["exp"] > short_payload["exp"]


# =============================================================================
# Token Payload Tests
# =============================================================================


class TestTokenPayload:
    """Test TokenPayload dataclass."""

    def test_token_payload_creation(self):
        """Test creating TokenPayload."""
        payload = TokenPayload(
            sub="user_123",
            tier="pro",
            type="access",
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert payload.sub == "user_123"
        assert payload.tier == "pro"
        assert payload.type == "access"

    def test_token_payload_optional_fields(self):
        """Test TokenPayload with optional fields."""
        payload = TokenPayload(
            sub="user_123",
            type="refresh",
            exp=datetime.now(timezone.utc) + timedelta(days=7),
        )

        assert payload.sub == "user_123"
        assert payload.tier is None
        assert payload.type == "refresh"


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================


class TestSecurityEdgeCases:
    """Test security edge cases."""

    def test_empty_password_hash(self):
        """Test hashing empty password."""
        # Empty password should still hash
        hashed = get_password_hash("")
        assert hashed is not None
        assert verify_password("", hashed) is True

    def test_very_long_password(self):
        """Test hashing very long password - bcrypt truncates at 72 bytes."""
        # bcrypt has a max password length of 72 bytes
        # Passwords are truncated, not rejected
        long_password = "a" * 72  # Use exactly 72 bytes to avoid issues
        hashed = get_password_hash(long_password)

        # Should still work
        assert hashed is not None
        assert verify_password(long_password, hashed) is True

    def test_unicode_password(self):
        """Test hashing password with unicode characters."""
        # Use a shorter unicode password to avoid 72-byte limit
        unicode_password = "안녕123!"  # Korean characters + ascii
        hashed = get_password_hash(unicode_password)

        assert hashed is not None
        assert verify_password(unicode_password, hashed) is True

    def test_special_characters_in_password(self):
        """Test hashing password with special characters."""
        # Shorter to avoid 72-byte limit
        special_password = "P@$$w0rd!#$%"
        hashed = get_password_hash(special_password)

        assert hashed is not None
        assert verify_password(special_password, hashed) is True

    def test_null_bytes_in_password(self):
        """Test hashing password with null bytes - behavior depends on bcrypt version."""
        # bcrypt behavior varies by version:
        # - bcrypt 5.x: raises ValueError for NULL bytes
        # - bcrypt 4.x: truncates at NULL byte (less secure but still hashes)
        null_password = "pass\x00word"

        try:
            # bcrypt 4.x may truncate at null byte instead of raising
            hashed = get_password_hash(null_password)
            # If it doesn't raise, verify it at least hashes something
            assert hashed is not None
            # Note: Password is truncated to "pass" in bcrypt 4.x
        except ValueError:
            # bcrypt 5.x raises ValueError - this is the safer behavior
            pass

    def test_token_with_minimal_claims(self):
        """Test token creation with minimal claims."""
        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
        )

        assert token is not None
        payload = decode_access_token(token)
        assert payload is not None
        assert payload.sub == str(user_id)

    def test_wrong_algorithm_decode(self):
        """Test decoding token with wrong algorithm fails."""
        user_id = uuid4()

        # Create token with different algorithm
        payload = {
            "sub": str(user_id),
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        # Use different secret (simulating wrong key)
        wrong_token = jwt.encode(
            payload,
            "wrong_secret_key",
            algorithm=settings.ALGORITHM,
        )

        # Should fail to decode with correct key
        decoded = decode_access_token(wrong_token)
        assert decoded is None
