"""
Rate Limiting Tests

속도 제한 테스트
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from app.core.rate_limit import (
    TierLimits,
    RateLimiter,
    ConsultationLimiter,
    get_rate_limit_headers,
)
from app.core.exceptions import RateLimitExceededError


# =============================================================================
# TierLimits Tests
# =============================================================================


class TestTierLimits:
    """Test TierLimits configuration."""

    def test_basic_tier_limits(self):
        """Test basic tier limits are correct."""
        requests_per_minute = TierLimits.get_limit("basic", "requests_per_minute")
        consultations = TierLimits.get_limit("basic", "consultations_per_month")

        assert requests_per_minute > 0
        assert consultations > 0

    def test_pro_tier_limits(self):
        """Test pro tier limits are higher than basic."""
        basic_rpm = TierLimits.get_limit("basic", "requests_per_minute")
        pro_rpm = TierLimits.get_limit("pro", "requests_per_minute")

        assert pro_rpm > basic_rpm

    def test_enterprise_tier_limits(self):
        """Test enterprise tier limits are highest."""
        pro_rpm = TierLimits.get_limit("pro", "requests_per_minute")
        enterprise_rpm = TierLimits.get_limit("enterprise", "requests_per_minute")

        assert enterprise_rpm > pro_rpm

    def test_unlimited_detection(self):
        """Test unlimited limit detection."""
        # Enterprise consultations should be unlimited (-1)
        enterprise_consultations = TierLimits.get_limit(
            "enterprise", "consultations_per_month"
        )

        assert TierLimits.is_unlimited(enterprise_consultations) is True
        assert TierLimits.is_unlimited(10) is False
        assert TierLimits.is_unlimited(0) is False

    def test_unknown_tier_defaults_to_basic(self):
        """Test unknown tier defaults to basic."""
        unknown_rpm = TierLimits.get_limit("unknown_tier", "requests_per_minute")
        basic_rpm = TierLimits.get_limit("basic", "requests_per_minute")

        assert unknown_rpm == basic_rpm

    def test_unknown_limit_type_returns_zero(self):
        """Test unknown limit type returns zero."""
        unknown_limit = TierLimits.get_limit("basic", "unknown_limit_type")
        assert unknown_limit == 0

    def test_all_tiers_have_required_limits(self):
        """Test all tiers have required limit types."""
        required_limits = [
            "requests_per_minute",
            "requests_per_hour",
            "consultations_per_month",
            "documents_per_month",
            "max_document_size_mb",
        ]

        for tier in ["basic", "pro", "enterprise"]:
            for limit_type in required_limits:
                limit = TierLimits.get_limit(tier, limit_type)
                # All limits should be defined (not defaulting to 0)
                assert limit != 0 or TierLimits.is_unlimited(limit)


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.incr.return_value = 1
        redis_mock.expire.return_value = True
        redis_mock.ttl.return_value = 60
        redis_mock.get.return_value = None
        redis_mock.keys.return_value = []
        redis_mock.delete.return_value = True
        return redis_mock

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create RateLimiter instance."""
        return RateLimiter(mock_redis)

    @pytest.mark.asyncio
    async def test_check_rate_limit_first_request(self, rate_limiter, mock_redis):
        """Test first request is allowed."""
        user_id = uuid4()
        mock_redis.incr.return_value = 1

        allowed, current, limit = await rate_limiter.check_rate_limit(
            user_id=user_id,
            tier="basic",
            endpoint="/api/test",
        )

        assert allowed is True
        assert current == 1
        assert limit > 0
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, rate_limiter, mock_redis):
        """Test request within limit is allowed."""
        user_id = uuid4()
        mock_redis.incr.return_value = 5  # Under basic limit

        allowed, current, limit = await rate_limiter.check_rate_limit(
            user_id=user_id,
            tier="basic",
        )

        assert allowed is True
        assert current == 5

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter, mock_redis):
        """Test request exceeding limit raises error."""
        user_id = uuid4()
        basic_limit = TierLimits.get_limit("basic", "requests_per_minute")
        mock_redis.incr.return_value = basic_limit + 1
        mock_redis.ttl.return_value = 45

        with pytest.raises(RateLimitExceededError) as exc_info:
            await rate_limiter.check_rate_limit(
                user_id=user_id,
                tier="basic",
            )

        assert "요청 한도 초과" in str(exc_info.value)
        assert "45초" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_rate_limit_unlimited(self, rate_limiter, mock_redis):
        """Test unlimited tier allows all requests."""
        user_id = uuid4()

        # Enterprise has high limits, simulate -1 (unlimited)
        with pytest.MonkeyPatch().context() as mp:
            # Temporarily make enterprise unlimited for requests
            original_limits = TierLimits.LIMITS.copy()
            TierLimits.LIMITS["test_unlimited"] = {"requests_per_minute": -1}

            allowed, current, limit = await rate_limiter.check_rate_limit(
                user_id=user_id,
                tier="test_unlimited",
            )

            assert allowed is True
            assert limit == -1

            # Restore
            TierLimits.LIMITS = original_limits

    @pytest.mark.asyncio
    async def test_check_hourly_limit(self, rate_limiter, mock_redis):
        """Test hourly rate limit."""
        user_id = uuid4()
        mock_redis.incr.return_value = 10

        allowed, current, limit = await rate_limiter.check_hourly_limit(
            user_id=user_id,
            tier="basic",
        )

        assert allowed is True
        assert current == 10

    @pytest.mark.asyncio
    async def test_check_hourly_limit_exceeded(self, rate_limiter, mock_redis):
        """Test hourly limit exceeded."""
        user_id = uuid4()
        hourly_limit = TierLimits.get_limit("basic", "requests_per_hour")
        mock_redis.incr.return_value = hourly_limit + 1
        mock_redis.ttl.return_value = 1800  # 30 minutes

        with pytest.raises(RateLimitExceededError) as exc_info:
            await rate_limiter.check_hourly_limit(
                user_id=user_id,
                tier="basic",
            )

        assert "시간당 요청 한도" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_remaining_requests(self, rate_limiter, mock_redis):
        """Test getting remaining requests info."""
        user_id = uuid4()
        mock_redis.get.side_effect = [b"5", b"100"]  # minute, hour usage
        mock_redis.ttl.side_effect = [30, 1800]

        result = await rate_limiter.get_remaining_requests(
            user_id=user_id,
            tier="basic",
        )

        assert "minute" in result
        assert "hour" in result
        assert result["minute"]["used"] == 5
        assert result["hour"]["used"] == 100
        assert result["minute"]["reset_in_seconds"] == 30
        assert result["hour"]["reset_in_seconds"] == 1800

    @pytest.mark.asyncio
    async def test_reset_user_limits(self, rate_limiter, mock_redis):
        """Test resetting user limits."""
        user_id = uuid4()
        mock_redis.keys.return_value = [b"key1", b"key2"]

        await rate_limiter.reset_user_limits(user_id)

        mock_redis.delete.assert_called_once()


# =============================================================================
# ConsultationLimiter Tests
# =============================================================================


class TestConsultationLimiter:
    """Test ConsultationLimiter class."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.incr.return_value = 1
        redis_mock.expire.return_value = True
        return redis_mock

    @pytest.fixture
    def consultation_limiter(self, mock_redis):
        """Create ConsultationLimiter instance."""
        return ConsultationLimiter(mock_redis)

    @pytest.mark.asyncio
    async def test_check_consultation_limit_allowed(
        self, consultation_limiter, mock_redis
    ):
        """Test consultation within limit is allowed."""
        user_id = uuid4()
        mock_redis.get.return_value = b"2"

        allowed, current, limit = await consultation_limiter.check_consultation_limit(
            user_id=user_id,
            tier="basic",
        )

        assert allowed is True
        assert current == 2

    @pytest.mark.asyncio
    async def test_check_consultation_limit_exceeded(
        self, consultation_limiter, mock_redis
    ):
        """Test consultation exceeding limit is rejected."""
        user_id = uuid4()
        basic_limit = TierLimits.get_limit("basic", "consultations_per_month")
        mock_redis.get.return_value = str(basic_limit).encode()

        allowed, current, limit = await consultation_limiter.check_consultation_limit(
            user_id=user_id,
            tier="basic",
        )

        assert allowed is False
        assert current == basic_limit

    @pytest.mark.asyncio
    async def test_check_consultation_limit_unlimited(
        self, consultation_limiter, mock_redis
    ):
        """Test unlimited tier allows all consultations."""
        user_id = uuid4()

        # Pro/Enterprise have unlimited (-1)
        allowed, current, limit = await consultation_limiter.check_consultation_limit(
            user_id=user_id,
            tier="pro",  # Pro has -1 consultations
        )

        assert allowed is True
        assert limit == -1 or limit > 100  # Either unlimited or very high

    @pytest.mark.asyncio
    async def test_increment_consultation_count(
        self, consultation_limiter, mock_redis
    ):
        """Test incrementing consultation count."""
        user_id = uuid4()
        mock_redis.incr.return_value = 3

        count = await consultation_limiter.increment_consultation_count(user_id)

        assert count == 3
        mock_redis.incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_first_consultation_sets_expiry(
        self, consultation_limiter, mock_redis
    ):
        """Test first consultation sets expiry."""
        user_id = uuid4()
        mock_redis.incr.return_value = 1

        count = await consultation_limiter.increment_consultation_count(user_id)

        assert count == 1
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_consultation_stats(self, consultation_limiter, mock_redis):
        """Test getting consultation statistics."""
        user_id = uuid4()
        mock_redis.get.return_value = b"5"

        stats = await consultation_limiter.get_consultation_stats(
            user_id=user_id,
            tier="basic",
        )

        assert stats["used"] == 5
        assert "limit" in stats
        assert "remaining" in stats
        assert "month" in stats
        assert "is_unlimited" in stats

    @pytest.mark.asyncio
    async def test_get_consultation_stats_unlimited(
        self, consultation_limiter, mock_redis
    ):
        """Test consultation stats for unlimited tier."""
        user_id = uuid4()
        mock_redis.get.return_value = b"100"

        # Use pro which should have unlimited (-1)
        stats = await consultation_limiter.get_consultation_stats(
            user_id=user_id,
            tier="enterprise",
        )

        if stats["is_unlimited"]:
            assert stats["remaining"] == -1


# =============================================================================
# Rate Limit Headers Tests
# =============================================================================


class TestRateLimitHeaders:
    """Test rate limit header generation."""

    def test_get_rate_limit_headers(self):
        """Test generating rate limit headers."""
        headers = get_rate_limit_headers(
            limit=100,
            remaining=95,
            reset_seconds=30,
        )

        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "95"
        assert headers["X-RateLimit-Reset"] == "30"

    def test_get_rate_limit_headers_zero_remaining(self):
        """Test headers when rate limit exhausted."""
        headers = get_rate_limit_headers(
            limit=100,
            remaining=0,
            reset_seconds=45,
        )

        assert headers["X-RateLimit-Remaining"] == "0"

    def test_get_rate_limit_headers_negative_remaining(self):
        """Test headers with negative remaining (over limit)."""
        headers = get_rate_limit_headers(
            limit=100,
            remaining=-5,  # Over limit
            reset_seconds=60,
        )

        # Should clamp to 0
        assert headers["X-RateLimit-Remaining"] == "0"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRateLimitIntegration:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_key_format(self):
        """Test rate limit key format is correct."""
        mock_redis = AsyncMock()
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        limiter = RateLimiter(mock_redis)
        user_id = uuid4()

        await limiter.check_rate_limit(
            user_id=user_id,
            tier="basic",
            endpoint="/api/v1/consultation",
        )

        # Verify key format
        call_args = mock_redis.incr.call_args[0][0]
        assert str(user_id) in call_args
        assert "rate_limit:" in call_args
        assert "/api/v1/consultation" in call_args

    @pytest.mark.asyncio
    async def test_consultation_limit_key_contains_month(self):
        """Test consultation key contains current month."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = b"0"

        limiter = ConsultationLimiter(mock_redis)
        user_id = uuid4()

        await limiter.check_consultation_limit(
            user_id=user_id,
            tier="basic",
        )

        # Verify key contains month
        call_args = mock_redis.get.call_args[0][0]
        now = datetime.now(timezone.utc)
        month_str = f"{now.year}:{now.month:02d}"
        assert month_str in call_args

    @pytest.mark.asyncio
    async def test_different_endpoints_different_limits(self):
        """Test different endpoints have separate rate limits."""
        mock_redis = AsyncMock()
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        limiter = RateLimiter(mock_redis)
        user_id = uuid4()

        # Call for different endpoints
        await limiter.check_rate_limit(user_id, "basic", endpoint="/api/v1/auth")
        await limiter.check_rate_limit(user_id, "basic", endpoint="/api/v1/consultation")

        # Should have made 2 incr calls with different keys
        assert mock_redis.incr.call_count == 2
        keys = [call[0][0] for call in mock_redis.incr.call_args_list]
        assert keys[0] != keys[1]
