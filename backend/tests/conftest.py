"""
Pytest Configuration and Fixtures

테스트 공통 설정 및 픽스처
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import NullPool, create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash
from app.db.session import get_db
from app.main import create_application
from app.models.base import Base
from app.models.user import User, UserTier


def get_tier_value(tier) -> str:
    """Helper to get tier value regardless of whether it's enum or string."""
    if hasattr(tier, 'value'):
        return tier.value
    return str(tier)


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()


# =============================================================================
# Application Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def app(db_session: AsyncSession) -> FastAPI:
    """Create test FastAPI application with overridden dependencies."""
    test_app = create_application()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    test_app.dependency_overrides[get_db] = override_get_db

    return test_app


@pytest.fixture(scope="function")
def client(app: FastAPI) -> Generator:
    """Create a synchronous test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(scope="function")
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# User Fixtures
# =============================================================================


@pytest.fixture
def test_user_data() -> dict:
    """Create test user registration data."""
    return {
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User",
        "terms_accepted": True,
    }


@pytest.fixture
def test_user_data_pro() -> dict:
    """Create pro tier test user data."""
    return {
        "email": "pro@example.com",
        "password": "ProPassword123!",
        "full_name": "Pro User",
        "terms_accepted": True,
    }


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user in the database."""
    user = User(
        id=uuid4(),
        email="testuser@example.com",
        password_hash=get_password_hash("TestPassword123!"),
        full_name="Test User",
        tier=UserTier.BASIC,
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_user_unverified(db_session: AsyncSession) -> User:
    """Create an unverified test user in the database."""
    user = User(
        id=uuid4(),
        email="unverified@example.com",
        password_hash=get_password_hash("TestPassword123!"),
        full_name="Unverified User",
        tier=UserTier.BASIC,
        is_active=True,
        is_verified=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_user_pro(db_session: AsyncSession) -> User:
    """Create a pro tier test user in the database."""
    user = User(
        id=uuid4(),
        email="prouser@example.com",
        password_hash=get_password_hash("ProPassword123!"),
        full_name="Pro User",
        tier=UserTier.PRO,
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_admin_user(db_session: AsyncSession) -> User:
    """Create an admin test user in the database."""
    user = User(
        id=uuid4(),
        email="admin@example.com",
        password_hash=get_password_hash("AdminPassword123!"),
        full_name="Admin User",
        tier=UserTier.ENTERPRISE,
        is_active=True,
        is_verified=True,
        is_admin=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# =============================================================================
# Authentication Fixtures
# =============================================================================


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers with a valid test token."""
    # Use correct API signature: user_id (UUID), tier (str)
    token = create_access_token(
        user_id=test_user.id,
        tier=get_tier_value(test_user.tier),
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def pro_auth_headers(test_user_pro: User) -> dict:
    """Create authentication headers for pro user."""
    # Use correct API signature: user_id (UUID), tier (str)
    token = create_access_token(
        user_id=test_user_pro.id,
        tier=get_tier_value(test_user_pro.tier),
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(test_admin_user: User) -> dict:
    """Create authentication headers for admin user."""
    # Use correct API signature: user_id (UUID), tier (str)
    token = create_access_token(
        user_id=test_admin_user.id,
        tier=get_tier_value(test_admin_user.tier),
    )
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    redis_mock.ttl.return_value = 60
    redis_mock.delete.return_value = True
    redis_mock.keys.return_value = []
    return redis_mock


@pytest.fixture
def mock_email_service():
    """Create a mock email service."""
    email_mock = MagicMock()
    email_mock.send_verification_email = AsyncMock(return_value=True)
    email_mock.send_password_reset_email = AsyncMock(return_value=True)
    email_mock.send_welcome_email = AsyncMock(return_value=True)
    return email_mock


# =============================================================================
# Utility Functions
# =============================================================================


def create_test_token(
    user_id: str,
    tier: str = "basic",
) -> str:
    """Helper function to create test tokens."""
    from uuid import UUID
    # Convert string to UUID if needed
    uid = UUID(user_id) if isinstance(user_id, str) else user_id
    return create_access_token(
        user_id=uid,
        tier=tier,
    )
