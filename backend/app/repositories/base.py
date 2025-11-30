"""
Base Repository

공통 데이터 액세스 패턴
"""

from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base


ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic base repository with common CRUD operations.

    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, db: AsyncSession):
                super().__init__(User, db)
    """

    def __init__(self, model: type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: UUID) -> ModelType | None:
        """Get a single record by ID."""
        result = await self.db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_by_ids(self, ids: list[UUID]) -> list[ModelType]:
        """Get multiple records by IDs."""
        result = await self.db.execute(select(self.model).where(self.model.id.in_(ids)))
        return list(result.scalars().all())

    async def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: str | None = None,
        order_desc: bool = True,
    ) -> list[ModelType]:
        """Get all records with pagination."""
        query = select(self.model)

        if order_by:
            order_column = getattr(self.model, order_by, None)
            if order_column is not None:
                query = query.order_by(order_column.desc() if order_desc else order_column.asc())
        else:
            query = query.order_by(self.model.created_at.desc())

        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Count all records."""
        result = await self.db.execute(select(func.count()).select_from(self.model))
        return result.scalar() or 0

    async def create(self, **kwargs: Any) -> ModelType:
        """Create a new record."""
        instance = self.model(**kwargs)
        self.db.add(instance)
        await self.db.flush()
        await self.db.refresh(instance)
        return instance

    async def update(self, id: UUID, **kwargs: Any) -> ModelType | None:
        """Update a record by ID."""
        instance = await self.get(id)
        if instance is None:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        await self.db.flush()
        await self.db.refresh(instance)
        return instance

    async def update_bulk(self, ids: list[UUID], **kwargs: Any) -> int:
        """Update multiple records by IDs."""
        result = await self.db.execute(
            update(self.model).where(self.model.id.in_(ids)).values(**kwargs)
        )
        return result.rowcount

    async def delete(self, id: UUID) -> bool:
        """Delete a record by ID."""
        result = await self.db.execute(delete(self.model).where(self.model.id == id))
        return result.rowcount > 0

    async def delete_bulk(self, ids: list[UUID]) -> int:
        """Delete multiple records by IDs."""
        result = await self.db.execute(delete(self.model).where(self.model.id.in_(ids)))
        return result.rowcount

    async def exists(self, id: UUID) -> bool:
        """Check if a record exists."""
        result = await self.db.execute(
            select(func.count()).select_from(self.model).where(self.model.id == id)
        )
        return (result.scalar() or 0) > 0

    async def get_or_create(self, defaults: dict[str, Any] | None = None, **kwargs: Any) -> tuple[ModelType, bool]:
        """
        Get an existing record or create a new one.

        Returns:
            Tuple of (instance, created) where created is True if a new record was created.
        """
        query = select(self.model)
        for key, value in kwargs.items():
            query = query.where(getattr(self.model, key) == value)

        result = await self.db.execute(query)
        instance = result.scalar_one_or_none()

        if instance:
            return instance, False

        create_kwargs = {**kwargs, **(defaults or {})}
        instance = await self.create(**create_kwargs)
        return instance, True
