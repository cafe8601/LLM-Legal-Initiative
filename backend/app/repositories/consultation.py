"""
Consultation Repository

법률 상담 데이터 액세스 레이어
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.consultation import (
    Consultation,
    ConsultationTurn,
    ModelOpinion,
    PeerReview,
    ConsultationStatus,
    ConsultationCategory,
)
from app.repositories.base import BaseRepository


class ConsultationRepository(BaseRepository[Consultation]):
    """Repository for Consultation model."""

    def __init__(self, db: AsyncSession):
        super().__init__(Consultation, db)

    async def get_with_turns(self, consultation_id: UUID) -> Consultation | None:
        """Get consultation with all turns loaded."""
        result = await self.db.execute(
            select(Consultation)
            .where(Consultation.id == consultation_id)
            .options(
                selectinload(Consultation.turns).selectinload(ConsultationTurn.model_opinions),
                selectinload(Consultation.turns).selectinload(ConsultationTurn.peer_reviews),
                selectinload(Consultation.turns).selectinload(ConsultationTurn.citations),
            )
        )
        return result.scalar_one_or_none()

    async def get_user_consultations(
        self,
        user_id: UUID,
        *,
        skip: int = 0,
        limit: int = 20,
        status: str | None = None,
        category: str | None = None,
    ) -> list[Consultation]:
        """Get consultations for a specific user."""
        query = select(Consultation).where(Consultation.user_id == user_id)

        if status:
            query = query.where(Consultation.status == status)
        if category:
            query = query.where(Consultation.category == category)

        query = query.order_by(Consultation.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count_user_consultations(
        self,
        user_id: UUID,
        *,
        since: datetime | None = None,
    ) -> int:
        """Count consultations for a user, optionally since a specific date."""
        query = select(func.count()).select_from(Consultation).where(
            Consultation.user_id == user_id
        )

        if since:
            query = query.where(Consultation.created_at >= since)

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def count_user_consultations_this_month(self, user_id: UUID) -> int:
        """Count user's consultations for current month."""
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return await self.count_user_consultations(user_id, since=start_of_month)

    async def search_consultations(
        self,
        user_id: UUID,
        query: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> list[Consultation]:
        """Search user's consultations by title or summary."""
        search_pattern = f"%{query}%"
        result = await self.db.execute(
            select(Consultation)
            .where(
                and_(
                    Consultation.user_id == user_id,
                    or_(
                        Consultation.title.ilike(search_pattern),
                        Consultation.summary.ilike(search_pattern),
                    ),
                )
            )
            .order_by(Consultation.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        consultation_id: UUID,
        status: str,
    ) -> Consultation | None:
        """Update consultation status."""
        return await self.update(consultation_id, status=status)

    async def get_recent_by_category(
        self,
        category: str,
        *,
        limit: int = 10,
        days: int = 30,
    ) -> list[Consultation]:
        """Get recent consultations by category."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.db.execute(
            select(Consultation)
            .where(
                and_(
                    Consultation.category == category,
                    Consultation.created_at >= since,
                    Consultation.status == ConsultationStatus.COMPLETED,
                )
            )
            .order_by(Consultation.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_statistics(
        self,
        user_id: UUID | None = None,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get consultation statistics."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        base_query = select(Consultation).where(Consultation.created_at >= since)

        if user_id:
            base_query = base_query.where(Consultation.user_id == user_id)

        # Total count
        total_result = await self.db.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total = total_result.scalar() or 0

        # By status
        status_result = await self.db.execute(
            select(Consultation.status, func.count(Consultation.id))
            .where(Consultation.created_at >= since)
            .group_by(Consultation.status)
        )
        by_status = {row[0]: row[1] for row in status_result.all()}

        # By category
        category_result = await self.db.execute(
            select(Consultation.category, func.count(Consultation.id))
            .where(Consultation.created_at >= since)
            .group_by(Consultation.category)
        )
        by_category = {row[0]: row[1] for row in category_result.all()}

        # Average tokens and cost
        avg_result = await self.db.execute(
            select(
                func.avg(Consultation.total_tokens_used),
                func.avg(Consultation.total_cost),
            ).where(Consultation.created_at >= since)
        )
        avg_row = avg_result.one()

        return {
            "total": total,
            "by_status": by_status,
            "by_category": by_category,
            "average_tokens": float(avg_row[0] or 0),
            "average_cost": float(avg_row[1] or 0),
        }


class ConsultationTurnRepository(BaseRepository[ConsultationTurn]):
    """Repository for ConsultationTurn model."""

    def __init__(self, db: AsyncSession):
        super().__init__(ConsultationTurn, db)

    async def get_with_details(self, turn_id: UUID) -> ConsultationTurn | None:
        """Get turn with all related data loaded."""
        result = await self.db.execute(
            select(ConsultationTurn)
            .where(ConsultationTurn.id == turn_id)
            .options(
                selectinload(ConsultationTurn.model_opinions),
                selectinload(ConsultationTurn.peer_reviews),
                selectinload(ConsultationTurn.citations),
            )
        )
        return result.scalar_one_or_none()

    async def get_consultation_turns(
        self,
        consultation_id: UUID,
        *,
        include_details: bool = False,
    ) -> list[ConsultationTurn]:
        """Get all turns for a consultation."""
        query = select(ConsultationTurn).where(
            ConsultationTurn.consultation_id == consultation_id
        )

        if include_details:
            query = query.options(
                selectinload(ConsultationTurn.model_opinions),
                selectinload(ConsultationTurn.peer_reviews),
                selectinload(ConsultationTurn.citations),
            )

        query = query.order_by(ConsultationTurn.turn_number)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_latest_turn(self, consultation_id: UUID) -> ConsultationTurn | None:
        """Get the latest turn for a consultation."""
        result = await self.db.execute(
            select(ConsultationTurn)
            .where(ConsultationTurn.consultation_id == consultation_id)
            .order_by(ConsultationTurn.turn_number.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_next_turn_number(self, consultation_id: UUID) -> int:
        """Get the next turn number for a consultation."""
        result = await self.db.execute(
            select(func.max(ConsultationTurn.turn_number)).where(
                ConsultationTurn.consultation_id == consultation_id
            )
        )
        max_turn = result.scalar()
        return (max_turn or 0) + 1

    async def update_status(
        self,
        turn_id: UUID,
        status: str,
        **kwargs: Any,
    ) -> ConsultationTurn | None:
        """Update turn status with optional additional fields."""
        return await self.update(turn_id, status=status, **kwargs)

    async def set_chairman_response(
        self,
        turn_id: UUID,
        response: str,
        *,
        tokens_used: int = 0,
        processing_time_ms: int | None = None,
    ) -> ConsultationTurn | None:
        """Set the chairman's final response."""
        return await self.update(
            turn_id,
            chairman_response=response,
            tokens_used=tokens_used,
            processing_time_ms=processing_time_ms,
            processing_completed_at=datetime.now(timezone.utc),
            status=ConsultationStatus.COMPLETED,
        )


class ModelOpinionRepository(BaseRepository[ModelOpinion]):
    """Repository for ModelOpinion model."""

    def __init__(self, db: AsyncSession):
        super().__init__(ModelOpinion, db)

    async def get_turn_opinions(self, turn_id: UUID) -> list[ModelOpinion]:
        """Get all opinions for a turn."""
        result = await self.db.execute(
            select(ModelOpinion)
            .where(ModelOpinion.turn_id == turn_id)
            .order_by(ModelOpinion.created_at)
        )
        return list(result.scalars().all())

    async def get_by_model(
        self,
        turn_id: UUID,
        model_name: str,
    ) -> ModelOpinion | None:
        """Get opinion by model name for a turn."""
        result = await self.db.execute(
            select(ModelOpinion).where(
                and_(
                    ModelOpinion.turn_id == turn_id,
                    ModelOpinion.model_name == model_name,
                )
            )
        )
        return result.scalar_one_or_none()


class PeerReviewRepository(BaseRepository[PeerReview]):
    """Repository for PeerReview model."""

    def __init__(self, db: AsyncSession):
        super().__init__(PeerReview, db)

    async def get_turn_reviews(self, turn_id: UUID) -> list[PeerReview]:
        """Get all peer reviews for a turn."""
        result = await self.db.execute(
            select(PeerReview)
            .where(PeerReview.turn_id == turn_id)
            .order_by(PeerReview.created_at)
        )
        return list(result.scalars().all())

    async def get_opinion_reviews(self, opinion_id: UUID) -> list[PeerReview]:
        """Get all reviews for a specific opinion."""
        result = await self.db.execute(
            select(PeerReview)
            .where(PeerReview.reviewed_opinion_id == opinion_id)
            .order_by(PeerReview.created_at)
        )
        return list(result.scalars().all())

    async def get_average_scores(self, turn_id: UUID) -> dict[str, float]:
        """Get average review scores for a turn."""
        result = await self.db.execute(
            select(
                func.avg(PeerReview.accuracy_score),
                func.avg(PeerReview.completeness_score),
                func.avg(PeerReview.practicality_score),
                func.avg(PeerReview.legal_basis_score),
                func.avg(PeerReview.overall_score),
            ).where(PeerReview.turn_id == turn_id)
        )
        row = result.one()
        return {
            "accuracy": float(row[0] or 0),
            "completeness": float(row[1] or 0),
            "practicality": float(row[2] or 0),
            "legal_basis": float(row[3] or 0),
            "overall": float(row[4] or 0),
        }
