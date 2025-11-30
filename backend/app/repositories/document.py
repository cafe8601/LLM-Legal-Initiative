"""
Document Repository

문서 및 인용 데이터 액세스 레이어
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, Citation, DocumentType
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model."""

    def __init__(self, db: AsyncSession):
        super().__init__(Document, db)

    async def get_user_documents(
        self,
        user_id: UUID,
        *,
        skip: int = 0,
        limit: int = 50,
        file_type: str | None = None,
        include_deleted: bool = False,
    ) -> list[Document]:
        """Get documents for a specific user."""
        query = select(Document).where(Document.user_id == user_id)

        if not include_deleted:
            query = query.where(Document.is_deleted == False)

        if file_type:
            query = query.where(Document.file_type == file_type)

        query = query.order_by(Document.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count_user_documents(
        self,
        user_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> int:
        """Count documents for a user."""
        query = select(func.count()).select_from(Document).where(
            Document.user_id == user_id
        )

        if not include_deleted:
            query = query.where(Document.is_deleted == False)

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_by_storage_path(self, storage_path: str) -> Document | None:
        """Get document by storage path."""
        result = await self.db.execute(
            select(Document).where(Document.storage_path == storage_path)
        )
        return result.scalar_one_or_none()

    async def soft_delete(self, document_id: UUID) -> Document | None:
        """Soft delete a document."""
        return await self.update(
            document_id,
            is_deleted=True,
            deleted_at=datetime.now(timezone.utc),
        )

    async def restore(self, document_id: UUID) -> Document | None:
        """Restore a soft-deleted document."""
        return await self.update(
            document_id,
            is_deleted=False,
            deleted_at=None,
        )

    async def get_unprocessed(self, *, limit: int = 100) -> list[Document]:
        """Get documents that haven't been processed yet."""
        result = await self.db.execute(
            select(Document)
            .where(
                and_(
                    Document.is_processed == False,
                    Document.is_deleted == False,
                )
            )
            .order_by(Document.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_as_processed(
        self,
        document_id: UUID,
        *,
        extracted_text: str | None = None,
        ocr_confidence: float | None = None,
        page_count: int | None = None,
    ) -> Document | None:
        """Mark document as processed with OCR results."""
        update_data: dict[str, Any] = {
            "is_processed": True,
            "ocr_completed": True,
        }

        if extracted_text is not None:
            update_data["extracted_text"] = extracted_text
        if ocr_confidence is not None:
            update_data["ocr_confidence"] = ocr_confidence
        if page_count is not None:
            update_data["page_count"] = page_count

        return await self.update(document_id, **update_data)

    async def search_by_content(
        self,
        user_id: UUID,
        query: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> list[Document]:
        """Search documents by extracted text content."""
        search_pattern = f"%{query}%"
        result = await self.db.execute(
            select(Document)
            .where(
                and_(
                    Document.user_id == user_id,
                    Document.is_deleted == False,
                    or_(
                        Document.original_name.ilike(search_pattern),
                        Document.extracted_text.ilike(search_pattern),
                        Document.description.ilike(search_pattern),
                    ),
                )
            )
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_storage_usage(self, user_id: UUID) -> int:
        """Get total storage usage in bytes for a user."""
        result = await self.db.execute(
            select(func.sum(Document.file_size)).where(
                and_(
                    Document.user_id == user_id,
                    Document.is_deleted == False,
                )
            )
        )
        return result.scalar() or 0


class CitationRepository(BaseRepository[Citation]):
    """Repository for Citation model."""

    def __init__(self, db: AsyncSession):
        super().__init__(Citation, db)

    async def get_turn_citations(
        self,
        turn_id: UUID,
        *,
        order_by_relevance: bool = True,
    ) -> list[Citation]:
        """Get all citations for a turn."""
        query = select(Citation).where(Citation.turn_id == turn_id)

        if order_by_relevance:
            query = query.order_by(Citation.relevance_score.desc())
        else:
            query = query.order_by(Citation.display_order)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_case_number(self, case_number: str) -> list[Citation]:
        """Get citations by case number."""
        result = await self.db.execute(
            select(Citation).where(Citation.case_number == case_number)
        )
        return list(result.scalars().all())

    async def get_by_law_number(self, law_number: str) -> list[Citation]:
        """Get citations by law number."""
        result = await self.db.execute(
            select(Citation).where(Citation.law_number == law_number)
        )
        return list(result.scalars().all())

    async def search_citations(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        category: str | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> list[Citation]:
        """Search citations by title or content."""
        search_pattern = f"%{query}%"
        base_query = select(Citation).where(
            or_(
                Citation.title.ilike(search_pattern),
                Citation.content.ilike(search_pattern),
            )
        )

        if doc_type:
            base_query = base_query.where(Citation.doc_type == doc_type)
        if category:
            base_query = base_query.where(Citation.category == category)

        result = await self.db.execute(
            base_query.order_by(Citation.relevance_score.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create_bulk(self, citations_data: list[dict[str, Any]]) -> list[Citation]:
        """Create multiple citations at once."""
        citations = [Citation(**data) for data in citations_data]
        self.db.add_all(citations)
        await self.db.flush()

        for citation in citations:
            await self.db.refresh(citation)

        return citations

    async def get_statistics_by_type(self) -> dict[str, int]:
        """Get citation count by document type."""
        result = await self.db.execute(
            select(Citation.doc_type, func.count(Citation.id)).group_by(Citation.doc_type)
        )
        return {row[0] or "unknown": row[1] for row in result.all()}

    async def get_most_cited(
        self,
        *,
        doc_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, int]]:
        """Get most frequently cited sources."""
        query = select(Citation.source, func.count(Citation.id).label("count"))

        if doc_type:
            query = query.where(Citation.doc_type == doc_type)

        query = query.group_by(Citation.source).order_by(func.count(Citation.id).desc()).limit(limit)

        result = await self.db.execute(query)
        return [(row[0], row[1]) for row in result.all()]
