"""
Document Models

문서 업로드 및 인용 관련 모델
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.consultation import ConsultationTurn


class DocumentType(str):
    """Document type values."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    WORD = "word"
    EXCEL = "excel"


class Document(Base):
    """User uploaded documents."""

    __tablename__ = "documents"

    # Foreign Keys
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # File info
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)  # bytes

    # Storage
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_bucket: Mapped[str] = mapped_column(String(100), nullable=False)

    # OCR/Extraction
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Metadata
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)  # Soft delete
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, name={self.original_name})>"


class Citation(Base):
    """Legal citations from RAG search."""

    __tablename__ = "citations"

    # Foreign Keys
    turn_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultation_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Citation info
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Document type
    doc_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # law, precedent, etc.
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Legal identifiers
    case_number: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 사건번호
    law_number: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 법률번호
    article_number: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 조문번호

    # Search metadata
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    search_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Citation order
    display_order: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    turn: Mapped["ConsultationTurn"] = relationship(
        "ConsultationTurn",
        back_populates="citations",
    )

    def __repr__(self) -> str:
        return f"<Citation(title={self.title[:50]}, source={self.source})>"
