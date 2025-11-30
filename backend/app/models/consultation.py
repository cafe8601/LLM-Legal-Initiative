"""
Consultation Models

법률 상담 관련 모델
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.document import Citation
    from app.models.memory import UserMemory


class ConsultationStatus(str):
    """Consultation status values."""

    PENDING = "pending"  # 대기 중
    PROCESSING = "processing"  # 처리 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패


class ConsultationCategory(str):
    """Legal consultation categories."""

    GENERAL = "general"  # 일반 법률
    CONTRACT = "contract"  # 계약 법률
    INTELLECTUAL_PROPERTY = "intellectual-property"  # 지식재산권
    LABOR = "labor"  # 노동 법률
    CRIMINAL = "criminal"  # 형사 법률
    ADMINISTRATIVE = "administrative"  # 행정 법률
    CORPORATE = "corporate"  # 회사/상법
    FAMILY = "family"  # 가족/상속법
    REAL_ESTATE = "real-estate"  # 부동산


class Consultation(Base):
    """Main consultation session."""

    __tablename__ = "consultations"

    # Foreign Keys
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Consultation info
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(
        String(50),
        default=ConsultationCategory.GENERAL,
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=ConsultationStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Summary (generated after completion)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="consultations")
    turns: Mapped[list["ConsultationTurn"]] = relationship(
        "ConsultationTurn",
        back_populates="consultation",
        order_by="ConsultationTurn.turn_number",
        lazy="dynamic",
    )
    memories: Mapped[list["UserMemory"]] = relationship(
        "UserMemory",
        back_populates="consultation",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<Consultation(id={self.id}, title={self.title}, status={self.status})>"


class ConsultationTurn(Base):
    """Single turn in a consultation (question + response)."""

    __tablename__ = "consultation_turns"

    # Foreign Keys
    consultation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Turn info
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # User input
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    attached_document_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Chairman's final response
    chairman_response: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default=ConsultationStatus.PENDING,
        nullable=False,
    )

    # Processing metadata
    processing_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    processing_completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Token usage
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    consultation: Mapped["Consultation"] = relationship(
        "Consultation",
        back_populates="turns",
    )
    model_opinions: Mapped[list["ModelOpinion"]] = relationship(
        "ModelOpinion",
        back_populates="turn",
        lazy="selectin",
    )
    peer_reviews: Mapped[list["PeerReview"]] = relationship(
        "PeerReview",
        back_populates="turn",
        lazy="selectin",
    )
    citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        back_populates="turn",
        lazy="selectin",
    )


class ModelOpinion(Base):
    """Individual LLM model's opinion (Stage 1)."""

    __tablename__ = "model_opinions"

    # Foreign Keys
    turn_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultation_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Model info
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Opinion content
    opinion_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Structured analysis (from v4.1 prompt)
    legal_basis: Mapped[str | None] = mapped_column(Text, nullable=True)  # 법적 근거
    risk_assessment: Mapped[str | None] = mapped_column(Text, nullable=True)  # 위험 요소
    recommendations: Mapped[str | None] = mapped_column(Text, nullable=True)  # 권고 사항
    confidence_level: Mapped[str | None] = mapped_column(String(20), nullable=True)  # 확신도

    # Processing metadata
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Raw response (for debugging)
    raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    turn: Mapped["ConsultationTurn"] = relationship(
        "ConsultationTurn",
        back_populates="model_opinions",
    )

    def __repr__(self) -> str:
        return f"<ModelOpinion(model={self.model_name}, turn_id={self.turn_id})>"


class PeerReview(Base):
    """Peer review of model opinions (Stage 2)."""

    __tablename__ = "peer_reviews"

    # Foreign Keys
    turn_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultation_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewed_opinion_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_opinions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Reviewer info
    reviewer_model: Mapped[str] = mapped_column(String(50), nullable=False)

    # Review content
    review_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Evaluation scores (1-5 scale)
    accuracy_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completeness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    practicality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    legal_basis_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Structured feedback
    strengths: Mapped[str | None] = mapped_column(Text, nullable=True)
    weaknesses: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggestions: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Processing metadata
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    turn: Mapped["ConsultationTurn"] = relationship(
        "ConsultationTurn",
        back_populates="peer_reviews",
    )

    def __repr__(self) -> str:
        return f"<PeerReview(reviewer={self.reviewer_model}, opinion_id={self.reviewed_opinion_id})>"
