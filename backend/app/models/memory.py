"""
Memory Models

v4.1 법률 자문 메모리 시스템 모델
- Session Memory: 현재 상담 세션 컨텍스트
- Short-term Memory: 최근 상담 내역 (7일)
- Long-term Memory: 사용자별 중요 법률 패턴 및 선호도
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.consultation import Consultation


class MemoryType(str, Enum):
    """Memory type classifications."""

    SESSION = "session"  # 현재 세션 컨텍스트
    SHORT_TERM = "short_term"  # 최근 상담 (7일)
    LONG_TERM = "long_term"  # 장기 패턴 및 선호도


class MemoryPriority(str, Enum):
    """Memory priority levels for retention."""

    LOW = "low"  # 일반 정보
    MEDIUM = "medium"  # 중요 정보
    HIGH = "high"  # 핵심 정보 (장기 보존)
    CRITICAL = "critical"  # 필수 정보 (영구 보존)


class UserMemory(Base):
    """
    사용자별 메모리 저장소.

    3가지 메모리 유형을 통합 관리:
    - Session: 현재 상담 세션의 컨텍스트
    - Short-term: 최근 7일간의 상담 요약
    - Long-term: 사용자의 법률 패턴 및 선호도
    """

    __tablename__ = "user_memories"

    # Foreign Keys
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    consultation_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Memory classification
    memory_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        default=MemoryPriority.MEDIUM.value,
        nullable=False,
    )

    # Memory content
    key: Mapped[str] = mapped_column(String(255), nullable=False)  # 검색용 키
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 실제 메모리 내용
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)  # 요약 (LLM 생성)

    # Metadata
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 법률 분야
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 검색 태그
    embedding_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 벡터 임베딩 ID

    # Relevance scoring
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Expiration
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)  # Short-term용
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memories")
    consultation: Mapped["Consultation | None"] = relationship(
        "Consultation",
        back_populates="memories",
    )

    # Composite indexes for efficient queries
    __table_args__ = (
        Index("ix_user_memories_user_type", "user_id", "memory_type"),
        Index("ix_user_memories_user_active", "user_id", "is_active"),
        Index("ix_user_memories_category", "user_id", "category"),
    )

    def __repr__(self) -> str:
        return f"<UserMemory(user_id={self.user_id}, type={self.memory_type}, key={self.key})>"


class ConversationHistory(Base):
    """
    상담 대화 이력.

    각 턴의 질문-응답을 저장하여 컨텍스트 구축에 사용.
    법률 자문 위원들이 이전 대화를 검색할 수 있도록 함.
    """

    __tablename__ = "conversation_histories"

    # Foreign Keys
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    consultation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("consultation_turns.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Conversation details
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tokens_count: Mapped[int] = mapped_column(Integer, default=0)

    # Searchability
    keywords: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 추출된 키워드
    legal_entities: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 법률 엔티티
    embedding_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Composite indexes
    __table_args__ = (
        Index("ix_conv_history_user_consultation", "user_id", "consultation_id"),
        Index("ix_conv_history_consultation_turn", "consultation_id", "turn_number"),
    )

    def __repr__(self) -> str:
        return f"<ConversationHistory(consultation_id={self.consultation_id}, turn={self.turn_number}, role={self.role})>"


class LegalPattern(Base):
    """
    사용자별 법률 패턴 인식.

    장기 메모리의 핵심 구성요소로, 사용자의:
    - 자주 묻는 법률 분야
    - 선호하는 답변 스타일
    - 반복되는 법률 이슈
    """

    __tablename__ = "legal_patterns"

    # Foreign Keys
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Pattern identification
    pattern_type: Mapped[str] = mapped_column(String(50), nullable=False)  # category, issue, style
    pattern_key: Mapped[str] = mapped_column(String(255), nullable=False)
    pattern_value: Mapped[str] = mapped_column(Text, nullable=False)

    # Statistics
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    last_occurrence_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Learning data
    source_consultations: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 출처 상담 ID들
    related_patterns: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 연관 패턴들

    # Composite index
    __table_args__ = (
        Index("ix_legal_patterns_user_type", "user_id", "pattern_type"),
    )

    def __repr__(self) -> str:
        return f"<LegalPattern(user_id={self.user_id}, type={self.pattern_type}, key={self.pattern_key})>"
