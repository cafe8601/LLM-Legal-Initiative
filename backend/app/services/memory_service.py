"""
Memory Service

v4.1 법률 자문 메모리 시스템 서비스
- Session Memory: 현재 상담 세션 컨텍스트 관리
- Short-term Memory: 최근 상담 내역 조회 (7일)
- Long-term Memory: 사용자별 법률 패턴 및 선호도 학습

법률 자문 위원들이 메모리를 검색하여 개인화된 자문 제공
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.memory import (
    UserMemory,
    ConversationHistory,
    LegalPattern,
    MemoryType,
    MemoryPriority,
)
from app.models.consultation import Consultation, ConsultationTurn

logger = logging.getLogger(__name__)


class MemoryService:
    """
    v4.1 메모리 시스템 서비스.

    3가지 메모리 유형을 통합 관리:
    - Session Memory: 현재 세션 컨텍스트
    - Short-term Memory: 최근 7일 상담 요약
    - Long-term Memory: 사용자 패턴 및 선호도
    """

    # Memory configuration
    SHORT_TERM_DAYS = 7
    LONG_TERM_MIN_OCCURRENCES = 3
    MAX_SESSION_MEMORIES = 10
    MAX_SHORT_TERM_RESULTS = 5
    MAX_LONG_TERM_PATTERNS = 10

    def __init__(self, db: AsyncSession):
        self.db = db

    # ========================================
    # Session Memory (현재 세션 컨텍스트)
    # ========================================

    async def save_session_memory(
        self,
        user_id: UUID,
        consultation_id: UUID,
        key: str,
        content: str,
        category: str | None = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: list[str] | None = None,
    ) -> UserMemory:
        """현재 세션의 중요 컨텍스트 저장."""
        memory = UserMemory(
            user_id=user_id,
            consultation_id=consultation_id,
            memory_type=MemoryType.SESSION.value,
            priority=priority.value,
            key=key,
            content=content,
            category=category,
            tags=tags,
            is_active=True,
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)
        logger.info(f"Session memory saved: user={user_id}, key={key}")
        return memory

    async def get_session_memory(
        self,
        user_id: UUID,
        consultation_id: UUID,
    ) -> str:
        """
        현재 세션의 메모리를 LLM 프롬프트용 문자열로 반환.

        Returns:
            세션 메모리 문자열 (LLM 프롬프트에 주입)
        """
        result = await self.db.execute(
            select(UserMemory)
            .where(
                and_(
                    UserMemory.user_id == user_id,
                    UserMemory.consultation_id == consultation_id,
                    UserMemory.memory_type == MemoryType.SESSION.value,
                    UserMemory.is_active == True,
                )
            )
            .order_by(desc(UserMemory.created_at))
            .limit(self.MAX_SESSION_MEMORIES)
        )
        memories = result.scalars().all()

        if not memories:
            return ""

        # Format for LLM prompt
        lines = ["[현재 세션 컨텍스트]"]
        for mem in memories:
            lines.append(f"- {mem.key}: {mem.content}")

        return "\n".join(lines)

    async def clear_session_memory(
        self,
        user_id: UUID,
        consultation_id: UUID,
    ) -> int:
        """세션 종료 시 세션 메모리 정리."""
        result = await self.db.execute(
            select(UserMemory).where(
                and_(
                    UserMemory.user_id == user_id,
                    UserMemory.consultation_id == consultation_id,
                    UserMemory.memory_type == MemoryType.SESSION.value,
                )
            )
        )
        memories = result.scalars().all()
        count = len(memories)

        for mem in memories:
            mem.is_active = False

        await self.db.commit()
        logger.info(f"Cleared {count} session memories for consultation {consultation_id}")
        return count

    # ========================================
    # Short-term Memory (최근 7일 상담)
    # ========================================

    async def save_conversation_turn(
        self,
        user_id: UUID,
        consultation_id: UUID,
        turn_id: UUID | None,
        turn_number: int,
        role: str,
        content: str,
        category: str | None = None,
        keywords: list[str] | None = None,
        legal_entities: list[str] | None = None,
    ) -> ConversationHistory:
        """대화 턴을 히스토리에 저장."""
        history = ConversationHistory(
            user_id=user_id,
            consultation_id=consultation_id,
            turn_id=turn_id,
            turn_number=turn_number,
            role=role,
            content=content,
            category=category,
            keywords=keywords,
            legal_entities=legal_entities,
            tokens_count=len(content.split()),  # 간단한 토큰 추정
        )
        self.db.add(history)
        await self.db.commit()
        await self.db.refresh(history)
        return history

    async def get_short_term_memory(
        self,
        user_id: UUID,
        current_consultation_id: UUID | None = None,
        category: str | None = None,
    ) -> str:
        """
        최근 7일간의 상담 내역을 LLM 프롬프트용 문자열로 반환.

        Args:
            user_id: 사용자 ID
            current_consultation_id: 현재 상담 ID (제외할 경우)
            category: 필터링할 법률 분야

        Returns:
            단기 메모리 문자열 (LLM 프롬프트에 주입)
        """
        since = datetime.now(timezone.utc) - timedelta(days=self.SHORT_TERM_DAYS)

        query = (
            select(ConversationHistory)
            .where(
                and_(
                    ConversationHistory.user_id == user_id,
                    ConversationHistory.created_at >= since,
                )
            )
            .order_by(desc(ConversationHistory.created_at))
        )

        # 현재 상담 제외
        if current_consultation_id:
            query = query.where(
                ConversationHistory.consultation_id != current_consultation_id
            )

        # 카테고리 필터
        if category:
            query = query.where(ConversationHistory.category == category)

        result = await self.db.execute(query.limit(self.MAX_SHORT_TERM_RESULTS * 2))
        histories = result.scalars().all()

        if not histories:
            return ""

        # Format for LLM prompt
        lines = ["[최근 상담 내역 (7일)]"]
        consultation_summaries = {}

        for hist in histories:
            cid = str(hist.consultation_id)
            if cid not in consultation_summaries:
                consultation_summaries[cid] = {
                    "date": hist.created_at.strftime("%Y-%m-%d"),
                    "category": hist.category,
                    "snippets": [],
                }
            if len(consultation_summaries[cid]["snippets"]) < 2:
                # 각 상담에서 최대 2개 스니펫
                snippet = hist.content[:200] + "..." if len(hist.content) > 200 else hist.content
                consultation_summaries[cid]["snippets"].append(f"{hist.role}: {snippet}")

        for cid, summary in list(consultation_summaries.items())[:self.MAX_SHORT_TERM_RESULTS]:
            lines.append(f"\n[{summary['date']}] ({summary['category'] or '일반'})")
            for snippet in summary["snippets"]:
                lines.append(f"  {snippet}")

        return "\n".join(lines)

    async def search_conversation_history(
        self,
        user_id: UUID,
        query: str,
        limit: int = 10,
    ) -> list[ConversationHistory]:
        """키워드로 대화 이력 검색."""
        search_pattern = f"%{query}%"
        result = await self.db.execute(
            select(ConversationHistory)
            .where(
                and_(
                    ConversationHistory.user_id == user_id,
                    or_(
                        ConversationHistory.content.ilike(search_pattern),
                        ConversationHistory.keywords.contains([query]),
                    ),
                )
            )
            .order_by(desc(ConversationHistory.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    # ========================================
    # Long-term Memory (패턴 및 선호도)
    # ========================================

    async def update_legal_pattern(
        self,
        user_id: UUID,
        pattern_type: str,
        pattern_key: str,
        pattern_value: str,
        consultation_id: UUID | None = None,
    ) -> LegalPattern:
        """
        법률 패턴 업데이트 또는 생성.

        패턴이 이미 존재하면 occurrence_count 증가,
        없으면 새로 생성.
        """
        # 기존 패턴 검색
        result = await self.db.execute(
            select(LegalPattern).where(
                and_(
                    LegalPattern.user_id == user_id,
                    LegalPattern.pattern_type == pattern_type,
                    LegalPattern.pattern_key == pattern_key,
                )
            )
        )
        pattern = result.scalar_one_or_none()

        if pattern:
            # 기존 패턴 업데이트
            pattern.occurrence_count += 1
            pattern.last_occurrence_at = datetime.now(timezone.utc)
            pattern.confidence_score = min(
                1.0,
                pattern.confidence_score + 0.1
            )
            if consultation_id:
                sources = pattern.source_consultations or []
                sources.append(str(consultation_id))
                pattern.source_consultations = sources[-10:]  # 최근 10개만 유지
        else:
            # 새 패턴 생성
            pattern = LegalPattern(
                user_id=user_id,
                pattern_type=pattern_type,
                pattern_key=pattern_key,
                pattern_value=pattern_value,
                occurrence_count=1,
                confidence_score=0.3,
                last_occurrence_at=datetime.now(timezone.utc),
                source_consultations=[str(consultation_id)] if consultation_id else None,
            )
            self.db.add(pattern)

        await self.db.commit()
        await self.db.refresh(pattern)
        return pattern

    async def get_long_term_memory(
        self,
        user_id: UUID,
        category: str | None = None,
    ) -> str:
        """
        사용자의 장기 메모리를 LLM 프롬프트용 문자열로 반환.

        자주 묻는 법률 분야, 반복되는 이슈, 선호 스타일 등.
        """
        query = (
            select(LegalPattern)
            .where(
                and_(
                    LegalPattern.user_id == user_id,
                    LegalPattern.occurrence_count >= self.LONG_TERM_MIN_OCCURRENCES,
                )
            )
            .order_by(desc(LegalPattern.confidence_score))
            .limit(self.MAX_LONG_TERM_PATTERNS)
        )

        result = await self.db.execute(query)
        patterns = result.scalars().all()

        if not patterns:
            return ""

        # Format for LLM prompt
        lines = ["[사용자 법률 패턴 및 선호도]"]

        # 그룹별 정리
        by_type: dict[str, list[LegalPattern]] = {}
        for p in patterns:
            if p.pattern_type not in by_type:
                by_type[p.pattern_type] = []
            by_type[p.pattern_type].append(p)

        type_names = {
            "category": "자주 묻는 분야",
            "issue": "반복되는 이슈",
            "style": "선호 스타일",
        }

        for ptype, plist in by_type.items():
            lines.append(f"\n{type_names.get(ptype, ptype)}:")
            for p in plist[:3]:  # 각 유형당 최대 3개
                confidence = "높음" if p.confidence_score > 0.7 else "중간" if p.confidence_score > 0.4 else "낮음"
                lines.append(f"  - {p.pattern_key}: {p.pattern_value} (확신도: {confidence})")

        return "\n".join(lines)

    async def learn_from_consultation(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: str,
        keywords: list[str],
        summary: str | None = None,
    ) -> None:
        """
        완료된 상담에서 장기 패턴 학습.

        상담이 완료되면 호출하여 사용자 패턴을 업데이트.
        """
        # 카테고리 패턴 업데이트
        await self.update_legal_pattern(
            user_id=user_id,
            pattern_type="category",
            pattern_key=category,
            pattern_value=f"{category} 관련 상담",
            consultation_id=consultation_id,
        )

        # 키워드 기반 이슈 패턴 업데이트
        for keyword in keywords[:5]:  # 상위 5개 키워드만
            await self.update_legal_pattern(
                user_id=user_id,
                pattern_type="issue",
                pattern_key=keyword,
                pattern_value=f"{keyword} 관련 법률 문의",
                consultation_id=consultation_id,
            )

        logger.info(f"Learned patterns from consultation {consultation_id}")

    # ========================================
    # 통합 메모리 조회 (법률 자문 위원용)
    # ========================================

    async def get_memory_context(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: str | None = None,
    ) -> dict[str, str]:
        """
        모든 메모리를 통합하여 반환.

        법률 자문 위원들이 사용할 컨텍스트 생성.

        Returns:
            {
                "session_memory": "...",
                "short_term_memory": "...",
                "long_term_memory": "..."
            }
        """
        session_mem = await self.get_session_memory(user_id, consultation_id)
        short_term_mem = await self.get_short_term_memory(
            user_id, consultation_id, category
        )
        long_term_mem = await self.get_long_term_memory(user_id, category)

        return {
            "session_memory": session_mem,
            "short_term_memory": short_term_mem,
            "long_term_memory": long_term_mem,
        }

    # ========================================
    # 메모리 검색 (법률 자문 위원용)
    # ========================================

    async def search_all_memories(
        self,
        user_id: UUID,
        query: str,
        include_history: bool = True,
        include_patterns: bool = True,
    ) -> dict[str, Any]:
        """
        모든 메모리 유형에서 검색.

        법률 자문 위원이 관련 이전 정보를 찾을 때 사용.
        """
        results: dict[str, Any] = {
            "conversation_history": [],
            "legal_patterns": [],
        }

        if include_history:
            results["conversation_history"] = await self.search_conversation_history(
                user_id, query
            )

        if include_patterns:
            search_pattern = f"%{query}%"
            pattern_result = await self.db.execute(
                select(LegalPattern)
                .where(
                    and_(
                        LegalPattern.user_id == user_id,
                        or_(
                            LegalPattern.pattern_key.ilike(search_pattern),
                            LegalPattern.pattern_value.ilike(search_pattern),
                        ),
                    )
                )
                .order_by(desc(LegalPattern.confidence_score))
                .limit(10)
            )
            results["legal_patterns"] = list(pattern_result.scalars().all())

        return results

    # ========================================
    # 유틸리티
    # ========================================

    async def cleanup_expired_memories(self) -> int:
        """만료된 메모리 정리 (정기 작업용)."""
        now = datetime.now(timezone.utc)

        # 만료된 short-term 메모리 비활성화
        result = await self.db.execute(
            select(UserMemory).where(
                and_(
                    UserMemory.expires_at != None,
                    UserMemory.expires_at < now,
                    UserMemory.is_active == True,
                )
            )
        )
        expired = result.scalars().all()

        for mem in expired:
            mem.is_active = False

        await self.db.commit()
        logger.info(f"Cleaned up {len(expired)} expired memories")
        return len(expired)

    async def get_memory_statistics(
        self,
        user_id: UUID,
    ) -> dict[str, Any]:
        """사용자 메모리 통계."""
        # 메모리 카운트
        memory_count = await self.db.execute(
            select(func.count())
            .select_from(UserMemory)
            .where(
                and_(
                    UserMemory.user_id == user_id,
                    UserMemory.is_active == True,
                )
            )
        )

        # 대화 이력 카운트
        history_count = await self.db.execute(
            select(func.count())
            .select_from(ConversationHistory)
            .where(ConversationHistory.user_id == user_id)
        )

        # 패턴 카운트
        pattern_count = await self.db.execute(
            select(func.count())
            .select_from(LegalPattern)
            .where(LegalPattern.user_id == user_id)
        )

        return {
            "active_memories": memory_count.scalar() or 0,
            "conversation_turns": history_count.scalar() or 0,
            "legal_patterns": pattern_count.scalar() or 0,
        }
