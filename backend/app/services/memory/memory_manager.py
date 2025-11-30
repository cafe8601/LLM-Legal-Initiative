"""
Memory Manager

통합 메모리 관리자.
PostgreSQL 기반 영구 메모리와 인메모리 캐시를 통합합니다.

Features:
- 통합 메모리 인터페이스
- 자동 캐시 동기화
- 메모리 유형별 최적화된 접근
- 법률 자문 위원 컨텍스트 관리
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.memory.session_cache import ConsultationCache, get_consultation_cache
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """메모리 유형."""
    SESSION = "session"          # 인메모리 세션 캐시
    SHORT_TERM = "short_term"    # PostgreSQL 단기 메모리 (7일)
    LONG_TERM = "long_term"      # PostgreSQL 장기 메모리 (패턴)
    CACHE = "cache"              # 인메모리 캐시 (RAG 결과 등)


@dataclass
class MemoryContext:
    """메모리 컨텍스트 (LLM 프롬프트용)."""
    session_memory: str
    short_term_memory: str
    long_term_memory: str
    cached_context: Optional[dict] = None

    def to_prompt_string(self) -> str:
        """LLM 프롬프트용 문자열로 변환."""
        parts = []

        if self.session_memory:
            parts.append(self.session_memory)

        if self.short_term_memory:
            parts.append(self.short_term_memory)

        if self.long_term_memory:
            parts.append(self.long_term_memory)

        return "\n\n".join(parts)

    @property
    def has_context(self) -> bool:
        """컨텍스트 존재 여부."""
        return bool(self.session_memory or self.short_term_memory or self.long_term_memory)


class MemoryManager:
    """
    통합 메모리 관리자.

    인메모리 캐시와 PostgreSQL 영구 저장소를 통합하여
    법률 자문 위원들에게 최적의 컨텍스트를 제공합니다.
    """

    def __init__(
        self,
        db: AsyncSession,
        cache: Optional[ConsultationCache] = None,
    ):
        """
        초기화.

        Args:
            db: SQLAlchemy AsyncSession
            cache: 세션 캐시 (None이면 싱글톤 사용)
        """
        self.db = db
        self.memory_service = MemoryService(db)
        self.cache = cache or get_consultation_cache()

    # ========================================
    # 통합 메모리 접근
    # ========================================

    async def store(
        self,
        user_id: UUID,
        consultation_id: UUID,
        key: str,
        value: Any,
        memory_type: MemoryType,
        category: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        메모리 저장 (통합 인터페이스).

        Args:
            user_id: 사용자 ID
            consultation_id: 상담 ID
            key: 메모리 키
            value: 저장할 값
            memory_type: 메모리 유형
            category: 법률 분야
            ttl: TTL (CACHE 유형만 적용)
        """
        if memory_type == MemoryType.SESSION:
            # 인메모리 + PostgreSQL 세션 메모리
            self.cache.set(str(consultation_id), key, value, ttl=ttl)

            # 중요한 세션 데이터는 PostgreSQL에도 저장
            if isinstance(value, str):
                await self.memory_service.save_session_memory(
                    user_id=user_id,
                    consultation_id=consultation_id,
                    key=key,
                    content=value,
                    category=category,
                )

        elif memory_type == MemoryType.CACHE:
            # 인메모리 캐시만
            self.cache.set(str(consultation_id), f"cache:{key}", value, ttl=ttl or 300)

        elif memory_type == MemoryType.SHORT_TERM:
            # PostgreSQL 단기 메모리
            if isinstance(value, str):
                await self.memory_service.save_session_memory(
                    user_id=user_id,
                    consultation_id=consultation_id,
                    key=key,
                    content=value,
                    category=category,
                )

        elif memory_type == MemoryType.LONG_TERM:
            # PostgreSQL 장기 패턴
            if isinstance(value, dict):
                await self.memory_service.update_legal_pattern(
                    user_id=user_id,
                    pattern_type=value.get("type", "general"),
                    pattern_key=key,
                    pattern_value=str(value.get("value", "")),
                    consultation_id=consultation_id,
                )

    async def retrieve(
        self,
        user_id: UUID,
        consultation_id: UUID,
        key: str,
        memory_type: MemoryType,
    ) -> Optional[Any]:
        """
        메모리 조회 (통합 인터페이스).

        Args:
            user_id: 사용자 ID
            consultation_id: 상담 ID
            key: 메모리 키
            memory_type: 메모리 유형

        Returns:
            저장된 값 또는 None
        """
        if memory_type == MemoryType.SESSION:
            # 인메모리 캐시 먼저 확인
            cached = self.cache.get(str(consultation_id), key)
            if cached is not None:
                return cached

            # PostgreSQL에서 조회
            session_mem = await self.memory_service.get_session_memory(
                user_id, consultation_id
            )
            return session_mem if session_mem else None

        elif memory_type == MemoryType.CACHE:
            return self.cache.get(str(consultation_id), f"cache:{key}")

        elif memory_type == MemoryType.SHORT_TERM:
            return await self.memory_service.get_short_term_memory(
                user_id, consultation_id
            )

        elif memory_type == MemoryType.LONG_TERM:
            return await self.memory_service.get_long_term_memory(user_id)

        return None

    # ========================================
    # 법률 자문 위원용 컨텍스트
    # ========================================

    async def get_council_context(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: Optional[str] = None,
    ) -> MemoryContext:
        """
        법률 자문 위원을 위한 통합 컨텍스트 조회.

        모든 메모리 유형을 통합하여 LLM 프롬프트에 주입할
        컨텍스트를 생성합니다.

        Args:
            user_id: 사용자 ID
            consultation_id: 상담 ID
            category: 법률 분야 필터

        Returns:
            MemoryContext 객체
        """
        session_key = str(consultation_id)

        # 캐시 확인
        cached = self.cache.get_memory_snapshot(session_key)
        if cached:
            return MemoryContext(**cached)

        # PostgreSQL에서 메모리 조회
        memory_context = await self.memory_service.get_memory_context(
            user_id=user_id,
            consultation_id=consultation_id,
            category=category,
        )

        context = MemoryContext(
            session_memory=memory_context.get("session_memory", ""),
            short_term_memory=memory_context.get("short_term_memory", ""),
            long_term_memory=memory_context.get("long_term_memory", ""),
        )

        # 캐시에 저장
        self.cache.set_memory_snapshot(session_key, {
            "session_memory": context.session_memory,
            "short_term_memory": context.short_term_memory,
            "long_term_memory": context.long_term_memory,
        })

        return context

    async def invalidate_context_cache(
        self,
        consultation_id: UUID,
    ) -> None:
        """컨텍스트 캐시 무효화."""
        session_key = str(consultation_id)
        self.cache.delete(session_key, "mem:snapshot")

    # ========================================
    # 에이전트 상태 관리
    # ========================================

    def store_agent_state(
        self,
        consultation_id: UUID,
        agent_id: str,
        state: dict,
    ) -> None:
        """에이전트 상태 저장."""
        self.cache.set_agent_state(str(consultation_id), agent_id, state)

    def get_agent_state(
        self,
        consultation_id: UUID,
        agent_id: str,
    ) -> Optional[dict]:
        """에이전트 상태 조회."""
        return self.cache.get_agent_state(str(consultation_id), agent_id)

    def get_all_agent_states(
        self,
        consultation_id: UUID,
    ) -> dict[str, dict]:
        """모든 에이전트 상태 조회."""
        return self.cache.get_all_agent_states(str(consultation_id))

    # ========================================
    # RAG 결과 캐싱
    # ========================================

    def cache_rag_results(
        self,
        consultation_id: UUID,
        query_hash: str,
        results: list[dict],
        ttl: int = 300,
    ) -> None:
        """RAG 검색 결과 캐시."""
        self.cache.cache_rag_result(
            str(consultation_id), query_hash, results, ttl
        )

    def get_cached_rag_results(
        self,
        consultation_id: UUID,
        query_hash: str,
    ) -> Optional[list[dict]]:
        """캐시된 RAG 결과 조회."""
        return self.cache.get_cached_rag_result(str(consultation_id), query_hash)

    # ========================================
    # 세션 관리
    # ========================================

    async def initialize_session(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: Optional[str] = None,
    ) -> MemoryContext:
        """
        세션 초기화.

        새 상담 세션 시작 시 호출하여 컨텍스트를 준비합니다.

        Returns:
            초기 MemoryContext
        """
        logger.info(f"Initializing session: consultation={consultation_id}")

        # 이전 세션 캐시 정리
        self.cache.clear_session(str(consultation_id))

        # 사용자의 메모리 컨텍스트 로드
        context = await self.get_council_context(
            user_id=user_id,
            consultation_id=consultation_id,
            category=category,
        )

        # 세션 시작 정보 저장
        self.cache.set_context(str(consultation_id), "session_start", {
            "user_id": str(user_id),
            "category": category,
            "started_at": "now",
        })

        return context

    async def finalize_session(
        self,
        user_id: UUID,
        consultation_id: UUID,
        category: str,
        keywords: list[str],
        summary: Optional[str] = None,
    ) -> None:
        """
        세션 종료.

        상담 완료 시 호출하여 학습 데이터를 저장합니다.
        """
        logger.info(f"Finalizing session: consultation={consultation_id}")

        # 장기 패턴 학습
        await self.memory_service.learn_from_consultation(
            user_id=user_id,
            consultation_id=consultation_id,
            category=category,
            keywords=keywords,
            summary=summary,
        )

        # PostgreSQL 세션 메모리 정리
        await self.memory_service.clear_session_memory(user_id, consultation_id)

        # 인메모리 캐시 정리
        self.cache.clear_session(str(consultation_id))

    # ========================================
    # 유틸리티
    # ========================================

    async def search_memories(
        self,
        user_id: UUID,
        query: str,
    ) -> dict[str, Any]:
        """
        메모리 검색.

        모든 메모리 유형에서 관련 정보를 검색합니다.
        """
        return await self.memory_service.search_all_memories(
            user_id=user_id,
            query=query,
        )

    async def get_statistics(
        self,
        user_id: UUID,
    ) -> dict:
        """사용자 메모리 통계."""
        db_stats = await self.memory_service.get_memory_statistics(user_id)
        cache_stats = self.cache.get_statistics()

        return {
            "database": db_stats,
            "cache": cache_stats,
        }

    async def cleanup(self) -> dict:
        """
        메모리 정리.

        만료된 메모리와 캐시를 정리합니다.
        """
        db_cleaned = await self.memory_service.cleanup_expired_memories()
        cache_cleaned = self.cache.cleanup_expired()

        return {
            "database_cleaned": db_cleaned,
            "cache_cleaned": cache_cleaned,
        }


# ========================================
# 팩토리 함수
# ========================================

_memory_manager_cache: dict[str, MemoryManager] = {}


def get_memory_manager(db: AsyncSession) -> MemoryManager:
    """
    MemoryManager 인스턴스 획득.

    동일한 DB 세션에 대해 동일한 인스턴스를 반환합니다.
    """
    # DB 세션 ID 기반 캐싱
    session_id = id(db)
    cache_key = str(session_id)

    if cache_key not in _memory_manager_cache:
        _memory_manager_cache[cache_key] = MemoryManager(db)

    return _memory_manager_cache[cache_key]


def clear_memory_manager_cache() -> None:
    """MemoryManager 캐시 초기화 (테스트용)."""
    _memory_manager_cache.clear()
