"""
Enhanced Memory System

MAS multiagent v4 기반 확장 메모리 시스템.
기존 PostgreSQL 기반 메모리와 빠른 인메모리 캐시를 통합합니다.

Components:
- SessionCache: 빠른 인메모리 세션 캐시 (TTL, LFU 정책)
- ConsultationCache: 법률 상담 전용 캐시
- MemoryManager: 통합 메모리 인터페이스
- MemoryContext: LLM 프롬프트용 메모리 컨텍스트

Usage:
    from app.services.memory import MemoryManager, get_memory_manager

    # 메모리 관리자 획득
    memory_manager = get_memory_manager(db)

    # 컨텍스트 로드
    context = await memory_manager.get_council_context(user_id, consultation_id)

    # 세션 초기화
    await memory_manager.initialize_session(user_id, consultation_id)
"""

from app.services.memory.session_cache import (
    SessionCache,
    ConsultationCache,
    CacheEntry,
    get_consultation_cache,
)
from app.services.memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryContext,
    get_memory_manager,
    clear_memory_manager_cache,
)

__all__ = [
    # Session Cache
    "SessionCache",
    "ConsultationCache",
    "CacheEntry",
    "get_consultation_cache",
    # Memory Manager
    "MemoryManager",
    "MemoryType",
    "MemoryContext",
    "get_memory_manager",
    "clear_memory_manager_cache",
]
