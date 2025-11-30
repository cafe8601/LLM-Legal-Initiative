"""
Session Cache

빠른 인메모리 세션 캐시.
MAS multiagent v4의 SessionMemory를 법률 자문 시스템에 맞게 확장.

Features:
- O(1) 접근 시간
- 자동 TTL 만료
- 세션별 격리
- 메타데이터 추적
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리."""
    value: Any
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None  # None = 무제한

    @property
    def is_expired(self) -> bool:
        """만료 여부 확인."""
        if self.ttl_seconds is None:
            return False
        expiry = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry

    @property
    def age_seconds(self) -> float:
        """엔트리 나이 (초)."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


class SessionCache:
    """
    세션 캐시.

    법률 자문 세션 동안 빠른 접근이 필요한 데이터를 저장합니다.
    스레드 안전합니다.

    동시성 최적화:
    - 전역 락: 세션 생성/삭제에만 사용
    - 세션별 락: 개별 세션 내 작업에 사용 (더 높은 동시성)
    """

    # 기본 설정
    DEFAULT_TTL = 3600  # 1시간
    MAX_ENTRIES_PER_SESSION = 100

    def __init__(
        self,
        default_ttl: int = DEFAULT_TTL,
        max_entries: int = MAX_ENTRIES_PER_SESSION,
    ):
        """
        초기화.

        Args:
            default_ttl: 기본 TTL (초)
            max_entries: 세션당 최대 엔트리 수
        """
        self._sessions: dict[str, dict[str, CacheEntry]] = {}
        self._global_lock = threading.RLock()  # 세션 생성/삭제용
        self._session_locks: dict[str, threading.RLock] = {}  # 세션별 락
        self._lock = self._global_lock  # 하위 호환성
        self.default_ttl = default_ttl
        self.max_entries = max_entries

    def _get_session_lock(self, session_key: str) -> threading.RLock:
        """세션별 락 획득. 없으면 생성."""
        with self._global_lock:
            if session_key not in self._session_locks:
                self._session_locks[session_key] = threading.RLock()
            return self._session_locks[session_key]

    def _ensure_session_exists(self, session_key: str) -> None:
        """세션이 존재하는지 확인하고 없으면 생성."""
        with self._global_lock:
            if session_key not in self._sessions:
                self._sessions[session_key] = {}
                self._session_locks[session_key] = threading.RLock()

    def _get_session_key(self, session_id: str | UUID) -> str:
        """세션 키 정규화."""
        return str(session_id)

    def set(
        self,
        session_id: str | UUID,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        값 저장.

        Args:
            session_id: 세션 ID
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초), None이면 기본값 사용
        """
        session_key = self._get_session_key(session_id)
        now = datetime.now(timezone.utc)

        # 세션이 존재하는지 확인 (전역 락 사용)
        self._ensure_session_exists(session_key)

        # 세션별 락으로 실제 데이터 작업 (더 높은 동시성)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            session_data = self._sessions[session_key]

            # 최대 엔트리 수 초과 시 오래된 항목 제거
            if len(session_data) >= self.max_entries and key not in session_data:
                self._evict_oldest(session_key)

            # 엔트리 생성/업데이트
            if key in session_data:
                entry = session_data[key]
                entry.value = value
                entry.updated_at = now
                entry.access_count += 1
            else:
                session_data[key] = CacheEntry(
                    value=value,
                    created_at=now,
                    updated_at=now,
                    ttl_seconds=ttl if ttl is not None else self.default_ttl,
                )

    def get(
        self,
        session_id: str | UUID,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        값 조회.

        Args:
            session_id: 세션 ID
            key: 캐시 키
            default: 기본값

        Returns:
            저장된 값 또는 기본값
        """
        session_key = self._get_session_key(session_id)

        # 세션 존재 여부 빠른 확인 (전역 락)
        with self._global_lock:
            if session_key not in self._sessions:
                return default

        # 세션별 락으로 데이터 접근 (더 높은 동시성)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            # 세션이 다른 스레드에 의해 삭제되었을 수 있음
            if session_key not in self._sessions:
                return default

            entry = self._sessions[session_key].get(key)
            if entry is None:
                return default

            # 만료 확인
            if entry.is_expired:
                del self._sessions[session_key][key]
                return default

            # 접근 횟수 증가
            entry.access_count += 1
            return entry.value

    def has(self, session_id: str | UUID, key: str) -> bool:
        """키 존재 여부 확인."""
        return self.get(session_id, key) is not None

    def delete(self, session_id: str | UUID, key: str) -> bool:
        """
        값 삭제.

        Returns:
            삭제 성공 여부
        """
        session_key = self._get_session_key(session_id)

        # 세션 존재 여부 빠른 확인 (전역 락)
        with self._global_lock:
            if session_key not in self._sessions:
                return False

        # 세션별 락으로 삭제 (더 높은 동시성)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            if session_key not in self._sessions:
                return False

            if key in self._sessions[session_key]:
                del self._sessions[session_key][key]
                return True

            return False

    def clear_session(self, session_id: str | UUID) -> int:
        """
        세션 전체 삭제.

        Returns:
            삭제된 엔트리 수
        """
        session_key = self._get_session_key(session_id)

        # 세션 삭제는 전역 락 사용 (세션 구조 변경)
        with self._global_lock:
            if session_key not in self._sessions:
                return 0

            count = len(self._sessions[session_key])
            del self._sessions[session_key]

            # 세션 락도 정리
            if session_key in self._session_locks:
                del self._session_locks[session_key]

            logger.info(f"Cleared session {session_key}: {count} entries")
            return count

    def get_all(self, session_id: str | UUID) -> dict[str, Any]:
        """세션의 모든 값 조회."""
        session_key = self._get_session_key(session_id)

        # 세션 존재 여부 빠른 확인 (전역 락)
        with self._global_lock:
            if session_key not in self._sessions:
                return {}

        # 세션별 락으로 데이터 접근 (더 높은 동시성)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            if session_key not in self._sessions:
                return {}

            # 만료된 항목 제거하며 반환
            result = {}
            expired_keys = []

            for key, entry in self._sessions[session_key].items():
                if entry.is_expired:
                    expired_keys.append(key)
                else:
                    result[key] = entry.value

            for key in expired_keys:
                del self._sessions[session_key][key]

            return result

    def keys(self, session_id: str | UUID) -> list[str]:
        """세션의 모든 키 조회."""
        return list(self.get_all(session_id).keys())

    def size(self, session_id: str | UUID) -> int:
        """세션의 엔트리 수."""
        return len(self.get_all(session_id))

    def get_metadata(
        self,
        session_id: str | UUID,
        key: str,
    ) -> Optional[dict]:
        """
        엔트리 메타데이터 조회.

        Returns:
            메타데이터 딕셔너리 또는 None
        """
        session_key = self._get_session_key(session_id)

        # 세션 존재 여부 빠른 확인 (전역 락)
        with self._global_lock:
            if session_key not in self._sessions:
                return None

        # 세션별 락으로 데이터 접근 (더 높은 동시성)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            if session_key not in self._sessions:
                return None

            entry = self._sessions[session_key].get(key)
            if entry is None or entry.is_expired:
                return None

            return {
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "age_seconds": entry.age_seconds,
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "is_expired": entry.is_expired,
            }

    def _evict_oldest(self, session_key: str) -> None:
        """가장 오래된 항목 제거."""
        if session_key not in self._sessions:
            return

        session_data = self._sessions[session_key]
        if not session_data:
            return

        # 접근 횟수가 가장 적은 항목 제거 (LFU 방식)
        oldest_key = min(
            session_data.keys(),
            key=lambda k: (session_data[k].access_count, session_data[k].updated_at),
        )
        del session_data[oldest_key]

    def cleanup_expired(self) -> int:
        """
        모든 세션에서 만료된 항목 정리.

        Returns:
            제거된 항목 수
        """
        removed = 0
        empty_sessions = []

        # 모든 세션 키 복사 (순회 중 수정 방지)
        with self._global_lock:
            session_keys = list(self._sessions.keys())

        # 각 세션에 대해 세션별 락으로 정리 (더 높은 동시성)
        for session_key in session_keys:
            session_lock = self._get_session_lock(session_key)
            with session_lock:
                if session_key not in self._sessions:
                    continue

                session_data = self._sessions[session_key]
                expired_keys = [
                    key for key, entry in session_data.items()
                    if entry.is_expired
                ]

                for key in expired_keys:
                    del session_data[key]
                    removed += 1

                # 빈 세션 표시
                if not session_data:
                    empty_sessions.append(session_key)

        # 빈 세션 정리 (전역 락 사용)
        if empty_sessions:
            with self._global_lock:
                for session_key in empty_sessions:
                    if session_key in self._sessions and not self._sessions[session_key]:
                        del self._sessions[session_key]
                        if session_key in self._session_locks:
                            del self._session_locks[session_key]

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache entries")

        return removed

    def get_statistics(self) -> dict:
        """캐시 통계."""
        with self._global_lock:
            total_entries = sum(len(s) for s in self._sessions.values())
            total_sessions = len(self._sessions)
            total_session_locks = len(self._session_locks)

            return {
                "total_sessions": total_sessions,
                "total_entries": total_entries,
                "entries_per_session": total_entries / total_sessions if total_sessions > 0 else 0,
                "max_entries_per_session": self.max_entries,
                "default_ttl": self.default_ttl,
                "session_locks_count": total_session_locks,  # 동시성 디버깅용
            }


# ========================================
# 법률 상담 전용 캐시 헬퍼
# ========================================


class ConsultationCache(SessionCache):
    """
    법률 상담 전용 캐시.

    상담 세션에서 자주 사용되는 데이터를 캐싱합니다.
    """

    # 캐시 키 프리픽스
    PREFIX_CONTEXT = "ctx:"  # 컨텍스트
    PREFIX_MEMORY = "mem:"   # 메모리
    PREFIX_RAG = "rag:"      # RAG 결과
    PREFIX_AGENT = "agent:"  # 에이전트 상태

    def set_context(
        self,
        session_id: str | UUID,
        context_type: str,
        context_data: dict,
    ) -> None:
        """컨텍스트 저장."""
        key = f"{self.PREFIX_CONTEXT}{context_type}"
        self.set(session_id, key, context_data)

    def get_context(
        self,
        session_id: str | UUID,
        context_type: str,
    ) -> Optional[dict]:
        """컨텍스트 조회."""
        key = f"{self.PREFIX_CONTEXT}{context_type}"
        return self.get(session_id, key)

    def set_memory_snapshot(
        self,
        session_id: str | UUID,
        memory_data: dict,
    ) -> None:
        """메모리 스냅샷 저장."""
        key = f"{self.PREFIX_MEMORY}snapshot"
        self.set(session_id, key, memory_data)

    def get_memory_snapshot(
        self,
        session_id: str | UUID,
    ) -> Optional[dict]:
        """메모리 스냅샷 조회."""
        key = f"{self.PREFIX_MEMORY}snapshot"
        return self.get(session_id, key)

    def cache_rag_result(
        self,
        session_id: str | UUID,
        query_hash: str,
        results: list[dict],
        ttl: int = 300,  # 5분
    ) -> None:
        """RAG 검색 결과 캐시."""
        key = f"{self.PREFIX_RAG}{query_hash}"
        self.set(session_id, key, results, ttl=ttl)

    def get_cached_rag_result(
        self,
        session_id: str | UUID,
        query_hash: str,
    ) -> Optional[list[dict]]:
        """캐시된 RAG 결과 조회."""
        key = f"{self.PREFIX_RAG}{query_hash}"
        return self.get(session_id, key)

    def set_agent_state(
        self,
        session_id: str | UUID,
        agent_id: str,
        state: dict,
    ) -> None:
        """에이전트 상태 저장."""
        key = f"{self.PREFIX_AGENT}{agent_id}"
        self.set(session_id, key, state)

    def get_agent_state(
        self,
        session_id: str | UUID,
        agent_id: str,
    ) -> Optional[dict]:
        """에이전트 상태 조회."""
        key = f"{self.PREFIX_AGENT}{agent_id}"
        return self.get(session_id, key)

    def get_all_agent_states(
        self,
        session_id: str | UUID,
    ) -> dict[str, dict]:
        """모든 에이전트 상태 조회."""
        all_data = self.get_all(session_id)
        prefix_len = len(self.PREFIX_AGENT)

        return {
            key[prefix_len:]: value
            for key, value in all_data.items()
            if key.startswith(self.PREFIX_AGENT)
        }


# 싱글톤 인스턴스
_consultation_cache: Optional[ConsultationCache] = None


def get_consultation_cache() -> ConsultationCache:
    """ConsultationCache 싱글톤 인스턴스 획득."""
    global _consultation_cache

    if _consultation_cache is None:
        _consultation_cache = ConsultationCache()

    return _consultation_cache
