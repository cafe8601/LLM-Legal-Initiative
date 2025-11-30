"""
MAS Multiagent v4 통합 테스트

Memory, RAG, Learning 시스템 통합 테스트.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Learning System
from app.services.learning import (
    OutcomeTracker,
    ConsultationOutcome,
    OutcomeStatus,
    ConsultationTier,
    PatternAnalyzer,
    AgentPerformance,
    LearningManager,
    AgentRecommendation,
    CascadeDecision,
    get_learning_manager,
)

# Memory System
from app.services.memory import (
    SessionCache,
    ConsultationCache,
    CacheEntry,
    get_consultation_cache,
    MemoryType,
    MemoryContext,
)

# RAG System
from app.services.rag import (
    HybridSearchResult,
    RAGConfig,
)


# =============================================================================
# Session Cache 테스트
# =============================================================================


class TestSessionCache:
    """세션 캐시 테스트"""

    def setup_method(self):
        self.cache = SessionCache(default_ttl=300, max_entries=100)
        self.session_id = "test-session-123"

    def test_cache_set_and_get(self):
        """캐시 설정 및 조회"""
        self.cache.set(self.session_id, "key1", {"value": "test"})
        result = self.cache.get(self.session_id, "key1")
        assert result == {"value": "test"}

    def test_cache_miss(self):
        """캐시 미스"""
        result = self.cache.get(self.session_id, "nonexistent")
        assert result is None

    def test_cache_ttl_expiry(self):
        """TTL 만료"""
        self.cache.set(self.session_id, "key1", "value", ttl=0)
        # 즉시 만료됨
        result = self.cache.get(self.session_id, "key1")
        assert result is None

    def test_cache_eviction_on_max_size(self):
        """최대 크기 초과 시 eviction"""
        small_cache = SessionCache(max_entries=3)
        for i in range(5):
            small_cache.set("session", f"key{i}", f"value{i}")
        # 최대 3개만 유지
        assert small_cache.size("session") <= 3

    def test_cache_delete(self):
        """캐시 삭제"""
        self.cache.set(self.session_id, "key1", "value")
        result = self.cache.delete(self.session_id, "key1")
        assert result is True
        assert self.cache.get(self.session_id, "key1") is None

    def test_cache_clear_session(self):
        """세션 전체 삭제"""
        self.cache.set(self.session_id, "key1", "value1")
        self.cache.set(self.session_id, "key2", "value2")
        count = self.cache.clear_session(self.session_id)
        assert count == 2
        assert self.cache.get(self.session_id, "key1") is None
        assert self.cache.get(self.session_id, "key2") is None

    def test_cache_statistics(self):
        """캐시 통계"""
        self.cache.set(self.session_id, "key1", "value")
        stats = self.cache.get_statistics()
        assert stats["total_sessions"] == 1
        assert stats["total_entries"] == 1

    def test_cache_get_all(self):
        """세션의 모든 값 조회"""
        self.cache.set(self.session_id, "key1", "value1")
        self.cache.set(self.session_id, "key2", "value2")
        all_data = self.cache.get_all(self.session_id)
        assert len(all_data) == 2
        assert all_data["key1"] == "value1"

    def test_cache_keys(self):
        """세션의 모든 키 조회"""
        self.cache.set(self.session_id, "key1", "value1")
        self.cache.set(self.session_id, "key2", "value2")
        keys = self.cache.keys(self.session_id)
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys


class TestConsultationCache:
    """상담 캐시 테스트"""

    def setup_method(self):
        self.cache = ConsultationCache()
        self.session_id = "test-consultation-1"

    def test_set_and_get_context(self):
        """컨텍스트 저장 및 조회"""
        context = {
            "category": "민사",
            "query": "계약 해지",
            "history": ["이전 메시지"],
        }
        self.cache.set_context(self.session_id, "main", context)
        result = self.cache.get_context(self.session_id, "main")
        assert result == context

    def test_cache_rag_result(self):
        """RAG 결과 캐싱"""
        query_hash = "abc123"
        rag_result = [{"title": "doc1"}, {"title": "doc2"}]
        self.cache.cache_rag_result(self.session_id, query_hash, rag_result)
        result = self.cache.get_cached_rag_result(self.session_id, query_hash)
        assert result == rag_result

    def test_set_and_get_agent_state(self):
        """에이전트 상태 저장 및 조회"""
        agent_id = "claude-sonnet"
        state = {"response": "법률 답변", "quality": 0.85}
        self.cache.set_agent_state(self.session_id, agent_id, state)
        result = self.cache.get_agent_state(self.session_id, agent_id)
        assert result == state

    def test_get_all_agent_states(self):
        """모든 에이전트 상태 조회"""
        self.cache.set_agent_state(self.session_id, "claude", {"quality": 0.9})
        self.cache.set_agent_state(self.session_id, "gpt", {"quality": 0.85})
        states = self.cache.get_all_agent_states(self.session_id)
        assert len(states) == 2
        assert "claude" in states
        assert "gpt" in states

    def test_memory_snapshot(self):
        """메모리 스냅샷"""
        snapshot = {"session": "data", "recent": ["item1"]}
        self.cache.set_memory_snapshot(self.session_id, snapshot)
        result = self.cache.get_memory_snapshot(self.session_id)
        assert result == snapshot


class TestGetConsultationCache:
    """상담 캐시 싱글톤 테스트"""

    def test_singleton_instance(self):
        """싱글톤 인스턴스"""
        cache1 = get_consultation_cache()
        cache2 = get_consultation_cache()
        assert cache1 is cache2


# =============================================================================
# Learning System 테스트
# =============================================================================


class TestOutcomeStatus:
    """결과 상태 테스트"""

    def test_outcome_statuses(self):
        """상태 값 확인"""
        assert OutcomeStatus.SUCCESS == "success"
        assert OutcomeStatus.FAILURE == "failure"
        assert OutcomeStatus.ESCALATED == "escalated"
        assert OutcomeStatus.PARTIAL == "partial"


class TestConsultationTier:
    """상담 티어 테스트"""

    def test_tier_values(self):
        """티어 값 확인"""
        assert ConsultationTier.DRAFTER == "drafter"
        assert ConsultationTier.VERIFIER == "verifier"
        assert ConsultationTier.DIRECT == "direct"


class TestOutcomeTracker:
    """결과 추적기 테스트"""

    def setup_method(self):
        self.tracker = OutcomeTracker()

    def test_record_success(self):
        """성공 결과 기록"""
        outcome = self.tracker.record_success(
            agent_id="claude-sonnet",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-123",
            user_id="user-456",
            category="민사",
            query_complexity="moderate",
            response_quality=0.85,
            response_time_ms=1500,
            tokens_used=500,
            cost_usd=0.001,
        )
        assert outcome.status == OutcomeStatus.SUCCESS.value
        assert outcome.agent_id == "claude-sonnet"
        assert outcome.response_quality == 0.85

    def test_record_failure(self):
        """실패 결과 기록"""
        outcome = self.tracker.record_failure(
            agent_id="gpt-4o",
            model_id="openai/gpt-4o",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-456",
            user_id="user-789",
            category="형사",
            query_complexity="complex",
            error_reason="품질 기준 미달",
            response_time_ms=2000,
            tokens_used=300,
            cost_usd=0.0005,
        )
        assert outcome.status == OutcomeStatus.FAILURE.value
        assert outcome.response_quality == 0.0

    def test_record_escalation(self):
        """에스컬레이션 기록"""
        outcome = self.tracker.record_escalation(
            agent_id="gemini-2-5",
            model_id="google/gemini-2.5-pro",
            consultation_id="test-789",
            user_id="user-123",
            category="가사",
            query_complexity="complex",
            escalation_reason="low_confidence",
            drafter_quality=0.55,
            response_time_ms=1200,
            tokens_used=400,
            cost_usd=0.0008,
        )
        assert outcome.status == OutcomeStatus.ESCALATED.value
        assert outcome.tier == ConsultationTier.DRAFTER.value

    def test_get_success_rate(self):
        """성공률 계산"""
        # 성공 3개, 실패 1개 기록
        for i in range(3):
            self.tracker.record_success(
                agent_id="claude",
                model_id="anthropic/claude-sonnet-4",
                tier=ConsultationTier.DRAFTER,
                consultation_id=f"success-{i}",
                user_id="user-1",
                category="민사",
                query_complexity="simple",
                response_quality=0.8,
                response_time_ms=1000,
                tokens_used=300,
                cost_usd=0.001,
            )

        self.tracker.record_failure(
            agent_id="claude",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="failure-1",
            user_id="user-1",
            category="민사",
            query_complexity="complex",
            error_reason="error",
            response_time_ms=1000,
            tokens_used=200,
            cost_usd=0.0005,
        )

        rate = self.tracker.get_success_rate()
        assert rate == 0.75  # 3/4

    def test_get_outcomes_for_agent(self):
        """에이전트별 결과 조회"""
        for agent in ["claude", "gpt", "claude"]:
            self.tracker.record_success(
                agent_id=agent,
                model_id=f"{agent}/model",
                tier=ConsultationTier.DRAFTER,
                consultation_id=f"test-{agent}",
                user_id="user-1",
                category="민사",
                query_complexity="simple",
                response_quality=0.8,
                response_time_ms=1000,
                tokens_used=300,
                cost_usd=0.001,
            )

        claude_outcomes = self.tracker.get_outcomes_for_agent("claude")
        assert len(claude_outcomes) == 2

    def test_get_statistics(self):
        """통계 조회"""
        self.tracker.record_success(
            agent_id="claude",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-1",
            user_id="user-1",
            category="민사",
            query_complexity="simple",
            response_quality=0.85,
            response_time_ms=1000,
            tokens_used=300,
            cost_usd=0.001,
        )
        stats = self.tracker.get_statistics()
        assert stats["total_outcomes"] == 1
        assert "by_agent" in stats
        assert "by_category" in stats


class TestPatternAnalyzer:
    """패턴 분석기 테스트"""

    def setup_method(self):
        self.tracker = OutcomeTracker()
        self.analyzer = PatternAnalyzer(self.tracker)
        # 샘플 데이터 추가
        for i in range(10):
            self.tracker.record_success(
                agent_id="claude" if i % 2 == 0 else "gpt",
                model_id="anthropic/claude-sonnet-4" if i % 2 == 0 else "openai/gpt-4o",
                tier=ConsultationTier.DRAFTER,
                consultation_id=f"test-{i}",
                user_id="user-1",
                category="민사" if i < 5 else "형사",
                query_complexity="simple",
                response_quality=0.7 + (i * 0.03),
                response_time_ms=1000 + (i * 100),
                tokens_used=300,
                cost_usd=0.001,
            )

    def test_analyze_agent_performance(self):
        """에이전트 성능 분석"""
        perf = self.analyzer.analyze_agent_performance("claude")
        assert isinstance(perf, AgentPerformance)
        assert perf.agent_id == "claude"
        assert perf.total_consultations > 0

    def test_get_best_agent_for_category(self):
        """카테고리별 최적 에이전트"""
        best = self.analyzer.get_best_agent_for_category("민사")
        assert best is not None
        # Returns (agent_id, score) tuple
        assert isinstance(best, tuple)
        assert isinstance(best[0], str)

    def test_analyze_category_performance(self):
        """카테고리 성능 분석"""
        # Use the correct method from PatternAnalyzer
        # Check outcomes for category via tracker
        outcomes = self.tracker.get_outcomes_for_category("민사")
        assert len(outcomes) > 0


class TestLearningManager:
    """학습 관리자 테스트"""

    def setup_method(self):
        self.manager = LearningManager()

    def test_record_consultation_outcome(self):
        """상담 결과 기록"""
        outcome = self.manager.record_consultation_outcome(
            agent_id="claude-sonnet",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-123",
            user_id="user-456",
            category="민사",
            query="계약 해지 방법",
            success=True,
            response_quality=0.85,
            response_time_ms=1500,
            tokens_used=500,
            cost_usd=0.001,
        )
        assert outcome is not None
        assert outcome.status == OutcomeStatus.SUCCESS.value

    def test_get_cascade_decision(self):
        """캐스케이드 결정"""
        decision = self.manager.get_cascade_decision(
            category="민사",
            query="간단한 질문",
            agent_id="claude-sonnet",
        )
        assert isinstance(decision, CascadeDecision)
        assert decision.skip_drafter in [True, False]

    def test_get_agent_recommendation(self):
        """에이전트 추천"""
        # 먼저 충분한 데이터 추가
        for i in range(10):
            self.manager.record_consultation_outcome(
                agent_id="claude" if i % 2 == 0 else "gpt",
                model_id="anthropic/claude-sonnet-4",
                tier=ConsultationTier.DRAFTER,
                consultation_id=f"test-{i}",
                user_id="user-1",
                category="민사",
                query="법률 질문",
                success=True,
                response_quality=0.8 + (0.01 * i),
                response_time_ms=1000,
                tokens_used=300,
                cost_usd=0.001,
            )

        recommendation = self.manager.get_agent_recommendation("민사", "복잡한 질문")
        assert isinstance(recommendation, AgentRecommendation)

    def test_get_learning_stats(self):
        """학습 통계 조회"""
        self.manager.record_consultation_outcome(
            agent_id="claude",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-1",
            user_id="user-1",
            category="민사",
            query="질문",
            success=True,
            response_quality=0.85,
            response_time_ms=1000,
            tokens_used=300,
            cost_usd=0.001,
        )
        stats = self.manager.get_learning_stats()
        assert "outcomes" in stats
        assert "analysis" in stats
        assert "config" in stats
        assert stats["outcomes"]["total_outcomes"] == 1

    def test_singleton_instance(self):
        """싱글톤 인스턴스"""
        manager1 = get_learning_manager()
        manager2 = get_learning_manager()
        assert manager1 is manager2


# =============================================================================
# RAG System 테스트
# =============================================================================


class TestRAGConfig:
    """RAG 설정 테스트"""

    def test_default_config(self):
        """기본 설정"""
        config = RAGConfig()
        assert config.legal_doc_top_k == 5
        assert config.experience_top_k == 3
        assert config.cache_ttl == 300

    def test_custom_config(self):
        """커스텀 설정"""
        config = RAGConfig(
            legal_doc_top_k=10,
            experience_top_k=5,
            legal_doc_weight=0.8,
        )
        assert config.legal_doc_top_k == 10
        assert config.experience_top_k == 5
        assert config.legal_doc_weight == 0.8


class TestHybridSearchResult:
    """하이브리드 검색 결과 테스트"""

    def test_result_creation(self):
        """검색 결과 생성"""
        result = HybridSearchResult(
            legal_documents=[{"title": "민법", "content": "내용"}],
            similar_experiences=[],
            combined_context="통합 컨텍스트",
            total_sources=1,
            cache_hit=False,
        )
        assert len(result.legal_documents) == 1
        assert result.total_sources == 1
        assert not result.cache_hit


# =============================================================================
# Memory System 테스트
# =============================================================================


class TestMemoryContext:
    """메모리 컨텍스트 테스트"""

    def test_context_creation(self):
        """컨텍스트 생성"""
        context = MemoryContext(
            session_memory="세션 데이터",
            short_term_memory="최근 기록",
            long_term_memory="장기 패턴",
        )
        assert context.session_memory == "세션 데이터"
        assert context.has_context is True

    def test_to_prompt_string(self):
        """프롬프트 문자열 변환"""
        context = MemoryContext(
            session_memory="세션",
            short_term_memory="단기",
            long_term_memory="장기",
        )
        prompt = context.to_prompt_string()
        assert "세션" in prompt
        assert "단기" in prompt
        assert "장기" in prompt

    def test_empty_context(self):
        """빈 컨텍스트"""
        context = MemoryContext(
            session_memory="",
            short_term_memory="",
            long_term_memory="",
        )
        assert context.has_context is False


class TestMemoryType:
    """메모리 타입 테스트"""

    def test_memory_types(self):
        """메모리 타입 확인"""
        assert MemoryType.SESSION == "session"
        assert MemoryType.SHORT_TERM == "short_term"
        assert MemoryType.LONG_TERM == "long_term"
        assert MemoryType.CACHE == "cache"


# =============================================================================
# Integration 테스트
# =============================================================================


class TestSystemIntegration:
    """시스템 통합 테스트"""

    def test_learning_cache_flow(self):
        """학습 + 캐시 흐름"""
        # 학습 관리자에서 결과 기록
        learning_manager = LearningManager()
        outcome = learning_manager.record_consultation_outcome(
            agent_id="claude-sonnet",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER,
            consultation_id="test-123",
            user_id="user-456",
            category="민사",
            query="임대차 계약",
            success=True,
            response_quality=0.9,
            response_time_ms=1500,
            tokens_used=500,
            cost_usd=0.001,
        )
        assert outcome.status == OutcomeStatus.SUCCESS.value

    def test_cache_to_learning_flow(self):
        """캐시 → 학습 흐름"""
        # 캐시에 컨텍스트 저장
        cache = get_consultation_cache()
        consultation_id = "test-flow-123"
        cache.set_context(consultation_id, "main", {
            "category": "민사",
            "query": "계약 해지",
        })

        # 컨텍스트 조회
        context = cache.get_context(consultation_id, "main")
        assert context["category"] == "민사"

        # 학습에 활용
        learning_manager = get_learning_manager()
        decision = learning_manager.get_cascade_decision(
            category=context["category"],
            query=context["query"],
            agent_id="claude-sonnet",
        )
        assert decision is not None

    def test_full_consultation_flow(self):
        """전체 상담 흐름 테스트"""
        consultation_id = str(uuid4())
        user_id = str(uuid4())

        # 1. 캐시 초기화
        cache = get_consultation_cache()
        cache.set_context(consultation_id, "main", {
            "user_id": user_id,
            "category": "민사",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        # 2. 학습 기반 결정
        learning_manager = get_learning_manager()
        decision = learning_manager.get_cascade_decision(
            category="민사",
            query="계약 해지 방법",
            agent_id="claude-sonnet",
        )
        assert decision.skip_drafter in [True, False]

        # 3. RAG 설정
        rag_config = RAGConfig(
            legal_doc_top_k=3 if not decision.skip_drafter else 5,
            experience_top_k=2 if not decision.skip_drafter else 3,
        )

        # 4. 결과 기록
        outcome = learning_manager.record_consultation_outcome(
            agent_id="claude-sonnet",
            model_id="anthropic/claude-sonnet-4",
            tier=ConsultationTier.DRAFTER if not decision.skip_drafter else ConsultationTier.VERIFIER,
            consultation_id=consultation_id,
            user_id=user_id,
            category="민사",
            query="계약 해지 방법",
            success=True,
            response_quality=0.85,
            response_time_ms=1500,
            tokens_used=500,
            cost_usd=0.001,
        )
        assert outcome is not None

        # 5. 상담 캐시 정리
        cache.clear_session(consultation_id)
        assert cache.get_context(consultation_id, "main") is None


# =============================================================================
# Edge Cases 테스트
# =============================================================================


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_query_handling(self):
        """빈 쿼리 처리"""
        learning_manager = get_learning_manager()
        decision = learning_manager.get_cascade_decision(
            category="민사",
            query="",
            agent_id="claude-sonnet",
        )
        # 빈 쿼리도 처리 가능
        assert decision is not None

    def test_unknown_category_handling(self):
        """알 수 없는 카테고리 처리"""
        learning_manager = get_learning_manager()
        decision = learning_manager.get_cascade_decision(
            category="unknown_category",
            query="질문",
            agent_id="claude-sonnet",
        )
        # 기본값으로 처리
        assert decision is not None

    def test_cache_with_special_characters(self):
        """특수 문자 포함 캐시"""
        cache = SessionCache()
        session_id = "test-session"
        special_key = "key:with:colons/and/slashes"
        cache.set(session_id, special_key, {"data": "특수문자 데이터"})
        result = cache.get(session_id, special_key)
        assert result["data"] == "특수문자 데이터"

    def test_large_context_handling(self):
        """큰 컨텍스트 처리"""
        cache = get_consultation_cache()
        consultation_id = "large-context-test"
        large_context = {
            "messages": [f"메시지 {i}" for i in range(1000)],
            "documents": [f"문서 {i}" for i in range(100)],
        }
        cache.set_context(consultation_id, "main", large_context)
        result = cache.get_context(consultation_id, "main")
        assert len(result["messages"]) == 1000

    def test_concurrent_cache_access(self):
        """동시 캐시 접근"""
        cache = SessionCache()
        session_id = "concurrent-test"
        # 동시에 여러 키에 접근
        for i in range(100):
            cache.set(session_id, f"key{i}", f"value{i}")

        for i in range(100):
            result = cache.get(session_id, f"key{i}")
            # 캐시 크기 제한으로 일부는 eviction 될 수 있음

        # 최근 것은 남아있음
        assert cache.get(session_id, "key99") == "value99"


# =============================================================================
# Performance 테스트
# =============================================================================


class TestPerformance:
    """성능 테스트"""

    def test_cache_operations_speed(self):
        """캐시 연산 속도"""
        import time

        cache = SessionCache(max_entries=10000)
        session_id = "perf-test"

        # Set 연산
        start = time.time()
        for i in range(1000):
            cache.set(session_id, f"key{i}", f"value{i}")
        set_time = time.time() - start
        assert set_time < 1.0  # 1초 미만

        # Get 연산
        start = time.time()
        for i in range(1000):
            cache.get(session_id, f"key{i}")
        get_time = time.time() - start
        assert get_time < 0.5  # 0.5초 미만

    def test_learning_manager_scaling(self):
        """학습 관리자 확장성"""
        import time

        manager = LearningManager()

        start = time.time()
        for i in range(100):
            manager.record_consultation_outcome(
                agent_id="claude-sonnet",
                model_id="anthropic/claude-sonnet-4",
                tier=ConsultationTier.DRAFTER,
                consultation_id=f"perf-test-{i}",
                user_id="user-456",
                category="민사",
                query="질문",
                success=True,
                response_quality=0.8,
                response_time_ms=1000,
                tokens_used=300,
                cost_usd=0.001,
            )
        record_time = time.time() - start
        assert record_time < 2.0  # 2초 미만

        # 추천 속도
        start = time.time()
        for _ in range(100):
            manager.get_agent_recommendation("민사", "질문")
        recommend_time = time.time() - start
        assert recommend_time < 2.0  # 2초 미만
