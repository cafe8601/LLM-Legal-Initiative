"""
Enhanced CascadeCouncil Tests

확장 CascadeCouncil 테스트 - MAS multiagent v4 통합
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.enhanced_cascade_council import (
    EnhancedCascadeCouncil,
    EnhancedCouncilConfig,
    EnhancedCouncilResult,
    create_enhanced_council,
)
from app.services.llm.cascade_council import (
    CascadeCouncilResult,
    CascadeOpinion,
    CascadeSynthesis,
)
from app.services.learning.learning_manager import (
    CascadeDecision,
    AgentRecommendation,
)
from app.services.rag.hybrid_orchestrator import HybridSearchResult


# =============================================================================
# Configuration Tests
# =============================================================================


class TestEnhancedCouncilConfig:
    """Test EnhancedCouncilConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancedCouncilConfig()

        assert config.enable_learning is True
        assert config.auto_record_outcomes is True
        assert config.min_drafter_confidence == 0.6

        assert config.enable_experience_rag is True
        assert config.enable_legal_rag is True
        assert config.max_context_tokens == 6000

        assert config.enable_memory is True
        assert config.memory_ttl == 3600

        assert config.use_intelligent_routing is True
        assert config.drafter_success_threshold == 0.65

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnhancedCouncilConfig(
            enable_learning=False,
            max_context_tokens=4000,
            use_intelligent_routing=False,
        )

        assert config.enable_learning is False
        assert config.max_context_tokens == 4000
        assert config.use_intelligent_routing is False


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEnhancedCascadeCouncilInit:
    """Test EnhancedCascadeCouncil initialization."""

    @pytest.mark.asyncio
    async def test_init_with_default_config(self, db_session: AsyncSession):
        """Test initialization with default config."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            assert council.db == db_session
            assert council.config.enable_learning is True
            assert council._learning_manager is None
            assert council._memory_manager is None
            assert council._rag_orchestrator is None
            assert council.stats["total_consultations"] == 0

    @pytest.mark.asyncio
    async def test_init_with_custom_config(self, db_session: AsyncSession):
        """Test initialization with custom config."""
        config = EnhancedCouncilConfig(
            enable_learning=False,
            enable_experience_rag=False,
        )

        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session, config=config)

            assert council.config.enable_learning is False
            assert council.config.enable_experience_rag is False


# =============================================================================
# Manager Acquisition Tests
# =============================================================================


class TestManagerAcquisition:
    """Test lazy initialization of managers."""

    @pytest.mark.asyncio
    async def test_get_learning_manager(self, db_session: AsyncSession):
        """Test learning manager lazy initialization."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            with patch("app.services.llm.enhanced_cascade_council.get_learning_manager") as mock_get:
                mock_manager = MagicMock()
                mock_get.return_value = mock_manager

                council = EnhancedCascadeCouncil(db=db_session)

                # First call initializes
                manager1 = council._get_learning_manager()
                assert manager1 == mock_manager
                mock_get.assert_called_once()

                # Second call returns cached
                manager2 = council._get_learning_manager()
                assert manager2 == manager1
                assert mock_get.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_get_memory_manager(self, db_session: AsyncSession):
        """Test memory manager lazy initialization."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            with patch("app.services.llm.enhanced_cascade_council.get_memory_manager") as mock_get:
                mock_manager = MagicMock()
                mock_get.return_value = mock_manager

                council = EnhancedCascadeCouncil(db=db_session)

                manager = council._get_memory_manager()
                assert manager == mock_manager
                mock_get.assert_called_once_with(db_session)

    @pytest.mark.asyncio
    async def test_get_rag_orchestrator(self, db_session: AsyncSession):
        """Test RAG orchestrator lazy initialization."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            with patch("app.services.llm.enhanced_cascade_council.HybridRAGOrchestrator") as mock_class:
                mock_orchestrator = MagicMock()
                mock_class.return_value = mock_orchestrator

                council = EnhancedCascadeCouncil(db=db_session)

                orchestrator = council._get_rag_orchestrator()
                assert orchestrator == mock_orchestrator
                mock_class.assert_called_once()


# =============================================================================
# Routing Decision Tests
# =============================================================================


class TestRoutingDecision:
    """Test learning-based routing decision."""

    @pytest.mark.asyncio
    async def test_routing_decision_enabled(self, db_session: AsyncSession):
        """Test routing decision when intelligent routing is enabled."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Mock learning manager
            mock_learning_manager = MagicMock()
            mock_learning_manager.get_cascade_decision.return_value = CascadeDecision(
                use_cascade=True,
                skip_drafter=False,
                predicted_drafter_success=0.75,
                decision_reason="High historical success rate",
            )
            mock_learning_manager.get_agent_recommendation.return_value = AgentRecommendation(
                recommended_agent="claude",
                confidence=0.85,
                reason="Best for contract law",
                use_drafter=True,
                drafter_success_probability=0.8,
                similar_cases=5,
            )
            council._learning_manager = mock_learning_manager

            result = await council._get_routing_decision(
                query="계약서 검토해주세요",
                category="contract",
            )

            assert result is not None
            assert result["drafter_prediction"] == 0.75
            assert result["skip_drafter"] is False
            assert len(result["recommendations"]) == 4  # 4 agents

    @pytest.mark.asyncio
    async def test_routing_decision_disabled(self, db_session: AsyncSession):
        """Test routing decision when intelligent routing is disabled."""
        config = EnhancedCouncilConfig(use_intelligent_routing=False)

        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session, config=config)

            result = await council._get_routing_decision(
                query="계약서 검토해주세요",
                category="contract",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_routing_decision_error_handling(self, db_session: AsyncSession):
        """Test routing decision handles errors gracefully."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Mock learning manager to raise exception
            mock_learning_manager = MagicMock()
            mock_learning_manager.get_cascade_decision.side_effect = Exception("DB error")
            council._learning_manager = mock_learning_manager

            result = await council._get_routing_decision(
                query="계약서 검토해주세요",
                category="contract",
            )

            assert result is None  # Returns None on error


# =============================================================================
# Memory Context Tests
# =============================================================================


class TestMemoryContext:
    """Test memory context loading."""

    @pytest.mark.asyncio
    async def test_load_memory_context_enabled(self, db_session: AsyncSession):
        """Test memory context loading when enabled."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Mock memory manager
            mock_memory_manager = AsyncMock()
            mock_context = MagicMock()
            mock_context.session_memory = "Session data"
            mock_context.short_term_memory = "Short term data"
            mock_context.long_term_memory = "Long term data"
            mock_memory_manager.get_council_context.return_value = mock_context
            council._memory_manager = mock_memory_manager

            result = await council._load_memory_context(
                user_id=uuid4(),
                consultation_id=uuid4(),
                category="contract",
            )

            assert result["session_memory"] == "Session data"
            assert result["short_term_memory"] == "Short term data"
            assert result["long_term_memory"] == "Long term data"

    @pytest.mark.asyncio
    async def test_load_memory_context_disabled(self, db_session: AsyncSession):
        """Test memory context loading when disabled."""
        config = EnhancedCouncilConfig(enable_memory=False)

        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session, config=config)

            result = await council._load_memory_context(
                user_id=uuid4(),
                consultation_id=uuid4(),
                category="contract",
            )

            assert result == {}

    @pytest.mark.asyncio
    async def test_load_memory_context_error_handling(self, db_session: AsyncSession):
        """Test memory context loading handles errors gracefully."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Mock memory manager to raise exception
            mock_memory_manager = AsyncMock()
            mock_memory_manager.get_council_context.side_effect = Exception("Memory error")
            council._memory_manager = mock_memory_manager

            result = await council._load_memory_context(
                user_id=uuid4(),
                consultation_id=uuid4(),
                category="contract",
            )

            assert result == {}


# =============================================================================
# RAG Context Tests
# =============================================================================


class TestRAGContext:
    """Test RAG context searching."""

    @pytest.mark.asyncio
    async def test_search_rag_context_enabled(self, db_session: AsyncSession):
        """Test RAG context search when enabled."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Mock RAG orchestrator
            mock_rag = AsyncMock()
            mock_result = HybridSearchResult(
                legal_documents=[{"title": "민법 제1조"}],
                similar_experiences=[],
                combined_context="관련 법률 자료...",
                total_sources=1,
                cache_hit=False,
            )
            mock_rag.search.return_value = mock_result
            council._rag_orchestrator = mock_rag

            result = await council._search_rag_context(
                query="계약 해지 조건",
                consultation_id="test-123",
                category="contract",
            )

            assert result is not None
            assert len(result.legal_documents) == 1
            assert result.combined_context == "관련 법률 자료..."

    @pytest.mark.asyncio
    async def test_search_rag_context_disabled(self, db_session: AsyncSession):
        """Test RAG context search when disabled."""
        config = EnhancedCouncilConfig(
            enable_experience_rag=False,
            enable_legal_rag=False,
        )

        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session, config=config)

            result = await council._search_rag_context(
                query="계약 해지 조건",
                consultation_id="test-123",
                category="contract",
            )

            assert result is None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_extract_agent_id(self, db_session: AsyncSession):
        """Test agent ID extraction from model name."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            assert council._extract_agent_id("claude-3-opus") == "claude"
            assert council._extract_agent_id("anthropic-claude") == "claude"
            assert council._extract_agent_id("gpt-4") == "gpt"
            assert council._extract_agent_id("openai-gpt-4") == "gpt"
            assert council._extract_agent_id("gemini-pro") == "gemini"
            assert council._extract_agent_id("google-gemini") == "gemini"
            assert council._extract_agent_id("grok-1") == "grok"
            assert council._extract_agent_id("x-ai-grok") == "grok"
            assert council._extract_agent_id("unknown-model") == "unknown"

    @pytest.mark.asyncio
    async def test_extract_keywords(self, db_session: AsyncSession):
        """Test keyword extraction from text."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            keywords = council._extract_keywords(
                "계약 해지 후 손해배상 청구 관련 문의입니다. 임대차 계약입니다."
            )

            assert "계약" in keywords
            assert "손해배상" in keywords
            assert "임대차" in keywords
            assert len(keywords) <= 10  # Max 10 keywords

    @pytest.mark.asyncio
    async def test_calculate_quality_score(self, db_session: AsyncSession):
        """Test quality score calculation."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Create mock result with all components
            mock_result = MagicMock(spec=CascadeCouncilResult)
            mock_result.synthesis = MagicMock()
            mock_result.synthesis.content = "Synthesis content..."
            mock_result.opinions = [
                MagicMock(error=None),
                MagicMock(error=None),
                MagicMock(error="Error"),
            ]
            mock_result.savings_percentage = 40.0
            mock_result.reviews = [MagicMock()]

            score = council._calculate_quality_score(mock_result)

            # Should have positive score
            assert score > 0
            assert score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_quality_score_minimal(self, db_session: AsyncSession):
        """Test quality score with minimal result."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = EnhancedCascadeCouncil(db=db_session)

            # Create mock result with minimal components
            mock_result = MagicMock(spec=CascadeCouncilResult)
            mock_result.synthesis = None
            mock_result.opinions = []
            mock_result.savings_percentage = 0.0
            mock_result.reviews = []

            score = council._calculate_quality_score(mock_result)

            assert score == 0.0


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Test statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_stats(self, db_session: AsyncSession):
        """Test getting stats."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_instance.get_cascade_stats.return_value = {"cascade_calls": 10}
            mock_orchestrator.return_value = mock_instance

            council = EnhancedCascadeCouncil(db=db_session)

            stats = council.get_stats()

            assert "enhanced_stats" in stats
            assert "cascade_stats" in stats
            assert "config" in stats
            assert stats["enhanced_stats"]["total_consultations"] == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Test factory function."""

    @pytest.mark.asyncio
    async def test_create_enhanced_council(self, db_session: AsyncSession):
        """Test create_enhanced_council factory."""
        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = create_enhanced_council(db=db_session)

            assert isinstance(council, EnhancedCascadeCouncil)
            assert council.db == db_session

    @pytest.mark.asyncio
    async def test_create_enhanced_council_with_config(self, db_session: AsyncSession):
        """Test create_enhanced_council with custom config."""
        config = EnhancedCouncilConfig(enable_learning=False)

        with patch("app.services.llm.enhanced_cascade_council.CascadeCouncilOrchestrator"):
            council = create_enhanced_council(db=db_session, config=config)

            assert council.config.enable_learning is False


# =============================================================================
# Enhanced Result Tests
# =============================================================================


class TestEnhancedCouncilResult:
    """Test EnhancedCouncilResult dataclass."""

    def test_enhanced_result_creation(self):
        """Test creating EnhancedCouncilResult."""
        result = EnhancedCouncilResult(
            opinions=[],
            reviews=[],
            synthesis=None,
            total_latency_ms=1000,
            total_tokens_used=500,
            stage_timings={},
            errors=[],
            total_estimated_cost=0.01,
            total_cost_saved=0.005,
            drafter_success_count=2,
            escalation_count=1,
            learning_applied=True,
            drafter_prediction_confidence=0.8,
            agent_recommendations=[{"agent_id": "claude"}],
            legal_docs_used=3,
            experiences_used=2,
            rag_cache_hit=True,
            consultation_outcome_id="outcome_123",
        )

        assert result.learning_applied is True
        assert result.drafter_prediction_confidence == 0.8
        assert len(result.agent_recommendations) == 1
        assert result.legal_docs_used == 3
        assert result.experiences_used == 2
        assert result.rag_cache_hit is True
        assert result.consultation_outcome_id == "outcome_123"

    def test_enhanced_result_defaults(self):
        """Test EnhancedCouncilResult default values."""
        result = EnhancedCouncilResult(
            opinions=[],
            reviews=[],
            synthesis=None,
            total_latency_ms=0,
            total_tokens_used=0,
            stage_timings={},
            errors=[],
            total_estimated_cost=0,
            total_cost_saved=0,
            drafter_success_count=0,
            escalation_count=0,
        )

        assert result.learning_applied is False
        assert result.drafter_prediction_confidence == 0.0
        assert result.agent_recommendations == []
        assert result.legal_docs_used == 0
        assert result.experiences_used == 0
        assert result.rag_cache_hit is False
        assert result.consultation_outcome_id is None
