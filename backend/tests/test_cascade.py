"""
CascadeFlow 통합 테스트

CascadeService와 CascadeCouncilOrchestrator의 핵심 기능 테스트.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm.cascade_service import (
    CascadeService,
    CascadeServiceFactory,
    CascadeModel,
    CascadeModelTier,
    CascadePair,
    QualityValidator,
    QueryComplexity,
    QualityCheckResult,
    CLAUDE_CASCADE,
    GPT_CASCADE,
    GEMINI_CASCADE,
    CHAIRMAN_CASCADE,
)
from app.services.llm.cascade_council import (
    CascadeCouncilOrchestrator,
    CascadeOpinion,
)


# =============================================================================
# QualityValidator 테스트
# =============================================================================


class TestQualityValidator:
    """품질 검증기 테스트"""

    def setup_method(self):
        self.validator = QualityValidator(
            min_length=100,
            max_length=10000,
            confidence_threshold=0.7,
            require_legal_basis=True,
        )

    def test_short_response_fails(self):
        """짧은 응답은 품질 검증 실패"""
        result = self.validator.check_response_quality(
            query="임대차 계약 해지 방법",
            response="잘 모르겠습니다.",  # 너무 짧음
        )
        assert not result.passed
        assert result.confidence < 0.7
        assert any("짧" in r for r in result.reasons)

    def test_response_with_legal_basis_passes(self):
        """법적 근거가 포함된 응답은 높은 점수"""
        response = """
        임대차 계약 해지에 대해 설명드리겠습니다.

        주택임대차보호법 제6조에 따르면, 임대인이 임대차기간이 끝나기
        6개월 전부터 2개월 전까지의 기간에 임차인에게 갱신거절의 통지를
        하지 아니한 경우에는 그 기간이 끝난 때에 전 임대차와 동일한
        조건으로 다시 임대차한 것으로 봅니다.

        대법원 판례에 따르면, 임차인의 차임연체가 2기 이상인 경우
        임대인은 계약을 해지할 수 있습니다.

        따라서 귀하의 경우, 계약 해지를 위해서는 위 법률 요건을
        충족해야 합니다.
        """
        result = self.validator.check_response_quality(
            query="임대차 계약 해지 방법",
            response=response,
        )
        assert result.confidence >= 0.5  # 법적 근거 포함
        assert result.metrics["legal_patterns"] >= 1

    def test_low_quality_patterns_detected(self):
        """저품질 패턴 감지"""
        response = """
        임대차 계약 해지에 대해 말씀드리자면, 확실하지 않습니다만
        일반적인 정보만 드릴 수 있습니다. 자세한 내용은 법률 전문가와
        상담하시기 바랍니다. 더 많은 정보가 필요합니다.
        """
        result = self.validator.check_response_quality(
            query="임대차 계약 해지 방법",
            response=response,
        )
        assert result.metrics["low_quality_patterns"] > 0

    def test_query_complexity_simple(self):
        """단순 쿼리 복잡도 분류"""
        simple_queries = [
            "임대차 뜻이 뭔가요?",
            "계약서 작성 비용 얼마인가요?",
            "등기 신청 어디서 하나요?",
        ]
        for query in simple_queries:
            complexity = self.validator.classify_query_complexity(query)
            assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]

    def test_query_complexity_complex(self):
        """복잡 쿼리 복잡도 분류"""
        complex_queries = [
            "상속 분쟁으로 인한 유류분 반환 청구 소송을 제기하려고 합니다.",
            "임대인이 계약 해지 후 보증금을 반환하지 않아 가압류 신청을 고려중입니다.",
            "형사 고소를 당했는데 어떻게 대응해야 할까요?",
        ]
        for query in complex_queries:
            complexity = self.validator.classify_query_complexity(query)
            assert complexity == QueryComplexity.COMPLEX


# =============================================================================
# CascadePair 테스트
# =============================================================================


class TestCascadePair:
    """캐스케이드 모델 쌍 테스트"""

    def test_claude_cascade_cost_ratio(self):
        """Claude 캐스케이드 비용 비율"""
        assert CLAUDE_CASCADE.cost_ratio > 1
        assert CLAUDE_CASCADE.drafter.cost_per_1m_tokens < CLAUDE_CASCADE.verifier.cost_per_1m_tokens

    def test_gpt_cascade_cost_ratio(self):
        """GPT 캐스케이드 비용 비율"""
        assert GPT_CASCADE.cost_ratio > 1
        # GPT-4o-mini는 GPT-5.1보다 훨씬 저렴
        assert GPT_CASCADE.drafter.cost_per_1m_tokens < 0.5

    def test_potential_savings_calculation(self):
        """잠재적 비용 절감률 계산"""
        # 60% 드래프터 사용 가정 시 절감률
        savings = GPT_CASCADE.potential_savings
        assert 0 < savings < 1
        assert savings > 0.3  # 최소 30% 절감 기대

    def test_chairman_cascade_verifier_is_opus(self):
        """의장 캐스케이드의 검증자는 Opus"""
        assert "opus" in CHAIRMAN_CASCADE.verifier.openrouter_id.lower()


# =============================================================================
# CascadeService 테스트 (Mock)
# =============================================================================


class TestCascadeService:
    """캐스케이드 서비스 테스트"""

    @pytest.fixture
    def mock_openai_client(self):
        """OpenAI 클라이언트 목"""
        mock_client = AsyncMock()

        # 응답 목 설정
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        법률 상담에 대한 답변입니다.

        민법 제750조에 따르면 고의 또는 과실로 인한 위법행위로
        타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있습니다.

        대법원 판례에서도 이를 확인할 수 있습니다.
        따라서 귀하의 경우 손해배상 청구가 가능합니다.
        """
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 500

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_cascade_service_drafter_success(self, mock_openai_client):
        """드래프터 응답이 품질 검증 통과 시 드래프터만 사용"""
        service = CascadeService(
            cascade_pair=GPT_CASCADE,
            quality_validator=QualityValidator(
                confidence_threshold=0.5,  # 낮은 임계값
                require_legal_basis=False,
            ),
        )

        with patch.object(service, "_client", mock_openai_client):
            result = await service.execute(
                system_prompt="법률 전문가입니다.",
                user_prompt="계약 해지 방법?",
            )

            assert result.tier_used == CascadeModelTier.DRAFTER
            assert not result.escalated
            assert result.content != ""

    @pytest.mark.asyncio
    async def test_cascade_service_force_verifier(self, mock_openai_client):
        """force_verifier=True면 검증자 직접 사용"""
        service = CascadeService(cascade_pair=GPT_CASCADE)

        with patch.object(service, "_client", mock_openai_client):
            result = await service.execute(
                system_prompt="법률 전문가입니다.",
                user_prompt="복잡한 상속 분쟁 사안입니다.",
                force_verifier=True,
            )

            assert result.tier_used == CascadeModelTier.VERIFIER
            assert not result.drafter_attempted

    def test_cascade_stats_initial(self):
        """초기 통계 상태"""
        service = CascadeService(cascade_pair=CLAUDE_CASCADE)
        stats = service.get_stats_summary()
        assert "아직 요청이 없습니다" in stats.get("message", "")


# =============================================================================
# CascadeServiceFactory 테스트
# =============================================================================


class TestCascadeServiceFactory:
    """캐스케이드 서비스 팩토리 테스트"""

    def test_create_claude_cascade(self):
        """Claude 캐스케이드 생성"""
        service = CascadeServiceFactory.create_claude_cascade()
        assert service.cascade_pair == CLAUDE_CASCADE

    def test_create_gpt_cascade(self):
        """GPT 캐스케이드 생성"""
        service = CascadeServiceFactory.create_gpt_cascade()
        assert service.cascade_pair == GPT_CASCADE

    def test_create_gemini_cascade(self):
        """Gemini 캐스케이드 생성"""
        service = CascadeServiceFactory.create_gemini_cascade()
        assert service.cascade_pair == GEMINI_CASCADE

    def test_create_chairman_cascade(self):
        """의장 캐스케이드 생성"""
        service = CascadeServiceFactory.create_chairman_cascade()
        assert service.cascade_pair == CHAIRMAN_CASCADE

    def test_create_all_council_cascades(self):
        """모든 위원회 캐스케이드 생성"""
        services = CascadeServiceFactory.create_all_council_cascades()
        assert "claude" in services
        assert "gpt" in services
        assert "gemini" in services
        assert "grok" in services

    def test_create_for_invalid_provider_raises(self):
        """지원하지 않는 제공자는 예외 발생"""
        with pytest.raises(ValueError):
            CascadeServiceFactory.create_for_provider("invalid_provider")


# =============================================================================
# CascadeCouncilOrchestrator 테스트
# =============================================================================


class TestCascadeCouncilOrchestrator:
    """캐스케이드 위원회 오케스트레이터 테스트"""

    def test_orchestrator_initialization(self):
        """오케스트레이터 초기화"""
        orchestrator = CascadeCouncilOrchestrator(
            enable_peer_review=True,
            enable_cascade=True,
            max_concurrent=4,
        )
        assert orchestrator.enable_peer_review
        assert orchestrator.enable_cascade
        assert orchestrator.max_concurrent == 4

    def test_orchestrator_cascade_disabled(self):
        """캐스케이드 비활성화 옵션"""
        orchestrator = CascadeCouncilOrchestrator(enable_cascade=False)
        assert not orchestrator.enable_cascade

    def test_get_cascade_services(self):
        """캐스케이드 서비스 획득"""
        orchestrator = CascadeCouncilOrchestrator()
        services = orchestrator._get_cascade_services()
        assert "claude" in services
        assert "gpt" in services
        assert "gemini" in services
        assert "grok" in services

    def test_get_chairman_cascade(self):
        """의장 캐스케이드 획득"""
        orchestrator = CascadeCouncilOrchestrator()
        chairman = orchestrator._get_chairman_cascade()
        assert chairman is not None
        assert chairman.cascade_pair == CHAIRMAN_CASCADE

    def test_provider_key_extraction(self):
        """제공자 키 추출"""
        orchestrator = CascadeCouncilOrchestrator()

        assert orchestrator._get_provider_key("anthropic/claude-sonnet-4") == "claude"
        assert orchestrator._get_provider_key("openai/gpt-4o") == "gpt"
        assert orchestrator._get_provider_key("google/gemini-2.5-pro") == "gemini"
        assert orchestrator._get_provider_key("x-ai/grok-2") == "grok"

    def test_cascade_stats(self):
        """캐스케이드 통계"""
        orchestrator = CascadeCouncilOrchestrator()
        stats = orchestrator.get_cascade_stats()
        assert "services" in stats
        assert "total" in stats
        assert "requests" in stats["total"]

    @pytest.mark.asyncio
    async def test_orchestrator_close(self):
        """오케스트레이터 정리"""
        orchestrator = CascadeCouncilOrchestrator()
        # 서비스 초기화
        _ = orchestrator._get_cascade_services()
        _ = orchestrator._get_chairman_cascade()

        await orchestrator.close()

        assert orchestrator._cascade_services is None
        assert orchestrator._chairman_cascade is None


# =============================================================================
# 통합 테스트 (E2E Mock)
# =============================================================================


class TestCascadeIntegration:
    """캐스케이드 통합 테스트"""

    def test_cascade_opinion_dataclass(self):
        """CascadeOpinion 데이터클래스"""
        opinion = CascadeOpinion(
            model="anthropic/claude-3-5-haiku",
            display_name="Claude Haiku 위원",
            content="법률 의견입니다.",
            drafter_used=True,
            escalated=False,
            cost_saved=0.001,
        )
        assert opinion.drafter_used
        assert not opinion.escalated
        assert opinion.cost_saved > 0

    def test_cost_savings_calculation(self):
        """비용 절감 계산 정확성"""
        # GPT 캐스케이드로 1000 토큰 처리 시
        drafter_cost = (1000 / 1_000_000) * GPT_CASCADE.drafter.cost_per_1m_tokens
        verifier_cost = (1000 / 1_000_000) * GPT_CASCADE.verifier.cost_per_1m_tokens

        # 드래프터로 처리 성공 시 절감액
        saved = verifier_cost - drafter_cost
        assert saved > 0

        # 절감률
        savings_rate = saved / verifier_cost
        assert 0.9 < savings_rate < 1.0  # 90% 이상 절감


# =============================================================================
# 품질 검증 엣지 케이스 테스트
# =============================================================================


class TestQualityValidatorEdgeCases:
    """품질 검증기 엣지 케이스"""

    def test_empty_response(self):
        """빈 응답"""
        validator = QualityValidator()
        result = validator.check_response_quality(
            query="테스트",
            response="",
        )
        assert not result.passed
        assert result.confidence < 0.5

    def test_very_long_response(self):
        """매우 긴 응답"""
        validator = QualityValidator(max_length=100)
        result = validator.check_response_quality(
            query="테스트",
            response="a" * 200,
        )
        # Check for "깁" (conjugated form of 길다 "to be long") or max_length indicator (>)
        assert any("깁" in r or ">" in r for r in result.reasons)

    def test_empty_query_handling(self):
        """빈 쿼리 처리"""
        validator = QualityValidator()
        complexity = validator.classify_query_complexity("")
        assert complexity == QueryComplexity.SIMPLE

    def test_unicode_handling(self):
        """유니코드 처리"""
        validator = QualityValidator(min_length=10)
        result = validator.check_response_quality(
            query="한글 테스트 질문입니다",
            response="한글로 작성된 응답입니다. 민법 제750조에 따르면 손해배상이 가능합니다.",
        )
        # 유니코드도 정상 처리됨
        assert result.metrics["length"] > 0
