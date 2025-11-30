"""
v4.1 Prompt Integration Tests

v4.1 한글 프롬프트 시스템 통합 테스트.
- 프롬프트 생성 및 메모리 주입 테스트
- LLMModel, TaskComplexity enum 검증
- 모델 파라미터 설정 테스트
- Council Orchestrator 메모리 통합 테스트
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm.legal_prompts_v4_1 import (
    LLMModel,
    TaskComplexity,
    MODEL_PARAMETERS,
    LEGAL_EXPERT_PROMPTS_KR,
    CHAIRMAN_PROMPTS_KR,
    STAGE2_REVIEW_PROMPTS_KR,
    STAGE2_EVALUATION_CRITERIA_KR,
    CLAUDE_OPUS_EFFORT_PARAMS,
    GPT_51_REASONING_PARAMS,
    inject_memory_and_rag,
    get_legal_expert_prompt_kr,
    get_chairman_prompt_kr,
    get_stage2_review_prompt_kr,
    get_model_parameters,
    select_optimal_model,
)


class TestLLMModelEnum:
    """LLMModel enum 테스트."""

    def test_all_models_defined(self):
        """모든 지원 모델이 정의되어 있는지 확인."""
        expected_models = [
            "openai/gpt-5.1",
            "openai/gpt-5.1-none",
            "anthropic/claude-opus-4-5-20251101",
            "anthropic/claude-sonnet-4-5-20250929",
            "google/gemini-3-pro-preview",
            "x-ai/grok-4",
            "x-ai/grok-4-1-fast-reasoning",
            "x-ai/grok-4-1-fast-non-reasoning",
        ]
        actual_models = [m.value for m in LLMModel]
        for model in expected_models:
            assert model in actual_models, f"Missing model: {model}"

    def test_model_enum_values(self):
        """모델 enum 값이 올바른지 확인."""
        assert LLMModel.GPT_51.value == "openai/gpt-5.1"
        assert LLMModel.CLAUDE_OPUS_45.value == "anthropic/claude-opus-4-5-20251101"
        assert LLMModel.GEMINI_3_PRO.value == "google/gemini-3-pro-preview"
        assert LLMModel.GROK_4.value == "x-ai/grok-4"


class TestTaskComplexityEnum:
    """TaskComplexity enum 테스트."""

    def test_all_complexities_defined(self):
        """모든 복잡도 레벨이 정의되어 있는지 확인."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"

    def test_complexity_count(self):
        """복잡도 레벨이 3개인지 확인."""
        assert len(TaskComplexity) == 3


class TestPromptTemplates:
    """프롬프트 템플릿 테스트."""

    def test_legal_expert_prompts_have_memory_placeholders(self):
        """법률 전문가 프롬프트에 메모리 placeholder가 있는지 확인."""
        required_placeholders = [
            "{{SESSION_MEMORY}}",
            "{{SHORT_TERM_MEMORY}}",
            "{{LONG_TERM_MEMORY}}",
            "{{LEGAL_RAG_RESULTS}}",
        ]
        for model, prompt in LEGAL_EXPERT_PROMPTS_KR.items():
            for placeholder in required_placeholders:
                assert placeholder in prompt, f"Missing {placeholder} in {model}"

    def test_chairman_prompts_have_memory_placeholders(self):
        """의장 프롬프트에 메모리 placeholder가 있는지 확인."""
        required_placeholders = [
            "{{SESSION_MEMORY}}",
            "{{SHORT_TERM_MEMORY}}",
            "{{LONG_TERM_MEMORY}}",
            "{{LEGAL_RAG_RESULTS}}",
        ]
        for model, prompt in CHAIRMAN_PROMPTS_KR.items():
            for placeholder in required_placeholders:
                assert placeholder in prompt, f"Missing {placeholder} in chairman {model}"

    def test_stage2_prompts_have_rag_placeholder(self):
        """Stage 2 프롬프트에 RAG placeholder가 있는지 확인."""
        for model, prompt in STAGE2_REVIEW_PROMPTS_KR.items():
            assert "{{LEGAL_RAG_RESULTS}}" in prompt, f"Missing RAG in stage2 {model}"

    def test_all_prompts_are_korean(self):
        """모든 프롬프트가 한국어인지 확인."""
        korean_keywords = ["한국어", "법률", "클라이언트", "분석", "권고"]
        for model, prompt in LEGAL_EXPERT_PROMPTS_KR.items():
            has_korean = any(kw in prompt for kw in korean_keywords)
            assert has_korean, f"Prompt for {model} may not be Korean"

    def test_prompts_have_irac_format(self):
        """법률 전문가 프롬프트에 IRAC 형식이 포함되어 있는지 확인."""
        irac_keywords = ["Issue", "Rule", "Application", "Conclusion", "IRAC", "쟁점", "규칙", "적용", "결론", "법적 근거", "분석"]
        for model, prompt in LEGAL_EXPERT_PROMPTS_KR.items():
            has_irac = any(kw in prompt for kw in irac_keywords)
            assert has_irac, f"Missing IRAC format in {model}"


class TestInjectMemoryAndRag:
    """inject_memory_and_rag 함수 테스트."""

    def test_inject_all_memories(self):
        """모든 메모리가 올바르게 주입되는지 확인."""
        template = """
        Session: {{SESSION_MEMORY}}
        Short: {{SHORT_TERM_MEMORY}}
        Long: {{LONG_TERM_MEMORY}}
        RAG: {{LEGAL_RAG_RESULTS}}
        """
        result = inject_memory_and_rag(
            template,
            session_memory="세션 내용",
            short_term_memory="단기 내용",
            long_term_memory="장기 내용",
            rag_results="RAG 결과",
        )
        assert "세션 내용" in result
        assert "단기 내용" in result
        assert "장기 내용" in result
        assert "RAG 결과" in result

    def test_inject_default_values(self):
        """기본값이 올바르게 주입되는지 확인."""
        template = """
        Session: {{SESSION_MEMORY}}
        Short: {{SHORT_TERM_MEMORY}}
        Long: {{LONG_TERM_MEMORY}}
        RAG: {{LEGAL_RAG_RESULTS}}
        """
        result = inject_memory_and_rag(template)
        assert "[세션 컨텍스트 없음]" in result
        assert "[최근 상담 없음]" in result
        assert "[클라이언트 이력 없음]" in result
        assert "[RAG 결과 없음]" in result

    def test_inject_partial_memories(self):
        """일부 메모리만 제공될 때 나머지는 기본값으로 채워지는지 확인."""
        template = "Session: {{SESSION_MEMORY}}, RAG: {{LEGAL_RAG_RESULTS}}"
        result = inject_memory_and_rag(
            template,
            session_memory="세션 있음",
        )
        assert "세션 있음" in result
        assert "[RAG 결과 없음]" in result


class TestGetLegalExpertPromptKr:
    """get_legal_expert_prompt_kr 함수 테스트."""

    def test_get_prompt_for_known_model(self):
        """알려진 모델에 대한 프롬프트 획득."""
        prompt = get_legal_expert_prompt_kr(
            LLMModel.GPT_51.value,
            session_memory="테스트 세션",
        )
        assert "테스트 세션" in prompt
        assert "GPT-5.1" in prompt or "법률" in prompt

    def test_get_prompt_for_unknown_model_uses_default(self):
        """알려지지 않은 모델은 기본 프롬프트(Sonnet) 사용."""
        prompt = get_legal_expert_prompt_kr(
            "unknown/model",
            session_memory="테스트",
        )
        assert "테스트" in prompt
        # Default is Sonnet prompt
        assert "법률" in prompt

    def test_memory_injection_in_prompt(self):
        """프롬프트에 메모리가 올바르게 주입되는지 확인."""
        prompt = get_legal_expert_prompt_kr(
            LLMModel.CLAUDE_SONNET_45.value,
            session_memory="이전 대화 내용",
            short_term_memory="7일 내 상담",
            long_term_memory="전체 이력",
            rag_results="검색된 법률 문서",
        )
        assert "이전 대화 내용" in prompt
        assert "7일 내 상담" in prompt
        assert "전체 이력" in prompt
        assert "검색된 법률 문서" in prompt


class TestGetChairmanPromptKr:
    """get_chairman_prompt_kr 함수 테스트."""

    def test_get_opus_chairman_prompt(self):
        """Opus 의장 프롬프트 획득."""
        prompt = get_chairman_prompt_kr(
            LLMModel.CLAUDE_OPUS_45.value,
            session_memory="의장 세션",
        )
        assert "의장" in prompt
        assert "의장 세션" in prompt

    def test_chairman_prompt_has_synthesis_protocol(self):
        """의장 프롬프트에 합성 프로토콜이 있는지 확인."""
        prompt = get_chairman_prompt_kr(LLMModel.CLAUDE_OPUS_45.value)
        # Check for synthesis-related keywords
        synthesis_keywords = ["합성", "합의", "충돌", "검증"]
        has_synthesis = any(kw in prompt for kw in synthesis_keywords)
        assert has_synthesis, "Chairman prompt should contain synthesis protocol"


class TestGetStage2ReviewPromptKr:
    """get_stage2_review_prompt_kr 함수 테스트."""

    def test_stage2_prompt_includes_question_and_opinions(self):
        """Stage 2 프롬프트에 질문과 의견이 포함되는지 확인."""
        prompt = get_stage2_review_prompt_kr(
            LLMModel.CLAUDE_SONNET_45.value,
            original_question="계약 위반 시 손해배상은?",
            anonymized_opinions="의견 A: ... 의견 B: ...",
            rag_results="관련 판례",
        )
        assert "계약 위반 시 손해배상은?" in prompt
        assert "의견 A:" in prompt
        assert "관련 판례" in prompt

    def test_stage2_prompt_includes_evaluation_criteria(self):
        """Stage 2 프롬프트에 평가 기준이 포함되는지 확인."""
        prompt = get_stage2_review_prompt_kr(
            LLMModel.CLAUDE_SONNET_45.value,
            original_question="질문",
            anonymized_opinions="의견",
        )
        # Check evaluation criteria keywords
        criteria_keywords = ["정확성", "완전성", "메모리 통합", "유용성", "명확성"]
        for keyword in criteria_keywords:
            assert keyword in prompt, f"Missing evaluation criteria: {keyword}"


class TestGetModelParameters:
    """get_model_parameters 함수 테스트."""

    def test_get_claude_opus_params_by_complexity(self):
        """Claude Opus 복잡도별 파라미터 테스트."""
        # Simple
        params_simple = get_model_parameters(
            LLMModel.CLAUDE_OPUS_45.value,
            TaskComplexity.SIMPLE,
        )
        assert params_simple.get("effort") == "low"

        # Medium
        params_medium = get_model_parameters(
            LLMModel.CLAUDE_OPUS_45.value,
            TaskComplexity.MEDIUM,
        )
        assert params_medium.get("effort") == "medium"

        # Complex
        params_complex = get_model_parameters(
            LLMModel.CLAUDE_OPUS_45.value,
            TaskComplexity.COMPLEX,
        )
        assert params_complex.get("effort") == "high"

    def test_get_gpt51_params_by_complexity(self):
        """GPT-5.1 복잡도별 파라미터 테스트."""
        # Simple
        params_simple = get_model_parameters(
            LLMModel.GPT_51.value,
            TaskComplexity.SIMPLE,
        )
        assert params_simple.get("reasoning_effort") == "none"

        # Complex
        params_complex = get_model_parameters(
            LLMModel.GPT_51.value,
            TaskComplexity.COMPLEX,
        )
        assert params_complex.get("reasoning_effort") == "high"

    def test_gemini_temperature_always_1(self):
        """Gemini 3 Pro의 temperature가 항상 1.0인지 확인."""
        params = get_model_parameters(LLMModel.GEMINI_3_PRO.value)
        assert params.get("temperature") == 1.0

    def test_grok_params(self):
        """Grok 모델 파라미터 테스트."""
        params = get_model_parameters(LLMModel.GROK_4.value)
        assert params.get("temperature") == 0.7
        assert params.get("max_tokens") == 8000


class TestSelectOptimalModel:
    """select_optimal_model 함수 테스트."""

    def test_chairman_always_opus(self):
        """의장은 항상 Opus 모델."""
        model = select_optimal_model("chairman", TaskComplexity.SIMPLE)
        assert model == LLMModel.CLAUDE_OPUS_45.value

        model = select_optimal_model("chairman", TaskComplexity.COMPLEX)
        assert model == LLMModel.CLAUDE_OPUS_45.value

    def test_stage2_cost_priority(self):
        """Stage 2 비용 우선 시 Grok non-reasoning 사용."""
        model = select_optimal_model("stage2", cost_priority=True)
        assert model == LLMModel.GROK_4_FAST_NON_REASONING.value

    def test_stage2_quality_priority(self):
        """Stage 2 품질 우선 시 Sonnet 사용."""
        model = select_optimal_model("stage2", cost_priority=False)
        assert model == LLMModel.CLAUDE_SONNET_45.value

    def test_stage1_complex_uses_opus(self):
        """Stage 1 복잡한 작업은 Opus 사용."""
        model = select_optimal_model("stage1", TaskComplexity.COMPLEX)
        assert model == LLMModel.CLAUDE_OPUS_45.value


class TestModelParametersConfiguration:
    """MODEL_PARAMETERS 설정 테스트."""

    def test_all_models_have_parameters(self):
        """모든 모델에 파라미터가 정의되어 있는지 확인."""
        for model in LLMModel:
            assert model.value in MODEL_PARAMETERS, f"Missing params for {model.value}"

    def test_all_models_have_max_tokens(self):
        """모든 모델에 max_tokens가 정의되어 있는지 확인."""
        for model, params in MODEL_PARAMETERS.items():
            assert "max_tokens" in params, f"Missing max_tokens for {model}"

    def test_claude_opus_effort_params(self):
        """Claude Opus effort 파라미터 설정 확인."""
        assert "low" in CLAUDE_OPUS_EFFORT_PARAMS
        assert "medium" in CLAUDE_OPUS_EFFORT_PARAMS
        assert "high" in CLAUDE_OPUS_EFFORT_PARAMS

    def test_gpt51_reasoning_params(self):
        """GPT-5.1 reasoning 파라미터 설정 확인."""
        assert "none" in GPT_51_REASONING_PARAMS
        assert "low" in GPT_51_REASONING_PARAMS
        assert "medium" in GPT_51_REASONING_PARAMS
        assert "high" in GPT_51_REASONING_PARAMS


class TestPromptMappingCompleteness:
    """프롬프트 매핑 완전성 테스트."""

    def test_legal_expert_prompts_coverage(self):
        """법률 전문가 프롬프트가 모든 모델을 커버하는지 확인."""
        covered_models = set(LEGAL_EXPERT_PROMPTS_KR.keys())
        # At least main models should be covered
        required = {
            LLMModel.GPT_51.value,
            LLMModel.CLAUDE_OPUS_45.value,
            LLMModel.CLAUDE_SONNET_45.value,
            LLMModel.GEMINI_3_PRO.value,
            LLMModel.GROK_4.value,
        }
        assert required.issubset(covered_models), "Missing legal expert prompts"

    def test_chairman_prompts_coverage(self):
        """의장 프롬프트가 주요 모델을 커버하는지 확인."""
        covered_models = set(CHAIRMAN_PROMPTS_KR.keys())
        required = {
            LLMModel.GPT_51.value,
            LLMModel.CLAUDE_OPUS_45.value,
            LLMModel.GEMINI_3_PRO.value,
            LLMModel.GROK_4.value,
        }
        assert required.issubset(covered_models), "Missing chairman prompts"

    def test_stage2_prompts_coverage(self):
        """Stage 2 프롬프트가 주요 모델을 커버하는지 확인."""
        covered_models = set(STAGE2_REVIEW_PROMPTS_KR.keys())
        required = {
            LLMModel.GPT_51.value,
            LLMModel.CLAUDE_SONNET_45.value,
            LLMModel.GEMINI_3_PRO.value,
            LLMModel.GROK_4.value,
        }
        assert required.issubset(covered_models), "Missing stage2 prompts"


class TestStage2EvaluationCriteria:
    """Stage 2 평가 기준 테스트."""

    def test_evaluation_criteria_has_five_categories(self):
        """평가 기준이 5가지 카테고리를 포함하는지 확인."""
        criteria = STAGE2_EVALUATION_CRITERIA_KR
        categories = ["정확성", "완전성", "메모리 통합", "유용성", "명확성"]
        for category in categories:
            assert category in criteria, f"Missing category: {category}"

    def test_evaluation_criteria_has_point_system(self):
        """평가 기준에 점수 시스템이 있는지 확인."""
        criteria = STAGE2_EVALUATION_CRITERIA_KR
        assert "10점" in criteria
        assert "50점" in criteria or "총점" in criteria
