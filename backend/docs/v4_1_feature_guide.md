# v4.1 프롬프트 시스템 사용 가이드

## 개요

v4.1 프롬프트 시스템은 한글 최적화된 법률 자문 프롬프트와 4-Memory 시스템을 통합한 업그레이드입니다.

### 주요 기능

1. **한글 최적화 프롬프트**: 모든 LLM 모델별 최적화된 한글 프롬프트
2. **4-Memory 시스템**: 세션/단기/장기 기억 + RAG 컨텍스트 통합
3. **TaskComplexity 기반 파라미터 최적화**: 작업 복잡도에 따른 자동 파라미터 조정
4. **IRAC 형식 법률 분석**: Issue-Rule-Application-Conclusion 구조화된 분석
5. **5가지 평가 기준**: 정확성, 완전성, 메모리 통합, 유용성, 명확성

---

## API 사용법

### 상담 생성 (v4.1 메모리 시스템 포함)

```http
POST /api/v1/consultations
Content-Type: application/json
Authorization: Bearer <token>

{
  "title": "근로계약 해지 관련 상담",
  "category": "labor",
  "initial_query": "정규직 근로자를 경영상 이유로 해고할 때 필요한 절차와 요건이 무엇인가요?",
  "complexity": "complex",
  "memory_context": {
    "session_memory": "이전 대화: 사용자가 50인 미만 중소기업 운영 중임을 언급함",
    "short_term_memory": "최근 7일: 관련 상담 없음",
    "long_term_memory": "클라이언트 이력: 2022년 노동법 관련 상담 진행, 당시 취업규칙 변경 자문 제공"
  }
}
```

### 후속 질문 추가 (메모리 업데이트 포함)

```http
POST /api/v1/consultations/{consultation_id}/turns
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "해고예고수당은 얼마나 지급해야 하나요?",
  "complexity": "medium",
  "memory_context": {
    "session_memory": "이전 대화: 경영상 해고 절차 논의, 정리해고 4요건 설명됨",
    "short_term_memory": "최근 7일: 관련 상담 없음",
    "long_term_memory": "클라이언트 이력: 50인 미만 중소기업, 2022년 취업규칙 변경 자문"
  }
}
```

---

## TaskComplexity 설정

작업 복잡도에 따라 LLM 파라미터가 자동 최적화됩니다.

| 복잡도 | 용도 | GPT-5.1 | Claude Opus | Gemini 3 Pro |
|--------|------|---------|-------------|--------------|
| `simple` | 단순 분류, 평가 | reasoning_effort: none | effort: low | thinking_level: high |
| `medium` | 일반 법률 분석 | reasoning_effort: medium | effort: medium | thinking_level: high |
| `complex` | 의장 합성, 윤리 분석 | reasoning_effort: high | effort: high | thinking_level: high |

### Python 코드 예시

```python
from app.services.llm.factory import LLMClientFactory
from app.services.llm.legal_prompts_v4_1 import TaskComplexity

# 복잡한 분석용 의장 생성
chairman = LLMClientFactory.create_chairman(
    thinking_budget=32000,
    complexity=TaskComplexity.COMPLEX,
)

# 일반 분석용 위원회 생성
council_members = LLMClientFactory.create_all_council_members(
    complexity=TaskComplexity.MEDIUM,
)
```

---

## 프롬프트 함수 사용법

### 법률 전문가 프롬프트 획득

```python
from app.services.llm.legal_prompts_v4_1 import (
    get_legal_expert_prompt_kr,
    LLMModel,
)

prompt = get_legal_expert_prompt_kr(
    model=LLMModel.GPT_51.value,
    session_memory="현재 세션 대화 내용...",
    short_term_memory="최근 7일 상담 이력...",
    long_term_memory="전체 클라이언트 이력...",
    rag_results="검색된 법령 및 판례...",
)
```

### 의장 프롬프트 획득

```python
from app.services.llm.legal_prompts_v4_1 import get_chairman_prompt_kr

prompt = get_chairman_prompt_kr(
    model=LLMModel.CLAUDE_OPUS_45.value,
    session_memory="...",
    short_term_memory="...",
    long_term_memory="...",
    rag_results="...",
)
```

### Stage 2 평가 프롬프트 생성

```python
from app.services.llm.legal_prompts_v4_1 import get_stage2_review_prompt_kr

prompt = get_stage2_review_prompt_kr(
    model=LLMModel.CLAUDE_SONNET_45.value,
    original_question="사용자의 법률 질문",
    anonymized_opinions="의견 A: ... 의견 B: ...",
    rag_results="검증용 RAG 결과",
)
```

---

## 4-Memory 시스템

### 메모리 유형

| 메모리 | 설명 | 용도 |
|--------|------|------|
| Session Memory | 현재 대화 세션의 맥락 | 대화 흐름 유지 |
| Short-term Memory | 최근 7일 상담 이력 | 연속성 있는 사안 처리 |
| Long-term Memory | 전체 클라이언트 이력 | 일관된 자문, 과거 참조 |
| RAG Results | 법률 데이터베이스 검색 결과 | 권위 있는 법적 근거 |

### 프롬프트 내 Placeholder

```
{{SESSION_MEMORY}} - 세션 기억 주입 위치
{{SHORT_TERM_MEMORY}} - 단기 기억 주입 위치
{{LONG_TERM_MEMORY}} - 장기 기억 주입 위치
{{LEGAL_RAG_RESULTS}} - RAG 결과 주입 위치
```

---

## 모델별 파라미터

### GPT-5.1

```python
{
    "reasoning_effort": "none" | "low" | "medium" | "high",
    "temperature": 0.7,
    "max_tokens": 4000-12000,  # reasoning_effort에 따라 조정
}
```

### Claude Opus 4.5

```python
{
    "effort": "low" | "medium" | "high",
    "temperature": 0.7,
    "max_tokens": 4000-16000,  # effort에 따라 조정
}
```

### Gemini 3 Pro

```python
{
    "thinking_level": "high",  # 항상 high 권장
    "temperature": 1.0,        # 절대 변경 금지!
    "media_resolution": "medium",
    "max_tokens": 8000,
}
```

### Grok 4

```python
{
    "temperature": 0.7,
    "max_tokens": 8000,
}
```

---

## Stage 2 평가 기준

5가지 기준으로 각 10점, 총 50점 만점:

1. **정확성 (Accuracy)**: RAG 인용 정확성, 법령/판례 적용 정확성
2. **완전성 (Completeness)**: 모든 법적 쟁점 다룸, 위험 요소 포괄적 식별
3. **메모리 통합 (Memory Integration)**: 세션/단기/장기 기억 활용도
4. **유용성 (Usefulness)**: 실행 가능한 권고, 실무적 적용 가능성
5. **명확성 (Clarity)**: 논리적 구조, 법률 용어 설명, 가독성

---

## IRAC 형식 분석

모든 법률 전문가 프롬프트는 IRAC 형식을 따릅니다:

```markdown
#### 쟁점 1: [쟁점명]
- **Issue**: 구체적인 법적 질문
- **Rule**: 적용 법령/판례 [RAG-ID: xxx]
- **Application**: 사실관계에 법규 적용
- **Conclusion**: 명확한 법적 결론
```

---

## 최적 모델 선택

```python
from app.services.llm.legal_prompts_v4_1 import select_optimal_model

# 의장은 항상 Opus
chairman_model = select_optimal_model("chairman")
# → "anthropic/claude-opus-4-5-20251101"

# Stage 2 품질 우선
stage2_model = select_optimal_model("stage2", cost_priority=False)
# → "anthropic/claude-sonnet-4-5-20250929"

# Stage 2 비용 우선
stage2_model = select_optimal_model("stage2", cost_priority=True)
# → "x-ai/grok-4-1-fast-non-reasoning"

# Stage 1 복잡한 작업
stage1_model = select_optimal_model("stage1", TaskComplexity.COMPLEX)
# → "anthropic/claude-opus-4-5-20251101"
```

---

## 테스트

```bash
# v4.1 통합 테스트 실행
cd backend
pytest tests/test_llm_v41.py -v
```

---

## 주의사항

1. **Gemini temperature**: 반드시 1.0 유지 (변경 시 thinking 기능 비활성화)
2. **RAG 인용 필수**: 모든 법적 진술에 RAG 인용 포함
3. **일관성 확인**: 장기 기억의 과거 자문과 일관성 필수 확인
4. **한국어 응답**: 모든 응답은 한국어로, 법률 용어는 영문 병기
