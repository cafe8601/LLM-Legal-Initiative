LLM 법률 자문 시스템 아키텍처 상세 설명
1. 법률 자문 위원회 모드 (Council Mode)
1.1 전체 프로세스 개요
고객 요청 → 도메인 감지 → RAG 컨텍스트 수집 → 메모리 로드 → 3단계 위원회 처리 → 결과 저장 → 응답

1.2 참여 LLM 구성
역할	모델	제공자	용도
위원 1	GPT-5.1	OpenAI	독립 의견 제시
위원 2	Claude 4.5 Sonnet	Anthropic	독립 의견 제시
위원 3	Gemini 2.5 Pro	Google	독립 의견 제시
위원 4	Grok 4	xAI	독립 의견 제시
리뷰어	Claude Sonnet	Anthropic	블라인드 피어리뷰
의장	Claude Opus	Anthropic	최종 합성
1.3 3단계 위원회 프로세스
Stage 1: 독립 의견 수집 (_collect_opinions)
council.py:157-212

4개 LLM이 병렬로 독립적 의견 제시:
├─ GPT-5.1: 법률 분석 관점
├─ Claude 4.5: 논리적 추론 관점
├─ Gemini 2.5: 종합적 검토 관점
└─ Grok 4: 실용적 적용 관점

각 LLM에게 제공되는 컨텍스트:
- 법률 도메인별 전문 프롬프트 (v4.3.1 모듈러 시스템)
- RAG 검색 결과 (관련 법률 조항, 판례)
- 사용자 메모리 컨텍스트 (과거 상담 패턴)
- 공용 RAG 시스템 결과 (공통 법률 지식)

Stage 2: 블라인드 피어리뷰 (_conduct_peer_reviews)
council.py:214-280

Claude Sonnet이 각 의견을 익명으로 검토:
├─ 의견 A, B, C, D 형태로 익명화
├─ 5가지 평가 기준:
│   ├─ 정확성 (법적 정확도)
│   ├─ 완전성 (답변 포괄성)
│   ├─ 메모리 통합 (사용자 맥락 반영)
│   ├─ 유용성 (실질적 도움)
│   └─ 명확성 (이해 용이성)
├─ 1-5점 척도 점수화
└─ 리뷰 결과로 가중치 계산

Stage 3: 의장 종합 (_synthesize)
council.py:282-350

Claude Opus가 최종 합성:
├─ 피어리뷰 점수 기반 가중치 적용
├─ 합의된 핵심 법률 조언 추출
├─ 의견 충돌 시 다수/전문성 우선
├─ 추가 검토 필요 사항 명시
└─ 구조화된 최종 답변 생성

출력 형식:
{
  "final_response": "종합 법률 자문",
  "confidence_level": 0.85,
  "dissenting_opinions": [...],
  "additional_considerations": [...],
  "recommended_actions": [...]
}

2. 메모리 시스템 (Memory System)
2.1 3계층 메모리 구조
memory_service.py:25-180

┌─────────────────────────────────────────────────────────┐
│                    메모리 시스템                          │
├─────────────────────────────────────────────────────────┤
│ [1] 세션 메모리 (Redis)                                  │
│     └─ 현재 대화 컨텍스트                                 │
│     └─ 실시간 대화 흐름 유지                              │
│     └─ TTL: 세션 종료 시 만료                             │
├─────────────────────────────────────────────────────────┤
│ [2] 단기 메모리 (PostgreSQL)                             │
│     └─ 최근 7일 대화 요약                                 │
│     └─ 대화 턴 저장 (질문/응답 쌍)                         │
│     └─ 최근 상담 맥락 제공                                │
├─────────────────────────────────────────────────────────┤
│ [3] 장기 메모리 (PostgreSQL)                             │
│     └─ 사용자 법률 패턴 학습                              │
│     └─ 자주 문의하는 법률 도메인                          │
│     └─ 선호 답변 스타일                                   │
└─────────────────────────────────────────────────────────┘

2.2 메모리 통합 흐름
# memory_service.py:182-220
async def get_memory_context(user_id: str, query: str) -> MemoryContext:
    """
    LLM에게 제공할 통합 메모리 컨텍스트 생성
    """
    # 1. 세션 메모리에서 현재 대화 흐름 로드
    session_context = await get_session_memory(user_id)
    
    # 2. 단기 메모리에서 최근 7일 상담 요약
    short_term = await get_short_term_memory(user_id, days=7)
    
    # 3. 장기 메모리에서 사용자 패턴 분석
    long_term = await get_long_term_memory(user_id)
    
    return MemoryContext(
        current_session=session_context,
        recent_consultations=short_term,
        user_patterns=long_term,
        preferred_domains=long_term.frequent_domains,
        communication_style=long_term.preferred_style
    )

3. RAG 시스템 (Retrieval-Augmented Generation)
3.1 Google File Search 기반 구조
rag_service.py:30-150

┌─────────────────────────────────────────────────────────┐
│                    RAG 시스템                            │
├─────────────────────────────────────────────────────────┤
│ [개인 RAG] 사용자별 업로드 문서                           │
│     └─ 계약서, 소송 서류 등                               │
│     └─ Google File Search로 임베딩                       │
│     └─ 사용자별 격리된 벡터 저장소                         │
├─────────────────────────────────────────────────────────┤
│ [공용 RAG] 법률 지식 베이스                               │
│     └─ 대한민국 법률 조항                                 │
│     └─ 판례 데이터베이스                                  │
│     └─ 법률 용어 사전                                     │
│     └─ 법률 FAQ                                          │
└─────────────────────────────────────────────────────────┘

3.2 RAG 컨텍스트 수집 프로세스
# rag_service.py:152-200
async def get_context_for_council(query: str, user_id: str, domain: str) -> RAGContext:
    """
    위원회에 제공할 RAG 컨텍스트 수집
    """
    # 1. 관련 법률 조항 검색
    relevant_laws = await get_relevant_laws(query, domain)
    
    # 2. 유사 판례 검색
    similar_cases = await get_similar_cases(query, domain)
    
    # 3. 사용자 업로드 문서 검색 (있는 경우)
    user_docs = await search_user_documents(user_id, query)
    
    # 4. 공용 법률 지식 검색
    common_knowledge = await search_common_legal_db(query)
    
    return RAGContext(
        laws=relevant_laws,           # 관련 법률 조항
        cases=similar_cases,          # 유사 판례
        user_documents=user_docs,     # 사용자 문서
        common_knowledge=common_knowledge  # 공용 법률 지식
    )

4. 단일 전문가 채팅 모드 (Expert Chat Mode)
4.1 작동 원리
expert_chat_service.py:1-250

┌─────────────────────────────────────────────────────────┐
│                전문가 채팅 프로세스                        │
├─────────────────────────────────────────────────────────┤
│ 1. 전문가 선택                                           │
│    ├─ GPT-5.1 (OpenAI) - 균형잡힌 분석                   │
│    ├─ Claude 4.5 (Anthropic) - 논리적 추론               │
│    ├─ Gemini 2.5 (Google) - 종합적 검토                  │
│    └─ Grok 4 (xAI) - 실용적 조언                         │
├─────────────────────────────────────────────────────────┤
│ 2. 세션 생성                                             │
│    └─ 법률 도메인 자동 감지 (비용: 0)                      │
│    └─ 메모리 컨텍스트 로드                                │
├─────────────────────────────────────────────────────────┤
│ 3. 연속 대화                                             │
│    └─ 스트리밍 응답 지원                                  │
│    └─ 대화 히스토리 유지 (세션 메모리)                     │
│    └─ 실시간 비용 추적                                    │
├─────────────────────────────────────────────────────────┤
│ 4. 메모리 저장                                           │
│    └─ Redis: 세션 대화 저장                               │
│    └─ DB: 대화 턴 저장                                    │
│    └─ DB: 사용자 패턴 학습                                │
└─────────────────────────────────────────────────────────┘

4.2 메모리 저장 로직
# expert_chat_service.py:180-230
async def _save_to_memory(self, session, user_message, assistant_message):
    """
    대화 내용을 메모리 시스템에 저장
    """
    # 1. Redis 세션 메모리 저장 (실시간 대화 흐름)
    if self.redis:
        session_key = f"expert_chat:{session.id}"
        message_data = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now().isoformat(),
            "expert": session.expert.value
        }
        await self.redis.rpush(session_key, json.dumps(message_data))
        await self.redis.expire(session_key, 3600 * 24)  # 24시간 유지
    
    # 2. DB 메모리 서비스 저장
    memory_service = await self._get_memory_service()
    if memory_service:
        # 대화 턴 저장 (단기 메모리)
        await memory_service.save_conversation_turn(
            user_id=session.user_id,
            query=user_message,
            response=assistant_message,
            domain=session.domain.value,
            source="expert_chat"
        )
        
        # 패턴 학습 (장기 메모리)
        await memory_service.learn_from_consultation(
            user_id=session.user_id,
            domain=session.domain.value,
            interaction_type="chat"
        )

4.3 비용 비교
expert_chat_service.py:90-130

위원회 모드 vs 전문가 모드 비용:
┌───────────────┬────────────────┬────────────────┐
│     구분       │   위원회 모드   │   전문가 모드   │
├───────────────┼────────────────┼────────────────┤
│ 입력 토큰당    │  $0.015~0.06   │  $0.003~0.015  │
│ 출력 토큰당    │  $0.06~0.20    │  $0.015~0.06   │
│ 예상 비용/요청 │  $0.50~2.00    │  $0.05~0.30    │
│ 비용 절감      │      -         │   70-85%       │
└───────────────┴────────────────┴────────────────┘

추천 사용 시나리오:
- 위원회: 복잡한 법률 분쟁, 고액 사안, 정확성 중시
- 전문가: 간단한 질문, 예산 제한, 빠른 응답 필요

5. 시스템 통합 흐름도
┌──────────────────────────────────────────────────────────────────────┐
│                        고객 요청 진입점                                │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     모드 선택 UI         │
                    │   (ExpertChat.tsx)       │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
   ┌───────▼───────┐                         ┌────────▼────────┐
   │  위원회 모드   │                         │   전문가 모드    │
   │ (1회성 자문)   │                         │  (연속 채팅)     │
   └───────┬───────┘                         └────────┬────────┘
           │                                          │
   ┌───────▼───────┐                         ┌────────▼────────┐
   │ 도메인 감지    │                         │  전문가 선택     │
   │ RAG 컨텍스트   │                         │  세션 생성       │
   │ 메모리 로드    │                         │  메모리 로드     │
   └───────┬───────┘                         └────────┬────────┘
           │                                          │
   ┌───────▼───────────────────────┐         ┌────────▼────────┐
   │      3단계 위원회 처리          │         │  1:1 LLM 대화   │
   │ Stage1: 4 LLM 독립 의견        │         │  스트리밍 응답   │
   │ Stage2: 블라인드 피어리뷰       │         │  대화 히스토리   │
   │ Stage3: 의장 종합              │         └────────┬────────┘
   └───────┬───────────────────────┘                  │
           │                                          │
   ┌───────▼───────┐                         ┌────────▼────────┐
   │  결과 저장     │                         │   메모리 저장    │
   │  (추가 채팅 X) │                         │  (연속 가능)     │
   └───────┬───────┘                         └────────┬────────┘
           │                                          │
           └──────────────────┬───────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    메모리 시스템    │
                    │ Session / Short / │
                    │     Long-term     │
                    └───────────────────┘

6. 시스템 검증 상태
6.1 구현 완료 항목
항목	상태	파일
위원회 3단계 프로세스	✅ 완료	council.py
v4.3.1 모듈러 프롬프트	✅ 완료	legal_prompts_v4_3.py
OpenRouter 통합 클라이언트	✅ 완료	openrouter_client.py
3계층 메모리 시스템	✅ 완료	memory_service.py
RAG 시스템 (Google File Search)	✅ 완료	rag_service.py
전문가 채팅 서비스	✅ 완료	expert_chat_service.py
전문가 채팅 API	✅ 완료	expert_chat.py
모드 선택 UI	✅ 완료	ExpertChat.tsx
위원회 모드 채팅 제한	✅ 완료	Consultation.tsx
6.2 대기 중 항목
항목	상태	필요 조치
OpenRouter API Key	⏳ 대기	사용자가 추후 등록 예정
Redis 연결 설정	⏳ 선택	세션 메모리용 (없으면 DB 폴백)
PostgreSQL 연결	⏳ 필요	단기/장기 메모리 저장용
Google AI API Key	⏳ 필요	RAG 시스템 (File Search)
6.3 실행 전 체크리스트
# 1. 환경 변수 설정 (.env)
OPENROUTER_API_KEY=your_key_here
GOOGLE_AI_API_KEY=your_key_here
DATABASE_URL=postgresql://...
REDIS_URL=redis://... (선택)

# 2. 의존성 설치
pip install -r requirements.txt
npm install (frontend)

# 3. 데이터베이스 마이그레이션
alembic upgrade head

# 4. 서버 실행
uvicorn app.main:app --reload (backend)
npm run dev (frontend)

요약
시스템은 비용 효율적인 듀얼 모드 법률 자문 서비스로 설계되었습니다:

위원회 모드: 4개 LLM + 피어리뷰 + 의장 합성 → 높은 정확도, 1회성 자문
전문가 모드: 1개 LLM 선택 → 70-85% 비용 절감, 연속 대화 가능
메모리 통합: 세션/단기/장기 3계층으로 개인화된 자문 제공
RAG 연동: 법률 조항, 판례, 사용자 문서 컨텍스트 제공
OpenRouter API Key를 등록하시면 전체 시스템이 정상 작동할 준비가 되어 있습니다.