"""
RAG Enhanced Orchestrator

LLM 오케스트레이터에서 RAG 서비스 통합.
Phase 6: backend-rag 구현.

Stage 1 LLM들이 법률 상담 시 RAG 컨텍스트를 활용합니다.

4개 메모리 시스템 통합:
1. 세션 메모리 (현재 대화)
2. 단기 메모리 (최근 7일)
3. 장기 메모리 (전체 이력)
4. RAG 컨텍스트 (법률 문서)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.services.llm.rag_service import RAGService, get_rag_service
from app.services.llm.council import (
    CouncilOrchestrator,
    CouncilResult,
    CouncilOpinion,
)
from app.services.llm.base import LegalContext
from app.schemas.legal_document import RAGSearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGEnhancedResult:
    """RAG가 통합된 상담 결과."""

    council_result: CouncilResult
    rag_context: str
    citations: list[dict] = field(default_factory=list)
    rag_search_time_ms: int = 0
    total_time_ms: int = 0


class RAGEnhancedOrchestrator:
    """
    RAG가 통합된 LLM 오케스트레이터.

    Google File Search 기반 RAG 컨텍스트를 각 LLM에 제공하여
    정확한 법률 인용과 근거 기반 답변을 생성합니다.

    Features:
    - 4개 메모리 시스템 통합 (세션/단기/장기/RAG)
    - Stage 1 LLM들에게 RAG 컨텍스트 제공
    - 자동 인용(Citation) 추출
    - 카테고리 기반 문서 필터링
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        council_orchestrator: Optional[CouncilOrchestrator] = None,
        rag_max_tokens: int = 8000,
    ):
        """
        RAG Enhanced Orchestrator 초기화.

        Args:
            rag_service: RAG 서비스 인스턴스 (DI용)
            council_orchestrator: Council 오케스트레이터 인스턴스 (DI용)
            rag_max_tokens: RAG 컨텍스트 최대 토큰 수
        """
        self.rag = rag_service or get_rag_service()
        self.council = council_orchestrator or CouncilOrchestrator()
        self.rag_max_tokens = rag_max_tokens
        self._initialized = False

    async def initialize(self) -> None:
        """서비스 초기화."""
        if self._initialized:
            return
        await self.rag.initialize()
        self._initialized = True
        logger.info("RAG Enhanced Orchestrator initialized")

    async def process_consultation(
        self,
        user_query: str,
        category: str = "general",
        session_memory: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        conversation_history: Optional[list[dict]] = None,
        document_texts: Optional[list[str]] = None,
        user_tier: str = "basic",
    ) -> RAGEnhancedResult:
        """
        RAG 컨텍스트를 포함한 법률 상담 처리.

        4개 메모리 시스템을 통합하여 LLM에 제공합니다:
        1. 세션 메모리: 현재 대화 컨텍스트
        2. 단기 메모리: 최근 7일 상담 이력
        3. 장기 메모리: 전체 고객 이력
        4. RAG 컨텍스트: 검색된 법률 문서

        Args:
            user_query: 사용자 질문
            category: 법률 분야 (contract, labor, criminal 등)
            session_memory: 현재 세션 메모리
            short_term_memory: 단기 메모리 (7일)
            long_term_memory: 장기 메모리
            conversation_history: 이전 대화 이력
            document_texts: 첨부 문서 텍스트
            user_tier: 사용자 티어

        Returns:
            RAGEnhancedResult with 상담 결과 및 인용
        """
        total_start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # 1. RAG 검색 수행
        rag_start_time = time.time()
        rag_context = await self.rag.get_context_for_llm(
            query=user_query,
            category=category,
            max_tokens=self.rag_max_tokens,
        )
        rag_search_time_ms = int((time.time() - rag_start_time) * 1000)

        logger.debug(f"RAG search completed in {rag_search_time_ms}ms")

        # 2. 4개 메모리 시스템 통합
        integrated_context = self._build_integrated_context(
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_context=rag_context,
        )

        # 3. RAG 컨텍스트를 포함한 LegalContext 생성
        rag_documents = await self.rag.get_context_for_council(
            query=user_query,
            category=category,
            max_documents=10,
        )

        legal_context = LegalContext(
            query=user_query,
            category=category,
            conversation_history=conversation_history or [],
            rag_context=rag_documents,
            document_texts=document_texts or [],
            user_tier=user_tier,
        )

        # 4. Council 오케스트레이터 호출
        council_result = await self.council.process_consultation(
            context=legal_context,
            session_memory=session_memory,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            rag_results=rag_context,
        )

        # 5. 검색된 인용 추가
        citation_results = await self.rag.search(
            query=user_query,
            category=category,
            top_k=5,
        )

        citations = [
            {
                "title": r.title,
                "content": r.snippet if hasattr(r, 'snippet') else r.content[:500],
                "source": r.source,
                "doc_type": r.doc_type.value if hasattr(r.doc_type, 'value') else str(r.doc_type),
                "category": r.category.value if hasattr(r.category, 'value') else str(r.category),
                "relevance_score": r.relevance_score,
                "case_number": r.case_number,
                "law_number": r.law_number,
            }
            for r in citation_results.results
        ]

        total_time_ms = int((time.time() - total_start_time) * 1000)

        return RAGEnhancedResult(
            council_result=council_result,
            rag_context=rag_context,
            citations=citations,
            rag_search_time_ms=rag_search_time_ms,
            total_time_ms=total_time_ms,
        )

    def _build_integrated_context(
        self,
        session_memory: str,
        short_term_memory: str,
        long_term_memory: str,
        rag_context: str,
    ) -> str:
        """4개 메모리 시스템을 통합된 컨텍스트로 구성."""
        parts = []

        if session_memory:
            parts.append(f"## 세션 메모리 (현재 대화)\n{session_memory}")

        if short_term_memory:
            parts.append(f"## 단기 메모리 (최근 7일)\n{short_term_memory}")

        if long_term_memory:
            parts.append(f"## 장기 메모리 (전체 이력)\n{long_term_memory}")

        if rag_context:
            parts.append(f"{rag_context}")

        return "\n\n".join(parts)

    async def process_with_grounding(
        self,
        user_query: str,
        category: str = "general",
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Grounded Generation을 사용한 빠른 답변 생성.

        Council 3-Stage 파이프라인 없이 RAG 기반으로 직접 답변을 생성합니다.
        간단한 질문이나 빠른 응답이 필요한 경우 사용합니다.

        Args:
            user_query: 사용자 질문
            category: 법률 분야
            system_prompt: 커스텀 시스템 프롬프트

        Returns:
            Grounded generation 결과 (answer + citations)
        """
        if not self._initialized:
            await self.initialize()

        result = await self.rag.search_with_answer(
            query=user_query,
            category=category,
            system_prompt=system_prompt,
        )

        return {
            "answer": result.generated_answer,
            "citations": [
                {
                    "title": chunk.title,
                    "content": chunk.content,
                    "source": chunk.source,
                    "relevance_score": chunk.relevance_score,
                }
                for chunk in result.grounding_chunks
            ],
            "citation_texts": result.citations,
            "model": result.model_used,
            "processing_time_ms": result.processing_time_ms,
        }

    async def search_similar_cases(
        self,
        case_description: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        유사 판례 검색.

        Args:
            case_description: 사건 설명
            top_k: 최대 결과 수

        Returns:
            유사 판례 목록
        """
        if not self._initialized:
            await self.initialize()

        results = await self.rag.get_similar_cases(
            case_description=case_description,
            top_k=top_k,
        )

        return [
            {
                "title": r.title,
                "content": r.content,
                "source": r.source,
                "case_number": r.case_number,
                "court": r.court,
                "decision_date": str(r.decision_date) if r.decision_date else None,
                "relevance_score": r.relevance_score,
            }
            for r in results
        ]

    async def search_relevant_laws(
        self,
        topic: str,
        category: str = "general",
        top_k: int = 5,
    ) -> list[dict]:
        """
        관련 법률 조문 검색.

        Args:
            topic: 검색 주제
            category: 법률 분야
            top_k: 최대 결과 수

        Returns:
            관련 법률 목록
        """
        if not self._initialized:
            await self.initialize()

        results = await self.rag.get_relevant_laws(
            topic=topic,
            category=category,
            top_k=top_k,
        )

        return [
            {
                "title": r.title,
                "content": r.content,
                "source": r.source,
                "law_number": r.law_number,
                "article_number": r.article_number,
                "relevance_score": r.relevance_score,
            }
            for r in results
        ]


# Singleton instance
_rag_orchestrator: Optional[RAGEnhancedOrchestrator] = None


def get_rag_orchestrator() -> RAGEnhancedOrchestrator:
    """RAGEnhancedOrchestrator 싱글톤 인스턴스 반환."""
    global _rag_orchestrator
    if _rag_orchestrator is None:
        _rag_orchestrator = RAGEnhancedOrchestrator()
    return _rag_orchestrator
