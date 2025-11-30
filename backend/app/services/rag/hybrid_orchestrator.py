"""
Hybrid RAG Orchestrator

법률 문서 RAG와 경험 RAG를 통합하는 오케스트레이터.

Features:
- 법률 문서 검색 (법령, 판례, 헌재결정 등)
- 상담 경험 검색 (유사 케이스)
- 가중치 기반 결과 통합
- 컨텍스트 최적화
- 병렬 검색 최적화
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.vector_store import VectorStore
from app.services.rag.experience_rag import ExperienceRAG, ExperienceSearchResult
from app.services.memory.session_cache import get_consultation_cache

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """통합 검색 결과."""
    # 법률 문서
    legal_documents: list[dict] = field(default_factory=list)

    # 유사 경험
    similar_experiences: list[ExperienceSearchResult] = field(default_factory=list)

    # 통합 컨텍스트
    combined_context: str = ""

    # 메타데이터
    total_sources: int = 0
    cache_hit: bool = False


@dataclass
class RAGConfig:
    """RAG 설정."""
    # 법률 문서 검색
    legal_doc_top_k: int = 5
    legal_doc_min_similarity: float = 0.5
    legal_doc_weight: float = 0.7  # 법률 문서 가중치

    # 경험 검색
    experience_top_k: int = 3
    experience_min_similarity: float = 0.6
    experience_weight: float = 0.3  # 경험 가중치

    # 캐싱
    cache_ttl: int = 300  # 5분

    # 컨텍스트 제한
    max_context_tokens: int = 4000


class HybridRAGOrchestrator:
    """
    하이브리드 RAG 오케스트레이터.

    법률 문서와 상담 경험을 통합하여 최적의 컨텍스트를 제공합니다.
    """

    def __init__(
        self,
        db: AsyncSession,
        config: Optional[RAGConfig] = None,
    ):
        """
        초기화.

        Args:
            db: SQLAlchemy AsyncSession
            config: RAG 설정
        """
        self.db = db
        self.config = config or RAGConfig()
        self.vector_store = VectorStore(db)
        self.experience_rag = ExperienceRAG(db)
        self.cache = get_consultation_cache()

    async def search(
        self,
        query: str,
        consultation_id: str,
        category: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_experiences: bool = True,
        include_legal_docs: bool = True,
    ) -> HybridSearchResult:
        """
        통합 검색 수행.

        Args:
            query: 검색 쿼리
            consultation_id: 상담 ID (캐시 키)
            category: 법률 분야 필터
            agent_id: 에이전트 필터 (경험 검색용)
            include_experiences: 경험 검색 포함
            include_legal_docs: 법률 문서 검색 포함

        Returns:
            HybridSearchResult
        """
        # 캐시 확인
        cache_key = self._get_cache_key(query, category, agent_id)
        cached = self.cache.get_cached_rag_result(consultation_id, cache_key)
        if cached:
            return HybridSearchResult(
                legal_documents=cached.get("legal_docs", []),
                similar_experiences=[],  # 경험은 직렬화 복잡 → 캐시 제외
                combined_context=cached.get("context", ""),
                total_sources=cached.get("total", 0),
                cache_hit=True,
            )

        result = HybridSearchResult()

        # 병렬 검색 최적화: 두 검색을 동시에 실행
        tasks = []
        task_names = []

        if include_legal_docs:
            tasks.append(self._search_legal_documents(query, category))
            task_names.append("legal_docs")

        if include_experiences:
            tasks.append(self._search_experiences(query, category, agent_id))
            task_names.append("experiences")

        # 병렬 실행
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, res in zip(task_names, results):
                if isinstance(res, Exception):
                    logger.warning(f"Search failed for {name}: {res}")
                    continue

                if name == "legal_docs":
                    result.legal_documents = res
                elif name == "experiences":
                    result.similar_experiences = res

        # 컨텍스트 통합
        result.combined_context = self._build_combined_context(
            legal_docs=result.legal_documents,
            experiences=result.similar_experiences,
        )

        result.total_sources = len(result.legal_documents) + len(result.similar_experiences)

        # 캐시 저장
        self.cache.cache_rag_result(
            consultation_id,
            cache_key,
            {
                "legal_docs": result.legal_documents,
                "context": result.combined_context,
                "total": result.total_sources,
            },
            ttl=self.config.cache_ttl,
        )

        logger.info(f"Hybrid search: {len(result.legal_documents)} docs, {len(result.similar_experiences)} experiences")
        return result

    async def _search_legal_documents(
        self,
        query: str,
        category: Optional[str],
    ) -> list[dict]:
        """법률 문서 검색."""
        try:
            results = await self.vector_store.hybrid_search(
                query=query,
                category=category,
                top_k=self.config.legal_doc_top_k,
                vector_weight=0.7,  # 벡터 70%, 키워드 30%
            )
            return results
        except Exception as e:
            logger.error(f"Legal document search failed: {e}")
            return []

    async def _search_experiences(
        self,
        query: str,
        category: Optional[str],
        agent_id: Optional[str],
    ) -> list[ExperienceSearchResult]:
        """경험 검색."""
        try:
            results = await self.experience_rag.search_similar_experiences(
                query=query,
                category=category,
                agent_id=agent_id,
                success_only=True,  # 성공 케이스만
                top_k=self.config.experience_top_k,
                min_similarity=self.config.experience_min_similarity,
            )
            return results
        except Exception as e:
            logger.error(f"Experience search failed: {e}")
            return []

    def _build_combined_context(
        self,
        legal_docs: list[dict],
        experiences: list[ExperienceSearchResult],
    ) -> str:
        """통합 컨텍스트 구성."""
        parts = []
        current_tokens = 0

        # 1. 법률 문서 컨텍스트
        if legal_docs:
            parts.append("[관련 법률 자료]")

            for doc in legal_docs:
                doc_text = self._format_legal_doc(doc)
                doc_tokens = len(doc_text.split())

                if current_tokens + doc_tokens > self.config.max_context_tokens * self.config.legal_doc_weight:
                    break

                parts.append(doc_text)
                current_tokens += doc_tokens

        # 2. 유사 경험 컨텍스트
        if experiences:
            parts.append("\n[유사 상담 사례]")

            remaining_tokens = self.config.max_context_tokens - current_tokens
            for exp_result in experiences:
                exp_text = self._format_experience(exp_result)
                exp_tokens = len(exp_text.split())

                if current_tokens + exp_tokens > self.config.max_context_tokens:
                    break

                parts.append(exp_text)
                current_tokens += exp_tokens

        return "\n".join(parts)

    def _format_legal_doc(self, doc: dict) -> str:
        """법률 문서 포맷팅."""
        doc_type = doc.get("doc_type", "")
        title = doc.get("title", "")
        content = doc.get("content", "")
        score = doc.get("relevance_score", 0)

        # 타입별 포맷
        type_labels = {
            "law": "법령",
            "precedent": "판례",
            "constitutional": "헌재결정",
        }
        type_label = type_labels.get(doc_type, "문서")

        # 내용 요약 (너무 길면 자름)
        if len(content) > 500:
            content = content[:500] + "..."

        return f"""
[{type_label}] {title} (관련도: {score:.2f})
{content}
"""

    def _format_experience(self, exp_result: ExperienceSearchResult) -> str:
        """경험 포맷팅."""
        exp = exp_result.experience
        similarity = exp_result.similarity

        # 응답 요약 (너무 길면 자름)
        summary = exp.response_summary
        if len(summary) > 300:
            summary = summary[:300] + "..."

        feedback_text = ""
        if exp.user_feedback:
            feedback_text = f" (평점: {exp.user_feedback}/5)"

        return f"""
[{exp.category}] 유사도 {similarity*100:.0f}%{feedback_text}
질문: {exp.query[:100]}...
핵심 답변: {summary}
"""

    def _get_cache_key(
        self,
        query: str,
        category: Optional[str],
        agent_id: Optional[str],
    ) -> str:
        """캐시 키 생성."""
        key_parts = [query, category or "", agent_id or ""]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    # ========================================
    # 고급 검색 메서드
    # ========================================

    async def search_for_cascade_drafter(
        self,
        query: str,
        consultation_id: str,
        agent_id: str,
        category: Optional[str] = None,
    ) -> str:
        """
        CascadeFlow 드래프터를 위한 최적화된 검색.

        드래프터는 빠른 응답이 필요하므로 경량 컨텍스트를 제공합니다.
        """
        # 드래프터용 경량 설정
        original_config = self.config
        self.config = RAGConfig(
            legal_doc_top_k=3,
            experience_top_k=2,
            max_context_tokens=2000,  # 더 작은 컨텍스트
        )

        try:
            result = await self.search(
                query=query,
                consultation_id=consultation_id,
                category=category,
                agent_id=agent_id,
            )
            return result.combined_context
        finally:
            self.config = original_config

    async def search_for_cascade_verifier(
        self,
        query: str,
        consultation_id: str,
        agent_id: str,
        category: Optional[str] = None,
        drafter_response: Optional[str] = None,
    ) -> str:
        """
        CascadeFlow 검증자를 위한 포괄적 검색.

        검증자는 드래프터 응답을 포함한 더 많은 컨텍스트를 받습니다.
        """
        result = await self.search(
            query=query,
            consultation_id=consultation_id,
            category=category,
            agent_id=agent_id,
        )

        context_parts = [result.combined_context]

        # 드래프터 응답 추가 (있는 경우)
        if drafter_response:
            context_parts.append(f"\n[이전 분석 (검토 필요)]\n{drafter_response[:1000]}")

        return "\n".join(context_parts)

    async def get_agent_specific_context(
        self,
        query: str,
        agent_id: str,
        category: Optional[str] = None,
    ) -> str:
        """
        에이전트 특화 컨텍스트.

        해당 에이전트의 성공 패턴을 포함한 컨텍스트를 제공합니다.
        """
        # 에이전트의 성공 패턴 조회
        success_patterns = await self.experience_rag.get_agent_success_patterns(
            agent_id=agent_id,
            category=category,
            limit=3,
        )

        if not success_patterns:
            return ""

        parts = [f"[{agent_id} 성공 패턴]"]
        for pattern in success_patterns:
            parts.append(f"- {pattern.category}: {pattern.response_summary[:200]}...")

        return "\n".join(parts)

    # ========================================
    # 통계 및 유틸리티
    # ========================================

    async def get_statistics(self) -> dict:
        """통합 통계."""
        vector_stats = await self.vector_store.get_stats()
        experience_stats = await self.experience_rag.get_statistics()

        return {
            "legal_documents": vector_stats,
            "experiences": experience_stats,
            "config": {
                "legal_doc_top_k": self.config.legal_doc_top_k,
                "experience_top_k": self.config.experience_top_k,
                "max_context_tokens": self.config.max_context_tokens,
            },
        }

    async def index_consultation_experience(
        self,
        consultation_id: str,
        user_id: str,
        query: str,
        response_summary: str,
        category: str,
        keywords: list[str],
        agent_id: str,
        model_id: str,
        success: bool,
        quality_score: float,
        user_feedback: Optional[int] = None,
    ) -> str:
        """
        상담 완료 후 경험 인덱싱.

        학습 시스템과 연동하여 상담 경험을 저장합니다.
        """
        return await self.experience_rag.index_experience(
            consultation_id=consultation_id,
            user_id=user_id,
            query=query,
            response_summary=response_summary,
            category=category,
            keywords=keywords,
            agent_id=agent_id,
            model_id=model_id,
            success=success,
            quality_score=quality_score,
            user_feedback=user_feedback,
        )
