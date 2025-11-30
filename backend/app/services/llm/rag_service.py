"""
RAG Service

OpenRouter + PostgreSQL pgvector 기반 법률 문서 검색 및 RAG 서비스.
단일 API Key(OpenRouter)로 임베딩 및 LLM 기능 통합.

v4.3.2: Google File Search → pgvector 마이그레이션
- 모든 기능을 OpenRouter API Key 하나로 통합
- 로컬 PostgreSQL + pgvector로 벡터 검색
- OpenRouter 임베딩 API 사용

Features:
- 로컬 pgvector 벡터 검색 (Google File Search 대체)
- OpenRouter 기반 임베딩 생성
- 하이브리드 검색 (벡터 + 키워드)
- Grounded Generation (RAG)
"""

import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.rag.embedding_service import EmbeddingService, get_embedding_service
from app.services.rag.vector_store import VectorStore
from app.schemas.legal_document import (
    LegalDocument,
    LegalDocumentMetadata,
    LegalDocumentType,
    LegalCategory,
    RAGSearchResult,
    RAGSearchResponse,
    GroundedGenerationResponse,
    CorpusStats,
)

logger = logging.getLogger(__name__)


# Category to document type/filter mapping
CATEGORY_FILTERS: dict[str, Optional[list[str]]] = {
    "general": None,
    "contract": ["민사", "계약", "civil"],
    "intellectual_property": ["지식재산권", "특허", "상표", "저작권"],
    "labor": ["노동", "근로", "고용", "labor"],
    "criminal": ["형사", "범죄", "criminal"],
    "administrative": ["행정", "공법", "administrative"],
    "corporate": ["회사", "상법", "기업", "commercial"],
    "family": ["가족", "가사", "상속", "family"],
    "real_estate": ["부동산", "물권", "등기", "real_estate"],
    "tax": ["세법", "조세", "tax"],
    "constitutional": ["헌법", "constitutional"],
    "other": None,
}


class RAGService:
    """
    OpenRouter + pgvector 기반 RAG 서비스.

    단일 API Key(OpenRouter)로 모든 기능 통합:
    - 임베딩: OpenRouter → text-embedding-3-small
    - 벡터 저장: PostgreSQL + pgvector
    - LLM 생성: OpenRouter → 선택한 모델

    Features:
    - 법률 문서 시맨틱 검색
    - 하이브리드 검색 (벡터 + 키워드)
    - Grounded Generation (RAG)
    - 위원회 컨텍스트 제공
    - 유사 판례/관련 법률 검색
    """

    def __init__(
        self,
        db: Optional[AsyncSession] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        RAG 서비스 초기화.

        Args:
            db: SQLAlchemy AsyncSession
            embedding_service: 임베딩 서비스 (DI용)
        """
        self._db = db
        self._embedding_service = embedding_service
        self._vector_store: Optional[VectorStore] = None
        self._initialized = False

    async def _get_vector_store(self) -> VectorStore:
        """벡터 저장소 lazy 초기화."""
        if self._vector_store is None:
            if self._db is None:
                # DB 세션이 없으면 새로 생성
                from app.core.database import async_session_maker
                self._db = async_session_maker()

            embedding_service = self._embedding_service or get_embedding_service()
            self._vector_store = VectorStore(self._db, embedding_service)

        return self._vector_store

    async def initialize(self) -> None:
        """서비스 초기화."""
        if self._initialized:
            return
        vector_store = await self._get_vector_store()
        await vector_store.initialize()
        self._initialized = True
        logger.info("RAG service initialized with pgvector")

    async def search(
        self,
        query: str,
        category: str = "general",
        doc_type: Optional[LegalDocumentType] = None,
        top_k: int = 10,
        min_relevance: float = 0.5
    ) -> RAGSearchResponse:
        """
        법률 문서 검색.

        Args:
            query: 검색 쿼리
            category: 법률 분야 (contract, labor, criminal 등)
            doc_type: 문서 타입 필터
            top_k: 최대 결과 수
            min_relevance: 최소 관련성 점수

        Returns:
            RAGSearchResponse with 검색된 문서들
        """
        start_time = time.time()

        vector_store = await self._get_vector_store()
        await vector_store.initialize()

        # 카테고리 필터 적용
        category_filter = self._get_category_filter(category)
        doc_type_filter = doc_type.value if doc_type else None

        try:
            results = await vector_store.search(
                query=query,
                category=category_filter,
                doc_type=doc_type_filter,
                top_k=top_k,
                min_similarity=min_relevance,
            )

            # RAGSearchResult 형식으로 변환
            search_results = []
            for result in results:
                search_results.append(RAGSearchResult(
                    doc_id=result.get("doc_id", ""),
                    title=result.get("title", "제목 없음"),
                    content=result.get("content", ""),
                    doc_type=self._parse_doc_type(result.get("doc_type")),
                    source=result.get("source", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    snippet=result.get("content", "")[:500],
                    case_number=result.get("case_number"),
                    law_number=result.get("law_number"),
                    article_number=result.get("article_number"),
                    court=result.get("court"),
                    decision_date=None,
                    category=self._parse_category(result.get("category")),
                    keywords=result.get("keywords", []),
                    file_id=result.get("doc_id"),
                ))

            processing_time = int((time.time() - start_time) * 1000)

            return RAGSearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            return RAGSearchResponse(
                query=query,
                results=[],
                total_results=0,
                processing_time_ms=processing_time,
            )

    def _get_category_filter(self, category: str) -> Optional[str]:
        """카테고리 문자열을 필터 값으로 변환."""
        filters = CATEGORY_FILTERS.get(category.lower())
        if filters and len(filters) > 0:
            return filters[0]
        return None

    def _parse_doc_type(self, doc_type_str: Optional[str]) -> LegalDocumentType:
        """문자열을 LegalDocumentType으로 변환."""
        if not doc_type_str:
            return LegalDocumentType.LAW
        try:
            return LegalDocumentType(doc_type_str.lower())
        except ValueError:
            mapping = {
                "법률": LegalDocumentType.LAW,
                "법령": LegalDocumentType.LAW,
                "판례": LegalDocumentType.PRECEDENT,
                "헌법재판소": LegalDocumentType.CONSTITUTIONAL,
                "논문": LegalDocumentType.ARTICLE,
                "해설": LegalDocumentType.COMMENTARY,
            }
            return mapping.get(doc_type_str, LegalDocumentType.LAW)

    def _parse_category(self, category_str: Optional[str]) -> LegalCategory:
        """문자열을 LegalCategory로 변환."""
        if not category_str:
            return LegalCategory.OTHER
        try:
            return LegalCategory(category_str.lower())
        except ValueError:
            mapping = {
                "민사": LegalCategory.CIVIL,
                "형사": LegalCategory.CRIMINAL,
                "행정": LegalCategory.ADMINISTRATIVE,
                "헌법": LegalCategory.CONSTITUTIONAL,
                "노동": LegalCategory.LABOR,
                "세법": LegalCategory.TAX,
                "상사": LegalCategory.COMMERCIAL,
                "가사": LegalCategory.FAMILY,
                "부동산": LegalCategory.REAL_ESTATE,
                "지식재산권": LegalCategory.INTELLECTUAL_PROPERTY,
            }
            return mapping.get(category_str, LegalCategory.OTHER)

    def _parse_court(self, court_str: Optional[str]) -> Optional[str]:
        """법원 문자열 파싱."""
        return court_str if court_str else None

    async def search_with_answer(
        self,
        query: str,
        category: str = "general",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> GroundedGenerationResponse:
        """
        검색과 답변 생성을 동시에 수행 (Grounded Generation).

        검색된 문서를 컨텍스트로 OpenRouter LLM에 전달하여 답변 생성.

        Args:
            query: 사용자 질문
            category: 법률 분야
            system_prompt: 시스템 프롬프트 (옵션)
            temperature: 생성 온도

        Returns:
            GroundedGenerationResponse with 생성된 답변과 출처 정보
        """
        start_time = time.time()

        # 1. 문서 검색
        search_response = await self.search(query, category, top_k=5)

        if not search_response.results:
            return GroundedGenerationResponse(
                query=query,
                generated_answer="관련 법률 문서를 찾지 못했습니다.",
                grounding_chunks=[],
                citations=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="",
                corpus_id="pgvector",
            )

        # 2. 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(search_response.results, 1):
            context_parts.append(f"""
[문서 {i}]
제목: {result.title}
출처: {result.source}
유형: {result.doc_type.value if hasattr(result.doc_type, 'value') else result.doc_type}
{f'사건번호: {result.case_number}' if result.case_number else ''}
{f'법률번호: {result.law_number}' if result.law_number else ''}

내용:
{result.content[:2000]}
""")

        context = "\n---\n".join(context_parts)

        # 3. 기본 시스템 프롬프트
        if not system_prompt:
            system_prompt = """당신은 한국 법률 전문가입니다.
제공된 법률 문서를 기반으로 정확하고 신뢰할 수 있는 답변을 제공합니다.

## 응답 지침
1. 답변 시 반드시 관련 법률 조문이나 판례를 인용하고 출처를 명시하세요.
2. 확실하지 않은 내용에 대해서는 명확히 그 한계를 밝히세요.
3. 법률 용어는 정확하게 사용하되, 일반인이 이해할 수 있도록 설명을 추가하세요.
4. 가능한 경우 관련 판례의 사건번호를 함께 제시하세요.

## 인용 형식
- 법률: 「법률명」 제○조 제○항
- 판례: 대법원 20XX. XX. XX. 선고 20XX다XXXXX 판결"""

        # 4. OpenRouter LLM 호출
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "HTTP-Referer": settings.OPENROUTER_SITE_URL,
                        "X-Title": settings.OPENROUTER_APP_NAME,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "google/gemini-2.5-flash-preview",  # 빠르고 저렴한 모델
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"""다음 법률 문서를 참고하여 질문에 답변해주세요.

## 참고 문서
{context}

## 질문
{query}

답변 시 참고한 문서의 번호 [문서 N]을 인용해주세요."""},
                        ],
                        "temperature": temperature,
                        "max_tokens": 4096,
                    },
                )
                response.raise_for_status()
                result = response.json()

            generated_answer = result["choices"][0]["message"]["content"]
            model_used = result.get("model", "google/gemini-2.5-flash-preview")

            # 5. 출처 정보 구성
            citations = [
                f"{r.title} ({r.source})"
                for r in search_response.results
            ]

            processing_time = int((time.time() - start_time) * 1000)

            return GroundedGenerationResponse(
                query=query,
                generated_answer=generated_answer,
                grounding_chunks=search_response.results,
                citations=citations,
                processing_time_ms=processing_time,
                model_used=model_used,
                corpus_id="pgvector",
            )

        except Exception as e:
            logger.error(f"Grounded generation error: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            return GroundedGenerationResponse(
                query=query,
                generated_answer=f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                grounding_chunks=search_response.results,
                citations=[],
                processing_time_ms=processing_time,
                model_used="",
                corpus_id="pgvector",
            )

    async def get_context_for_llm(
        self,
        query: str,
        category: str = "general",
        max_tokens: int = 8000
    ) -> str:
        """
        LLM 컨텍스트 윈도우에 제공할 검색 결과 포맷팅.

        Stage 1 LLM들이 사용할 RAG 컨텍스트를 생성합니다.

        Args:
            query: 검색 쿼리
            category: 법률 분야
            max_tokens: 최대 토큰 수 (대략적 추정)

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        response = await self.search(query, category, top_k=15)

        if not response.results:
            return "관련 법률 문서를 찾지 못했습니다."

        context_parts = ["## 관련 법률 문서\n"]
        current_length = 0
        char_limit = max_tokens * 4  # 대략 1 토큰 = 4자

        for i, result in enumerate(response.results, 1):
            citation_text = f"""
### [{i}] {result.title}
- 출처: {result.source}
- 유형: {result.doc_type.value if hasattr(result.doc_type, 'value') else result.doc_type}
- 분야: {result.category.value if hasattr(result.category, 'value') else result.category}
{f'- 사건번호: {result.case_number}' if result.case_number else ''}
{f'- 법률번호: {result.law_number}' if result.law_number else ''}
- 관련도: {result.relevance_score:.2f}

**내용:**
{result.content}

---
"""
            if current_length + len(citation_text) > char_limit:
                break

            context_parts.append(citation_text)
            current_length += len(citation_text)

        return "\n".join(context_parts)

    async def get_context_for_council(
        self,
        query: str,
        category: str = "general",
        max_documents: int = 5,
    ) -> list[dict]:
        """
        위원회 자문을 위한 RAG 컨텍스트 조회.

        LegalContext.rag_context 형식으로 반환합니다.

        Args:
            query: 법률 질문
            category: 법률 분야
            max_documents: 최대 문서 수

        Returns:
            RAG 컨텍스트 딕셔너리 리스트
        """
        response = await self.search(
            query=query,
            category=category,
            top_k=max_documents,
        )

        context = []
        for result in response.results:
            context.append({
                "title": result.title,
                "content": result.content[:2000],
                "source": result.source,
                "source_url": result.source_url if hasattr(result, 'source_url') else None,
                "doc_type": result.doc_type.value if hasattr(result.doc_type, 'value') else str(result.doc_type),
                "category": result.category.value if hasattr(result.category, 'value') else str(result.category),
                "case_number": result.case_number,
                "law_number": result.law_number,
                "relevance_score": result.relevance_score,
            })

        return context

    async def ingest_document(
        self,
        doc_id: str,
        content: str,
        metadata: LegalDocumentMetadata
    ) -> dict:
        """
        새 법률 문서 업로드.

        OpenRouter 임베딩 + pgvector에 저장합니다.

        Args:
            doc_id: 문서 ID
            content: 문서 내용
            metadata: 문서 메타데이터

        Returns:
            업로드 결과
        """
        vector_store = await self._get_vector_store()
        await vector_store.initialize()

        doc_type_val = metadata.doc_type if isinstance(metadata.doc_type, str) else metadata.doc_type.value
        category_val = metadata.category if isinstance(metadata.category, str) else metadata.category.value

        vector_ids = await vector_store.add_document(
            doc_id=doc_id,
            title=metadata.title,
            content=content,
            doc_type=doc_type_val,
            category=category_val,
            source=metadata.source,
            case_number=metadata.case_number,
            law_number=metadata.law_number,
            article_number=metadata.article_number,
            court=metadata.court if isinstance(metadata.court, str) else (metadata.court.value if metadata.court else None),
            decision_date=metadata.decision_date,
            keywords=metadata.keywords,
        )

        return {
            "doc_id": doc_id,
            "vector_ids": vector_ids,
            "chunks": len(vector_ids),
            "status": "uploaded",
        }

    async def delete_document(self, doc_id: str) -> bool:
        """문서 삭제."""
        vector_store = await self._get_vector_store()
        await vector_store.initialize()
        return await vector_store.delete_document(doc_id)

    async def get_similar_cases(
        self,
        case_description: str,
        top_k: int = 5
    ) -> list[RAGSearchResult]:
        """
        유사 판례 검색.

        사건 설명을 기반으로 유사한 판례를 검색합니다.

        Args:
            case_description: 사건 설명
            top_k: 최대 결과 수

        Returns:
            유사 판례 목록
        """
        response = await self.search(
            query=case_description,
            doc_type=LegalDocumentType.PRECEDENT,
            top_k=top_k
        )
        return response.results

    async def get_relevant_laws(
        self,
        topic: str,
        category: str = "general",
        top_k: int = 5
    ) -> list[RAGSearchResult]:
        """
        관련 법률 조문 검색.

        주제를 기반으로 관련 법률 조문을 검색합니다.

        Args:
            topic: 검색 주제
            category: 법률 분야
            top_k: 최대 결과 수

        Returns:
            관련 법률 목록
        """
        response = await self.search(
            query=topic,
            category=category,
            doc_type=LegalDocumentType.LAW,
            top_k=top_k
        )
        return response.results

    async def get_stats(self) -> CorpusStats:
        """Corpus 통계 조회."""
        vector_store = await self._get_vector_store()
        await vector_store.initialize()
        stats = await vector_store.get_stats()

        return CorpusStats(
            corpus_id="pgvector-local",
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_documents", 0),
            documents_by_type=stats.get("documents_by_type", {}),
            documents_by_category=stats.get("documents_by_category", {}),
            last_upload=None,
        )


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
