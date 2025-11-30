"""
RAG Services

OpenRouter + PostgreSQL pgvector 기반 RAG 서비스 모듈.
단일 API Key(OpenRouter)로 모든 임베딩 및 검색 기능 통합.

Components:
- EmbeddingService: OpenRouter 기반 임베딩 생성
- VectorStore: PostgreSQL + pgvector 벡터 저장소
- ExperienceRAG: 상담 경험 기반 RAG 시스템
- HybridRAGOrchestrator: 법률 문서 + 경험 통합 검색
- GoogleFileSearchService: (Legacy) Google File Search 서비스

v4.4.0: MAS multiagent v4 학습/경험 RAG 통합
- 상담 경험 벡터 저장 및 검색
- 유사 케이스 기반 컨텍스트 제공
- 하이브리드 검색 (법률 문서 + 경험)

v4.3.2: OpenRouter 통합 완료
- 임베딩: OpenRouter API (text-embedding-3-small)
- 벡터 저장: PostgreSQL + pgvector
- LLM: OpenRouter API (다양한 모델)
"""

from app.services.rag.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
)

from app.services.rag.vector_store import VectorStore

# Experience RAG (MAS multiagent v4)
from app.services.rag.experience_rag import (
    ExperienceRAG,
    ConsultationExperience,
    ExperienceSearchResult,
)

from app.services.rag.hybrid_orchestrator import (
    HybridRAGOrchestrator,
    HybridSearchResult,
    RAGConfig,
)

# Legacy Google File Search (이전 버전 호환성)
# 주의: Google API Key 필요 - 새 프로젝트는 pgvector 사용 권장
try:
    from app.services.rag.google_file_search import (
        GoogleFileSearchService,
        get_file_search_service,
    )
except ImportError:
    # google-generativeai 패키지 없으면 무시
    GoogleFileSearchService = None  # type: ignore
    get_file_search_service = None  # type: ignore


__all__ = [
    # Core RAG services
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
    "VectorStore",
    # Experience RAG (MAS multiagent v4)
    "ExperienceRAG",
    "ConsultationExperience",
    "ExperienceSearchResult",
    # Hybrid RAG Orchestrator
    "HybridRAGOrchestrator",
    "HybridSearchResult",
    "RAGConfig",
    # Legacy Google services
    "GoogleFileSearchService",
    "get_file_search_service",
]
