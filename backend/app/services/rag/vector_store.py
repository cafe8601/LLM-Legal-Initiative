"""
Vector Store Service

PostgreSQL + pgvector 기반 로컬 벡터 저장소.
OpenRouter 임베딩 서비스와 통합하여 단일 API Key로 운영.

Features:
- 법률 문서 벡터 저장
- 시맨틱 유사도 검색
- 메타데이터 필터링
- 하이브리드 검색 (벡터 + 키워드)
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """
    PostgreSQL + pgvector 기반 벡터 저장소.

    Prerequisites:
    - PostgreSQL with pgvector extension
    - CREATE EXTENSION IF NOT EXISTS vector;
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        벡터 저장소 초기화.

        Args:
            db: SQLAlchemy AsyncSession
            embedding_service: 임베딩 서비스 (None이면 기본 서비스 사용)
        """
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()
        self._initialized = False

    async def initialize(self) -> None:
        """pgvector 확장 및 테이블 확인."""
        if self._initialized:
            return

        try:
            # pgvector 확장 확인/생성
            await self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await self.db.commit()

            # 테이블 존재 확인
            result = await self.db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'legal_document_vectors'
                )
            """))
            exists = result.scalar()

            if not exists:
                await self._create_tables()

            self._initialized = True
            logger.info("VectorStore initialized")

        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise

    async def _create_tables(self) -> None:
        """벡터 저장 테이블 생성."""
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS legal_document_vectors (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                doc_id VARCHAR(255) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                content_hash VARCHAR(64),

                -- 메타데이터
                doc_type VARCHAR(50) NOT NULL,
                category VARCHAR(50) NOT NULL,
                source VARCHAR(255),
                case_number VARCHAR(100),
                law_number VARCHAR(100),
                article_number VARCHAR(50),
                court VARCHAR(100),
                decision_date DATE,
                keywords TEXT[],

                -- 벡터 (1536 dimensions for OpenAI text-embedding-3-small)
                embedding vector(1536),

                -- 청크 정보 (원본이 청킹된 경우)
                chunk_index INTEGER DEFAULT 0,
                parent_doc_id VARCHAR(255),

                -- 타임스탬프
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))

        # 벡터 유사도 검색용 인덱스 (IVFFlat - 대규모 데이터용)
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_legal_vectors_embedding
            ON legal_document_vectors
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))

        # 메타데이터 필터링용 인덱스
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_legal_vectors_doc_type
            ON legal_document_vectors (doc_type)
        """))
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_legal_vectors_category
            ON legal_document_vectors (category)
        """))
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_legal_vectors_keywords
            ON legal_document_vectors USING GIN (keywords)
        """))

        # 전문 검색용 인덱스
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_legal_vectors_content_fts
            ON legal_document_vectors
            USING GIN (to_tsvector('simple', content))
        """))

        await self.db.commit()
        logger.info("Created legal_document_vectors table with indexes")

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        doc_type: str,
        category: str,
        source: str = "",
        case_number: Optional[str] = None,
        law_number: Optional[str] = None,
        article_number: Optional[str] = None,
        court: Optional[str] = None,
        decision_date: Optional[datetime] = None,
        keywords: Optional[list[str]] = None,
        chunk_documents: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> list[str]:
        """
        문서를 벡터화하여 저장.

        긴 문서는 자동으로 청킹하여 여러 벡터로 저장합니다.

        Args:
            doc_id: 문서 고유 ID
            title: 문서 제목
            content: 문서 내용
            doc_type: 문서 유형 (law, precedent, constitutional 등)
            category: 법률 분야 (civil, criminal 등)
            source: 출처
            case_number: 사건번호 (판례)
            law_number: 법률번호
            article_number: 조문번호
            court: 법원명
            decision_date: 선고일
            keywords: 키워드 목록
            chunk_documents: 청킹 활성화 여부
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩

        Returns:
            생성된 벡터 ID 리스트
        """
        await self.initialize()

        # 기존 문서 삭제 (업데이트 시)
        await self.delete_document(doc_id)

        # 청킹
        if chunk_documents and len(content) > chunk_size:
            chunks = self.embedding_service.chunk_text(content, chunk_size, chunk_overlap)
        else:
            chunks = [content]

        # 임베딩 생성 (배치)
        embeddings = await self.embedding_service.embed_texts(chunks)

        # 저장
        vector_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_doc_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id

            await self.db.execute(text("""
                INSERT INTO legal_document_vectors (
                    doc_id, title, content, doc_type, category, source,
                    case_number, law_number, article_number, court, decision_date,
                    keywords, embedding, chunk_index, parent_doc_id
                ) VALUES (
                    :doc_id, :title, :content, :doc_type, :category, :source,
                    :case_number, :law_number, :article_number, :court, :decision_date,
                    :keywords, :embedding, :chunk_index, :parent_doc_id
                )
            """), {
                "doc_id": chunk_doc_id,
                "title": title,
                "content": chunk,
                "doc_type": doc_type,
                "category": category,
                "source": source,
                "case_number": case_number,
                "law_number": law_number,
                "article_number": article_number,
                "court": court,
                "decision_date": decision_date,
                "keywords": keywords or [],
                "embedding": str(embedding),  # pgvector 형식
                "chunk_index": i,
                "parent_doc_id": doc_id if len(chunks) > 1 else None,
            })

            vector_ids.append(chunk_doc_id)

        await self.db.commit()
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        return vector_ids

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        doc_type: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """
        시맨틱 검색.

        벡터 유사도와 메타데이터 필터를 결합하여 검색합니다.

        Args:
            query: 검색 쿼리
            category: 법률 분야 필터
            doc_type: 문서 유형 필터
            keywords: 키워드 필터 (OR 조건)
            top_k: 최대 결과 수
            min_similarity: 최소 유사도 (0-1)

        Returns:
            검색 결과 리스트
        """
        await self.initialize()

        # 쿼리 임베딩 생성
        query_embedding = await self.embedding_service.embed_text(query)

        # 필터 조건 구성
        filters = []
        params = {
            "embedding": str(query_embedding),
            "min_similarity": min_similarity,
            "top_k": top_k,
        }

        if category:
            filters.append("category = :category")
            params["category"] = category

        if doc_type:
            filters.append("doc_type = :doc_type")
            params["doc_type"] = doc_type

        if keywords:
            filters.append("keywords && :keywords")  # Array overlap
            params["keywords"] = keywords

        filter_clause = " AND ".join(filters) if filters else "TRUE"

        # 벡터 유사도 검색 쿼리
        sql = text(f"""
            SELECT
                doc_id,
                title,
                content,
                doc_type,
                category,
                source,
                case_number,
                law_number,
                article_number,
                court,
                decision_date,
                keywords,
                chunk_index,
                parent_doc_id,
                1 - (embedding <=> :embedding::vector) AS similarity
            FROM legal_document_vectors
            WHERE {filter_clause}
              AND 1 - (embedding <=> :embedding::vector) >= :min_similarity
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        # 결과 포맷팅
        results = []
        for row in rows:
            results.append({
                "doc_id": row.doc_id,
                "title": row.title,
                "content": row.content,
                "doc_type": row.doc_type,
                "category": row.category,
                "source": row.source,
                "case_number": row.case_number,
                "law_number": row.law_number,
                "article_number": row.article_number,
                "court": row.court,
                "decision_date": row.decision_date.isoformat() if row.decision_date else None,
                "keywords": row.keywords or [],
                "relevance_score": float(row.similarity),
                "chunk_index": row.chunk_index,
                "parent_doc_id": row.parent_doc_id,
            })

        logger.debug(f"Search returned {len(results)} results for: {query[:50]}...")
        return results

    async def hybrid_search(
        self,
        query: str,
        category: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> list[dict]:
        """
        하이브리드 검색 (벡터 + 전문 검색).

        벡터 유사도와 키워드 매칭을 결합하여 검색합니다.

        Args:
            query: 검색 쿼리
            category: 법률 분야 필터
            doc_type: 문서 유형 필터
            top_k: 최대 결과 수
            vector_weight: 벡터 점수 가중치 (0-1)

        Returns:
            검색 결과 리스트
        """
        await self.initialize()

        query_embedding = await self.embedding_service.embed_text(query)
        keyword_weight = 1 - vector_weight

        # 필터 조건
        filters = []
        params = {
            "embedding": str(query_embedding),
            "query": query,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "top_k": top_k,
        }

        if category:
            filters.append("category = :category")
            params["category"] = category

        if doc_type:
            filters.append("doc_type = :doc_type")
            params["doc_type"] = doc_type

        filter_clause = " AND ".join(filters) if filters else "TRUE"

        # 하이브리드 검색 쿼리
        sql = text(f"""
            WITH vector_search AS (
                SELECT
                    doc_id,
                    1 - (embedding <=> :embedding::vector) AS vector_score
                FROM legal_document_vectors
                WHERE {filter_clause}
            ),
            keyword_search AS (
                SELECT
                    doc_id,
                    ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', :query)) AS keyword_score
                FROM legal_document_vectors
                WHERE {filter_clause}
                  AND to_tsvector('simple', content) @@ plainto_tsquery('simple', :query)
            )
            SELECT
                d.doc_id,
                d.title,
                d.content,
                d.doc_type,
                d.category,
                d.source,
                d.case_number,
                d.law_number,
                d.article_number,
                d.court,
                d.decision_date,
                d.keywords,
                d.chunk_index,
                d.parent_doc_id,
                COALESCE(v.vector_score, 0) * :vector_weight +
                COALESCE(k.keyword_score, 0) * :keyword_weight AS combined_score
            FROM legal_document_vectors d
            LEFT JOIN vector_search v ON d.doc_id = v.doc_id
            LEFT JOIN keyword_search k ON d.doc_id = k.doc_id
            WHERE {filter_clause}
            ORDER BY combined_score DESC
            LIMIT :top_k
        """)

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        results = []
        for row in rows:
            results.append({
                "doc_id": row.doc_id,
                "title": row.title,
                "content": row.content,
                "doc_type": row.doc_type,
                "category": row.category,
                "source": row.source,
                "case_number": row.case_number,
                "law_number": row.law_number,
                "article_number": row.article_number,
                "court": row.court,
                "decision_date": row.decision_date.isoformat() if row.decision_date else None,
                "keywords": row.keywords or [],
                "relevance_score": float(row.combined_score) if row.combined_score else 0.0,
                "chunk_index": row.chunk_index,
                "parent_doc_id": row.parent_doc_id,
            })

        return results

    async def delete_document(self, doc_id: str) -> bool:
        """
        문서 및 관련 청크 삭제.

        Args:
            doc_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """
        await self.initialize()

        try:
            # 원본 문서 및 청크 모두 삭제
            await self.db.execute(text("""
                DELETE FROM legal_document_vectors
                WHERE doc_id = :doc_id OR parent_doc_id = :doc_id
            """), {"doc_id": doc_id})

            await self.db.commit()
            logger.info(f"Deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def get_stats(self) -> dict:
        """
        벡터 저장소 통계 조회.

        Returns:
            통계 정보 딕셔너리
        """
        await self.initialize()

        # 전체 문서 수
        total_result = await self.db.execute(text(
            "SELECT COUNT(*) FROM legal_document_vectors"
        ))
        total_count = total_result.scalar()

        # 유형별 문서 수
        type_result = await self.db.execute(text("""
            SELECT doc_type, COUNT(*) as count
            FROM legal_document_vectors
            GROUP BY doc_type
        """))
        type_counts = {row.doc_type: row.count for row in type_result.fetchall()}

        # 분야별 문서 수
        category_result = await self.db.execute(text("""
            SELECT category, COUNT(*) as count
            FROM legal_document_vectors
            GROUP BY category
        """))
        category_counts = {row.category: row.count for row in category_result.fetchall()}

        return {
            "total_documents": total_count,
            "documents_by_type": type_counts,
            "documents_by_category": category_counts,
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
        }
