"""
Experience RAG System

상담 경험 기반 RAG 시스템.
MAS multiagent v4의 experience_embeddings를 법률 자문 시스템에 맞게 확장.

Features:
- 과거 상담 경험 벡터 저장
- 유사 상담 케이스 검색
- 위원별 성공 패턴 검색
- 법률 문서 RAG와 통합
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class ConsultationExperience:
    """상담 경험 데이터."""
    experience_id: str
    consultation_id: str
    user_id: str

    # 상담 내용
    query: str
    response_summary: str
    category: str
    keywords: list[str]

    # 결과
    agent_id: str
    model_id: str
    success: bool
    quality_score: float
    user_feedback: Optional[int] = None

    # 메타데이터
    created_at: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperienceSearchResult:
    """경험 검색 결과."""
    experience: ConsultationExperience
    similarity: float
    relevance_reason: str


class ExperienceRAG:
    """
    경험 기반 RAG 시스템.

    과거 상담 경험을 벡터화하여 저장하고
    유사한 케이스를 검색합니다.
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        초기화.

        Args:
            db: SQLAlchemy AsyncSession
            embedding_service: 임베딩 서비스
        """
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()
        self._initialized = False

    async def initialize(self) -> None:
        """테이블 초기화."""
        if self._initialized:
            return

        try:
            # pgvector 확장 확인
            await self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # 테이블 존재 확인
            result = await self.db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'consultation_experiences'
                )
            """))
            exists = result.scalar()

            if not exists:
                await self._create_tables()

            self._initialized = True
            logger.info("ExperienceRAG initialized")

        except Exception as e:
            logger.error(f"ExperienceRAG initialization failed: {e}")
            raise

    async def _create_tables(self) -> None:
        """테이블 생성."""
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS consultation_experiences (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                experience_id VARCHAR(255) UNIQUE NOT NULL,
                consultation_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,

                -- 상담 내용
                query TEXT NOT NULL,
                response_summary TEXT NOT NULL,
                category VARCHAR(50) NOT NULL,
                keywords TEXT[],

                -- 결과
                agent_id VARCHAR(50) NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                success BOOLEAN NOT NULL,
                quality_score FLOAT NOT NULL,
                user_feedback INTEGER,

                -- 벡터
                query_embedding vector(1536),
                response_embedding vector(1536),

                -- 메타데이터
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))

        # 인덱스 생성
        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_experiences_query_embedding
            ON consultation_experiences
            USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 100)
        """))

        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_experiences_category
            ON consultation_experiences (category)
        """))

        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_experiences_agent
            ON consultation_experiences (agent_id)
        """))

        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_experiences_success
            ON consultation_experiences (success)
        """))

        await self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_experiences_keywords
            ON consultation_experiences USING GIN (keywords)
        """))

        await self.db.commit()
        logger.info("Created consultation_experiences table")

    async def index_experience(
        self,
        consultation_id: str | UUID,
        user_id: str | UUID,
        query: str,
        response_summary: str,
        category: str,
        keywords: list[str],
        agent_id: str,
        model_id: str,
        success: bool,
        quality_score: float,
        user_feedback: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        상담 경험 인덱싱.

        Args:
            consultation_id: 상담 ID
            user_id: 사용자 ID
            query: 사용자 질문
            response_summary: 응답 요약
            category: 법률 분야
            keywords: 키워드 목록
            agent_id: 에이전트 ID
            model_id: 모델 ID
            success: 성공 여부
            quality_score: 품질 점수
            user_feedback: 사용자 피드백 (1-5)
            metadata: 추가 메타데이터

        Returns:
            생성된 experience_id
        """
        await self.initialize()

        # 경험 ID 생성
        experience_id = f"exp_{hashlib.sha256(f'{consultation_id}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"

        # 임베딩 생성
        query_embedding = await self.embedding_service.embed_text(query)
        response_embedding = await self.embedding_service.embed_text(response_summary)

        # 저장
        await self.db.execute(text("""
            INSERT INTO consultation_experiences (
                experience_id, consultation_id, user_id,
                query, response_summary, category, keywords,
                agent_id, model_id, success, quality_score, user_feedback,
                query_embedding, response_embedding, metadata
            ) VALUES (
                :experience_id, :consultation_id, :user_id,
                :query, :response_summary, :category, :keywords,
                :agent_id, :model_id, :success, :quality_score, :user_feedback,
                :query_embedding, :response_embedding, :metadata
            )
        """), {
            "experience_id": experience_id,
            "consultation_id": str(consultation_id),
            "user_id": str(user_id),
            "query": query,
            "response_summary": response_summary,
            "category": category,
            "keywords": keywords,
            "agent_id": agent_id,
            "model_id": model_id,
            "success": success,
            "quality_score": quality_score,
            "user_feedback": user_feedback,
            "query_embedding": str(query_embedding),
            "response_embedding": str(response_embedding),
            "metadata": metadata or {},
        })

        await self.db.commit()
        logger.info(f"Indexed experience: {experience_id}, category={category}, success={success}")

        return experience_id

    async def search_similar_experiences(
        self,
        query: str,
        category: Optional[str] = None,
        agent_id: Optional[str] = None,
        success_only: bool = False,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[ExperienceSearchResult]:
        """
        유사 경험 검색.

        Args:
            query: 검색 쿼리
            category: 법률 분야 필터
            agent_id: 에이전트 필터
            success_only: 성공 케이스만
            top_k: 최대 결과 수
            min_similarity: 최소 유사도

        Returns:
            ExperienceSearchResult 리스트
        """
        await self.initialize()

        # 쿼리 임베딩
        query_embedding = await self.embedding_service.embed_text(query)

        # 필터 조건
        filters = []
        params = {
            "embedding": str(query_embedding),
            "min_similarity": min_similarity,
            "top_k": top_k,
        }

        if category:
            filters.append("category = :category")
            params["category"] = category

        if agent_id:
            filters.append("agent_id = :agent_id")
            params["agent_id"] = agent_id

        if success_only:
            filters.append("success = TRUE")

        filter_clause = " AND ".join(filters) if filters else "TRUE"

        # 검색
        sql = text(f"""
            SELECT
                experience_id, consultation_id, user_id,
                query, response_summary, category, keywords,
                agent_id, model_id, success, quality_score, user_feedback,
                metadata, created_at,
                1 - (query_embedding <=> :embedding::vector) AS similarity
            FROM consultation_experiences
            WHERE {filter_clause}
              AND 1 - (query_embedding <=> :embedding::vector) >= :min_similarity
            ORDER BY query_embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        results = []
        for row in rows:
            experience = ConsultationExperience(
                experience_id=row.experience_id,
                consultation_id=row.consultation_id,
                user_id=row.user_id,
                query=row.query,
                response_summary=row.response_summary,
                category=row.category,
                keywords=row.keywords or [],
                agent_id=row.agent_id,
                model_id=row.model_id,
                success=row.success,
                quality_score=row.quality_score,
                user_feedback=row.user_feedback,
                created_at=row.created_at.isoformat() if row.created_at else "",
                metadata=row.metadata or {},
            )

            results.append(ExperienceSearchResult(
                experience=experience,
                similarity=float(row.similarity),
                relevance_reason=f"유사도 {row.similarity*100:.1f}%, {experience.category}",
            ))

        logger.debug(f"Found {len(results)} similar experiences for query")
        return results

    async def get_agent_success_patterns(
        self,
        agent_id: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[ConsultationExperience]:
        """
        에이전트의 성공 패턴 조회.

        특정 에이전트가 성공한 케이스들을 조회합니다.
        """
        await self.initialize()

        filters = ["agent_id = :agent_id", "success = TRUE"]
        params = {"agent_id": agent_id, "limit": limit}

        if category:
            filters.append("category = :category")
            params["category"] = category

        filter_clause = " AND ".join(filters)

        sql = text(f"""
            SELECT
                experience_id, consultation_id, user_id,
                query, response_summary, category, keywords,
                agent_id, model_id, success, quality_score, user_feedback,
                metadata, created_at
            FROM consultation_experiences
            WHERE {filter_clause}
            ORDER BY quality_score DESC, created_at DESC
            LIMIT :limit
        """)

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        return [
            ConsultationExperience(
                experience_id=row.experience_id,
                consultation_id=row.consultation_id,
                user_id=row.user_id,
                query=row.query,
                response_summary=row.response_summary,
                category=row.category,
                keywords=row.keywords or [],
                agent_id=row.agent_id,
                model_id=row.model_id,
                success=row.success,
                quality_score=row.quality_score,
                user_feedback=row.user_feedback,
                created_at=row.created_at.isoformat() if row.created_at else "",
                metadata=row.metadata or {},
            )
            for row in rows
        ]

    async def get_category_best_practices(
        self,
        category: str,
        limit: int = 5,
    ) -> list[ConsultationExperience]:
        """
        분야별 베스트 프랙티스 조회.

        해당 분야에서 가장 성공적인 상담 케이스들을 조회합니다.
        """
        await self.initialize()

        sql = text("""
            SELECT
                experience_id, consultation_id, user_id,
                query, response_summary, category, keywords,
                agent_id, model_id, success, quality_score, user_feedback,
                metadata, created_at
            FROM consultation_experiences
            WHERE category = :category AND success = TRUE
            ORDER BY
                COALESCE(user_feedback, 0) DESC,
                quality_score DESC
            LIMIT :limit
        """)

        result = await self.db.execute(sql, {"category": category, "limit": limit})
        rows = result.fetchall()

        return [
            ConsultationExperience(
                experience_id=row.experience_id,
                consultation_id=row.consultation_id,
                user_id=row.user_id,
                query=row.query,
                response_summary=row.response_summary,
                category=row.category,
                keywords=row.keywords or [],
                agent_id=row.agent_id,
                model_id=row.model_id,
                success=row.success,
                quality_score=row.quality_score,
                user_feedback=row.user_feedback,
                created_at=row.created_at.isoformat() if row.created_at else "",
                metadata=row.metadata or {},
            )
            for row in rows
        ]

    async def update_user_feedback(
        self,
        experience_id: str,
        feedback: int,
    ) -> bool:
        """사용자 피드백 업데이트."""
        await self.initialize()

        result = await self.db.execute(text("""
            UPDATE consultation_experiences
            SET user_feedback = :feedback
            WHERE experience_id = :experience_id
        """), {"experience_id": experience_id, "feedback": feedback})

        await self.db.commit()
        return result.rowcount > 0

    async def get_statistics(self) -> dict:
        """통계 조회."""
        await self.initialize()

        # 전체 통계
        total_result = await self.db.execute(text(
            "SELECT COUNT(*) FROM consultation_experiences"
        ))
        total = total_result.scalar()

        success_result = await self.db.execute(text(
            "SELECT COUNT(*) FROM consultation_experiences WHERE success = TRUE"
        ))
        successes = success_result.scalar()

        # 에이전트별
        agent_result = await self.db.execute(text("""
            SELECT agent_id, COUNT(*) as count,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
            FROM consultation_experiences
            GROUP BY agent_id
        """))
        by_agent = {
            row.agent_id: {
                "count": row.count,
                "success_rate": row.successes / row.count if row.count > 0 else 0,
            }
            for row in agent_result.fetchall()
        }

        # 카테고리별
        category_result = await self.db.execute(text("""
            SELECT category, COUNT(*) as count,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
            FROM consultation_experiences
            GROUP BY category
        """))
        by_category = {
            row.category: {
                "count": row.count,
                "success_rate": row.successes / row.count if row.count > 0 else 0,
            }
            for row in category_result.fetchall()
        }

        return {
            "total_experiences": total,
            "success_rate": successes / total if total > 0 else 0,
            "by_agent": by_agent,
            "by_category": by_category,
        }
