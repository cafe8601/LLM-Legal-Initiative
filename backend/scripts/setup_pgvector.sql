-- PostgreSQL + pgvector 설정 스크립트
-- 법률 문서 벡터 검색을 위한 테이블 및 인덱스 생성
--
-- 실행 방법:
--   psql -d legal_council -f setup_pgvector.sql
--
-- 또는 애플리케이션 시작 시 자동으로 실행됩니다.

-- 1. pgvector 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 법률 문서 벡터 테이블 생성
CREATE TABLE IF NOT EXISTS legal_document_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),

    -- 메타데이터
    doc_type VARCHAR(50) NOT NULL,           -- law, precedent, constitutional, article, commentary
    category VARCHAR(50) NOT NULL,           -- civil, criminal, administrative, labor, tax, etc.
    source VARCHAR(255),                     -- 출처 (법원명, 법률 출처 등)
    case_number VARCHAR(100),                -- 사건번호 (판례)
    law_number VARCHAR(100),                 -- 법률번호
    article_number VARCHAR(50),              -- 조문번호
    court VARCHAR(100),                      -- 법원명
    decision_date DATE,                      -- 선고일
    keywords TEXT[],                         -- 키워드 배열

    -- 벡터 (1536 dimensions for OpenAI text-embedding-3-small)
    embedding vector(1536),

    -- 청크 정보 (원본이 청킹된 경우)
    chunk_index INTEGER DEFAULT 0,
    parent_doc_id VARCHAR(255),

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. 인덱스 생성

-- 벡터 유사도 검색용 IVFFlat 인덱스 (대규모 데이터용)
-- lists 값은 데이터 크기에 따라 조정 (sqrt(문서수) 권장)
CREATE INDEX IF NOT EXISTS idx_legal_vectors_embedding
ON legal_document_vectors
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW 인덱스 (더 정확하지만 메모리 많이 사용)
-- 소규모 데이터나 높은 정확도가 필요할 때 사용
-- CREATE INDEX IF NOT EXISTS idx_legal_vectors_embedding_hnsw
-- ON legal_document_vectors
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- 메타데이터 필터링용 인덱스
CREATE INDEX IF NOT EXISTS idx_legal_vectors_doc_type
ON legal_document_vectors (doc_type);

CREATE INDEX IF NOT EXISTS idx_legal_vectors_category
ON legal_document_vectors (category);

CREATE INDEX IF NOT EXISTS idx_legal_vectors_doc_id
ON legal_document_vectors (doc_id);

CREATE INDEX IF NOT EXISTS idx_legal_vectors_parent_doc_id
ON legal_document_vectors (parent_doc_id)
WHERE parent_doc_id IS NOT NULL;

-- 키워드 검색용 GIN 인덱스
CREATE INDEX IF NOT EXISTS idx_legal_vectors_keywords
ON legal_document_vectors USING GIN (keywords);

-- 전문 검색용 GIN 인덱스 (한글 형태소 분석)
CREATE INDEX IF NOT EXISTS idx_legal_vectors_content_fts
ON legal_document_vectors
USING GIN (to_tsvector('simple', content));

-- 4. 업데이트 트리거 (updated_at 자동 갱신)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_legal_vectors_updated_at ON legal_document_vectors;
CREATE TRIGGER update_legal_vectors_updated_at
    BEFORE UPDATE ON legal_document_vectors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 5. 통계 조회 뷰
CREATE OR REPLACE VIEW legal_document_stats AS
SELECT
    COUNT(*) as total_documents,
    COUNT(DISTINCT parent_doc_id) as total_parent_documents,
    COUNT(*) FILTER (WHERE parent_doc_id IS NOT NULL) as total_chunks,
    doc_type,
    category,
    COUNT(*) as doc_count
FROM legal_document_vectors
GROUP BY GROUPING SETS ((), (doc_type), (category), (doc_type, category));

-- 6. 샘플 데이터 확인 쿼리 (테스트용)
-- SELECT * FROM legal_document_stats WHERE doc_type IS NOT NULL AND category IS NOT NULL;

COMMENT ON TABLE legal_document_vectors IS '법률 문서 벡터 저장소 - OpenRouter 임베딩 + pgvector';
COMMENT ON COLUMN legal_document_vectors.embedding IS 'OpenAI text-embedding-3-small (1536 dimensions)';
COMMENT ON COLUMN legal_document_vectors.chunk_index IS '청크 인덱스 (0부터 시작, 단일 문서면 0)';
COMMENT ON COLUMN legal_document_vectors.parent_doc_id IS '부모 문서 ID (청크인 경우)';
