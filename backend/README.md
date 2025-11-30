# LLM Legal Advisory Council - Backend

AI 기반 법률 자문 위원회 서비스의 백엔드 API 서버

## 기술 스택

- **Framework**: FastAPI 0.115+
- **Database**: PostgreSQL 16 + SQLAlchemy 2.0 (async)
- **Cache**: Redis 7
- **Authentication**: JWT (python-jose)
- **LLM APIs**: OpenAI, Anthropic, Google Gemini, xAI
- **RAG**: Google File Search API
- **Storage**: AWS S3
- **Payment**: Stripe

## 프로젝트 구조

```
backend/
├── app/
│   ├── api/v1/           # API 엔드포인트
│   ├── core/             # 핵심 설정 (config, security, exceptions)
│   ├── models/           # SQLAlchemy 모델
│   ├── schemas/          # Pydantic 스키마
│   ├── services/         # 비즈니스 로직
│   ├── repositories/     # 데이터 접근 계층
│   └── utils/            # 유틸리티
├── alembic/              # DB 마이그레이션
├── tests/                # 테스트
├── scripts/              # 유틸리티 스크립트
└── docker-compose.yml    # Docker 설정
```

## 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 파일 복사
cp .env.example .env

# .env 파일 편집하여 API 키 설정
```

### 2. Docker로 실행 (권장)

```bash
# 개발 환경 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f api
```

### 3. 로컬 실행 (개발)

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

### 4. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 주요 API 엔드포인트

### 인증
- `POST /api/v1/auth/register` - 회원가입
- `POST /api/v1/auth/login` - 로그인
- `POST /api/v1/auth/refresh` - 토큰 갱신

### 상담
- `POST /api/v1/consultations` - 새 상담 생성
- `GET /api/v1/consultations` - 상담 목록
- `POST /api/v1/consultations/{id}/turns` - 후속 질문

### 검색 (RAG)
- `GET /api/v1/search/legal` - 법률 문서 검색
- `GET /api/v1/search/similar-cases` - 유사 판례 검색

## LLM 구성

### Stage 1: 병렬 의견 수집 (4개 LLM)
- GPT-5.1 (OpenAI)
- Claude Sonnet 4.5 (Anthropic)
- Gemini 3 Pro (Google)
- Grok 4 (xAI)

### Stage 2: 블라인드 교차 평가
- Claude Sonnet 4.5 (Anthropic)

### 의장: 최종 종합
- Claude Opus 4.5 (Anthropic) - `effort: high`

## 데이터베이스 마이그레이션

```bash
# 마이그레이션 생성
alembic revision --autogenerate -m "description"

# 마이그레이션 적용
alembic upgrade head

# 롤백
alembic downgrade -1
```

## 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=app --cov-report=html

# 특정 테스트만
pytest tests/test_auth.py -v
```

## 환경 변수

주요 환경 변수 (`.env.example` 참조):

| 변수 | 설명 |
|------|------|
| `DATABASE_URL` | PostgreSQL 연결 URL |
| `REDIS_URL` | Redis 연결 URL |
| `SECRET_KEY` | JWT 서명 키 |
| `OPENAI_API_KEY` | OpenAI API 키 |
| `ANTHROPIC_API_KEY` | Anthropic API 키 |
| `GOOGLE_API_KEY` | Google API 키 |
| `XAI_API_KEY` | xAI API 키 |

## 라이선스

MIT License
