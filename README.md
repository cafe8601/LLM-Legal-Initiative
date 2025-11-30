# LLM Legal Advisory Council

AI 기반 법률 자문 서비스 플랫폼 - Multi-Agent System을 활용한 지능형 법률 상담 솔루션

## Overview

LLM Legal Advisory Council은 여러 AI 에이전트가 협력하여 법률 자문을 제공하는 Multi-Agent System(MAS) 기반 플랫폼입니다. 캐스케이드 의사결정 구조를 통해 복잡한 법률 문제에 대해 다각적인 분석과 균형 잡힌 조언을 제공합니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client (React + TypeScript)                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend Server                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Auth &    │  │  Rate Limit │  │   Monitoring & Health   │  │
│  │  Security   │  │   Control   │  │        Checks           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Multi-Agent System (MAS)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Cascade Council System                    │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐   │  │
│  │  │ Drafter │─▶│Reviewer │─▶│ Senior  │─▶│  Chairman   │   │  │
│  │  │ Agent   │  │ Agent   │  │ Counsel │  │   (Final)   │   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    RAG & Learning System                   │  │
│  │  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐  │  │
│  │  │ Vector Store  │  │ Pattern Analyzer│  │   Outcome    │  │  │
│  │  │   (pgvector)  │  │  & Learning    │  │   Tracker    │  │  │
│  │  └───────────────┘  └────────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ PostgreSQL  │  │    Redis    │  │   LLM Providers         │  │
│  │  Database   │  │    Cache    │  │ (OpenAI, Anthropic,     │  │
│  │             │  │             │  │  Google, OpenRouter)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Multi-Agent Cascade System
- **Drafter Agent**: 초안 작성 - 법률 문제 분석 및 초기 답변 생성
- **Reviewer Agent**: 검토 및 보완 - 법적 정확성 검증
- **Senior Counsel**: 심층 분석 - 복잡한 사례에 대한 전문적 검토
- **Chairman**: 최종 의사결정 - 모든 의견을 종합한 최종 자문

### RAG (Retrieval-Augmented Generation)
- 법률 문서 벡터 검색 (pgvector)
- 유사 사례 기반 답변 생성
- 하이브리드 검색 (키워드 + 시맨틱)

### Learning System
- **Pattern Analyzer**: 상담 패턴 분석 및 학습
- **Outcome Tracker**: 결과 추적 및 품질 개선
- 성공/실패 사례 기반 지속적 개선

### Monitoring & Observability
- 실시간 메트릭 수집 (Prometheus 호환)
- 컴포넌트별 헬스체크
- 구조화 로깅 (structlog)

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 16 + pgvector
- **Cache**: Redis 7
- **ORM**: SQLAlchemy 2.0 (async)
- **Authentication**: JWT (python-jose)
- **Logging**: structlog

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS

### LLM Providers
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (Multiple models)
- xAI (Grok)

### Infrastructure
- Docker & Docker Compose
- Alembic (Database migrations)

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 16
- Redis 7
- Docker (optional)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/cafe8601/LLM-Legal-Initiative.git
cd LLM-Legal-Initiative/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database settings

# Run database migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Docker Setup

```bash
# Development environment
docker-compose up -d

# Run tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Production environment
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration

### Environment Variables

```bash
# Application
APP_NAME="LLM Legal Advisory Council"
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/legal_council

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Cache TTL (seconds)
CACHE_DEFAULT_TTL=3600
CACHE_SESSION_TTL=7200
CACHE_RAG_TTL=300

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_TIMEOUT=5

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'console'
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token

### Consultations
- `POST /api/v1/consultations` - Create new consultation
- `GET /api/v1/consultations` - List user consultations
- `GET /api/v1/consultations/{id}` - Get consultation details

### Expert Chat
- `POST /api/v1/expert-chat/sessions` - Create chat session
- `POST /api/v1/expert-chat/sessions/{id}/messages` - Send message
- `GET /api/v1/expert-chat/sessions/{id}/messages` - Get messages

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check with component status
- `GET /metrics` - Application metrics (JSON)
- `GET /metrics/prometheus` - Prometheus format metrics

## Testing

```bash
cd backend

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_security.py -v

# Run tests in Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Test Coverage
- **258 tests passing**
- **50% code coverage**
- Key modules: security (100%), monitoring (85%+), schemas (95%+)

## Project Structure

```
LLM-Legal-Initiative/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API endpoints
│   │   ├── core/             # Configuration, security
│   │   ├── db/               # Database connection
│   │   ├── models/           # SQLAlchemy models
│   │   ├── repositories/     # Data access layer
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   │   ├── learning/     # ML learning system
│   │   │   ├── llm/          # LLM clients & cascade
│   │   │   ├── memory/       # Session & cache
│   │   │   ├── monitoring/   # Metrics & health
│   │   │   └── rag/          # RAG & vector search
│   │   └── utils/            # Utilities
│   ├── alembic/              # Database migrations
│   ├── tests/                # Test suite
│   └── docker-compose.yml    # Docker configuration
├── frontend/
│   ├── components/           # React components
│   ├── pages/                # Page components
│   └── types.ts              # TypeScript types
└── docs/                     # Documentation
```

## User Tiers

| Feature | Basic | Pro | Enterprise |
|---------|-------|-----|------------|
| Daily Consultations | 5 | 50 | Unlimited |
| Response Priority | Standard | High | Highest |
| Expert Chat | Limited | Full | Full + Priority |
| Document Analysis | Basic | Advanced | Advanced + Custom |
| API Access | - | REST | REST + Webhooks |

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI, Anthropic, Google for LLM APIs
- FastAPI community for the excellent framework
- All contributors and testers

---

**Disclaimer**: This system provides AI-generated legal information for educational purposes only. It does not constitute legal advice. Always consult with a qualified attorney for legal matters.
