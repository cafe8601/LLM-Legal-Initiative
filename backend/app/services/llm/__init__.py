"""
LLM Service Package

LLM 클라이언트 및 위원회 오케스트레이션 서비스

Components:
- Base: LLM client interface and response models
- Clients: OpenAI, Anthropic, Google, xAI implementations
- Factory: Client creation factory
- Council: 3-stage council orchestration
- Cascade: CascadeFlow 기반 지능형 모델 캐스케이딩
- RAG: Google File Search based retrieval
"""

from app.services.llm.base import (
    BaseLLMClient,
    LegalContext,
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.factory import LLMClientFactory
from app.services.llm.council import (
    CouncilOrchestrator,
    CouncilOpinion,
    CouncilResult,
    ChairmanSynthesis,
    PeerReview,
)
from app.services.llm.rag_service import (
    RAGService,
    RAGSearchResult,
    LegalDocument,
    get_rag_service,
)

# Client imports
from app.services.llm.openai_client import OpenAIClient
from app.services.llm.anthropic_client import (
    AnthropicClient,
    ClaudeSonnetClient,
    ClaudePeerReviewer,
    ClaudeChairman,
)
from app.services.llm.google_client import GeminiClient
from app.services.llm.xai_client import GrokClient

# OpenRouter unified client
from app.services.llm.openrouter_client import (
    OpenRouterClient,
    OpenRouterCouncilMember,
    OpenRouterPeerReviewer,
    OpenRouterChairman,
)

# CascadeFlow integration
from app.services.llm.cascade_service import (
    CascadeService,
    CascadeServiceFactory,
    CascadeResult,
    CascadePair,
    CascadeModel,
    CascadeModelTier,
    QualityValidator,
    QueryComplexity,
    CLAUDE_CASCADE,
    GPT_CASCADE,
    GEMINI_CASCADE,
    CHAIRMAN_CASCADE,
)
from app.services.llm.cascade_council import (
    CascadeCouncilOrchestrator,
    CascadeCouncilResult,
    CascadeOpinion,
    CascadeReview,
    CascadeSynthesis,
)

__all__ = [
    # Base
    "BaseLLMClient",
    "LegalContext",
    "LLMResponse",
    "ModelRole",
    "StreamingLLMClient",
    # Factory
    "LLMClientFactory",
    # Council
    "CouncilOrchestrator",
    "CouncilOpinion",
    "CouncilResult",
    "ChairmanSynthesis",
    "PeerReview",
    # RAG
    "RAGService",
    "RAGSearchResult",
    "LegalDocument",
    "get_rag_service",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "ClaudeSonnetClient",
    "ClaudePeerReviewer",
    "ClaudeChairman",
    "GeminiClient",
    "GrokClient",
    # OpenRouter
    "OpenRouterClient",
    "OpenRouterCouncilMember",
    "OpenRouterPeerReviewer",
    "OpenRouterChairman",
    # CascadeFlow
    "CascadeService",
    "CascadeServiceFactory",
    "CascadeResult",
    "CascadePair",
    "CascadeModel",
    "CascadeModelTier",
    "QualityValidator",
    "QueryComplexity",
    "CLAUDE_CASCADE",
    "GPT_CASCADE",
    "GEMINI_CASCADE",
    "CHAIRMAN_CASCADE",
    "CascadeCouncilOrchestrator",
    "CascadeCouncilResult",
    "CascadeOpinion",
    "CascadeReview",
    "CascadeSynthesis",
]
