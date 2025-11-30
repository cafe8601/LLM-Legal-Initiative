"""
Google Gemini Client

Stage 1 위원회 멤버 - Gemini 3 Pro with File Search grounding

v4.1 한글 프롬프트 적용:
- thinking_level 파라미터 (low/medium/high)
- 모델 파라미터 최적화
- 메모리 시스템 통합
"""

import logging
from typing import Any, AsyncIterator

from app.core.config import settings
from app.services.llm.base import (
    BaseLLMClient,
    LegalContext,
    LLMResponse,
    ModelRole,
    StreamingLLMClient,
)
from app.services.llm.legal_prompts_v4_1 import (
    get_legal_expert_prompt_kr,
    get_stage2_review_prompt_kr,
    get_model_parameters,
    LLMModel,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class GeminiClient(StreamingLLMClient):
    """
    Google Gemini 3 Pro client for legal consultation.

    Supports:
    - Standard generation
    - Grounded generation with Google File Search (RAG)

    v4.1 Features:
    - thinking_level 파라미터 (low/medium/high)
    - 정밀 인용 (예: 「근로기준법」 제23조 제1항)
    - 실무 해석 우선 (학설 < 실무)
    - 메모리 시스템 통합
    """

    def __init__(
        self,
        model_name: str | None = None,
        role: ModelRole = ModelRole.COUNCIL_MEMBER,
        temperature: float | None = None,
        thinking_level: str = "medium",
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ):
        model = model_name or settings.GEMINI_MODEL
        super().__init__(model_name=model, role=role)
        self.temperature = temperature or settings.GEMINI_TEMPERATURE
        self.api_key = settings.GOOGLE_API_KEY
        self.project_id = settings.GOOGLE_PROJECT_ID
        self.corpus_name = settings.GOOGLE_LEGAL_CORPUS_NAME
        self.thinking_level = thinking_level
        self.complexity = complexity

        # v4.1 파라미터 로드
        self._model_params = get_model_parameters(
            LLMModel.GEMINI_3_PRO.value, complexity
        )

    @property
    def display_name(self) -> str:
        return f"Gemini 3 Pro (thinking: {self.thinking_level})"

    def _get_council_member_prompt(self) -> str:
        """v4.1 한글 법률 전문가 프롬프트 반환"""
        return get_legal_expert_prompt_kr(
            model=LLMModel.GEMINI_3_PRO.value,
            session_memory="",
            short_term_memory="",
            long_term_memory="",
            rag_results="",
        )

    async def _initialize_client(self) -> None:
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized: {self.model_name}")
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate response using Gemini."""
        try:
            import google.generativeai as genai

            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )

            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = await self._client.generate_content_async(
                full_prompt,
                generation_config=generation_config,
            )

            # Extract content
            content = ""
            if response.candidates:
                content = response.candidates[0].content.parts[0].text

            # Calculate tokens (approximate)
            usage_metadata = getattr(response, "usage_metadata", None)
            total_tokens = 0
            if usage_metadata:
                total_tokens = (
                    getattr(usage_metadata, "prompt_token_count", 0) +
                    getattr(usage_metadata, "candidates_token_count", 0)
                )

            return LLMResponse(
                content=content,
                model=self.model_name,
                role=self.role,
                tokens_used=total_tokens,
                raw_response=response,
                metadata={
                    "finish_reason": (
                        response.candidates[0].finish_reason.name
                        if response.candidates else None
                    ),
                    "safety_ratings": (
                        [
                            {"category": r.category.name, "probability": r.probability.name}
                            for r in response.candidates[0].safety_ratings
                        ]
                        if response.candidates else []
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        try:
            import google.generativeai as genai

            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens,
            )

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = await self._client.generate_content_async(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise

    async def generate_with_grounding(
        self,
        context: LegalContext,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Generate response with Google File Search grounding.

        Uses the legal document corpus for RAG.
        """
        if not self._client:
            await self._initialize_client()

        try:
            import google.generativeai as genai

            if system_prompt is None:
                system_prompt = self._get_default_system_prompt()

            user_prompt = self._build_user_prompt(context)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Configure grounding with File Search
            # Note: This uses the Vertex AI semantic retrieval
            retrieval_config = genai.types.RetrievalConfig(
                mode=genai.types.RetrievalMode.CHUNKS,
                corpus_resource_name=(
                    f"projects/{self.project_id}/locations/global/"
                    f"corpora/{self.corpus_name}"
                ),
            )

            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=4096,
            )

            response = await self._client.generate_content_async(
                full_prompt,
                generation_config=generation_config,
                retrieval_config=retrieval_config,
            )

            # Extract grounding metadata
            grounding_chunks = []
            if hasattr(response, "grounding_metadata"):
                for chunk in getattr(response.grounding_metadata, "grounding_chunks", []):
                    grounding_chunks.append({
                        "source": getattr(chunk, "source", ""),
                        "content": getattr(chunk, "content", ""),
                    })

            content = ""
            if response.candidates:
                content = response.candidates[0].content.parts[0].text

            return LLMResponse(
                content=content,
                model=self.model_name,
                role=self.role,
                raw_response=response,
                metadata={
                    "grounded": True,
                    "grounding_chunks": grounding_chunks,
                    "corpus": self.corpus_name,
                },
            )

        except Exception as e:
            logger.error(f"Gemini grounding error: {e}")
            # Fallback to standard generation
            logger.info("Falling back to standard generation")
            return await self.generate(context, system_prompt)

    async def search_legal_corpus(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """
        Search the legal document corpus using semantic retrieval.

        Args:
            query: Search query
            top_k: Number of results
            category: Optional category filter

        Returns:
            List of relevant document chunks
        """
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine

            # Create search client
            client = discoveryengine.SearchServiceAsyncClient()

            # Build request
            request = discoveryengine.SearchRequest(
                serving_config=(
                    f"projects/{self.project_id}/locations/global/"
                    f"dataStores/{self.corpus_name}/servingConfigs/default_search"
                ),
                query=query,
                page_size=top_k,
            )

            # Add category filter if provided
            if category:
                request.filter = f'category = "{category}"'

            # Execute search
            response = await client.search(request)

            results = []
            async for result in response:
                doc = result.document
                results.append({
                    "id": doc.id,
                    "title": doc.struct_data.get("title", ""),
                    "content": doc.struct_data.get("content", ""),
                    "source": doc.struct_data.get("source", ""),
                    "doc_type": doc.struct_data.get("doc_type", ""),
                    "category": doc.struct_data.get("category", ""),
                    "relevance_score": result.relevance_score,
                })

            return results

        except ImportError:
            logger.warning(
                "google-cloud-discoveryengine not installed. "
                "RAG search disabled."
            )
            return []
        except Exception as e:
            logger.error(f"Corpus search error: {e}")
            return []
