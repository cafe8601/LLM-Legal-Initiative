"""
Embedding Service

OpenRouter를 통한 임베딩 생성 서비스.
단일 API Key로 모든 LLM 및 임베딩 기능 통합.

지원 임베딩 모델 (OpenRouter):
- openai/text-embedding-3-small (1536 dim, 가성비)
- openai/text-embedding-3-large (3072 dim, 고성능)
- google/text-embedding-004 (768 dim)
"""

import asyncio
import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# 임베딩 모델 설정
EMBEDDING_MODELS = {
    "openai-small": {
        "model": "openai/text-embedding-3-small",
        "dimensions": 1536,
        "max_tokens": 8191,
        "cost_per_1m": 0.02,  # USD
    },
    "openai-large": {
        "model": "openai/text-embedding-3-large",
        "dimensions": 3072,
        "max_tokens": 8191,
        "cost_per_1m": 0.13,
    },
    "google": {
        "model": "google/text-embedding-004",
        "dimensions": 768,
        "max_tokens": 2048,
        "cost_per_1m": 0.00,  # Free tier
    },
}

# 기본 모델 (비용 효율적)
DEFAULT_EMBEDDING_MODEL = "openai-small"


class EmbeddingService:
    """
    OpenRouter 기반 임베딩 서비스.

    Features:
    - OpenRouter API를 통한 임베딩 생성
    - 배치 처리 지원
    - 자동 청킹 (긴 텍스트 분할)
    - 캐싱 지원 (Redis)
    """

    def __init__(
        self,
        model_key: str = DEFAULT_EMBEDDING_MODEL,
        cache_enabled: bool = True,
    ):
        """
        임베딩 서비스 초기화.

        Args:
            model_key: 사용할 모델 키 (openai-small, openai-large, google)
            cache_enabled: 캐싱 활성화 여부
        """
        self.model_config = EMBEDDING_MODELS.get(model_key, EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])
        self.model = self.model_config["model"]
        self.dimensions = self.model_config["dimensions"]
        self.max_tokens = self.model_config["max_tokens"]
        self.cache_enabled = cache_enabled
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 lazy 초기화."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "HTTP-Referer": settings.OPENROUTER_SITE_URL,
                    "X-Title": settings.OPENROUTER_APP_NAME,
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """
        단일 텍스트 임베딩 생성.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (float 리스트)
        """
        if not text.strip():
            return [0.0] * self.dimensions

        # 텍스트 길이 제한
        text = self._truncate_text(text)

        client = await self._get_client()

        try:
            response = await client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": text,
                },
            )
            response.raise_for_status()
            result = response.json()

            embedding = result["data"][0]["embedding"]
            logger.debug(f"Generated embedding: dim={len(embedding)}, model={self.model}")
            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"임베딩 생성 실패: {e.response.text}")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise RuntimeError(f"임베딩 생성 실패: {e}")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        배치 텍스트 임베딩 생성.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []

        # 빈 텍스트 필터링 및 인덱스 추적
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text.strip():
                valid_indices.append(i)
                valid_texts.append(self._truncate_text(text))

        if not valid_texts:
            return [[0.0] * self.dimensions for _ in texts]

        client = await self._get_client()

        try:
            # OpenRouter는 배치 임베딩 지원
            response = await client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": valid_texts,
                },
            )
            response.raise_for_status()
            result = response.json()

            # 결과 정렬 (API가 순서 보장하지 않을 수 있음)
            embeddings_map = {item["index"]: item["embedding"] for item in result["data"]}
            valid_embeddings = [embeddings_map[i] for i in range(len(valid_texts))]

            # 원래 인덱스에 맞게 결과 재구성
            full_embeddings = [[0.0] * self.dimensions for _ in texts]
            for idx, embedding in zip(valid_indices, valid_embeddings):
                full_embeddings[idx] = embedding

            logger.info(f"Batch embedding: {len(valid_texts)} texts, model={self.model}")
            return full_embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Batch embedding API error: {e.response.status_code}")
            # 폴백: 개별 처리
            return await self._embed_texts_sequential(texts)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise RuntimeError(f"배치 임베딩 생성 실패: {e}")

    async def _embed_texts_sequential(self, texts: list[str]) -> list[list[float]]:
        """배치 실패 시 순차 처리."""
        results = []
        for text in texts:
            try:
                embedding = await self.embed_text(text)
                results.append(embedding)
            except Exception:
                results.append([0.0] * self.dimensions)
        return results

    def _truncate_text(self, text: str) -> str:
        """텍스트를 최대 토큰 길이에 맞게 자름."""
        # 대략적인 토큰 추정 (1 토큰 ≈ 4 한글 글자 또는 4 영문 글자)
        max_chars = self.max_tokens * 3  # 보수적 추정
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Text truncated to {max_chars} chars")
        return text

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        긴 텍스트를 청크로 분할.

        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (글자 수)
            overlap: 청크 간 오버랩 (글자 수)

        Returns:
            청크 리스트
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마침표, 물음표, 느낌표 등에서 끊기
                for sep in ['. ', '? ', '! ', '.\n', '\n\n', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    async def close(self) -> None:
        """리소스 정리."""
        if self._client:
            await self._client.aclose()
            self._client = None


# 싱글톤 인스턴스
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """임베딩 서비스 싱글톤 반환."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
