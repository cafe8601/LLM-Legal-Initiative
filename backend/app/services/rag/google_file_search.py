"""
Google File Search Service

Google Gemini File Search API를 활용한 법률 문서 검색 서비스.
Phase 6: backend-rag 구현.

Google File Search는 자동으로:
- 문서 임베딩 생성
- 벡터 인덱싱
- 시맨틱 검색
- 출처 기반 응답 생성
을 처리합니다.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.schemas.legal_document import (
    LegalDocument,
    LegalDocumentMetadata,
    LegalDocumentType,
    LegalCategory,
    CourtType,
    RAGSearchResult,
    CorpusStats,
)

logger = logging.getLogger(__name__)


class GoogleFileSearchService:
    """
    Google Gemini File Search를 활용한 법률 문서 검색 서비스.

    Features:
    - Corpus 관리 (생성, 조회, 삭제)
    - 문서 업로드 (자동 임베딩)
    - 시맨틱 검색 (메타데이터 필터링)
    - Grounded Generation (RAG)
    """

    def __init__(self):
        """서비스 초기화."""
        self._genai = None
        self._types = None
        self.model_name = getattr(settings, "GEMINI_RAG_MODEL", "gemini-2.0-flash")
        self.corpus_name = getattr(
            settings,
            "GOOGLE_LEGAL_CORPUS_NAME",
            "legal-documents-corpus"
        )
        self._corpus = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of Google GenAI."""
        if self._initialized:
            return

        try:
            import google.generativeai as genai
            from google.generativeai import types

            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self._genai = genai
            self._types = types
            self._initialized = True
            logger.info("Google GenAI client initialized")
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI: {e}")
            raise

    async def initialize_corpus(self) -> str:
        """
        법률 문서 Corpus 생성 또는 기존 Corpus 연결.

        Returns:
            Corpus resource name
        """
        await self._ensure_initialized()

        try:
            # 기존 corpus 검색
            corpora = self._genai.list_corpora()
            for corpus in corpora:
                if corpus.display_name == self.corpus_name:
                    self._corpus = corpus
                    logger.info(f"Connected to existing corpus: {corpus.name}")
                    return corpus.name

            # 새 corpus 생성
            self._corpus = self._genai.create_corpus(
                display_name=self.corpus_name,
                description="한국 법률 문서 (법령, 판례, 헌법재판소 결정, 논문)"
            )
            logger.info(f"Created new corpus: {self._corpus.name}")
            return self._corpus.name

        except Exception as e:
            logger.error(f"Corpus initialization failed: {e}")
            raise RuntimeError(f"Corpus 초기화 실패: {e}")

    async def upload_document(self, document: LegalDocument) -> dict:
        """
        법률 문서를 Google File Search에 업로드.

        Google이 자동으로 임베딩을 생성하고 인덱싱합니다.

        Args:
            document: 업로드할 법률 문서

        Returns:
            업로드된 문서 정보
        """
        if not self._corpus:
            await self.initialize_corpus()

        # 메타데이터를 문서 앞에 추가 (검색 품질 향상)
        formatted_content = self._format_document_with_metadata(document)

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(formatted_content)
            temp_path = f.name

        try:
            # 파일 업로드
            uploaded_file = self._genai.upload_file(
                path=temp_path,
                display_name=document.file_name,
                mime_type=document.mime_type
            )

            # Custom metadata 구성
            custom_metadata = self._build_custom_metadata(document.metadata)

            # Corpus에 문서 추가 (자동 청킹 및 임베딩)
            doc = self._genai.create_document(
                corpus=self._corpus.name,
                display_name=document.doc_id,
                document=uploaded_file,
                custom_metadata=custom_metadata
            )

            logger.info(f"Document uploaded: {document.doc_id}")

            return {
                "doc_id": document.doc_id,
                "document_name": doc.name,
                "file_name": document.file_name,
                "status": "uploaded",
                "uploaded_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise RuntimeError(f"문서 업로드 실패: {e}")

        finally:
            # 임시 파일 삭제
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _build_custom_metadata(
        self,
        metadata: LegalDocumentMetadata
    ) -> list:
        """문서 메타데이터를 Google CustomMetadata 형식으로 변환."""
        custom_metadata = [
            self._types.CustomMetadata(
                key="doc_type",
                string_value=metadata.doc_type if isinstance(metadata.doc_type, str)
                else metadata.doc_type.value
            ),
            self._types.CustomMetadata(
                key="category",
                string_value=metadata.category if isinstance(metadata.category, str)
                else metadata.category.value
            ),
            self._types.CustomMetadata(
                key="source",
                string_value=metadata.source
            ),
            self._types.CustomMetadata(
                key="title",
                string_value=metadata.title
            ),
        ]

        # Optional fields
        if metadata.case_number:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="case_number",
                    string_value=metadata.case_number
                )
            )

        if metadata.law_number:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="law_number",
                    string_value=metadata.law_number
                )
            )

        if metadata.article_number:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="article_number",
                    string_value=metadata.article_number
                )
            )

        if metadata.court:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="court",
                    string_value=metadata.court if isinstance(metadata.court, str)
                    else metadata.court.value
                )
            )

        if metadata.decision_date:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="decision_date",
                    string_value=metadata.decision_date.isoformat()
                )
            )

        if metadata.keywords:
            custom_metadata.append(
                self._types.CustomMetadata(
                    key="keywords",
                    string_value=",".join(metadata.keywords)
                )
            )

        return custom_metadata

    def _format_document_with_metadata(self, document: LegalDocument) -> str:
        """문서에 검색 가능한 메타데이터 헤더 추가."""
        meta = document.metadata

        # 문서 타입/카테고리 값 추출
        doc_type_val = meta.doc_type if isinstance(meta.doc_type, str) else meta.doc_type.value
        category_val = meta.category if isinstance(meta.category, str) else meta.category.value

        header_parts = [
            f"[문서 유형: {doc_type_val}]",
            f"[제목: {meta.title}]",
            f"[출처: {meta.source}]",
            f"[분야: {category_val}]",
        ]

        if meta.case_number:
            header_parts.append(f"[사건번호: {meta.case_number}]")
        if meta.court:
            court_val = meta.court if isinstance(meta.court, str) else meta.court.value
            header_parts.append(f"[법원: {court_val}]")
        if meta.decision_date:
            header_parts.append(f"[선고일: {meta.decision_date}]")
        if meta.law_number:
            header_parts.append(f"[법률번호: {meta.law_number}]")
        if meta.law_name:
            header_parts.append(f"[법률명: {meta.law_name}]")
        if meta.article_number:
            header_parts.append(f"[조문번호: {meta.article_number}]")
        if meta.keywords:
            header_parts.append(f"[키워드: {', '.join(meta.keywords)}]")
        if meta.summary:
            header_parts.append(f"[요약: {meta.summary}]")

        header = "\n".join(header_parts)
        return f"{header}\n\n---\n\n{document.content}"

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        doc_type: Optional[LegalDocumentType] = None,
        max_results: int = 10,
        min_relevance: float = 0.0
    ) -> list[dict]:
        """
        Google File Search를 사용한 법률 문서 검색.

        Args:
            query: 검색 쿼리
            category: 법률 분야 필터 (민사, 형사 등)
            doc_type: 문서 유형 필터
            max_results: 최대 결과 수
            min_relevance: 최소 관련성 점수

        Returns:
            검색 결과 리스트
        """
        if not self._corpus:
            await self.initialize_corpus()

        # 필터 조건 구성
        metadata_filters = []
        if category:
            metadata_filters.append(
                self._types.MetadataFilter(
                    key="category",
                    conditions=[self._types.Condition(
                        operation=self._types.Condition.Operator.EQUALS,
                        string_value=category
                    )]
                )
            )
        if doc_type:
            doc_type_value = doc_type if isinstance(doc_type, str) else doc_type.value
            metadata_filters.append(
                self._types.MetadataFilter(
                    key="doc_type",
                    conditions=[self._types.Condition(
                        operation=self._types.Condition.Operator.EQUALS,
                        string_value=doc_type_value
                    )]
                )
            )

        try:
            # Corpus에서 검색
            results = self._genai.query_corpus(
                corpus=self._corpus.name,
                query=query,
                results_count=max_results,
                metadata_filters=metadata_filters if metadata_filters else None
            )

            # 결과 포맷팅
            formatted_results = []
            for chunk in results.relevant_chunks:
                relevance_score = chunk.chunk_relevance_score
                if relevance_score < min_relevance:
                    continue

                formatted_results.append({
                    "content": chunk.chunk.data.string_value,
                    "relevance_score": relevance_score,
                    "document_name": chunk.chunk.document_metadata.document_name,
                    "source": self._extract_metadata(chunk, "source"),
                    "title": self._extract_metadata(chunk, "title"),
                    "doc_type": self._extract_metadata(chunk, "doc_type"),
                    "category": self._extract_metadata(chunk, "category"),
                    "case_number": self._extract_metadata(chunk, "case_number"),
                    "law_number": self._extract_metadata(chunk, "law_number"),
                    "article_number": self._extract_metadata(chunk, "article_number"),
                    "court": self._extract_metadata(chunk, "court"),
                    "decision_date": self._extract_metadata(chunk, "decision_date"),
                    "keywords": self._extract_metadata(chunk, "keywords"),
                })

            logger.debug(f"Search returned {len(formatted_results)} results for: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _extract_metadata(self, chunk, key: str) -> str:
        """청크에서 메타데이터 추출."""
        try:
            for meta in chunk.chunk.document_metadata.custom_metadata:
                if meta.key == key:
                    return meta.string_value
        except Exception:
            pass
        return ""

    async def search_with_generation(
        self,
        query: str,
        category: Optional[str] = None,
        doc_type: Optional[LegalDocumentType] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 4096
    ) -> dict:
        """
        검색 결과를 기반으로 답변 생성 (RAG/Grounded Generation).

        Google File Search의 grounding 기능을 활용하여
        검색된 문서를 기반으로 답변을 생성합니다.

        Args:
            query: 사용자 질문
            category: 법률 분야 필터
            doc_type: 문서 유형 필터
            system_instruction: 시스템 프롬프트
            temperature: 생성 온도
            max_output_tokens: 최대 출력 토큰

        Returns:
            생성된 답변과 출처 정보
        """
        if not self._corpus:
            await self.initialize_corpus()

        # 기본 시스템 프롬프트
        if not system_instruction:
            system_instruction = """당신은 한국 법률 전문가입니다.
검색된 법률 문서를 기반으로 정확하고 신뢰할 수 있는 답변을 제공합니다.

## 응답 지침
1. 답변 시 반드시 관련 법률 조문이나 판례를 인용하고 출처를 명시하세요.
2. 확실하지 않은 내용에 대해서는 명확히 그 한계를 밝히세요.
3. 법률 용어는 정확하게 사용하되, 일반인이 이해할 수 있도록 설명을 추가하세요.
4. 가능한 경우 관련 판례의 사건번호를 함께 제시하세요.

## 인용 형식
- 법률: 「법률명」 제○조 제○항
- 판례: 대법원 20XX. XX. XX. 선고 20XX다XXXXX 판결"""

        try:
            # 모델 생성 with File Search 도구
            model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                tools=[
                    self._types.Tool(
                        retrieval=self._types.Retrieval(
                            source=self._types.SemanticRetriever(
                                source=self._corpus.name
                            )
                        )
                    )
                ],
                generation_config=self._genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )

            # 응답 생성
            response = model.generate_content(query)

            # 출처 정보 추출
            citations = []
            grounding_chunks = []

            if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
                for chunk in getattr(
                    response.grounding_metadata,
                    'grounding_chunks',
                    []
                ):
                    citation_info = {}
                    if hasattr(chunk, 'retrieved_context'):
                        citation_info = {
                            "title": getattr(chunk.retrieved_context, 'title', ''),
                            "uri": getattr(chunk.retrieved_context, 'uri', ''),
                            "content": getattr(chunk.retrieved_context, 'text', '')
                        }
                    elif hasattr(chunk, 'web'):
                        citation_info = {
                            "title": getattr(chunk.web, 'title', ''),
                            "uri": getattr(chunk.web, 'uri', ''),
                            "content": ''
                        }

                    if citation_info:
                        citations.append(citation_info)
                        grounding_chunks.append(citation_info)

            return {
                "answer": response.text,
                "citations": citations,
                "grounding_chunks": grounding_chunks,
                "model": self.model_name,
                "corpus": self.corpus_name
            }

        except Exception as e:
            logger.error(f"Grounded generation failed: {e}")
            raise RuntimeError(f"RAG 답변 생성 실패: {e}")

    async def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제.

        Args:
            doc_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """
        if not self._corpus:
            await self.initialize_corpus()

        try:
            # Corpus 내 문서 검색
            documents = self._genai.list_documents(corpus=self._corpus.name)
            for doc in documents:
                if doc.display_name == doc_id:
                    self._genai.delete_document(name=doc.name)
                    logger.info(f"Document deleted: {doc_id}")
                    return True
            logger.warning(f"Document not found: {doc_id}")
            return False
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            raise RuntimeError(f"문서 삭제 실패: {e}")

    async def list_documents(
        self,
        category: Optional[str] = None,
        doc_type: Optional[LegalDocumentType] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Corpus 내 문서 목록 조회.

        Args:
            category: 카테고리 필터
            doc_type: 문서 타입 필터
            limit: 최대 결과 수

        Returns:
            문서 목록
        """
        if not self._corpus:
            await self.initialize_corpus()

        try:
            documents = self._genai.list_documents(corpus=self._corpus.name)

            result = []
            count = 0

            for doc in documents:
                if count >= limit:
                    break

                doc_info = {
                    "doc_id": doc.display_name,
                    "name": doc.name,
                    "create_time": str(doc.create_time) if doc.create_time else None,
                    "update_time": str(doc.update_time) if doc.update_time else None,
                    "metadata": {}
                }

                # 메타데이터 추출
                for meta in getattr(doc, 'custom_metadata', []):
                    doc_info["metadata"][meta.key] = meta.string_value

                # 필터 적용
                if category and doc_info["metadata"].get("category") != category:
                    continue
                if doc_type:
                    doc_type_value = doc_type if isinstance(doc_type, str) else doc_type.value
                    if doc_info["metadata"].get("doc_type") != doc_type_value:
                        continue

                result.append(doc_info)
                count += 1

            return result

        except Exception as e:
            logger.error(f"List documents failed: {e}")
            return []

    async def get_corpus_stats(self) -> CorpusStats:
        """
        Corpus 통계 정보 조회.

        Returns:
            Corpus 통계 정보
        """
        if not self._corpus:
            await self.initialize_corpus()

        try:
            documents = self._genai.list_documents(corpus=self._corpus.name)
            doc_list = list(documents)

            # 문서 유형별/카테고리별 통계
            type_counts: dict[str, int] = {}
            category_counts: dict[str, int] = {}
            last_upload = None

            for doc in doc_list:
                for meta in getattr(doc, 'custom_metadata', []):
                    if meta.key == "doc_type":
                        type_counts[meta.string_value] = type_counts.get(
                            meta.string_value, 0
                        ) + 1
                    if meta.key == "category":
                        category_counts[meta.string_value] = category_counts.get(
                            meta.string_value, 0
                        ) + 1

                # 최신 업로드 시간 추적
                if doc.create_time:
                    doc_time = doc.create_time
                    if last_upload is None or doc_time > last_upload:
                        last_upload = doc_time

            return CorpusStats(
                corpus_id=self._corpus.name,
                total_documents=len(doc_list),
                total_chunks=0,  # Google API에서 직접 제공하지 않음
                documents_by_type=type_counts,
                documents_by_category=category_counts,
                last_upload=last_upload
            )

        except Exception as e:
            logger.error(f"Get corpus stats failed: {e}")
            return CorpusStats(
                corpus_id=self._corpus.name if self._corpus else "",
                total_documents=0,
                total_chunks=0,
                documents_by_type={},
                documents_by_category={}
            )

    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[LegalDocumentMetadata] = None
    ) -> dict:
        """
        문서 업데이트 (삭제 후 재업로드).

        Google File Search는 문서 업데이트를 직접 지원하지 않으므로
        삭제 후 재업로드 방식으로 구현합니다.

        Args:
            doc_id: 문서 ID
            content: 새 내용 (None이면 기존 내용 유지)
            metadata: 새 메타데이터 (None이면 기존 메타데이터 유지)

        Returns:
            업데이트 결과
        """
        # 기존 문서 정보 조회
        docs = await self.list_documents()
        existing_doc = None
        for doc in docs:
            if doc["doc_id"] == doc_id:
                existing_doc = doc
                break

        if not existing_doc:
            raise ValueError(f"Document not found: {doc_id}")

        # 삭제
        await self.delete_document(doc_id)

        # 메타데이터 구성
        if metadata is None:
            # 기존 메타데이터에서 복원
            existing_meta = existing_doc.get("metadata", {})
            metadata = LegalDocumentMetadata(
                doc_id=doc_id,
                doc_type=LegalDocumentType(existing_meta.get("doc_type", "law")),
                title=existing_meta.get("title", ""),
                source=existing_meta.get("source", ""),
                category=LegalCategory(existing_meta.get("category", "other")),
                case_number=existing_meta.get("case_number"),
                law_number=existing_meta.get("law_number"),
                article_number=existing_meta.get("article_number"),
            )

        if content is None:
            raise ValueError("Content is required for update (original content not stored)")

        # 재업로드
        document = LegalDocument(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            file_name=f"{doc_id}.txt"
        )

        return await self.upload_document(document)


# Singleton instance
_file_search_service: Optional[GoogleFileSearchService] = None


def get_file_search_service() -> GoogleFileSearchService:
    """GoogleFileSearchService 싱글톤 인스턴스 반환."""
    global _file_search_service
    if _file_search_service is None:
        _file_search_service = GoogleFileSearchService()
    return _file_search_service
