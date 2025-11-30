"""
Documents API Endpoints

문서 업로드 및 관리 API
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import CurrentActiveUser, DBSession
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    FileSizeExceededError,
    InvalidFileTypeError,
    StorageQuotaExceededError,
    UnauthorizedError,
)
from app.services.document_service import DocumentService
from app.services.llm.rag_service import get_rag_service
from app.schemas.legal_document import (
    LegalDocumentType,
    LegalCategory,
    LegalDocumentMetadata,
    LegalDocumentCreate,
    LegalDocumentResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class DocumentResponse(BaseModel):
    """Document response."""

    id: str
    file_name: str
    original_name: str
    file_type: str
    file_size: int
    is_processed: bool
    ocr_completed: bool
    created_at: str


class DocumentDetailResponse(DocumentResponse):
    """Document detail with additional info."""

    mime_type: str
    description: str | None = None
    page_count: int | None = None
    ocr_confidence: float | None = None


class DocumentListResponse(BaseModel):
    """Paginated document list."""

    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    id: str
    file_name: str
    original_name: str
    message: str


class OCRResultResponse(BaseModel):
    """OCR result response."""

    document_id: str
    status: str  # processing, completed, failed
    extracted_text: str | None = None
    confidence: float | None = None
    page_count: int | None = None


class StorageStatsResponse(BaseModel):
    """Storage statistics response."""

    used_bytes: int
    quota_bytes: int
    used_percentage: float
    document_count: int
    file_size_limit: int


# =============================================================================
# Helper Functions
# =============================================================================


def _build_document_response(doc) -> DocumentResponse:
    """Build document response from model."""
    return DocumentResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        original_name=doc.original_name,
        file_type=doc.file_type,
        file_size=doc.file_size,
        is_processed=doc.is_processed,
        ocr_completed=doc.ocr_completed,
        created_at=doc.created_at.isoformat(),
    )


def _build_document_detail_response(doc) -> DocumentDetailResponse:
    """Build document detail response from model."""
    return DocumentDetailResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        original_name=doc.original_name,
        file_type=doc.file_type,
        file_size=doc.file_size,
        is_processed=doc.is_processed,
        ocr_completed=doc.ocr_completed,
        created_at=doc.created_at.isoformat(),
        mime_type=doc.mime_type,
        description=doc.description,
        page_count=doc.page_count,
        ocr_confidence=doc.ocr_confidence,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="문서 업로드",
    description="상담에 사용할 문서를 업로드합니다.",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="업로드할 문서 파일")],
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: CurrentActiveUser,
    description: str | None = Query(None, max_length=500, description="문서 설명"),
) -> DocumentUploadResponse:
    """
    Upload a document for consultation.

    Supported formats:
    - PDF (.pdf)
    - Images (.jpg, .jpeg, .png, .gif, .webp)
    - Word documents (.doc, .docx)
    - Excel spreadsheets (.xls, .xlsx)
    - Text files (.txt)

    Max file size varies by tier:
    - Basic: 5MB
    - Pro: 25MB
    - Enterprise: 100MB
    """
    service = DocumentService(db)

    try:
        # Read file content
        content = await file.read()

        # Upload document
        document = await service.upload_document(
            user=current_user,
            filename=file.filename or "unknown",
            content=content,
            content_type=file.content_type,
            description=description,
        )

        # Start OCR processing in background
        background_tasks.add_task(
            service.process_document_ocr,
            document.id,
        )

        return DocumentUploadResponse(
            id=str(document.id),
            file_name=document.file_name,
            original_name=document.original_name,
            message="문서가 업로드되었습니다. OCR 처리가 백그라운드에서 진행됩니다.",
        )

    except InvalidFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        )
    except FileSizeExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )
    except StorageQuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    finally:
        await service.close()


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="문서 목록 조회",
    description="사용자의 업로드된 문서 목록을 조회합니다.",
)
async def list_documents(
    db: DBSession,
    current_user: CurrentActiveUser,
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    file_type: str | None = Query(None, description="파일 타입 필터 (pdf, image, text, word, excel)"),
) -> DocumentListResponse:
    """
    List user's uploaded documents.
    """
    service = DocumentService(db)

    result = await service.list_documents(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        file_type=file_type,
    )

    return DocumentListResponse(
        items=[_build_document_response(doc) for doc in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        total_pages=result["total_pages"],
    )


@router.get(
    "/storage",
    response_model=StorageStatsResponse,
    summary="저장 공간 현황",
    description="사용자의 저장 공간 사용 현황을 조회합니다.",
)
async def get_storage_stats(
    db: DBSession,
    current_user: CurrentActiveUser,
) -> StorageStatsResponse:
    """
    Get storage usage statistics for the current user.
    """
    service = DocumentService(db)
    tier = current_user.tier.value if current_user.tier else "basic"

    stats = await service.get_storage_stats(current_user.id, tier)

    return StorageStatsResponse(**stats)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="문서 상세 조회",
    description="문서의 상세 정보를 조회합니다.",
)
async def get_document(
    document_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> DocumentDetailResponse:
    """
    Get document details.
    """
    service = DocumentService(db)

    try:
        document = await service.get_document(document_id, current_user.id)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다.",
            )

        return _build_document_detail_response(document)

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get(
    "/{document_id}/ocr",
    response_model=OCRResultResponse,
    summary="OCR 결과 조회",
    description="문서의 OCR 추출 결과를 조회합니다.",
)
async def get_ocr_result(
    document_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> OCRResultResponse:
    """
    Get OCR extraction result for a document.
    """
    service = DocumentService(db)

    try:
        result = await service.get_ocr_result(document_id, current_user.id)

        return OCRResultResponse(**result)

    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get(
    "/{document_id}/download",
    summary="문서 다운로드",
    description="문서를 다운로드합니다.",
)
async def download_document(
    document_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
    method: str = Query("redirect", description="다운로드 방식 (redirect, stream)"),
):
    """
    Download document.

    Methods:
    - redirect: Redirects to presigned S3 URL (default, recommended)
    - stream: Streams the file content directly
    """
    service = DocumentService(db)

    try:
        if method == "redirect":
            # Generate presigned URL and redirect
            url = await service.get_presigned_url(document_id, current_user.id)
            return RedirectResponse(url=url)

        elif method == "stream":
            # Stream file content directly
            content, filename, content_type = await service.download_document(
                document_id, current_user.id
            )

            return StreamingResponse(
                iter([content]),
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(content)),
                },
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="잘못된 다운로드 방식입니다. 'redirect' 또는 'stream'을 사용하세요.",
            )

    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    finally:
        await service.close()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="문서 삭제",
    description="문서를 삭제합니다 (소프트 삭제).",
)
async def delete_document(
    document_id: UUID,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> None:
    """
    Delete a document (soft delete).

    The document can be recovered by admin if needed.
    """
    service = DocumentService(db)

    try:
        await service.delete_document(document_id, current_user.id)

    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentResponse,
    summary="OCR 재처리",
    description="문서의 OCR 처리를 다시 수행합니다.",
)
async def reprocess_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: CurrentActiveUser,
) -> DocumentResponse:
    """
    Reprocess document OCR.

    Use this if the initial OCR extraction failed or produced poor results.
    """
    service = DocumentService(db)

    try:
        document = await service.get_document(document_id, current_user.id)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다.",
            )

        # Reset OCR status
        from app.repositories.document import DocumentRepository
        doc_repo = DocumentRepository(db)
        await doc_repo.update(
            document_id,
            is_processed=False,
            ocr_completed=False,
            extracted_text=None,
            ocr_confidence=None,
        )
        await db.commit()

        # Start OCR processing in background
        background_tasks.add_task(
            service.process_document_ocr,
            document_id,
        )

        # Refresh document
        document = await service.get_document(document_id, current_user.id)

        return _build_document_response(document)

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


# =============================================================================
# RAG (Legal Document Search) Endpoints
# =============================================================================


class RAGUploadRequest(BaseModel):
    """RAG 문서 업로드 요청."""

    title: str = Field(..., max_length=500, description="문서 제목")
    content: str = Field(..., min_length=10, description="문서 내용")
    doc_type: LegalDocumentType = Field(..., description="문서 타입")
    source: str = Field(..., description="출처")
    source_url: str | None = Field(default=None, description="출처 URL")
    category: LegalCategory = Field(default=LegalCategory.OTHER, description="법률 분야")
    keywords: list[str] = Field(default_factory=list, description="검색 키워드")
    summary: str | None = Field(default=None, max_length=2000, description="요약")

    # Optional legal-specific fields
    law_number: str | None = None
    law_name: str | None = None
    article_number: str | None = None
    case_number: str | None = None


class RAGUploadResponse(BaseModel):
    """RAG 문서 업로드 응답."""

    doc_id: str
    title: str
    message: str
    status: str


class RAGSearchResultItem(BaseModel):
    """RAG 검색 결과 항목."""

    doc_id: str
    title: str
    content: str
    doc_type: str
    source: str
    relevance_score: float
    snippet: str
    category: str
    case_number: str | None = None
    law_number: str | None = None


class RAGSearchResponseModel(BaseModel):
    """RAG 검색 응답."""

    query: str
    results: list[RAGSearchResultItem]
    total_results: int
    processing_time_ms: int


@router.post(
    "/rag/upload",
    response_model=RAGUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="RAG 법률 문서 업로드",
    description="법률 자문 검색에 사용할 문서를 RAG 시스템에 업로드합니다.",
)
async def upload_rag_document(
    request: RAGUploadRequest,
    current_user: CurrentActiveUser,
) -> RAGUploadResponse:
    """
    Upload a legal document to the RAG system.

    The document will be indexed and available for legal consultation searches.

    Supported document types:
    - law: 법률/법령
    - precedent: 판례
    - constitutional: 헌법재판소 결정
    - article: 법학 논문
    - commentary: 법률 해설
    """
    import uuid

    try:
        rag_service = get_rag_service()

        # Generate unique document ID
        doc_id = f"user_{current_user.id}_{uuid.uuid4().hex[:8]}"

        # Build metadata
        metadata = LegalDocumentMetadata(
            doc_id=doc_id,
            doc_type=request.doc_type,
            title=request.title,
            source=request.source,
            source_url=request.source_url,
            law_number=request.law_number,
            law_name=request.law_name,
            article_number=request.article_number,
            case_number=request.case_number,
            category=request.category,
            keywords=request.keywords,
            summary=request.summary,
        )

        # Upload to RAG system
        result = await rag_service.ingest_document(
            doc_id=doc_id,
            content=request.content,
            metadata=metadata,
        )

        return RAGUploadResponse(
            doc_id=doc_id,
            title=request.title,
            message="문서가 RAG 시스템에 성공적으로 업로드되었습니다.",
            status="indexed",
        )

    except Exception as e:
        logger.error(f"RAG upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 업로드 중 오류가 발생했습니다: {str(e)}",
        )


@router.post(
    "/rag/search",
    response_model=RAGSearchResponseModel,
    summary="RAG 법률 문서 검색",
    description="RAG 시스템에서 법률 문서를 검색합니다.",
)
async def search_rag_documents(
    query: str = Query(..., min_length=3, description="검색 쿼리"),
    category: str = Query("general", description="법률 분야"),
    doc_type: LegalDocumentType | None = Query(None, description="문서 타입 필터"),
    max_results: int = Query(10, ge=1, le=50, description="최대 결과 수"),
    current_user: CurrentActiveUser = None,
) -> RAGSearchResponseModel:
    """
    Search legal documents in the RAG system.

    Returns semantically similar documents based on the query.
    """
    try:
        rag_service = get_rag_service()

        response = await rag_service.search(
            query=query,
            category=category,
            doc_type=doc_type,
            top_k=max_results,
        )

        results = []
        for r in response.results:
            results.append(RAGSearchResultItem(
                doc_id=r.doc_id,
                title=r.title,
                content=r.content[:1000],  # Truncate for response
                doc_type=r.doc_type.value if hasattr(r.doc_type, 'value') else str(r.doc_type),
                source=r.source,
                relevance_score=r.relevance_score,
                snippet=r.snippet,
                category=r.category.value if hasattr(r.category, 'value') else str(r.category),
                case_number=r.case_number,
                law_number=r.law_number,
            ))

        return RAGSearchResponseModel(
            query=query,
            results=results,
            total_results=response.total_results,
            processing_time_ms=response.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}",
        )


@router.get(
    "/rag/stats",
    summary="RAG 시스템 통계",
    description="RAG 시스템의 통계 정보를 조회합니다.",
)
async def get_rag_stats(
    current_user: CurrentActiveUser,
) -> dict:
    """
    Get RAG system statistics.

    Returns information about the legal document corpus.
    """
    try:
        rag_service = get_rag_service()
        stats = await rag_service.get_stats()

        return {
            "corpus_id": stats.corpus_id,
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "documents_by_type": stats.documents_by_type,
            "documents_by_category": stats.documents_by_category,
            "last_upload": stats.last_upload.isoformat() if stats.last_upload else None,
        }

    except Exception as e:
        logger.error(f"RAG stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}",
        )


@router.delete(
    "/rag/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="RAG 문서 삭제",
    description="RAG 시스템에서 문서를 삭제합니다.",
)
async def delete_rag_document(
    doc_id: str,
    current_user: CurrentActiveUser,
) -> None:
    """
    Delete a document from the RAG system.

    Users can only delete documents they uploaded.
    """
    # Check ownership
    if not doc_id.startswith(f"user_{current_user.id}_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인이 업로드한 문서만 삭제할 수 있습니다.",
        )

    try:
        rag_service = get_rag_service()
        success = await rag_service.delete_document(doc_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG delete error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 삭제 중 오류가 발생했습니다: {str(e)}",
        )
