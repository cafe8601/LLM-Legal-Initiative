"""
Consultation Service

법률 상담 비즈니스 로직.
v4.1: 메모리 시스템 및 TaskComplexity 지원.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConsultationLimitExceededError,
    ConsultationNotFoundError,
    UnauthorizedError,
)
from app.core.rate_limit import ConsultationLimiter, TierLimits
from app.models.consultation import (
    Consultation,
    ConsultationCategory,
    ConsultationStatus,
    ConsultationTurn,
    ModelOpinion,
    PeerReview,
)
from app.models.user import User
from app.repositories.consultation import (
    ConsultationRepository,
    ConsultationTurnRepository,
    ModelOpinionRepository,
    PeerReviewRepository,
)
from app.repositories.document import CitationRepository
from app.schemas.consultation import (
    ConsultationCreate,
    ConsultationTurnCreate,
    ConsultationResponse,
    ConsultationDetailResponse,
    ConsultationTurnResponse,
)
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """v4.1 Memory context for LLM prompts."""

    session_memory: str = ""
    short_term_memory: str = ""
    long_term_memory: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for LLM injection."""
        return {
            "session_memory": self.session_memory,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
        }


class ConsultationService:
    """Service for managing legal consultations."""

    def __init__(
        self,
        db: AsyncSession,
        redis: Redis | None = None,
    ):
        self.db = db
        self.redis = redis
        self.consultation_repo = ConsultationRepository(db)
        self.turn_repo = ConsultationTurnRepository(db)
        self.opinion_repo = ModelOpinionRepository(db)
        self.review_repo = PeerReviewRepository(db)
        self.citation_repo = CitationRepository(db)
        self.memory_service = MemoryService(db)

    # =========================================================================
    # Consultation Limit Management
    # =========================================================================

    async def check_consultation_limit(self, user: User) -> bool:
        """
        Check if user can create a new consultation.

        Raises:
            ConsultationLimitExceededError: If monthly limit exceeded
        """
        limit = TierLimits.get_limit(user.tier.value, "consultations_per_month")

        if TierLimits.is_unlimited(limit):
            return True

        # Get current month usage
        current_count = await self.consultation_repo.count_user_consultations_this_month(
            user.id
        )

        if current_count >= limit:
            raise ConsultationLimitExceededError(
                f"월간 상담 한도({limit}회)를 초과했습니다. "
                f"Pro 또는 Enterprise 요금제로 업그레이드해주세요."
            )

        return True

    async def get_remaining_consultations(self, user: User) -> dict:
        """Get remaining consultation count for user."""
        limit = TierLimits.get_limit(user.tier.value, "consultations_per_month")
        current_count = await self.consultation_repo.count_user_consultations_this_month(
            user.id
        )

        return {
            "limit": limit,
            "used": current_count,
            "remaining": max(0, limit - current_count) if limit > 0 else -1,
            "is_unlimited": TierLimits.is_unlimited(limit),
        }

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create_consultation(
        self,
        user_id: UUID,
        data: ConsultationCreate,
    ) -> Consultation:
        """Create a new consultation with initial turn."""
        # Create consultation
        consultation = Consultation(
            user_id=user_id,
            title=data.title,
            category=ConsultationCategory(data.category),
            status=ConsultationStatus.PENDING,
        )

        consultation = await self.consultation_repo.create(
            user_id=user_id,
            title=data.title,
            category=data.category,
            status=ConsultationStatus.PENDING.value,
        )

        # Create initial turn
        turn = await self.turn_repo.create(
            consultation_id=consultation.id,
            turn_number=1,
            user_query=data.initial_query,
            status=ConsultationStatus.PENDING.value,
        )

        # Update consultation turn count
        consultation.turn_count = 1
        await self.db.commit()
        await self.db.refresh(consultation)

        logger.info(
            f"Consultation created: {consultation.id} for user {user_id}"
        )

        return consultation

    async def get_consultation(
        self,
        consultation_id: UUID,
        user_id: UUID,
    ) -> Consultation | None:
        """Get consultation with ownership check."""
        consultation = await self.consultation_repo.get_with_turns(consultation_id)

        if not consultation:
            return None

        if consultation.user_id != user_id:
            raise UnauthorizedError("이 상담에 접근할 권한이 없습니다.")

        return consultation

    async def get_consultation_detail(
        self,
        consultation_id: UUID,
        user_id: UUID,
    ) -> ConsultationDetailResponse | None:
        """Get detailed consultation with all turns."""
        consultation = await self.get_consultation(consultation_id, user_id)

        if not consultation:
            return None

        # Build response with turns
        turns = []
        for turn in consultation.turns:
            turn_response = await self._build_turn_response(turn)
            turns.append(turn_response)

        return ConsultationDetailResponse(
            id=str(consultation.id),
            title=consultation.title,
            category=consultation.category.value if consultation.category else "general",
            status=consultation.status.value if consultation.status else "pending",
            summary=consultation.summary,
            turn_count=consultation.turn_count or 0,
            created_at=consultation.created_at.isoformat(),
            updated_at=consultation.updated_at.isoformat(),
            turns=turns,
        )

    async def list_consultations(
        self,
        user_id: UUID,
        page: int = 1,
        page_size: int = 10,
        status: str | None = None,
        category: str | None = None,
    ) -> dict:
        """List user's consultations with pagination."""
        skip = (page - 1) * page_size

        consultations = await self.consultation_repo.get_user_consultations(
            user_id=user_id,
            skip=skip,
            limit=page_size,
            status=status,
            category=category,
        )

        total = await self.consultation_repo.count_user_consultations(user_id)
        total_pages = (total + page_size - 1) // page_size

        items = [
            ConsultationResponse(
                id=str(c.id),
                title=c.title,
                category=c.category.value if c.category else "general",
                status=c.status.value if c.status else "pending",
                summary=c.summary,
                turn_count=c.turn_count or 0,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in consultations
        ]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def add_turn(
        self,
        consultation_id: UUID,
        user_id: UUID,
        data: ConsultationTurnCreate,
    ) -> ConsultationTurn:
        """Add a follow-up question to consultation."""
        consultation = await self.get_consultation(consultation_id, user_id)

        if not consultation:
            raise ConsultationNotFoundError("상담을 찾을 수 없습니다.")

        # Check if previous turn is complete
        if consultation.status not in [
            ConsultationStatus.COMPLETED,
            ConsultationStatus.FAILED,
        ]:
            raise ValueError("이전 질문이 아직 처리 중입니다.")

        # Get next turn number
        next_turn = await self.turn_repo.get_next_turn_number(consultation_id)

        # Create new turn
        turn = await self.turn_repo.create(
            consultation_id=consultation_id,
            turn_number=next_turn,
            user_query=data.query,
            status=ConsultationStatus.PENDING.value,
        )

        # Update consultation
        await self.consultation_repo.update(
            consultation_id,
            turn_count=next_turn,
            status=ConsultationStatus.PENDING.value,
        )

        logger.info(f"Turn {next_turn} added to consultation {consultation_id}")

        return turn

    async def delete_consultation(
        self,
        consultation_id: UUID,
        user_id: UUID,
    ) -> None:
        """Delete a consultation and all related data."""
        consultation = await self.get_consultation(consultation_id, user_id)

        if not consultation:
            raise ConsultationNotFoundError("상담을 찾을 수 없습니다.")

        # Delete (soft delete or hard delete based on policy)
        await self.consultation_repo.delete(consultation_id)

        logger.info(f"Consultation {consultation_id} deleted by user {user_id}")

    # =========================================================================
    # Turn Processing
    # =========================================================================

    async def start_turn_processing(
        self,
        consultation_id: UUID,
        turn_number: int,
        user_id: UUID | None = None,
        memory_context: Optional[MemoryContext] = None,
        complexity: str = "medium",
    ) -> None:
        """
        Start processing a turn (called as background task).

        Args:
            consultation_id: Consultation UUID
            turn_number: Turn number to process
            user_id: User UUID for memory system
            memory_context: v4.1 memory context (session/short-term/long-term)
            complexity: Task complexity for v4.1 parameter optimization
        """
        consultation = await self.consultation_repo.get(consultation_id)
        if not consultation:
            return

        turns = await self.turn_repo.get_consultation_turns(
            consultation_id, include_details=False
        )
        turn = next((t for t in turns if t.turn_number == turn_number), None)
        if not turn:
            return

        # Get user_id from consultation if not provided
        if user_id is None:
            user_id = consultation.user_id

        # Load memory context from MemoryService if not provided
        if memory_context is None:
            memory_dict = await self.memory_service.get_memory_context(
                user_id=user_id,
                consultation_id=consultation_id,
                category=consultation.category.value if consultation.category else None,
            )
            memory_context = MemoryContext(
                session_memory=memory_dict["session_memory"],
                short_term_memory=memory_dict["short_term_memory"],
                long_term_memory=memory_dict["long_term_memory"],
            )

        try:
            # Update status
            await self._update_turn_status(turn.id, ConsultationStatus.ANALYZING)
            await self._publish_progress(consultation_id, "analyzing", 10)

            # RAG Search with v4.1 integration
            await self._publish_log(consultation_id, "법률 데이터베이스 검색 중...")
            await self._publish_progress(consultation_id, "analyzing", 25)

            # Get RAG context
            rag_results = await self._get_rag_context(
                turn.user_query,
                consultation.category.value if consultation.category else "general",
            )

            # Build LegalContext for Council
            from app.services.llm.base import LegalContext
            from app.services.llm.council import CouncilOrchestrator

            legal_context = LegalContext(
                query=turn.user_query,
                category=consultation.category.value if consultation.category else "general",
                conversation_history=await self._get_conversation_history(consultation_id, turn_number),
                rag_context=self._parse_rag_results(rag_results),
                user_tier="pro",  # TODO: Get from user
            )

            # Create Council Orchestrator with v4.1 memory system
            council = CouncilOrchestrator(
                enable_peer_review=True,
                max_concurrent=4,
                session_memory=memory_context.session_memory,
                short_term_memory=memory_context.short_term_memory,
                long_term_memory=memory_context.long_term_memory,
            )

            # Progress callback for real-time updates
            async def progress_callback(stage: str, progress: float, message: str):
                stage_mapping = {
                    "stage1": ("drafting", 30 + int(progress * 20)),
                    "stage2": ("reviewing", 50 + int(progress * 25)),
                    "stage3": ("synthesizing", 75 + int(progress * 20)),
                }
                step, pct = stage_mapping.get(stage, ("processing", int(progress * 100)))
                await self._publish_progress(consultation_id, step, pct)
                await self._publish_log(consultation_id, message)

            # Stage 1: LLM Opinions with v4.1 memory injection
            await self._update_turn_status(turn.id, ConsultationStatus.DRAFTING)
            await self._publish_log(consultation_id, "AI 법률 자문단 의견 생성 중 (v4.1 메모리 시스템 활성화)...")
            await self._publish_progress(consultation_id, "drafting", 30)

            logger.info(
                f"Turn {turn_number} processing with v4.1: "
                f"complexity={complexity}, "
                f"session_memory_len={len(memory_context.session_memory)}, "
                f"rag_results_len={len(rag_results)}"
            )

            # Run Council consultation (Stage 1 → Stage 2 → Stage 3)
            council_result = await council.consult(
                context=legal_context,
                progress_callback=progress_callback,
            )

            # Save opinions to database
            for opinion in council_result.opinions:
                if not opinion.error:
                    await self.opinion_repo.create(
                        turn_id=turn.id,
                        model_name=opinion.display_name,
                        model_version=opinion.model,
                        opinion_text=opinion.content,
                        legal_basis=opinion.legal_basis,
                        risk_assessment=opinion.risk_assessment,
                        recommendations=opinion.recommendations,
                        confidence_level=opinion.confidence_level,
                        tokens_input=0,
                        tokens_output=opinion.tokens_used,
                        processing_time_ms=opinion.latency_ms,
                    )

            # Save peer reviews to database
            for review in council_result.reviews:
                # Find the opinion being reviewed
                reviewed_opinion = next(
                    (o for o in council_result.opinions if o.display_name == review.reviewed_model),
                    None
                )
                if reviewed_opinion:
                    opinion_record = await self.opinion_repo.get_by_model(
                        turn.id, reviewed_opinion.display_name
                    )
                    if opinion_record:
                        await self.review_repo.create(
                            turn_id=turn.id,
                            reviewed_opinion_id=opinion_record.id,
                            reviewer_model=review.reviewer_model,
                            review_text=review.content,
                            accuracy_score=review.scores.get("정확성"),
                            completeness_score=review.scores.get("완전성"),
                            practicality_score=review.scores.get("유용성"),
                            legal_basis_score=review.scores.get("명확성"),
                            overall_score=review.scores.get("종합"),
                        )

            # Save chairman synthesis
            if council_result.synthesis:
                await self.turn_repo.update(
                    turn.id,
                    chairman_response=council_result.synthesis.content,
                    processing_time_ms=council_result.total_latency_ms,
                    tokens_used=council_result.total_tokens_used,
                )

            # Complete
            await self._update_turn_status(turn.id, ConsultationStatus.COMPLETED)
            await self.consultation_repo.update(
                consultation_id, status=ConsultationStatus.COMPLETED.value
            )
            await self._publish_progress(consultation_id, "complete", 100)

            # Save conversation to memory system
            await self._save_turn_to_memory(
                user_id=user_id,
                consultation_id=consultation_id,
                turn=turn,
                chairman_response=council_result.synthesis.content if council_result.synthesis else None,
                category=consultation.category.value if consultation.category else "general",
            )

            # Log errors if any
            if council_result.errors:
                for error in council_result.errors:
                    logger.warning(f"Council error: {error}")

            logger.info(
                f"Turn {turn_number} processing complete for {consultation_id}. "
                f"Opinions: {len(council_result.opinions)}, "
                f"Reviews: {len(council_result.reviews)}, "
                f"Total latency: {council_result.total_latency_ms}ms"
            )

        except Exception as e:
            logger.error(f"Turn processing failed: {e}")
            await self._update_turn_status(turn.id, ConsultationStatus.FAILED)
            await self.consultation_repo.update(
                consultation_id, status=ConsultationStatus.FAILED.value
            )
            await self._publish_error(consultation_id, str(e))
            raise

    async def _get_rag_context(
        self,
        query: str,
        category: str,
    ) -> str:
        """
        Get RAG context for the query.

        Returns formatted RAG results for injection into v4.1 prompts.
        """
        try:
            from app.services.llm.rag_service import get_rag_service

            rag_service = get_rag_service()
            context = await rag_service.get_context_for_llm(
                query=query,
                category=category,
                max_tokens=8000,
            )
            return context
        except Exception as e:
            logger.warning(f"RAG context fetch failed: {e}")
            return "[RAG 결과 없음 - 일반 법률 지식 기반 응답]"

    async def _get_conversation_history(
        self,
        consultation_id: UUID,
        current_turn: int,
    ) -> list[dict]:
        """Get conversation history for context."""
        try:
            turns = await self.turn_repo.get_consultation_turns(
                consultation_id, include_details=False
            )
            history = []
            for turn in turns:
                if turn.turn_number < current_turn:
                    history.append({
                        "role": "user",
                        "content": turn.user_query,
                    })
                    if turn.chairman_response:
                        history.append({
                            "role": "assistant",
                            "content": turn.chairman_response,
                        })
            return history
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
            return []

    def _parse_rag_results(self, rag_text: str) -> list[dict]:
        """Parse RAG results string into structured format."""
        if not rag_text or "[RAG 결과 없음" in rag_text:
            return []

        # Simple parsing - RAG service returns formatted text
        # Convert to list of dicts for LegalContext
        results = []
        sections = rag_text.split("[참고문헌")
        for section in sections[1:]:  # Skip first empty part
            try:
                # Extract source and content
                lines = section.strip().split("\n")
                if lines:
                    source_line = lines[0] if lines else ""
                    content = "\n".join(lines[1:]) if len(lines) > 1 else ""
                    results.append({
                        "source": source_line,
                        "content": content,
                        "doc_type": "legal_reference",
                    })
            except Exception:
                continue
        return results

    async def _save_turn_to_memory(
        self,
        user_id: UUID,
        consultation_id: UUID,
        turn: ConsultationTurn,
        chairman_response: str | None,
        category: str,
    ) -> None:
        """
        Save completed turn to memory system.

        Stores both the user query and assistant response for:
        - Session memory context building
        - Short-term memory (7-day recall)
        - Long-term pattern learning
        """
        try:
            # Save user query to conversation history
            await self.memory_service.save_conversation_turn(
                user_id=user_id,
                consultation_id=consultation_id,
                turn_id=turn.id,
                turn_number=turn.turn_number,
                role="user",
                content=turn.user_query,
                category=category,
                keywords=self._extract_keywords(turn.user_query),
            )

            # Save assistant response to conversation history
            if chairman_response:
                await self.memory_service.save_conversation_turn(
                    user_id=user_id,
                    consultation_id=consultation_id,
                    turn_id=turn.id,
                    turn_number=turn.turn_number,
                    role="assistant",
                    content=chairman_response,
                    category=category,
                )

            # Save session memory for important context
            await self.memory_service.save_session_memory(
                user_id=user_id,
                consultation_id=consultation_id,
                key=f"turn_{turn.turn_number}_summary",
                content=f"질문: {turn.user_query[:200]}...",
                category=category,
            )

            # Learn patterns for long-term memory
            keywords = self._extract_keywords(turn.user_query)
            await self.memory_service.learn_from_consultation(
                user_id=user_id,
                consultation_id=consultation_id,
                category=category,
                keywords=keywords,
            )

            logger.info(f"Turn {turn.turn_number} saved to memory system")

        except Exception as e:
            # Memory save failure shouldn't fail the consultation
            logger.warning(f"Failed to save turn to memory: {e}")

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for memory indexing."""
        # Simple keyword extraction - can be enhanced with NLP
        # Common legal terms to look for
        legal_terms = [
            "계약", "손해배상", "해고", "임대차", "상속", "이혼", "채권", "채무",
            "보증", "담보", "위자료", "양육권", "친권", "부동산", "등기", "소송",
            "재판", "항소", "상고", "조정", "중재", "형사", "민사", "행정",
            "노동", "근로", "임금", "퇴직금", "해약", "취소", "무효", "위반",
        ]

        keywords = []
        for term in legal_terms:
            if term in text:
                keywords.append(term)

        return keywords[:10]  # Return top 10 keywords

    async def _update_turn_status(
        self,
        turn_id: UUID,
        status: ConsultationStatus,
    ) -> None:
        """Update turn status."""
        update_data: dict[str, Any] = {"status": status.value}

        if status == ConsultationStatus.ANALYZING:
            update_data["processing_started_at"] = datetime.now(timezone.utc)
        elif status == ConsultationStatus.COMPLETED:
            update_data["processing_completed_at"] = datetime.now(timezone.utc)

        await self.turn_repo.update(turn_id, **update_data)

    # =========================================================================
    # SSE Streaming
    # =========================================================================

    async def stream_progress(
        self,
        consultation_id: UUID,
    ) -> AsyncGenerator[dict, None]:
        """Stream processing progress via Redis pub/sub."""
        if not self.redis:
            # Fallback for when Redis is not available
            yield {
                "type": "error",
                "data": {"message": "Streaming not available"},
            }
            return

        channel = f"consultation:{consultation_id}"
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield data

                    if data.get("type") in ["complete", "error"]:
                        break
        finally:
            await pubsub.unsubscribe(channel)

    async def _publish_progress(
        self,
        consultation_id: UUID,
        step: str,
        percentage: int,
    ) -> None:
        """Publish progress event to Redis."""
        if not self.redis:
            return

        await self.redis.publish(
            f"consultation:{consultation_id}",
            json.dumps({
                "type": "progress",
                "data": {"step": step, "percentage": percentage},
            }),
        )

    async def _publish_log(
        self,
        consultation_id: UUID,
        message: str,
    ) -> None:
        """Publish log event to Redis."""
        if not self.redis:
            return

        await self.redis.publish(
            f"consultation:{consultation_id}",
            json.dumps({
                "type": "log",
                "data": {
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }),
        )

    async def _publish_error(
        self,
        consultation_id: UUID,
        error: str,
    ) -> None:
        """Publish error event to Redis."""
        if not self.redis:
            return

        await self.redis.publish(
            f"consultation:{consultation_id}",
            json.dumps({
                "type": "error",
                "data": {"message": error},
            }),
        )

    # =========================================================================
    # Response Building
    # =========================================================================

    async def _build_turn_response(
        self,
        turn: ConsultationTurn,
    ) -> ConsultationTurnResponse:
        """Build turn response with all details."""
        # Get related data
        opinions = await self.opinion_repo.get_turn_opinions(turn.id)
        reviews = await self.review_repo.get_turn_reviews(turn.id)
        citations = await self.citation_repo.get_turn_citations(turn.id)

        return ConsultationTurnResponse(
            id=str(turn.id),
            turn_number=turn.turn_number,
            user_query=turn.user_query,
            chairman_response=turn.chairman_response,
            status=turn.status.value if turn.status else "pending",
            processing_time_ms=turn.processing_time_ms,
            model_opinions=[
                {
                    "model_name": o.model_name,
                    "opinion_text": o.opinion_text,
                    "legal_basis": o.legal_basis,
                    "risk_assessment": o.risk_assessment,
                    "recommendations": o.recommendations,
                    "confidence_level": o.confidence_level,
                }
                for o in opinions
            ],
            peer_reviews=[
                {
                    "reviewer_model": r.reviewer_model,
                    "reviewed_model": r.reviewed_opinion.model_name if r.reviewed_opinion else "unknown",
                    "review_text": r.review_text,
                    "overall_score": r.overall_score,
                }
                for r in reviews
            ],
            citations=[
                {
                    "title": c.title,
                    "content": c.content,
                    "source": c.source,
                    "source_url": c.source_url,
                    "doc_type": c.doc_type,
                    "case_number": c.case_number,
                    "law_number": c.law_number,
                    "relevance_score": c.relevance_score or 0.0,
                }
                for c in citations
            ],
        )

    # =========================================================================
    # Export
    # =========================================================================

    async def export_to_pdf(
        self,
        consultation_id: UUID,
        user_id: UUID,
    ) -> bytes:
        """Export consultation as PDF."""
        consultation = await self.get_consultation(consultation_id, user_id)

        if not consultation:
            raise ConsultationNotFoundError("상담을 찾을 수 없습니다.")

        # Note: PDF generation will be implemented with reportlab or weasyprint
        # For now, return placeholder
        raise NotImplementedError("PDF export will be implemented in Phase 7")
