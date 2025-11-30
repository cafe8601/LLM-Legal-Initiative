"""SQLAlchemy database models."""

from app.models.base import Base
from app.models.user import User, UserTier, RefreshToken, ContactSubmission
from app.models.consultation import (
    Consultation,
    ConsultationTurn,
    ModelOpinion,
    PeerReview,
    ConsultationStatus,
    ConsultationCategory,
)
from app.models.document import Document, Citation, DocumentType
from app.models.memory import (
    UserMemory,
    ConversationHistory,
    LegalPattern,
    MemoryType,
    MemoryPriority,
)

__all__ = [
    # Base
    "Base",
    # User
    "User",
    "UserTier",
    "RefreshToken",
    "ContactSubmission",
    # Consultation
    "Consultation",
    "ConsultationTurn",
    "ModelOpinion",
    "PeerReview",
    "ConsultationStatus",
    "ConsultationCategory",
    # Document
    "Document",
    "Citation",
    "DocumentType",
    # Memory
    "UserMemory",
    "ConversationHistory",
    "LegalPattern",
    "MemoryType",
    "MemoryPriority",
]
