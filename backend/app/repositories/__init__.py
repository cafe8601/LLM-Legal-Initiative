"""Repository layer for data access."""

from app.repositories.base import BaseRepository
from app.repositories.user import UserRepository
from app.repositories.consultation import ConsultationRepository
from app.repositories.document import DocumentRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "ConsultationRepository",
    "DocumentRepository",
]
