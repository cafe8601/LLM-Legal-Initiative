"""
API v1 Router

모든 v1 API 엔드포인트 통합
"""

from fastapi import APIRouter

from app.api.v1 import auth, consultations, documents, search, users, contact, expert_chat

api_router = APIRouter()

# Include all routers
api_router.include_router(auth.router)
api_router.include_router(consultations.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(users.router)
api_router.include_router(contact.router)
api_router.include_router(expert_chat.router)
