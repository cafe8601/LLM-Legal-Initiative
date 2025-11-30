"""
Database Seeding Script

개발 및 테스트 데이터 생성
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.session import async_session_factory, async_engine
from app.models.base import Base
from app.models.user import User, UserTier, ContactSubmission
from app.models.consultation import (
    Consultation,
    ConsultationTurn,
    ModelOpinion,
    PeerReview,
    ConsultationStatus,
    ConsultationCategory,
)
from app.models.document import Document, Citation, DocumentType


async def create_tables():
    """Create all tables if they don't exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created successfully")


async def seed_users(db: AsyncSession) -> list[User]:
    """Seed test users."""
    users_data = [
        {
            "email": "admin@legal-council.kr",
            "password_hash": get_password_hash("Admin123!@#"),
            "full_name": "관리자",
            "tier": UserTier.ENTERPRISE,
            "is_active": True,
            "is_verified": True,
            "is_admin": True,
            "preferred_language": "ko",
        },
        {
            "email": "pro@example.com",
            "password_hash": get_password_hash("ProUser123!"),
            "full_name": "프로 사용자",
            "phone": "010-1234-5678",
            "company": "테스트 회사",
            "tier": UserTier.PRO,
            "is_active": True,
            "is_verified": True,
            "preferred_language": "ko",
        },
        {
            "email": "basic@example.com",
            "password_hash": get_password_hash("Basic123!"),
            "full_name": "일반 사용자",
            "tier": UserTier.BASIC,
            "is_active": True,
            "is_verified": True,
            "consultation_count_this_month": 2,
            "preferred_language": "ko",
        },
        {
            "email": "unverified@example.com",
            "password_hash": get_password_hash("Test123!"),
            "full_name": "미인증 사용자",
            "tier": UserTier.BASIC,
            "is_active": True,
            "is_verified": False,
            "preferred_language": "ko",
        },
    ]

    users = []
    for user_data in users_data:
        user = User(**user_data)
        db.add(user)
        users.append(user)

    await db.flush()
    print(f"✅ Created {len(users)} test users")
    return users


async def seed_consultations(db: AsyncSession, users: list[User]) -> list[Consultation]:
    """Seed test consultations."""
    pro_user = users[1]  # Pro user

    consultations_data = [
        {
            "user_id": pro_user.id,
            "title": "임대차 계약 분쟁 관련 자문",
            "category": ConsultationCategory.CONTRACT,
            "status": ConsultationStatus.COMPLETED,
            "summary": "임대차 계약 해지 및 보증금 반환 관련 법률 자문 완료",
            "turn_count": 2,
            "total_tokens_used": 15000,
            "total_cost": 0.45,
        },
        {
            "user_id": pro_user.id,
            "title": "상표권 침해 대응 방안",
            "category": ConsultationCategory.INTELLECTUAL_PROPERTY,
            "status": ConsultationStatus.COMPLETED,
            "summary": "상표권 침해에 대한 법적 대응 방안 검토",
            "turn_count": 1,
            "total_tokens_used": 8000,
            "total_cost": 0.24,
        },
        {
            "user_id": pro_user.id,
            "title": "근로계약 해지 관련 문의",
            "category": ConsultationCategory.LABOR,
            "status": ConsultationStatus.PROCESSING,
            "turn_count": 1,
            "total_tokens_used": 5000,
            "total_cost": 0.15,
        },
    ]

    consultations = []
    for data in consultations_data:
        consultation = Consultation(**data)
        db.add(consultation)
        consultations.append(consultation)

    await db.flush()
    print(f"✅ Created {len(consultations)} test consultations")
    return consultations


async def seed_consultation_turns(
    db: AsyncSession,
    consultations: list[Consultation],
) -> list[ConsultationTurn]:
    """Seed consultation turns with model opinions."""
    turns = []

    # First consultation - 2 turns
    consultation = consultations[0]

    turn1 = ConsultationTurn(
        consultation_id=consultation.id,
        turn_number=1,
        user_query="임대차 계약 기간이 만료되었는데 임대인이 보증금 반환을 미루고 있습니다. 어떻게 해야 하나요?",
        status=ConsultationStatus.COMPLETED,
        chairman_response="""## 법률 자문 결과

### 1. 쟁점 요약
임대차 계약 종료 후 임대인의 보증금 반환 의무 이행 지연에 관한 사안입니다.

### 2. 법적 근거
- 민법 제618조 (임대차의 효력)
- 주택임대차보호법 제3조 (대항력)
- 주택임대차보호법 제3조의2 (보증금의 회수)

### 3. 권고 사항
1. 내용증명 발송을 통한 반환 요청
2. 임차권등기명령 신청 고려
3. 지급명령 또는 민사소송 제기 검토

### 4. 주의사항
- 소멸시효(10년) 내 청구권 행사 필요
- 증거자료(계약서, 입금증 등) 확보 중요
""",
        tokens_used=8000,
        estimated_cost=0.24,
        processing_time_ms=15000,
        processing_started_at=datetime.now(timezone.utc),
        processing_completed_at=datetime.now(timezone.utc),
    )
    db.add(turn1)
    await db.flush()

    # Add model opinions for turn1
    models = [
        ("GPT-5.1", "gpt-5.1-2025"),
        ("Claude Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ("Gemini 3 Pro", "gemini-3-pro"),
        ("Grok 4", "grok-4"),
    ]

    for model_name, model_version in models:
        opinion = ModelOpinion(
            turn_id=turn1.id,
            model_name=model_name,
            model_version=model_version,
            opinion_text=f"{model_name}의 법률 분석 의견...",
            legal_basis="민법 제618조, 주택임대차보호법 제3조",
            risk_assessment="임대인의 지연손해금 부담 가능성",
            recommendations="내용증명 발송 후 법적 조치 검토",
            confidence_level="high",
            tokens_input=2000,
            tokens_output=1500,
            processing_time_ms=3000,
        )
        db.add(opinion)

    turn2 = ConsultationTurn(
        consultation_id=consultation.id,
        turn_number=2,
        user_query="내용증명은 어떻게 작성하면 되나요?",
        status=ConsultationStatus.COMPLETED,
        chairman_response="""## 내용증명 작성 가이드

### 1. 필수 기재사항
- 발신인/수신인 정보
- 계약 내용 요약
- 반환 요구 금액
- 이행 기한

### 2. 작성 예시
[구체적인 내용증명 양식 제공...]
""",
        tokens_used=7000,
        estimated_cost=0.21,
        processing_time_ms=12000,
        processing_started_at=datetime.now(timezone.utc),
        processing_completed_at=datetime.now(timezone.utc),
    )
    db.add(turn2)

    turns.extend([turn1, turn2])

    # Second consultation - 1 turn
    consultation2 = consultations[1]
    turn3 = ConsultationTurn(
        consultation_id=consultation2.id,
        turn_number=1,
        user_query="경쟁사가 우리 회사 상표와 유사한 상표를 사용하고 있습니다. 대응 방법이 궁금합니다.",
        status=ConsultationStatus.COMPLETED,
        chairman_response="""## 상표권 침해 대응 가이드

### 1. 침해 여부 판단 기준
- 상표의 동일·유사성
- 지정상품의 동일·유사성
- 출처 혼동 가능성

### 2. 대응 방안
1. 경고장 발송
2. 상표권 침해금지 가처분
3. 손해배상 청구 소송
""",
        tokens_used=8000,
        estimated_cost=0.24,
        processing_time_ms=14000,
        processing_started_at=datetime.now(timezone.utc),
        processing_completed_at=datetime.now(timezone.utc),
    )
    db.add(turn3)
    turns.append(turn3)

    await db.flush()
    print(f"✅ Created {len(turns)} consultation turns with model opinions")
    return turns


async def seed_citations(db: AsyncSession, turns: list[ConsultationTurn]):
    """Seed test citations."""
    citations_data = [
        {
            "turn_id": turns[0].id,
            "title": "대법원 2020다12345 판결",
            "content": "임대차계약 종료 후 임대인의 보증금 반환의무는...",
            "source": "대법원",
            "source_url": "https://law.go.kr/LSW/precInfoP.do?precSeq=12345",
            "doc_type": "precedent",
            "category": "민사",
            "case_number": "2020다12345",
            "relevance_score": 0.95,
            "display_order": 1,
        },
        {
            "turn_id": turns[0].id,
            "title": "주택임대차보호법 제3조의2",
            "content": "임차인은 임대차가 끝난 후 보증금을 반환받을 때까지...",
            "source": "국가법령정보센터",
            "source_url": "https://law.go.kr/LSW/lsInfoP.do?lsiSeq=123456",
            "doc_type": "law",
            "category": "민사",
            "law_number": "법률 제18799호",
            "article_number": "제3조의2",
            "relevance_score": 0.92,
            "display_order": 2,
        },
        {
            "turn_id": turns[2].id,
            "title": "상표법 제108조",
            "content": "상표권 또는 전용사용권을 침해한 자는...",
            "source": "국가법령정보센터",
            "source_url": "https://law.go.kr/LSW/lsInfoP.do?lsiSeq=234567",
            "doc_type": "law",
            "category": "지식재산",
            "law_number": "법률 제19115호",
            "article_number": "제108조",
            "relevance_score": 0.88,
            "display_order": 1,
        },
    ]

    for data in citations_data:
        citation = Citation(**data)
        db.add(citation)

    await db.flush()
    print(f"✅ Created {len(citations_data)} test citations")


async def seed_contact_submissions(db: AsyncSession):
    """Seed test contact submissions."""
    submissions_data = [
        {
            "name": "홍길동",
            "email": "hong@example.com",
            "phone": "010-9876-5432",
            "company": "ABC 주식회사",
            "message": "법률 자문 서비스에 대해 문의드립니다. 기업 법무 관련 월정액 계약이 가능한지요?",
            "is_read": False,
        },
        {
            "name": "김영희",
            "email": "kim@example.com",
            "message": "서비스 이용 중 결제 관련 문의가 있습니다.",
            "is_read": True,
            "is_replied": True,
        },
    ]

    for data in submissions_data:
        submission = ContactSubmission(**data)
        db.add(submission)

    await db.flush()
    print(f"✅ Created {len(submissions_data)} contact submissions")


async def seed_documents(db: AsyncSession, users: list[User]):
    """Seed test documents."""
    pro_user = users[1]

    documents_data = [
        {
            "user_id": pro_user.id,
            "file_name": "contract_20241101.pdf",
            "original_name": "임대차계약서.pdf",
            "file_type": DocumentType.PDF,
            "mime_type": "application/pdf",
            "file_size": 524288,  # 512KB
            "storage_path": f"documents/{pro_user.id}/contract_20241101.pdf",
            "storage_bucket": "legal-council-documents",
            "extracted_text": "임대차 계약서\n\n1. 목적물의 표시...",
            "ocr_completed": True,
            "ocr_confidence": 0.98,
            "page_count": 3,
            "is_processed": True,
        },
        {
            "user_id": pro_user.id,
            "file_name": "trademark_cert.jpg",
            "original_name": "상표등록증.jpg",
            "file_type": DocumentType.IMAGE,
            "mime_type": "image/jpeg",
            "file_size": 1048576,  # 1MB
            "storage_path": f"documents/{pro_user.id}/trademark_cert.jpg",
            "storage_bucket": "legal-council-documents",
            "extracted_text": "상표등록증\n\n등록번호: 제40-1234567호...",
            "ocr_completed": True,
            "ocr_confidence": 0.95,
            "is_processed": True,
        },
    ]

    for data in documents_data:
        document = Document(**data)
        db.add(document)

    await db.flush()
    print(f"✅ Created {len(documents_data)} test documents")


async def main():
    """Main seeding function."""
    print("🌱 Starting database seeding...")
    print(f"   Database: {settings.DATABASE_URL}")

    # Create tables
    await create_tables()

    async with async_session_factory() as db:
        try:
            # Seed data
            users = await seed_users(db)
            consultations = await seed_consultations(db, users)
            turns = await seed_consultation_turns(db, consultations)
            await seed_citations(db, turns)
            await seed_contact_submissions(db)
            await seed_documents(db, users)

            # Commit all changes
            await db.commit()
            print("\n✅ Database seeding completed successfully!")

            # Print summary
            print("\n📊 Seed Data Summary:")
            print(f"   - Users: {len(users)}")
            print(f"   - Consultations: {len(consultations)}")
            print(f"   - Consultation Turns: {len(turns)}")
            print("\n🔑 Test Accounts:")
            print("   - Admin: admin@legal-council.kr / Admin123!@#")
            print("   - Pro User: pro@example.com / ProUser123!")
            print("   - Basic User: basic@example.com / Basic123!")

        except Exception as e:
            await db.rollback()
            print(f"\n❌ Error during seeding: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
