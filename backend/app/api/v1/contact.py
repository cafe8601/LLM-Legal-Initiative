"""
Contact API Endpoints

문의 폼 API
"""

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import DBSession
from app.models.user import ContactSubmission
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contact", tags=["Contact"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class ContactSubmissionRequest(BaseModel):
    """Contact form submission request."""

    name: str = Field(min_length=2, max_length=100, description="문의자 이름")
    email: EmailStr = Field(description="연락받을 이메일 주소")
    phone: str | None = Field(None, max_length=20, description="연락처 (선택)")
    company: str | None = Field(None, max_length=100, description="회사명 (선택)")
    inquiry_type: str = Field(
        default="general",
        pattern="^(general|enterprise|technical|partnership|other)$",
        description="문의 유형",
    )
    message: str = Field(min_length=10, max_length=5000, description="문의 내용")


class ContactSubmissionResponse(BaseModel):
    """Contact form submission response."""

    id: str
    message: str
    estimated_response_time: str


class EnterpriseInquiryRequest(BaseModel):
    """Enterprise inquiry request."""

    company_name: str = Field(min_length=2, max_length=200, description="회사명")
    contact_name: str = Field(min_length=2, max_length=100, description="담당자명")
    contact_email: EmailStr = Field(description="담당자 이메일")
    contact_phone: str = Field(max_length=20, description="담당자 연락처")
    company_size: str = Field(
        pattern="^(1-10|11-50|51-200|201-500|500+)$",
        description="회사 규모",
    )
    industry: str = Field(max_length=100, description="업종")
    expected_users: int = Field(ge=1, description="예상 사용자 수")
    use_case: str = Field(min_length=10, max_length=5000, description="활용 계획")
    additional_requirements: str | None = Field(
        None, max_length=2000, description="추가 요청 사항"
    )


class EnterpriseInquiryResponse(BaseModel):
    """Enterprise inquiry response."""

    id: str
    message: str
    next_steps: list[str]


# =============================================================================
# Helper Functions
# =============================================================================


async def _send_contact_notifications(
    submission_id: UUID,
    name: str,
    email: str,
    message: str,
    inquiry_type: str,
):
    """Send notification emails for contact submission."""
    try:
        email_service = EmailService()

        # Send notification to support team
        await email_service.send_contact_notification(
            submission_id=str(submission_id),
            name=name,
            email=email,
            message=message,
            inquiry_type=inquiry_type,
        )

        # Send confirmation to submitter
        await email_service.send_contact_confirmation(
            to_email=email,
            name=name,
        )

        logger.info(f"Contact notification emails sent for submission {submission_id}")

    except Exception as e:
        logger.error(f"Failed to send contact notification emails: {e}")
        # Don't raise - email failure shouldn't break the submission


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "",
    response_model=ContactSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="문의 제출",
    description="문의 폼을 제출합니다. 제출된 문의는 24시간 내에 답변됩니다.",
)
async def submit_contact_form(
    data: ContactSubmissionRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> ContactSubmissionResponse:
    """
    Submit a contact form inquiry.

    This will:
    - Save the submission to the database
    - Send notification email to support team
    - Send confirmation email to the submitter

    Response times:
    - General inquiries: 24-48 hours
    - Enterprise inquiries: 4-8 business hours
    - Technical issues: 12-24 hours
    """
    # Create submission record
    submission = ContactSubmission(
        name=data.name,
        email=data.email,
        phone=data.phone,
        company=data.company,
        message=f"[{data.inquiry_type}] {data.message}",
        is_read=False,
        is_replied=False,
    )

    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    logger.info(f"Contact submission created: {submission.id}")

    # Send notification emails in background
    background_tasks.add_task(
        _send_contact_notifications,
        submission.id,
        data.name,
        data.email,
        data.message,
        data.inquiry_type,
    )

    # Determine estimated response time based on inquiry type
    response_times = {
        "general": "24-48시간",
        "enterprise": "4-8 영업시간",
        "technical": "12-24시간",
        "partnership": "2-3 영업일",
        "other": "24-48시간",
    }

    return ContactSubmissionResponse(
        id=str(submission.id),
        message="문의가 접수되었습니다. 빠른 시일 내에 답변 드리겠습니다.",
        estimated_response_time=response_times.get(data.inquiry_type, "24-48시간"),
    )


@router.post(
    "/enterprise",
    response_model=EnterpriseInquiryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enterprise 문의",
    description="Enterprise 요금제 문의를 제출합니다. 담당자가 4-8 영업시간 내에 연락드립니다.",
)
async def submit_enterprise_inquiry(
    data: EnterpriseInquiryRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> EnterpriseInquiryResponse:
    """
    Submit an enterprise plan inquiry.

    This creates a high-priority inquiry for the sales team.
    A dedicated account manager will reach out within 4-8 business hours.
    """
    # Format the enterprise inquiry message
    message = f"""[Enterprise Inquiry]

Company: {data.company_name}
Industry: {data.industry}
Company Size: {data.company_size}
Expected Users: {data.expected_users}

Contact: {data.contact_name}
Phone: {data.contact_phone}

Use Case:
{data.use_case}

Additional Requirements:
{data.additional_requirements or 'N/A'}
"""

    # Create submission record
    submission = ContactSubmission(
        name=data.contact_name,
        email=data.contact_email,
        phone=data.contact_phone,
        company=data.company_name,
        message=message,
        is_read=False,
        is_replied=False,
    )

    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    logger.info(f"Enterprise inquiry created: {submission.id}")

    # Send notification emails in background
    background_tasks.add_task(
        _send_contact_notifications,
        submission.id,
        data.contact_name,
        data.contact_email,
        message,
        "enterprise",
    )

    return EnterpriseInquiryResponse(
        id=str(submission.id),
        message="Enterprise 문의가 접수되었습니다. 담당자가 곧 연락드리겠습니다.",
        next_steps=[
            "담당 매니저가 4-8 영업시간 내에 연락드립니다.",
            "귀사의 요구사항에 맞는 맞춤 플랜을 제안해 드립니다.",
            "무료 데모 및 POC 진행이 가능합니다.",
            "질문이 있으시면 enterprise@legalcouncil.ai로 문의해 주세요.",
        ],
    )


@router.get(
    "/pricing",
    summary="요금제 정보",
    description="각 요금제의 기능과 가격 정보를 조회합니다.",
)
async def get_pricing_info() -> dict:
    """
    Get pricing information for all tiers.

    Returns detailed pricing and feature comparison for each tier.
    """
    return {
        "tiers": [
            {
                "name": "Basic",
                "price": {
                    "monthly": 0,
                    "yearly": 0,
                },
                "currency": "KRW",
                "features": [
                    "월 3회 상담",
                    "기본 법률 분야 지원",
                    "24시간 이내 응답",
                    "이메일 지원",
                ],
                "limitations": [
                    "문서 업로드 5MB 제한",
                    "저장 공간 100MB",
                ],
                "cta": "무료로 시작하기",
            },
            {
                "name": "Pro",
                "price": {
                    "monthly": 49000,
                    "yearly": 470000,
                },
                "currency": "KRW",
                "features": [
                    "무제한 상담",
                    "모든 법률 분야 지원",
                    "우선 처리",
                    "문서 분석 기능",
                    "PDF 리포트 내보내기",
                    "전화 상담 지원",
                ],
                "limitations": [
                    "문서 업로드 25MB 제한",
                    "저장 공간 1GB",
                ],
                "cta": "Pro 시작하기",
                "popular": True,
            },
            {
                "name": "Enterprise",
                "price": {
                    "monthly": None,  # Custom pricing
                    "yearly": None,
                },
                "currency": "KRW",
                "features": [
                    "무제한 상담",
                    "모든 법률 분야 지원",
                    "최우선 처리",
                    "고급 문서 분석",
                    "API 액세스",
                    "커스텀 프롬프트",
                    "전담 계정 매니저",
                    "SLA 보장",
                    "온프레미스 설치 옵션",
                ],
                "limitations": [
                    "문서 업로드 100MB 제한",
                    "저장 공간 10GB",
                ],
                "cta": "문의하기",
            },
        ],
        "faq": [
            {
                "question": "무료 체험이 가능한가요?",
                "answer": "네, Basic 요금제는 무료로 시작하실 수 있습니다. 또한 Pro 요금제의 14일 무료 체험도 제공합니다.",
            },
            {
                "question": "결제 방법은 어떻게 되나요?",
                "answer": "신용카드, 체크카드, 계좌이체를 지원합니다. Enterprise 요금제는 청구서 결제도 가능합니다.",
            },
            {
                "question": "언제든 해지할 수 있나요?",
                "answer": "네, 언제든 해지 가능합니다. 해지 후에도 남은 기간은 계속 이용하실 수 있습니다.",
            },
            {
                "question": "데이터는 안전한가요?",
                "answer": "모든 데이터는 암호화되어 저장되며, 한국 내 데이터센터에서 관리됩니다. ISO 27001 인증을 준비 중입니다.",
            },
        ],
    }
