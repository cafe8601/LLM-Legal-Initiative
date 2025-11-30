"""
Email Service

이메일 발송 서비스 (SendGrid 사용)
"""

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Email service using SendGrid."""

    def __init__(self):
        self.from_email = settings.FROM_EMAIL
        self.from_name = settings.EMAIL_FROM_NAME
        self.api_key = settings.SENDGRID_API_KEY
        self.frontend_url = settings.FRONTEND_URL

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str | None = None,
    ) -> bool:
        """
        Send an email using SendGrid.

        Args:
            to_email: Recipient email
            subject: Email subject
            html_content: HTML email body
            text_content: Plain text email body (optional)

        Returns:
            True if email sent successfully
        """
        if not self.api_key:
            logger.warning(f"SendGrid API key not configured. Would send email to {to_email}")
            logger.debug(f"Subject: {subject}")
            logger.debug(f"Content: {html_content[:200]}...")
            return True  # Return True in development without API key

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Email, To, Content

            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
            )
            message.add_content(Content("text/html", html_content))

            if text_content:
                message.add_content(Content("text/plain", text_content))

            sg = SendGridAPIClient(self.api_key)
            response = sg.send(message)

            if response.status_code in (200, 201, 202):
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code}")
                return False

        except ImportError:
            logger.warning("SendGrid not installed. Skipping email send.")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    async def send_verification_email(
        self,
        to_email: str,
        user_name: str,
        token: str,
    ) -> bool:
        """
        Send email verification link.

        Args:
            to_email: User email
            user_name: User's name
            token: Verification token
        """
        verification_url = f"{self.frontend_url}/verify-email?token={token}"

        subject = "[법률 자문 위원회] 이메일 인증을 완료해 주세요"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #3182ce; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
                .button:hover {{ background: #2c5282; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚖️ AI 법률 자문 위원회</h1>
                </div>
                <div class="content">
                    <h2>안녕하세요, {user_name}님!</h2>
                    <p>법률 자문 위원회 서비스에 가입해 주셔서 감사합니다.</p>
                    <p>아래 버튼을 클릭하여 이메일 인증을 완료해 주세요:</p>

                    <div style="text-align: center;">
                        <a href="{verification_url}" class="button">이메일 인증하기</a>
                    </div>

                    <p style="color: #666; font-size: 14px;">
                        이 링크는 24시간 동안 유효합니다.<br>
                        버튼이 작동하지 않으면 아래 링크를 브라우저에 직접 입력해 주세요:
                    </p>
                    <p style="word-break: break-all; font-size: 12px; color: #999;">
                        {verification_url}
                    </p>
                </div>
                <div class="footer">
                    <p>본 이메일은 발신 전용입니다.</p>
                    <p>© 2024 AI 법률 자문 위원회. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        안녕하세요, {user_name}님!

        법률 자문 위원회 서비스에 가입해 주셔서 감사합니다.

        아래 링크를 클릭하여 이메일 인증을 완료해 주세요:
        {verification_url}

        이 링크는 24시간 동안 유효합니다.

        ---
        AI 법률 자문 위원회
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_password_reset_email(
        self,
        to_email: str,
        user_name: str,
        token: str,
    ) -> bool:
        """
        Send password reset link.

        Args:
            to_email: User email
            user_name: User's name
            token: Reset token
        """
        reset_url = f"{self.frontend_url}/reset-password?token={token}"

        subject = "[법률 자문 위원회] 비밀번호 재설정"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #e53e3e; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
                .button:hover {{ background: #c53030; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 6px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚖️ AI 법률 자문 위원회</h1>
                </div>
                <div class="content">
                    <h2>비밀번호 재설정 요청</h2>
                    <p>안녕하세요, {user_name}님!</p>
                    <p>비밀번호 재설정 요청이 접수되었습니다.</p>

                    <div style="text-align: center;">
                        <a href="{reset_url}" class="button">비밀번호 재설정</a>
                    </div>

                    <div class="warning">
                        <strong>⚠️ 주의:</strong> 이 링크는 1시간 동안만 유효합니다.
                        본인이 요청하지 않았다면 이 이메일을 무시해 주세요.
                    </div>

                    <p style="word-break: break-all; font-size: 12px; color: #999;">
                        링크: {reset_url}
                    </p>
                </div>
                <div class="footer">
                    <p>본 이메일은 발신 전용입니다.</p>
                    <p>© 2024 AI 법률 자문 위원회. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        안녕하세요, {user_name}님!

        비밀번호 재설정 요청이 접수되었습니다.

        아래 링크를 클릭하여 비밀번호를 재설정해 주세요:
        {reset_url}

        ⚠️ 주의: 이 링크는 1시간 동안만 유효합니다.
        본인이 요청하지 않았다면 이 이메일을 무시해 주세요.

        ---
        AI 법률 자문 위원회
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_welcome_email(
        self,
        to_email: str,
        user_name: str,
    ) -> bool:
        """
        Send welcome email after verification.

        Args:
            to_email: User email
            user_name: User's name
        """
        subject = "[법률 자문 위원회] 가입을 환영합니다! 🎉"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .feature {{ background: white; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #3182ce; }}
                .button {{ display: inline-block; background: #3182ce; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚖️ 환영합니다!</h1>
                </div>
                <div class="content">
                    <h2>{user_name}님, 가입을 환영합니다!</h2>
                    <p>AI 법률 자문 위원회 서비스를 이용해 주셔서 감사합니다.</p>

                    <h3>🌟 주요 기능</h3>
                    <div class="feature">
                        <strong>🤖 4개 AI 모델 협업</strong><br>
                        GPT-5.1, Claude Sonnet 4.5, Gemini 3 Pro, Grok 4가 다각도로 분석합니다.
                    </div>
                    <div class="feature">
                        <strong>🔍 블라인드 교차 평가</strong><br>
                        각 AI의 의견을 익명으로 상호 검증하여 정확도를 높입니다.
                    </div>
                    <div class="feature">
                        <strong>👨‍⚖️ 의장 종합</strong><br>
                        Claude Opus 4.5가 모든 의견을 종합하여 최종 자문을 제공합니다.
                    </div>

                    <div style="text-align: center;">
                        <a href="{self.frontend_url}/consultation/new" class="button">첫 상담 시작하기</a>
                    </div>
                </div>
                <div class="footer">
                    <p>© 2024 AI 법률 자문 위원회. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return await self.send_email(to_email, subject, html_content)

    async def send_contact_notification(
        self,
        submission_id: str,
        name: str,
        email: str,
        message: str,
        inquiry_type: str,
    ) -> bool:
        """
        Send notification to support team about new contact submission.

        Args:
            submission_id: Contact submission ID
            name: Submitter name
            email: Submitter email
            message: Message content
            inquiry_type: Type of inquiry
        """
        support_email = settings.SUPPORT_EMAIL or self.from_email

        inquiry_labels = {
            "general": "일반 문의",
            "enterprise": "Enterprise 문의",
            "technical": "기술 지원",
            "partnership": "파트너십 제안",
            "other": "기타",
        }

        subject = f"[새 문의] {inquiry_labels.get(inquiry_type, inquiry_type)} - {name}"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #2d3748; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .info-row {{ margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }}
                .label {{ font-weight: bold; color: #4a5568; }}
                .message-box {{ background: white; padding: 20px; border-radius: 6px; border-left: 4px solid #3182ce; margin-top: 15px; }}
                .priority-high {{ border-left-color: #e53e3e; }}
                .priority-normal {{ border-left-color: #3182ce; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>📬 새로운 문의가 접수되었습니다</h2>
                </div>
                <div class="content">
                    <div class="info-row">
                        <span class="label">문의 ID:</span> {submission_id}
                    </div>
                    <div class="info-row">
                        <span class="label">문의 유형:</span> {inquiry_labels.get(inquiry_type, inquiry_type)}
                    </div>
                    <div class="info-row">
                        <span class="label">문의자:</span> {name}
                    </div>
                    <div class="info-row">
                        <span class="label">이메일:</span> <a href="mailto:{email}">{email}</a>
                    </div>

                    <div class="message-box {'priority-high' if inquiry_type == 'enterprise' else 'priority-normal'}">
                        <strong>문의 내용:</strong>
                        <p style="white-space: pre-wrap;">{message}</p>
                    </div>

                    <p style="margin-top: 20px; color: #666;">
                        <strong>응답 기한:</strong>
                        {"4-8 영업시간" if inquiry_type == "enterprise" else "24-48시간"}
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return await self.send_email(support_email, subject, html_content)

    async def send_contact_confirmation(
        self,
        to_email: str,
        name: str,
    ) -> bool:
        """
        Send confirmation email to contact form submitter.

        Args:
            to_email: Submitter email
            name: Submitter name
        """
        subject = "[법률 자문 위원회] 문의가 접수되었습니다"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .info-box {{ background: #e6fffa; border: 1px solid #38b2ac; padding: 15px; border-radius: 6px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚖️ AI 법률 자문 위원회</h1>
                </div>
                <div class="content">
                    <h2>안녕하세요, {name}님!</h2>
                    <p>문의해 주셔서 감사합니다.</p>
                    <p>귀하의 문의가 정상적으로 접수되었습니다.</p>

                    <div class="info-box">
                        <strong>📋 안내 사항</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>일반 문의: 24-48시간 내 답변</li>
                            <li>기술 지원: 12-24시간 내 답변</li>
                            <li>Enterprise 문의: 4-8 영업시간 내 답변</li>
                        </ul>
                    </div>

                    <p>추가 문의사항이 있으시면 언제든 연락해 주세요.</p>
                    <p>
                        📧 이메일: support@legalcouncil.ai<br>
                        🌐 웹사이트: {self.frontend_url}
                    </p>
                </div>
                <div class="footer">
                    <p>본 이메일은 자동 발송된 확인 메일입니다.</p>
                    <p>© 2024 AI 법률 자문 위원회. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        안녕하세요, {name}님!

        문의해 주셔서 감사합니다.
        귀하의 문의가 정상적으로 접수되었습니다.

        📋 응답 예상 시간:
        - 일반 문의: 24-48시간 내 답변
        - 기술 지원: 12-24시간 내 답변
        - Enterprise 문의: 4-8 영업시간 내 답변

        추가 문의사항이 있으시면 언제든 연락해 주세요.

        ---
        AI 법률 자문 위원회
        """

        return await self.send_email(to_email, subject, html_content, text_content)

    async def send_consultation_complete_email(
        self,
        to_email: str,
        user_name: str,
        consultation_id: str,
        consultation_title: str,
    ) -> bool:
        """
        Send notification when consultation is complete.

        Args:
            to_email: User email
            user_name: User's name
            consultation_id: Consultation ID
            consultation_title: Consultation title
        """
        consultation_url = f"{self.frontend_url}/consultation/{consultation_id}"

        subject = f"[법률 자문 위원회] '{consultation_title}' 상담 완료"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #38a169 0%, #2f855a 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .button {{ display: inline-block; background: #38a169; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✅ 상담 완료</h1>
                </div>
                <div class="content">
                    <h2>{user_name}님, 법률 상담이 완료되었습니다.</h2>
                    <p><strong>상담 제목:</strong> {consultation_title}</p>
                    <p>4개 AI 모델의 분석과 의장의 종합 의견이 준비되었습니다.</p>

                    <div style="text-align: center;">
                        <a href="{consultation_url}" class="button">결과 확인하기</a>
                    </div>
                </div>
                <div class="footer">
                    <p>© 2024 AI 법률 자문 위원회. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return await self.send_email(to_email, subject, html_content)
