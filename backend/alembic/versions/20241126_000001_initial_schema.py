"""Initial schema

Revision ID: 20241126_000001
Revises:
Create Date: 2024-11-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20241126_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ========================================================================
    # Users Table
    # ========================================================================
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(100), nullable=False),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("company", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("is_verified", sa.Boolean(), default=False, nullable=False),
        sa.Column("is_admin", sa.Boolean(), default=False, nullable=False),
        sa.Column("tier", sa.String(20), default="basic", nullable=False, index=True),
        sa.Column("consultation_count_this_month", sa.Integer(), default=0, nullable=False),
        sa.Column("last_consultation_reset", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stripe_customer_id", sa.String(100), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(100), nullable=True),
        sa.Column("preferred_language", sa.String(10), default="ko", nullable=False),
        sa.Column("notification_email", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.CheckConstraint("tier IN ('basic', 'pro', 'enterprise')", name="ck_users_tier"),
    )

    # ========================================================================
    # Refresh Tokens Table
    # ========================================================================
    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), default=False),
        sa.Column("device_info", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_refresh_tokens_expires", "refresh_tokens", ["expires_at"])

    # ========================================================================
    # Contact Submissions Table
    # ========================================================================
    op.create_table(
        "contact_submissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("company", sa.String(100), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("is_read", sa.Boolean(), default=False),
        sa.Column("is_replied", sa.Boolean(), default=False),
        sa.Column("replied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replied_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("idx_contact_created_at", "contact_submissions", ["created_at"])

    # ========================================================================
    # Documents Table
    # ========================================================================
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("original_name", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("storage_bucket", sa.String(100), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("ocr_completed", sa.Boolean(), default=False),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_processed", sa.Boolean(), default=False),
        sa.Column("is_deleted", sa.Boolean(), default=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # ========================================================================
    # Consultations Table
    # ========================================================================
    op.create_table(
        "consultations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("category", sa.String(50), default="general", nullable=False, index=True),
        sa.Column("status", sa.String(20), default="pending", nullable=False, index=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("turn_count", sa.Integer(), default=0),
        sa.Column("total_tokens_used", sa.Integer(), default=0),
        sa.Column("total_cost", sa.Float(), default=0.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "category IN ('general', 'contract', 'intellectual-property', 'labor', 'criminal', 'administrative', 'corporate', 'family', 'real-estate')",
            name="ck_consultations_category",
        ),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name="ck_consultations_status"),
    )
    op.create_index("idx_consultations_created_at", "consultations", ["created_at"])

    # ========================================================================
    # Consultation Turns Table
    # ========================================================================
    op.create_table(
        "consultation_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("consultation_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("attached_document_ids", postgresql.JSON(), nullable=True),
        sa.Column("chairman_response", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), default="pending", nullable=False),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), default=0),
        sa.Column("estimated_cost", sa.Float(), default=0.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["consultation_id"], ["consultations.id"], ondelete="CASCADE"),
    )

    # ========================================================================
    # Model Opinions Table
    # ========================================================================
    op.create_table(
        "model_opinions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("model_name", sa.String(50), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("opinion_text", sa.Text(), nullable=False),
        sa.Column("legal_basis", sa.Text(), nullable=True),
        sa.Column("risk_assessment", sa.Text(), nullable=True),
        sa.Column("recommendations", sa.Text(), nullable=True),
        sa.Column("confidence_level", sa.String(20), nullable=True),
        sa.Column("tokens_input", sa.Integer(), default=0),
        sa.Column("tokens_output", sa.Integer(), default=0),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("raw_response", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["consultation_turns.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_opinions_model_name", "model_opinions", ["model_name"])

    # ========================================================================
    # Peer Reviews Table
    # ========================================================================
    op.create_table(
        "peer_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("reviewed_opinion_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reviewer_model", sa.String(50), nullable=False),
        sa.Column("review_text", sa.Text(), nullable=False),
        sa.Column("accuracy_score", sa.Integer(), nullable=True),
        sa.Column("completeness_score", sa.Integer(), nullable=True),
        sa.Column("practicality_score", sa.Integer(), nullable=True),
        sa.Column("legal_basis_score", sa.Integer(), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("strengths", sa.Text(), nullable=True),
        sa.Column("weaknesses", sa.Text(), nullable=True),
        sa.Column("suggestions", sa.Text(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), default=0),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["consultation_turns.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewed_opinion_id"], ["model_opinions.id"], ondelete="CASCADE"),
        sa.CheckConstraint("accuracy_score BETWEEN 1 AND 5", name="ck_peer_reviews_accuracy"),
        sa.CheckConstraint("completeness_score BETWEEN 1 AND 5", name="ck_peer_reviews_completeness"),
        sa.CheckConstraint("practicality_score BETWEEN 1 AND 5", name="ck_peer_reviews_practicality"),
        sa.CheckConstraint("legal_basis_score BETWEEN 1 AND 5", name="ck_peer_reviews_legal_basis"),
    )

    # ========================================================================
    # Citations Table
    # ========================================================================
    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("source_url", sa.String(500), nullable=True),
        sa.Column("doc_type", sa.String(50), nullable=True),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("case_number", sa.String(100), nullable=True),
        sa.Column("law_number", sa.String(100), nullable=True),
        sa.Column("article_number", sa.String(50), nullable=True),
        sa.Column("relevance_score", sa.Float(), default=0.0),
        sa.Column("search_query", sa.Text(), nullable=True),
        sa.Column("display_order", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["consultation_turns.id"], ondelete="CASCADE"),
    )

    # ========================================================================
    # Additional Indexes for Performance
    # ========================================================================
    # Full-text search index for consultations
    op.execute(
        """
        CREATE INDEX idx_consultations_title_trgm ON consultations
        USING gin (title gin_trgm_ops)
        """
    )

    # Full-text search index for citations
    op.execute(
        """
        CREATE INDEX idx_citations_title_trgm ON citations
        USING gin (title gin_trgm_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_citations_content_trgm ON citations
        USING gin (content gin_trgm_ops)
        """
    )


def downgrade() -> None:
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_citations_content_trgm")
    op.execute("DROP INDEX IF EXISTS idx_citations_title_trgm")
    op.execute("DROP INDEX IF EXISTS idx_consultations_title_trgm")

    # Drop tables in reverse order of creation (respecting foreign keys)
    op.drop_table("citations")
    op.drop_table("peer_reviews")
    op.drop_table("model_opinions")
    op.drop_table("consultation_turns")
    op.drop_table("consultations")
    op.drop_table("documents")
    op.drop_table("contact_submissions")
    op.drop_table("refresh_tokens")
    op.drop_table("users")

    # Drop extensions
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
