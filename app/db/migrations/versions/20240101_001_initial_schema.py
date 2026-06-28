"""Initial schema: documents, chunks, jobs, feedback, eval_runs

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision      = "001"
down_revision = None
branch_labels = None
depends_on    = None


def upgrade() -> None:
    # ── documents ──────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id",                 postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("filename",           sa.String(512),  nullable=False),
        sa.Column("original_filename",  sa.String(512),  nullable=False),
        sa.Column("file_type",          sa.String(32),   nullable=False),
        sa.Column("file_path",          sa.String(1024), nullable=True),
        sa.Column("file_size_bytes",    sa.BigInteger(), nullable=True),
        sa.Column("status",             sa.String(32),   nullable=False, server_default="pending"),
        sa.Column("error_message",      sa.Text(),       nullable=True),
        sa.Column("raw_text",           sa.Text(),       nullable=True),
        sa.Column("page_count",         sa.Integer(),    nullable=True),
        sa.Column("word_count",         sa.Integer(),    nullable=True),
        sa.Column("chunk_count",        sa.Integer(),    nullable=False, server_default="0"),
        sa.Column("title",              sa.String(512),  nullable=True),
        sa.Column("author",             sa.String(256),  nullable=True),
        sa.Column("department",         sa.String(128),  nullable=True),
        sa.Column("doc_category",       sa.String(128),  nullable=True),
        sa.Column("language",           sa.String(16),   nullable=True),
        sa.Column("doc_created_date",   sa.String(64),   nullable=True),
        sa.Column("tags",               postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("metadata_json",      postgresql.JSONB(), nullable=True),
        sa.Column("created_at",         sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at",         sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("indexed_at",         sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_documents_status",      "documents", ["status"])
    op.create_index("idx_documents_file_type",   "documents", ["file_type"])
    op.create_index("idx_documents_department",  "documents", ["department"])
    op.create_index("idx_documents_doc_category","documents", ["doc_category"])
    op.create_index("idx_documents_created_at",  "documents", ["created_at"])
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # ── chunks ─────────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id",               postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("doc_id",           postgresql.UUID(as_uuid=False), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text",             sa.Text(),       nullable=False),
        sa.Column("chunk_index",      sa.Integer(),    nullable=False),
        sa.Column("char_count",       sa.Integer(),    nullable=False, server_default="0"),
        sa.Column("token_count",      sa.Integer(),    nullable=True),
        sa.Column("chunking_strategy",sa.String(32),   nullable=False, server_default="recursive"),
        sa.Column("page_number",      sa.Integer(),    nullable=True),
        sa.Column("section_heading",  sa.String(512),  nullable=True),
        sa.Column("section_id",       sa.String(128),  nullable=True),
        sa.Column("is_embedded",      sa.Boolean(),    nullable=False, server_default="false"),
        sa.Column("embedding_model",  sa.String(128),  nullable=True),
        sa.Column("embedded_at",      sa.DateTime(timezone=True), nullable=True),
        sa.Column("department",       sa.String(128),  nullable=True),
        sa.Column("doc_category",     sa.String(128),  nullable=True),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
    )
    op.create_index("idx_chunks_doc_id",      "chunks", ["doc_id"])
    op.create_index("idx_chunks_is_embedded", "chunks", ["is_embedded"])

    # ── jobs ───────────────────────────────────────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id",              postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("doc_id",          postgresql.UUID(as_uuid=False), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_type",        sa.String(64),  nullable=False, server_default="ingest"),
        sa.Column("status",          sa.String(32),  nullable=False, server_default="pending"),
        sa.Column("celery_task_id",  sa.String(256), nullable=True),
        sa.Column("error",           sa.Text(),      nullable=True),
        sa.Column("progress_pct",    sa.Float(),     nullable=True),
        sa.Column("result_json",     postgresql.JSONB(), nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at",      sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("completed_at",    sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_jobs_doc_id", "jobs", ["doc_id"])
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.execute("""
        CREATE TRIGGER trg_jobs_updated_at
        BEFORE UPDATE ON jobs
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # ── feedback ───────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id",               postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("answer_id",        sa.String(128), nullable=False),
        sa.Column("query",            sa.Text(),      nullable=False),
        sa.Column("answer_text",      sa.Text(),      nullable=False),
        sa.Column("rating",           sa.Integer(),   nullable=False),
        sa.Column("correction",       sa.Text(),      nullable=True),
        sa.Column("bad_citation_ids", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("extra_json",       postgresql.JSONB(), nullable=True),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating"),
    )
    op.create_index("idx_feedback_answer_id", "feedback", ["answer_id"])

    # ── eval_runs ──────────────────────────────────────────────────────────────
    op.create_table(
        "eval_runs",
        sa.Column("id",                postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_name",          sa.String(256), nullable=False),
        sa.Column("dataset_name",      sa.String(256), nullable=False),
        sa.Column("faithfulness",      sa.Float(),     nullable=True),
        sa.Column("answer_relevance",  sa.Float(),     nullable=True),
        sa.Column("context_precision", sa.Float(),     nullable=True),
        sa.Column("context_recall",    sa.Float(),     nullable=True),
        sa.Column("llm_model",         sa.String(128), nullable=True),
        sa.Column("embedding_model",   sa.String(128), nullable=True),
        sa.Column("chunking_strategy", sa.String(64),  nullable=True),
        sa.Column("search_mode",       sa.String(32),  nullable=True),
        sa.Column("rerank_top_n",      sa.Integer(),   nullable=True),
        sa.Column("num_questions",     sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("results_json",      postgresql.JSONB(), nullable=True),
        sa.Column("created_at",        sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )

    # ── Enable pgcrypto for gen_random_uuid() ──────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("feedback")
    op.drop_table("jobs")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;")
