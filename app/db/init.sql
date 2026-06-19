-- =============================================================================
-- Enterprise Knowledge Intelligence Platform — Database Schema
-- =============================================================================
-- Run once to initialise a fresh Postgres database.
-- For migrations after Day 1, use Alembic (alembic upgrade head).
-- =============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- documents
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename            VARCHAR(512) NOT NULL,
    original_filename   VARCHAR(512) NOT NULL,
    file_type           VARCHAR(32)  NOT NULL,             -- pdf | docx | txt | md | html
    file_path           VARCHAR(1024),
    file_size_bytes     BIGINT,

    -- Processing state
    status              VARCHAR(32)  NOT NULL DEFAULT 'pending',
    -- pending | processing | indexed | failed | archived
    error_message       TEXT,

    -- Extracted content
    raw_text            TEXT,
    page_count          INTEGER,
    word_count          INTEGER,
    chunk_count         INTEGER      NOT NULL DEFAULT 0,

    -- Searchable / filterable metadata (denormalised)
    title               VARCHAR(512),
    author              VARCHAR(256),
    department          VARCHAR(128),
    doc_category        VARCHAR(128),
    language            VARCHAR(16),
    doc_created_date    VARCHAR(64),
    tags                TEXT[],

    -- Arbitrary extra metadata (stored as JSONB for flexibility)
    metadata_json       JSONB,

    -- Timestamps
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    indexed_at          TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_status      ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_file_type   ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_department  ON documents(department);
CREATE INDEX IF NOT EXISTS idx_documents_doc_category ON documents(doc_category);
CREATE INDEX IF NOT EXISTS idx_documents_created_at  ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_tags        ON documents USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_documents_metadata    ON documents USING GIN(metadata_json);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- chunks
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Content
    text                TEXT        NOT NULL,
    chunk_index         INTEGER     NOT NULL,
    char_count          INTEGER     NOT NULL DEFAULT 0,
    token_count         INTEGER,

    -- Chunking provenance
    chunking_strategy   VARCHAR(32) NOT NULL DEFAULT 'recursive',

    -- Source location within the document
    page_number         INTEGER,
    section_heading     VARCHAR(512),
    section_id          VARCHAR(128),

    -- Embedding status
    is_embedded         BOOLEAN     NOT NULL DEFAULT FALSE,
    embedding_model     VARCHAR(128),
    embedded_at         TIMESTAMPTZ,

    -- Denormalised filter fields (copied from parent document)
    department          VARCHAR(128),
    doc_category        VARCHAR(128),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_chunk_doc_index UNIQUE (doc_id, chunk_index)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id       ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_is_embedded  ON chunks(is_embedded);
CREATE INDEX IF NOT EXISTS idx_chunks_department   ON chunks(department);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy     ON chunks(chunking_strategy);

-- =============================================================================
-- jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    job_type            VARCHAR(64) NOT NULL DEFAULT 'ingest',
    -- ingest | re_embed | delete | re_chunk

    status              VARCHAR(32) NOT NULL DEFAULT 'pending',
    -- pending | running | success | failed | retrying

    celery_task_id      VARCHAR(256),
    error               TEXT,
    progress_pct        FLOAT,
    result_json         JSONB,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_doc_id   ON jobs(doc_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status   ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type     ON jobs(job_type);

DROP TRIGGER IF EXISTS trg_jobs_updated_at ON jobs;
CREATE TRIGGER trg_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- feedback
-- =============================================================================
CREATE TABLE IF NOT EXISTS feedback (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_id           VARCHAR(128) NOT NULL,
    query               TEXT        NOT NULL,
    answer_text         TEXT        NOT NULL,
    rating              INTEGER     NOT NULL CHECK (rating BETWEEN 1 AND 5),
    correction          TEXT,
    bad_citation_ids    TEXT[],
    extra_json          JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_answer_id ON feedback(answer_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating    ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created   ON feedback(created_at DESC);

-- =============================================================================
-- eval_runs  (evaluation framework — Day 20)
-- =============================================================================
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name            VARCHAR(256) NOT NULL,
    dataset_name        VARCHAR(256) NOT NULL,

    -- RAGAS metric scores (0.0 – 1.0)
    faithfulness        FLOAT,
    answer_relevance    FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,

    -- Config snapshot (so we can compare runs)
    llm_model           VARCHAR(128),
    embedding_model     VARCHAR(128),
    chunking_strategy   VARCHAR(64),
    search_mode         VARCHAR(32),
    rerank_top_n        INTEGER,

    num_questions       INTEGER     NOT NULL DEFAULT 0,
    results_json        JSONB,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at DESC);
