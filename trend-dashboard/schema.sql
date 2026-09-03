-- GALENOS Trends dashboard — SQLite schema
--
-- This is the data contract between your literature-mining pipeline output
-- files and this Flask app. Populate these tables using
-- scripts/import_notebook_outputs.py, or run scripts/generate_sample_data.py
-- to get a synthetic dataset that exercises the whole app immediately.

PRAGMA foreign_keys = ON;

-- One row per topic.
-- `keywords` is a JSON array parsed from the topics file's `Words` column.
CREATE TABLE IF NOT EXISTS topics (
    topic_id     INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    keywords     TEXT NOT NULL,             -- JSON array, e.g. ["ketamine","depression","..."]
    n_papers     INTEGER NOT NULL DEFAULT 0, -- sum of actual_count across all months
    is_trendy    INTEGER NOT NULL DEFAULT 0, -- 0/1, from the trendy-predictions file's `Trendy` column
    trend_rank   REAL,                       -- from `RankSum`; NULL if not trendy
    trend_mae    REAL                        -- from `ModelMAE`; the time-series model's mean absolute error for this topic (diagnostic, optional)
);

-- One row per topic per calendar month: actual mentions (from the monthly
-- mentions file's wide Topic0..TopicN columns) vs the predicted mentions
-- (from the trendy-predictions file's Pred_M0..Pred_M115 columns).
-- `predicted_count` is NULL for months the predictions file doesn't cover
-- (older months, before the model's prediction window begins).
CREATE TABLE IF NOT EXISTS monthly_counts (
    topic_id        INTEGER NOT NULL REFERENCES topics(topic_id),
    month           TEXT NOT NULL,      -- ISO date, first-of-month: 'YYYY-MM-01'
    actual_count    INTEGER NOT NULL,
    predicted_count REAL,
    PRIMARY KEY (topic_id, month)
);
CREATE INDEX IF NOT EXISTS idx_monthly_counts_topic_month
    ON monthly_counts(topic_id, month);

-- One row per paper, from the search-results file
-- (PaperId,PaperTitle,Citations,coFoS,Authors,Abstract,Lang,PubYear,PubDate).
CREATE TABLE IF NOT EXISTS papers (
    paper_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    abstract          TEXT,
    authors           TEXT,               -- as given in the Authors column
    publication_date  TEXT,               -- normalised ISO date 'YYYY-MM-DD'
    citations         INTEGER,            -- from Citations
    fields_of_study   TEXT,               -- from coFoS
    language          TEXT,               -- from Lang
    openalex_url      TEXT                -- derived from PaperId when it looks like an OpenAlex work id
);
CREATE INDEX IF NOT EXISTS idx_papers_pubdate ON papers(publication_date);

-- Many-to-many: which topics a paper is flagged for, melted from the
-- paper-topic assignment file's binary Topic0..TopicN columns.
-- `probability` is 1.0 for every row here, since that file is already
-- thresholded/binary rather than a continuous probability.
CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id),
    topic_id     INTEGER NOT NULL REFERENCES topics(topic_id),
    probability  REAL NOT NULL,
    PRIMARY KEY (paper_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_paper_topics_topic ON paper_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_paper_topics_paper ON paper_topics(paper_id);

-- Derived data (NOT imported from source files): a small sample of papers
-- per topic with 2D text-embedding coordinates, computed locally by
-- scripts/compute_embeddings.py, used for the front-page scatter plot.
-- A paper can appear more than once here if it's sampled under more than
-- one topic it's assigned to; its (embed_x, embed_y) will be identical
-- across those rows, since the coordinate is a property of the paper's
-- text, not of the topic.
CREATE TABLE IF NOT EXISTS topic_paper_samples (
    topic_id  INTEGER NOT NULL REFERENCES topics(topic_id),
    paper_id  TEXT NOT NULL REFERENCES papers(paper_id),
    embed_x   REAL NOT NULL,
    embed_y   REAL NOT NULL,
    PRIMARY KEY (topic_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_topic_paper_samples_topic ON topic_paper_samples(topic_id);
