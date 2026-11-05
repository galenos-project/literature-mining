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
    trend_mae    REAL,                       -- from `ModelMAE`; the time-series model's mean absolute error for this topic (diagnostic, optional)
    lineage_id   INTEGER                     -- from `LineageId`; links to topic_history (NULL if the topics file has none)
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
    openalex_url      TEXT,               -- derived from PaperId when it looks like an OpenAlex work id
    study_type        TEXT,               -- from StudyType: the pipeline's LLM study-design label (NULL if absent)
    grey_literature   INTEGER,            -- from GreyLiterature: 1 = preprint/thesis/conference/etc., 0 = published
    work_type         TEXT,               -- from WorkType: OpenAlex work type (article, review, preprint, ...)
    source_name       TEXT                -- from SourceName: journal / repository the paper appeared in
);
CREATE INDEX IF NOT EXISTS idx_papers_pubdate ON papers(publication_date);
-- Covering index for the study-type counts on the home page (avoids scanning
-- the whole papers table, abstracts and all).
CREATE INDEX IF NOT EXISTS idx_papers_study_type ON papers(study_type, grey_literature);

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

-- One row per topic lineage per monthly pipeline run, from the archived
-- runs in pipeline_data/history/<YYYY-MM>/topics.csv. A lineage is a chain of
-- topics, one per run, each matched to its predecessor in the previous run
-- (see map_to_previous_run in scripts/update_data_monthly.py); topic_id is
-- only meaningful within its own run. keyword_overlap / paper_overlap
-- describe the match to the previous run's topic, and are NULL for the first
-- run of a lineage.
CREATE TABLE IF NOT EXISTS topic_history (
    lineage_id       INTEGER NOT NULL,
    run_month        TEXT NOT NULL,      -- 'YYYY-MM' the pipeline ran in
    topic_id         INTEGER NOT NULL,
    name             TEXT NOT NULL,
    keywords         TEXT NOT NULL,      -- JSON array
    n_papers         INTEGER,            -- papers assigned to the topic in that run
    is_trendy        INTEGER NOT NULL DEFAULT 0,
    trend_rank       REAL,               -- RankSum in that run
    trendy_position  INTEGER,            -- 1 = highest RankSum among that run's trendy topics; NULL if not trendy
    keyword_overlap  INTEGER,            -- of 15
    paper_overlap    REAL,               -- 0..1
    PRIMARY KEY (lineage_id, run_month)
);

-- Derived data (NOT imported from source files): the number of papers of
-- each study type assigned to each topic, rebuilt by
-- scripts/import_notebook_outputs.py after every import (aggregating
-- paper_topics x papers live takes seconds). Used for the front-page topic
-- evidence map. study_type is NULL for papers not yet classified.
CREATE TABLE IF NOT EXISTS topic_study_types (
    topic_id    INTEGER NOT NULL REFERENCES topics(topic_id),
    study_type  TEXT,
    n_papers    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic_study_types_topic ON topic_study_types(topic_id);
