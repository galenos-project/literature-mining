# GALENOS Trends in the Published Mental Health Literature

A Flask dashboard for the [galenos-project/literature-mining](https://github.com/galenos-project/literature-mining)
pipeline, reporting the same corpus/topic/trend workflow described in
[Hastings et al., *BMJ Mental Health* 2026](https://mentalhealth.bmj.com/content/29/1/e302379):
OpenAlex search &rarr; BERTopic topic model &rarr; monthly mention counts &rarr;
GRU time-series model &rarr; trendiness ranking.

## What it does

- **Front page** — corpus stats, a "trending now" ticker, a multi-line
  chart of the top trending topics' mentions over the last 10 years, a
  topic comparison chart (search or pick up to 5 topics to see their
  actual and predicted monthly mentions on one chart), a topic evidence
  map (one bubble per topic, placed by its share of trials/systematic
  reviews and of pre-clinical studies; trending topics in red), and a
  breakdown of all papers by study type.
- **Topic page** — the topic's timeline, its papers by study type, its
  history across monthly updates, and related topics (sharing at least 8
  of 15 keywords, or more than half of the smaller topic's papers).
- **Drill-down** — click any month on a timeline to see the titles,
  abstracts, authors, venue, published/grey-literature status, study type
  and links of the papers that mentioned that topic that month.

## Quickstart (with synthetic sample data)

```bash
pip install -r requirements.txt
python scripts/init_db.py --reset
python scripts/generate_sample_data.py
python app.py
```

Then open http://127.0.0.1:5000. The sample data is entirely synthetic
(fake topics, fake papers, fake authors) — it exists only to make the app
runnable and demoable before you wire in real pipeline output.

## Wiring in real data

The app reads from a single SQLite database (`data/galenos.db`) defined by
`schema.sql`. Five tables — `topics`, `monthly_counts`, `papers`,
`paper_topics`, `topic_history` — populated from **five source files** (plus
the pipeline's history archive) by
`scripts/import_notebook_outputs.py`:

| flag               | file columns                                                                 | populates                                                                            |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--topics`       | `Topic,Name,Words` (optionally `LineageId,...`, as the pipeline writes)    | `topics` (id, name, keywords, lineage)                                             |
| `--monthly`      | `PubDate,Topic0,Topic1,...` (wide, one row per month)                      | `monthly_counts.actual_count`, and `topics.n_papers`                             |
| `--predictions`  | `,Topic,TopicName,Trendy,RankSum,ModelMAE,Pred_M0,...,Pred_M115`           | `monthly_counts.predicted_count`, `topics.is_trendy/trend_rank/trend_mae`        |
| `--papers`       | `PaperId,PaperTitle,Citations,coFoS,Authors,Abstract,Lang,PubYear,PubDate` | `papers`                                                                           |
| `--paper-topics` | same columns as`--papers`, plus binary `Topic0,Topic1,...`               | `paper_topics` (probability fixed at 1.0, since the source is already thresholded) |
| `--history-dir`  | the pipeline's `history/` folder of archived runs (see below)              | `topic_history` (the topic page's History panel)                                   |

```bash
python scripts/init_db.py --reset
python scripts/import_notebook_outputs.py \
    --topics topics.csv \
    --monthly monthly_mentions.csv \
    --predictions trendy_predictions.csv \
    --papers papers.csv \
    --paper-topics paper_topic_assignments.csv
```

All five flags are optional and independent — you can (re)load just one
file at a time as your pipeline output changes.

**Two things worth knowing about this import:**

- **Predicted-month alignment.** The predictions file's `Pred_M0..Pred_M115`
  columns carry no dates. The importer aligns them to the *last* 116 months
  of your `--monthly` file, so the most recent prediction lines up with the
  most recent actual month (working backwards from there). If your export
  uses a different starting point, adjust `align_predicted_months()` in
  `scripts/import_notebook_outputs.py`.

### The topic evidence map

The front page's evidence map plots each topic (with at least 50 classified
papers, `EVIDENCE_MAP_MIN_PAPERS` in `db.py`) by the share of its papers
that are trials or systematic reviews (x) and pre-clinical studies (y),
sized by papers; its table view lists every trending topic's full
study-type profile. It reads `topic_study_types`, a small derived table of
papers per study type per topic that `import_notebook_outputs.py` rebuilds
at the end of every import, together with `topic_relations` (the topic
page's related topics; thresholds `RELATED_MIN_KEYWORDS` and
`RELATED_MIN_PAPER_SHARE` in the importer). Running the importer with no
arguments just rebuilds these two tables.

## Monthly batch updates

`scripts/update_data_monthly.py` runs the full pipeline in one command,
instead of the notebooks by hand: refetch the corpus from OpenAlex → refit the
topic model on the full corpus → name topics → build the binary
paper-topic matrix → rebuild monthly mentions → retrain the trend model →
reload the live database.

```bash
pip install -r requirements-pipeline.txt   
# keys go in trend-dashboard/.env (gitignored; already-exported variables win):
#   OPENALEX_API_KEY=...  OPENALEX_MAILTO=you@example.org  OPENAI_API_KEY=...
#   LLM_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct   (optional DeepInfra model id; this is the default)

python scripts/update_data_monthly.py --data-dir pipeline_data
```

`pipeline_data/` holds the pipeline's own working CSVs (its record of the
full corpus, topics, etc. across runs) — separate from anything you import
manually via `import_notebook_outputs.py`. The corpus is a rolling window
of the 120 complete calendar months before the current one (e.g. a run in
October 2026 covers 2016-10-01 to 2026-09-30). Each run refetches that whole
window rather than only newly published papers, because OpenAlex keeps
adding papers to past months and updating existing records; papers that
have rolled out of the window are dropped. The refetch goes to
`pipeline_data/papers_fetching.csv` and only replaces `papers.csv` once
complete (an interrupted run resumes from it automatically), and it's
refused if the corpus would shrink by more than 5%. Duplicate records are
merged (same PaperId; same title and abstract; or same abstract, authors
and date with one title contained in the other -- see `merge_papers`),
keeping a published copy over a grey-literature one. The run then
refits everything, and does a full `init_db.py --reset` + reimport
(topic ids aren't stable across a full-corpus
refit, so this is a full rebuild each run, not an incremental patch).

### Study type and grey literature

Every paper gets two further labels, stored in `papers.csv` (and the
database's `papers` table):

- **`StudyType`** — one study design per paper, assigned by the LLM from the
  title and abstract (`classify_study_types`): `interventional`,
  `observational`, `qualitative`, `case_report`, `animal`, `in_vitro`,
  `systematic_review`, `other_review`, `protocol`, `other`. Computational
  studies take the label of the data they analyse (e.g. machine learning on
  patient records is `observational`, molecular docking is `in_vitro`). The
  definitions and tie-break rules are in `STUDY_TYPES` /
  `STUDY_TYPE_PROMPT`. Papers OpenAlex types as editorials, letters, peer
  reviews etc. are labelled `other` without an LLM call. Labels are cached
  in `pipeline_data/study_types.csv` by a hash of title + abstract, so only
  new or edited papers are classified each month (the first full pass is
  ~250k LLM calls; it is resumable). Bump `STUDY_TYPE_PROMPT_VERSION` after
  changing the categories or prompt to relabel everything.
- **`GreyLiterature`** — from OpenAlex metadata at fetch time
  (`is_grey_literature`): preprints, dissertations, conference abstracts and
  papers, reports and datasets are grey; anything else is published if it
  has a journal/book location or is indexed in PubMed or DOAJ. When the same
  paper appears both as a preprint and as a published article, the published
  copy is kept.

### Topic history

Topic ids are re-derived on every refit, so continuity between monthly
runs is tracked separately. After the topic model is refit, each new topic
is compared with the previous run's topics: it *continues* one if at
least 10 of their 15 keywords are shared and the shared papers (counting
only papers in both runs' corpora) make up at least 80% of both topics
(`TOPIC_MATCH_MIN_KEYWORDS` / `TOPIC_MATCH_MIN_PAPER_OVERLAP`). Matches are
1-to-1. A continuing topic keeps the previous topic's name and `LineageId`;
only new topics are named by the LLM. UMAP is seeded
(`UMAP_RANDOM_STATE`), so re-running the topic stage on the same corpus
reproduces the same topics.

Each completed run is archived to `pipeline_data/history/<YYYY-MM>/`
(the month the pipeline ran in): `topics.csv` (lineage, name, keywords,
trendiness, size, and the match scores against the previous run, recorded
for the closest previous topic even when unmatched so the thresholds can be
tuned) and `assignments.csv.gz` (`PaperId,Topic`; `-1` for papers in no
topic). The next run maps against the newest earlier month, so re-running
in the same month replaces that month's archive rather than matching
against it. The archive is the permanent record; the database's
`topic_history` table is rebuilt from it on every reload.

Useful flags: `--since YYYY-MM-DD` to override where an interrupted
fetch resumes from, `--skip-fetch` to re-run modelling on the existing corpus only
(e.g. for testing config changes), `--skip-db-reload` to just write the
CSVs without touching `data/galenos.db`.

## Project layout

```
app.py                          Flask routes + JSON API
db.py                           SQLite access layer
config.py
schema.sql                      Data contract (see above)
scripts/
  init_db.py                    Create/reset the database
  generate_sample_data.py       Synthetic demo dataset
  import_notebook_outputs.py    Template loader for real exports
templates/                      Jinja2 pages (base, index, topic, month, 404)
static/css/style.css            Design system
static/js/main.js               Plotly charts + dropdown + click-through
data/galenos.db                 SQLite database (created by the scripts above)
```
