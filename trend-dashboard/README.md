# GALENOS Trends in the Published Mental Health Literature

A Flask dashboard for the [galenos-project/literature-mining](https://github.com/galenos-project/literature-mining)
pipeline, reporting the same corpus/topic/trend workflow described in
[Hastings et al., *BMJ Mental Health* 2026](https://mentalhealth.bmj.com/content/29/1/e302379):
OpenAlex search &rarr; BERTopic topic model &rarr; monthly mention counts &rarr;
GRU time-series model &rarr; trendiness ranking.

## What it does

- **Front page** — corpus stats, a "trending now" ticker, a multi-line
  chart of the top trending topics' mentions over the last 10 years, a
  topic explorer with actual-vs-predicted timelines, and a paper-landscape
  scatter plot (a sample of papers per topic, positioned by text
  similarity; trending topics in colour, established topics in grey).
- **Topic explorer** — pick any topic from a dropdown to see its full
  monthly timeline (actual mentions vs the model's predicted mentions,
  matching the blue/orange convention in the paper's figures).
- **Drill-down** — click any month on a timeline to see the titles,
  abstracts, authors and links of the papers that mentioned that topic that
  month.

## Quickstart (with synthetic sample data)

```bash
pip install -r requirements.txt
python scripts/init_db.py --reset
python scripts/generate_sample_data.py
python scripts/compute_embeddings.py
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

### The paper-landscape scatter plot

No embeddings were saved by the original pipeline, so this dashboard
computes its own: `scripts/compute_embeddings.py` samples ~5 papers per
topic, embeds each paper's title + abstract with the `all-MiniLM-L6-v2`
sentence-transformer, projects those to 2D with UMAP (one shared
projection, so every topic's points land in the same comparable space),
and stores the result in `topic_paper_samples`. The first run downloads
the ~90 MB model; embedding a few thousand papers on CPU takes a couple
of minutes.

How tightly a topic's papers group in the plot is controlled by the UMAP
parameters, exposed as flags: `--n-neighbors` (lower ⇒ tighter, more
separated clusters), `--min-dist` (near 0.0 ⇒ dense clumps), `--metric`,
plus `--samples-per-topic` and `--seed`.

Run it any time after `--papers` and `--paper-topics` have been loaded:

```bash
python scripts/compute_embeddings.py --samples-per-topic 15 --n-neighbors 10 --min-dist 0.0
```

It's independent of the other import steps and safe to re-run (it clears
and rebuilds `topic_paper_samples` each time), so re-run it whenever the
underlying papers/topic assignments change.

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
refused if the corpus would shrink by more than 5%. The run then
refits everything, and does a full `init_db.py --reset` + reimport +
`compute_embeddings.py` (topic ids aren't stable across a full-corpus
refit, so this is a full rebuild each run, not an incremental patch).

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
  compute_embeddings.py          Derives the scatter-plot sample + 2D layout
templates/                      Jinja2 pages (base, index, topic, month, 404)
static/css/style.css            Design system
static/js/main.js               Plotly charts + dropdown + click-through
data/galenos.db                 SQLite database (created by the scripts above)
```
