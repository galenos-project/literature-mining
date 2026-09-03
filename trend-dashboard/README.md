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
`schema.sql`. Four tables — `topics`, `monthly_counts`, `papers`,
`paper_topics` — populated from **five source files** by
`scripts/import_notebook_outputs.py`:

| flag               | file columns                                                                 | populates                                                                            |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--topics`       | `Topic,Name,Words`                                                         | `topics` (id, name, keywords)                                                      |
| `--monthly`      | `PubDate,Topic0,Topic1,...` (wide, one row per month)                      | `monthly_counts.actual_count`, and `topics.n_papers`                             |
| `--predictions`  | `,Topic,TopicName,Trendy,RankSum,ModelMAE,Pred_M0,...,Pred_M115`           | `monthly_counts.predicted_count`, `topics.is_trendy/trend_rank/trend_mae`        |
| `--papers`       | `PaperId,PaperTitle,Citations,coFoS,Authors,Abstract,Lang,PubYear,PubDate` | `papers`                                                                           |
| `--paper-topics` | same columns as`--papers`, plus binary `Topic0,Topic1,...`               | `paper_topics` (probability fixed at 1.0, since the source is already thresholded) |

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
computes its own local, lightweight stand-in: `scripts/compute_embeddings.py`
samples ~5 papers per topic, fits a single shared TF-IDF + TruncatedSVD
(LSA) projection across all of them (so every topic's points land in the
same comparable 2D space), and stores the result in `topic_paper_samples`.
This deliberately avoids downloading a transformer model — it's pure
scikit-learn and runs in seconds even for ~1000 topics. If you'd rather use
real sentence embeddings later, swap out `embed_texts()` in that script for
a sentence-transformers call; everything downstream (sampling, storage, the
scatter plot itself) stays the same.

Run it any time after `--papers` and `--paper-topics` have been loaded:

```bash
python scripts/compute_embeddings.py --samples-per-topic 5
```

It's independent of the other import steps and safe to re-run (it clears
and rebuilds `topic_paper_samples` each time), so re-run it whenever the
underlying papers/topic assignments change.

## Monthly batch updates

`scripts/update_data_monthly.py` runs the full pipeline in one command,
instead of the notebooks by hand: fetch new OpenAlex papers → refit the
topic model on the full corpus → name topics → build the binary
paper-topic matrix → rebuild monthly mentions → retrain the trend model →
reload the live database.

```bash
pip install -r requirements-pipeline.txt   
export OPENALEX_MAILTO="you@example.org"   
export OPENAI_API_KEY="..."         

python scripts/update_data_monthly.py --data-dir pipeline_data
```

`pipeline_data/` holds the pipeline's own working CSVs (its record of the
full corpus, topics, etc. across runs) — separate from anything you import
manually via `import_notebook_outputs.py`. Re-running the command each
month fetches only papers published since the newest one already on file,
refits everything, and does a full `init_db.py --reset` + reimport +
`compute_embeddings.py` (topic ids aren't stable across a full-corpus
refit, so this is a full rebuild each run, not an incremental patch).

Useful flags: `--since YYYY-MM-DD` to override the auto-detected fetch
start date, `--skip-fetch` to re-run modelling on the existing corpus only
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
