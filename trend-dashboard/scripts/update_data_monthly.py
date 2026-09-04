"""Monthly batch update: the full literature-mining pipeline in one script.

Replaces running the original notebooks by hand each month:

  1. fetch_new_papers()          -- OpenAlex search, merged into the corpus
  2. fit_topic_model()            -- BERTopic refit on the FULL corpus
  3. name_topics()                 -- LLM-based short names from keywords
  4. build_paper_topic_matrix()    -- binary Topic0..N columns, thresholded
  5. build_monthly_mentions()      -- wide PubDate,Topic0,Topic1,... counts
  6. fit_trend_model()             -- GRU walk-forward prediction + Trendy/RankSum

...then writes all five CSVs to --data-dir and does a full
init_db --reset + import_notebook_outputs.py + compute_embeddings.py refresh
of the live database (topic ids aren't stable across a full-corpus refit, so
this is a full rebuild each run, not an incremental patch).

Usage:
    python scripts/update_data_monthly.py --data-dir pipeline_data
    python scripts/update_data_monthly.py --data-dir pipeline_data --fetch-only  # fetch + dedupe papers.csv, then stop
    python scripts/update_data_monthly.py --data-dir pipeline_data --topic-model-only  # topic stage only: refit + name + write topics.csv/paper_topic_assignments.csv, then stop
    python scripts/update_data_monthly.py --data-dir pipeline_data --skip-fetch  # rerun modelling only
    python scripts/update_data_monthly.py --data-dir pipeline_data --skip-fetch --skip-topic-model # rerun trend prediction only
"""
import argparse
import math
import os
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np 

import requests

import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

search_term = '("mental" OR "psychological" OR "behavioural" OR "psychology" OR "psychiatry" OR "neurological" OR "mind" OR "brain" OR "behaviour" OR "psychiatric") \
              AND ("anxiety" OR "depression" OR "psychosis") AND ("treatment" OR "therapy" OR "therapeutic" OR "mechanism" OR "intervention" OR "early" OR "diagnosis" OR "diagnostic" OR "translation")'
OPENALEX_FILTER = f"has_abstract:true,title_and_abstract.search:{search_term},language:en" 
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "user.name@example.com")  # OpenAlex's "polite pool"
FETCH_CITATIONS = True  # True fetches each paper's full citing-ID list (one extra API call per paper — slow)

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")  # optional; unlocks OpenAlex's premium rate limits
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")      # DeepInfra token (used with the base_url below)


ASSIGNMENT_THRESHOLD = 0.08     # per the paper's methodology
SLIDING_WINDOW_MONTHS = 6       # per the paper's methodology
TREND_N_MONTHS = 4              # of the last...
TREND_M_MONTHS = 6               # ...months, actual must exceed predicted (the paper's rule) -- AND:
TREND_RECENT_N_MONTHS = 2      # of the last...
TREND_RECENT_M_MONTHS = 3      # ...months, actual must ALSO exceed predicted (outperformance still current)
TREND_MIN_SLOPE = 0.0         # AND the OLS slope of actual over the last TREND_M_MONTHS must exceed this
TREND_HIDDEN_SIZE = 10          # GRU hidden units (both stacked layers)
TREND_EPOCHS = 10
TREND_BATCH_SIZE = 32
TREND_LEAVE_K_OUT = 10         # topics held out (and predicted) per training run; 1 == exact leave-one-topic-out
TREND_SIZE_NORM_EXPONENT = 0.6 # RankSum denominator is mean_actual ** this. 1.0 = original (÷mean, biased
                               # toward small/noisy topics); 0.5 = ÷Poisson SD, ~size-neutral; 0.0 = no size norm


# ==========================================================================
# 1. Fetch new papers from OpenAlex
# ==========================================================================

def reconstruct_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]} to save space."""
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_citing_ids(work, mailto, max_ids=2000):
    """Follow a work's cited_by_api_url to collect the OpenAlex ids of
    papers that cite it (pipe-delimited, to match the existing Citations
    column format). Slow -- one extra paginated API call per paper."""
    url = work.get("cited_by_api_url")
    if not url:
        return ""
    ids, cursor = [], "*"
    while cursor and len(ids) < max_ids:
        resp = requests.get(url, params={"cursor": cursor, "per-page": 200, "mailto": mailto}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ids.extend(r["id"].rsplit("/", 1)[-1] for r in data["results"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not data["results"]:
            break
    return "|".join(ids[:max_ids])


def normalize_work(work, mailto, fetch_citations):
    paper_id = work["id"].rsplit("/", 1)[-1]
    title = work.get("title") or work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    authors = "; ".join(
        a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")
    )
    concepts = sorted(work.get("concepts") or [], key=lambda c: -(c.get("score") or 0))[:3]
    fields_of_study = "; ".join(c["display_name"] for c in concepts)
    citations = fetch_citing_ids(work, mailto) if fetch_citations else ""

    return {
        "PaperId": paper_id,
        "PaperTitle": title,
        "Citations": citations,
        "coFoS": fields_of_study,
        "Authors": authors,
        "Abstract": abstract,
        "Lang": work.get("language") or "",
        "PubYear": work.get("publication_year"),
        "PubDate": work.get("publication_date") or "",
    }


def fetch_new_papers(since_date, filter_str=OPENALEX_FILTER, mailto=OPENALEX_MAILTO,
                      api_key=OPENALEX_API_KEY, fetch_citations=FETCH_CITATIONS, max_pages=None):
    """Page through OpenAlex works matching `filter_str` published on/after
    `since_date` (a date or 'YYYY-MM-DD' string).

    Returns (rows, resume_date, complete):
      - rows: list of dicts matching the papers.csv schema (whatever was
        successfully fetched -- possibly not all of it, see below). Papers
        dated beyond the current calendar month are excluded (OpenAlex
        occasionally has forward-dated entries; the dashboard would count
        them in a month that hasn't happened yet).
      - resume_date: the publication date to pass as --since on the next
        run to continue from here. Deliberately the same date as the last
        paper actually fetched (not the day after), since OpenAlex's
        from_publication_date filter is inclusive: if a failure happens
        mid-page, there could be other papers with that exact same date
        still unfetched. Re-including that one day means a handful of
        already-fetched papers get refetched (harmless -- merge_papers()
        dedupes by PaperId and by title+abstract), which is a small price
        for not silently
        missing same-day papers.
      - complete: False if a network error cut the fetch short (results
        are sorted oldest-first, so `rows` is everything from `since_date`
        up to `resume_date`, not a scattered partial sample); True if
        pagination ran to completion normally (including if it stopped
        early on hitting future-dated papers -- see below).

    Network errors (timeouts, connection resets, HTTP errors) are retried
    a few times with backoff before being treated as a real failure, so a
    single transient blip doesn't lose an otherwise-long fetch run.
    """
    if isinstance(since_date, date):
        since_date = since_date.isoformat()

    base_url = "https://api.openalex.org/works"
    full_filter = f"{filter_str},from_publication_date:{since_date}"
    cursor = "*"
    rows = []
    resume_date = since_date
    current_month = date.today().strftime("%Y-%m")
    page = 0
    max_retries = 3
    retry_backoff_seconds = (5, 15, 30)

    while cursor:
        data = None
        last_error = None
        params = {
            "filter": full_filter, "per-page": 200, "cursor": cursor,
            "sort": "publication_date:asc",  # so "resume from the latest date fetched" is valid,
                                              # and so future-dated entries (if any) all end up at the end
            "mailto": mailto,
        }
        if api_key:
            params["api_key"] = api_key

        for attempt in range(max_retries):
            try:
                resp = requests.get(base_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"  page {page + 1}: request failed ({e}); "
                      f"attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_backoff_seconds[attempt])

        if data is None:
            print(f"  giving up after {max_retries} attempts on page {page + 1}. "
                  f"{len(rows)} paper(s) fetched so far this run ({last_error}).")
            print(f"  RESUME FROM: --since {resume_date}")
            return rows, resume_date, False

        results = data.get("results", [])
        hit_future_date = False
        for work in results:
            row = normalize_work(work, mailto, fetch_citations)
            if row["PubDate"] and row["PubDate"][:7] > current_month:
                # Sorted ascending, so every remaining paper (this page and
                # all later pages) is also future-dated -- stop entirely
                # rather than paginating further just to skip them.
                print(f"  reached a future-dated paper ({row['PaperId']}, {row['PubDate']}); stopping fetch here")
                hit_future_date = True
                break
            rows.append(row)
            if row["PubDate"]:
                resume_date = max(resume_date, row["PubDate"])
        if hit_future_date:
            return rows, resume_date, True

        cursor = data.get("meta", {}).get("next_cursor")
        page += 1
        if not results or (max_pages and page >= max_pages):
            break
        time.sleep(0.1)  # be polite to the API

    return rows, resume_date, True


def _normalize_text(series):
    """Lowercase, collapse internal whitespace and strip surrounding
    whitespace/punctuation, so trivially different renderings of the same
    title or abstract compare equal."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.strip(".,;:-–— ")
    )


def merge_papers(existing_df, new_rows):
    """Append new_rows to existing_df and dedupe, in two passes:

      1. by PaperId -- new rows win on conflict, though the
         from_publication_date filter should make those rare;
      2. by normalized title + abstract -- OpenAlex sometimes indexes the
         same paper under more than one work id (a preprint and its
         published version, or a plain duplicate record), so identical
         content slips past the PaperId pass. The copy with the longest
         abstract is kept, ties broken by the earliest PubDate, on the
         assumption that's the most complete / canonical record.

    Rows with neither a title nor an abstract are dropped outright -- they
    carry nothing for the topic model. Both cleanups run over the whole
    merged corpus, so they also tidy up anything already sitting in an
    existing papers.csv from earlier runs.
    """
    new_df = pd.DataFrame(new_rows)
    if existing_df is None or existing_df.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing_df
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    if merged.empty:
        return merged

    n_before = len(merged)
    merged = merged.drop_duplicates(subset="PaperId", keep="last").reset_index(drop=True)
    n_after_id = len(merged)
    n_empty = 0

    if {"PaperTitle", "Abstract"}.issubset(merged.columns):
        norm_abstract = _normalize_text(merged["Abstract"])
        content_key = _normalize_text(merged["PaperTitle"]) + "\x1f" + norm_abstract

        # Drop rows with neither a title nor an abstract -- nothing to model.
        has_content = content_key.str.strip("\x1f ") != ""
        n_empty = int((~has_content).sum())
        merged = merged[has_content].reset_index(drop=True)
        norm_abstract = norm_abstract[has_content].reset_index(drop=True)
        content_key = content_key[has_content].reset_index(drop=True)

        # Visit the richest copy (longest abstract, then earliest PubDate)
        # first so duplicated(keep="first") drops the poorer copies.
        rank = pd.DataFrame({
            "abstract_len": norm_abstract.str.len(),
            "pubdate": pd.to_datetime(merged.get("PubDate"), errors="coerce"),
        })
        visit_order = rank.sort_values(
            ["abstract_len", "pubdate"], ascending=[False, True]
        ).index
        drop_idx = visit_order[content_key.loc[visit_order].duplicated(keep="first")]
        merged = merged.drop(index=drop_idx).reset_index(drop=True)

    n_after_content = len(merged)
    if n_before != n_after_id:
        print(f"  deduped {n_before - n_after_id} paper(s) sharing a PaperId")
    if n_empty:
        print(f"  dropped {n_empty} paper(s) with no title or abstract")
    if n_after_id - n_empty != n_after_content:
        print(f"  deduped {n_after_id - n_empty - n_after_content} paper(s) with a duplicate title + abstract")
    return merged

# ==========================================================================
# 2. Topic model (full-corpus refit)
# ==========================================================================

def build_documents(papers_df):
    titles = papers_df["PaperTitle"].fillna("")
    abstracts = papers_df["Abstract"].fillna("")
    return (titles + ". " + abstracts).str.strip().tolist()


def fit_topic_model(papers_df):
    """Refit BERTopic on the full corpus. Returns (topic_model, docs,
    topic_distr) where topic_distr is BERTopic's approximate_distribution()
    output: shape (n_docs, n_topics), columns = sequential topic ids
    0..n_topics-1 (the -1 outlier topic is excluded, matching the paper's
    methodology of only assigning documents to real topics above a
    probability threshold)."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    docs = build_documents(papers_df)

    # Define custom stopwords and initialize CountVectorizer
    custom_stopwords = ["figure", "fig","doi","https","org","disease","clinical","study","www"]
    with open('./data/pubmed.txt', 'r') as file:
        pubmed_stopwords = file.readlines()
        # Remove newline characters
        pubmed_stopwords = [line.strip() for line in pubmed_stopwords]
    all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords).union(pubmed_stopwords)) 

    vectorizer_model = CountVectorizer(
        stop_words=all_stopwords,
        min_df=5  # Only include words that appear in at least 5 documents
    )

    # Initialize Sentence-BERT model for embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Instantiate UMAP and HDBSCAN with desired parameters
    # Ensure these are NOT dictionaries
    umap_model = UMAP(n_neighbors=20, n_components=15, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=25, min_samples=15, metric='euclidean')

    # Initialize BERTopic with more words per topic
    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=15,  # The number of words to retrieve per topic
        verbose=False #,
        #nr_topics="auto"
    )
    topic_model.fit_transform(docs)

    topic_distr, _ = topic_model.approximate_distribution(docs)
    return topic_model, docs, topic_distr


def summarize_topic_model(topic_model):
    """Print how the current clustering parameters carved up the corpus:
    topic count, the share of documents left in the -1 outlier bucket, and
    the spread of topic sizes -- the numbers to watch when tuning
    min_cluster_size / n_neighbors for larger or smaller topics."""
    info = topic_model.get_topic_info()  # one row per topic, includes -1
    total = int(info["Count"].sum())
    n_outliers = int(info.loc[info["Topic"] == -1, "Count"].sum())
    sizes = info.loc[info["Topic"] != -1, "Count"]

    print(f"  {len(sizes)} topics; {n_outliers} / {total} docs "
          f"({n_outliers / total:.1%}) left as -1 outliers")
    if not sizes.empty:
        print(f"  topic sizes -- min {int(sizes.min())}, median {int(sizes.median())}, "
              f"mean {sizes.mean():.0f}, max {int(sizes.max())}")
        for thresh in (20, 30, 50):
            print(f"    {int((sizes < thresh).sum())} topic(s) with < {thresh} papers")


# ==========================================================================
# 3. Topic naming
# ==========================================================================

def name_topics(topic_model):
    topic_ids = sorted(t for t in topic_model.get_topics().keys() if t != -1)

    from openai import OpenAI

    SYSTEM_MSG = "You are a helpful expert assistant for working with topics from the scientific literature in the field of mental health."
    modelname = "Qwen/Qwen3-Next-80B-A3B-Instruct" #"meta-llama/Llama-3.3-70B-Instruct"
    client = OpenAI(
            api_key = OPENAI_API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
    )
    def generateFromPrompt(promptStr,maxTokens=100):
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": promptStr}
        ]
        completion = client.chat.completions.create(
        model=modelname,
        messages=messages)
        response=completion.choices[0].message.content
        return(response)

    results = {}
    for topic_id in topic_ids:
        keywords = [w for w, _ in topic_model.get_topic(topic_id)[:15]]
        try:
            name = generateFromPrompt(f"Please give a concise word or phrase to describe the main research topic within the field of mental health that unifies the following words and indicates the mental health relevance: {keywords}. Please return only the word or phrase with no explanation. Topic: ")
        except Exception as e:  # Keep the pipeline running even if naming fails?
            print(f"  LLM naming failed for topic {topic_id} ({e})")
        results[topic_id] = (name, keywords)
    return results


# ==========================================================================
# 4. Binary paper-topic assignment matrix
# ==========================================================================

def build_paper_topic_matrix(papers_df, topic_distr, threshold=ASSIGNMENT_THRESHOLD):
    """Return a DataFrame with the papers.csv columns plus binary
    Topic0..TopicN columns, matching the paper-topic assignment file
    format the importer expects."""
    n_topics = topic_distr.shape[1]
    binary = (topic_distr > threshold).astype(int)
    topic_cols = pd.DataFrame(binary, columns=[f"Topic{i}" for i in range(n_topics)])
    out = pd.concat([papers_df.reset_index(drop=True), topic_cols], axis=1)
    return out


# ==========================================================================
# 5. Monthly mention counts (wide format)
# ==========================================================================

def build_monthly_mentions(paper_topic_matrix):
    """Aggregate the binary paper-topic matrix into wide monthly mention
    counts, one row per calendar month.

    The current (in-progress) calendar month is dropped here: OpenAlex
    indexing lags real publication dates, so counts for the current month
    are always an undercount until the month is actually over. Excluding
    it once, at the source, means every downstream consumer -- the
    dashboard's chart, the trend model, anyone else reading
    monthly_mentions.csv directly -- sees an honest, complete series,
    rather than each one having to separately work around an incomplete
    final row.
    """
    topic_cols = [c for c in paper_topic_matrix.columns if re.match(r"^Topic\d+$", c)]
    df = paper_topic_matrix.copy()
    df["Month"] = pd.to_datetime(df["PubDate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["Month"])

    current_month = pd.Timestamp.now().to_period("M").to_timestamp()
    df = df[df["Month"] < current_month]

    monthly = df.groupby("Month")[topic_cols].sum().sort_index()
    monthly.index.name = "PubDate"
    monthly = monthly.reset_index()
    monthly["PubDate"] = monthly["PubDate"].dt.strftime("%Y-%m-%d")
    return monthly


# ==========================================================================
# 6. Trend prediction model
#
# This is a PyTorch port of a TensorFlow/Keras notebook, kept deliberately
# close to the original's specific (and slightly unusual) design rather
# than "cleaned up", since matching it was the point:
#
#   - LEAVE-k-TOPICS-OUT TRAINING: topics are split into consecutive
#     groups of TREND_LEAVE_K_OUT (k). For each group, a fresh GRU is
#     trained from scratch on every topic OUTSIDE the group, then used to
#     predict each of the k held-out topics. k == 1 is exact
#     leave-one-topic-out, as in the original notebook -- N full training
#     runs. k > 1 trades a tiny, unbiased change in each training set
#     (N-k vs N-1 topics pooled) for a roughly k-fold speedup; no held-out
#     topic ever contributes to the model that predicts it, so the
#     no-self-leakage property is unchanged. The k topics in a group share
#     one trained model, so they also share its ModelMAE in the output.
#   - SHARED MINMAX SCALING: one MinMaxScaler is fit per training run,
#     on the pooled raw values of every training topic (never the held-out
#     group), and that exact same fitted scaler is reused -- transform only,
#     never refit -- to scale each held-out topic before prediction.
#   - WINDOWING: `range(len(series) - look_back)` -- every look_back-month
#     window predicts the very next month, right up to the last month in
#     `series`. The original notebook used `- look_back - 1` here, dropping
#     one extra month, in order to dodge the current (incomplete) month
#     sneaking into training. That's now handled explicitly and once, in
#     build_monthly_mentions() (which drops the in-progress current month
#     before it ever reaches this function) -- so `series` is already
#     guaranteed to end on a complete month, and this windowing can use
#     all of it rather than quietly discarding one more.
#   - TRENDY FLAG: the paper's rule (actual > predicted in >=
#     TREND_N_MONTHS of the last TREND_M_MONTHS) AND-ed with two extra
#     gates -- a recency check (>= TREND_RECENT_N_MONTHS of the last
#     TREND_RECENT_M_MONTHS) and a positive OLS slope of actual over the
#     last TREND_M_MONTHS. See _trend_rank_and_flag().
#   - RankSum is computed for every topic,
#     weighting each of the last TREND_M_MONTHS months by
#     e^(1/(TREND_M_MONTHS - i)) -- i.e. putting more weight on the most
#     recent months, not a smooth decay. The weighted excess is then
#     divided by mean_actual ** TREND_SIZE_NORM_EXPONENT; the original
#     paper used exponent 1.0 (plain mean), 0.5 (Poisson SD) removes the
#     resulting bias toward small, naturally-noisier topics -- see
#     _trend_rank_and_flag().
# ==========================================================================

def make_windows(values, look_back):
    """(X, y) sliding windows from one series: every look_back-month window
    predicts the very next month, through to the end of `values`. Assumes
    `values` already ends on a complete month -- see build_monthly_mentions(),
    which drops the in-progress current month before series get here."""
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i + look_back])
        y.append(values[i + look_back])
    return X, y


def _trend_rank_and_flag(actual_tail, predicted_tail, mean_actual, n=TREND_N_MONTHS, m=TREND_M_MONTHS,
                          recent_n=TREND_RECENT_N_MONTHS, recent_m=TREND_RECENT_M_MONTHS,
                          min_slope=TREND_MIN_SLOPE, size_norm_exponent=TREND_SIZE_NORM_EXPONENT):
    """Trendiness scoring, extending the original paper's rule. A topic is
    flagged trendy only if ALL of:
      1. actual > predicted in >= n of the last m months (the paper's rule);
      2. actual > predicted in >= recent_n of the last recent_m months, so
         the outperformance is still current rather than fading;
      3. the ordinary-least-squares slope of actual over the last m months
         is > min_slope, so the counts are genuinely rising and not merely
         sitting above an under-shooting prediction.

    RankSum is unchanged and still returned for every topic regardless of
    the flag: it sums the positive monthly excesses (actual - predicted),
    each weighted by e^(1/(m-i)) (i=0 oldest of the m, i=m-1 most recent),
    divided by `max(mean_actual, 1) ** size_norm_exponent`. The original
    used exponent 1.0 (plain mean), which over-normalizes -- monthly counts
    are ~Poisson, so noise scales as sqrt(mean), and dividing an absolute
    excess by the mean leaves small topics with an inflated score from
    noise alone. 0.5 divides by the Poisson standard deviation instead
    (~size-neutral under the null); 0.0 disables it."""
    exceed_count = sum(1 for a, p in zip(actual_tail, predicted_tail) if a > p)
    recent_exceed = sum(
        1 for a, p in zip(actual_tail[-recent_m:], predicted_tail[-recent_m:]) if a > p
    )
    if len(actual_tail) >= 2:
        xs = np.arange(len(actual_tail), dtype=float)
        slope = float(np.polyfit(xs, np.asarray(actual_tail, dtype=float), 1)[0])
    else:
        slope = 0.0

    trendy = exceed_count >= n and recent_exceed >= recent_n and slope > min_slope

    terms = [
        0.0 if a < p else float((a - p) * math.exp(1 / (m - i)))
        for i, (a, p) in enumerate(zip(actual_tail, predicted_tail))
    ]
    rank_sum = sum(terms) / (max(mean_actual, 1.0) ** size_norm_exponent)
    return trendy, rank_sum


def _build_series(monthly_mentions_df):
    topic_cols = [c for c in monthly_mentions_df.columns if re.match(r"^Topic\d+$", c)]
    monthly_sorted = monthly_mentions_df.sort_values("PubDate")
    return {
        int(col.replace("Topic", "")): monthly_sorted[col].to_numpy(dtype=float)
        for col in topic_cols
    }


def fit_trend_model(monthly_mentions_df, window=SLIDING_WINDOW_MONTHS, epochs=TREND_EPOCHS,
                     batch_size=TREND_BATCH_SIZE, leave_k_out=TREND_LEAVE_K_OUT,
                     checkpoint_path=None):
    """Returns a DataFrame matching the trendy-predictions.csv format:
    ,Topic,TopicName,Trendy,RankSum,ModelMAE,Pred_M0,...,Pred_M{K-1}

    Trains one leave-`leave_k_out`-topics-out GRU per group of that many
    topics (see module note above): roughly ceil(N_topics / leave_k_out)
    full training runs. leave_k_out=1 is exact leave-one-topic-out (one run
    per topic, slow); the default trades a small, unbiased change in each
    training set for a ~leave_k_out-fold speedup. Writes an interim
    checkpoint to `checkpoint_path` after every group, mirroring the
    original notebook's crash-safety behaviour on long runs.
    """
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("  torch/scikit-learn not installed; using a trailing-mean baseline instead of the GRU")
        return _fit_trend_baseline(monthly_mentions_df, window)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TrendGRU(nn.Module):
        """Mirrors Sequential([GRU(10, return_sequences=True), GRU(10), Dense(1)])."""

        def __init__(self, hidden_size=TREND_HIDDEN_SIZE):
            super().__init__()
            self.gru1 = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru1(x)      # return_sequences=True: pass the full sequence on
            _, h2 = self.gru2(out)     # return_sequences=False: only the final hidden state
            return self.head(h2.squeeze(0))

    def train_one_model(train_topic_ids, series):
        # ONE scaler, shared by every training topic AND (in predict_topic
        # below) the held-out topic -- see the module note above for why
        # per-topic scaling on both sides actively hurts this, despite
        # fixing the original train/test mismatch.
        scaler = MinMaxScaler(feature_range=(0, 1))
        stacked = np.concatenate([series[t] for t in train_topic_ids]).reshape(-1, 1)
        scaler.fit(stacked)

        X, y = [], []
        for tid in train_topic_ids:
            scaled = scaler.transform(series[tid].reshape(-1, 1)).flatten()
            wx, wy = make_windows(scaled, window)
            X.extend(wx)
            y.extend(wy)

        X_t = torch.tensor(np.array(X, dtype=np.float32)).unsqueeze(-1).to(device)
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(-1).to(device)

        model = TrendGRU().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Training-set MAE, matching the original (computed on the same
        # data the model was just trained on, not a held-out split).
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_t).cpu().numpy()
        pred_actual = scaler.inverse_transform(pred_scaled)
        y_actual = scaler.inverse_transform(y_t.cpu().numpy())
        mae = float(np.mean(np.abs(pred_actual - y_actual)))
        return model, scaler, mae

    def predict_topic(model, scaler, topic_id, series):
        # Reuse the SAME scaler the model was trained with -- transform
        # only, no refit -- so the held-out topic is expressed on exactly
        # the numeric scale the model actually learned from.
        scaled = scaler.transform(series[topic_id].reshape(-1, 1)).flatten()
        X_test, y_test = make_windows(scaled, window)
        if not X_test:
            return np.array([]), np.array([])
        X_test_t = torch.tensor(np.array(X_test, dtype=np.float32)).unsqueeze(-1).to(device)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy()
        y_pred = scaler.inverse_transform(pred_scaled).flatten()
        y_actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        return y_actual, y_pred

    series = _build_series(monthly_mentions_df)
    topic_ids = sorted(series.keys())
    k = max(1, int(leave_k_out))
    groups = [topic_ids[i:i + k] for i in range(0, len(topic_ids), k)]
    print(f"  {len(topic_ids)} topics in {len(groups)} leave-{k}-out group(s); "
          f"one GRU trained per group")

    rows = []
    done = 0
    for gi, group in enumerate(groups):
        held_out = set(group)
        train_topic_ids = [t for t in topic_ids if t not in held_out]
        model, scaler, mae = train_one_model(train_topic_ids, series)

        for topic_id in group:
            done += 1
            actual, predicted = predict_topic(model, scaler, topic_id, series)

            if len(actual) == 0:
                print(f"  topic {topic_id}: not enough monthly history to predict; skipping")
                continue

            last_actual = actual[-TREND_M_MONTHS:]
            last_pred = predicted[-TREND_M_MONTHS:]
            mean_actual = float(np.mean(actual))
            trendy, rank_sum = _trend_rank_and_flag(last_actual, last_pred, mean_actual)

            row = {
                "": topic_id,
                "Topic": topic_id,
                "TopicName": "",  # informational only; the importer uses topics.csv's Name as authoritative
                "Trendy": bool(trendy),
                "RankSum": rank_sum,
                "ModelMAE": mae,  # shared by every topic in this leave-k-out group
            }
            for i, p in enumerate(predicted):
                row[f"Pred_M{i}"] = float(p)
            rows.append(row)

            print(f"  [{done}/{len(topic_ids)}] topic {topic_id} "
                  f"(group {gi + 1}/{len(groups)}): MAE={mae:.4f} trendy={trendy} rank={rank_sum:.4f}")

        if checkpoint_path:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(rows)


def _fit_trend_baseline(monthly_mentions_df, window):
    """Non-torch fallback: a trailing-mean prediction per topic (using that
    topic's own history, not leave-one-topic-out), scored with the same
    rank formula. Not a faithful reproduction of the original -- just keeps
    the pipeline runnable without torch installed."""
    series = _build_series(monthly_mentions_df)
    rows = []
    for topic_id, values in series.items():
        preds = [sum(values[t - window:t]) / window for t in range(window, len(values))]
        actual_tail = values[-TREND_M_MONTHS:]
        pred_tail = preds[-TREND_M_MONTHS:]
        mean_actual = float(np.mean(values)) if len(values) else 0.0
        trendy, rank_sum = _trend_rank_and_flag(actual_tail, pred_tail, mean_actual)
        mae = float(np.mean(np.abs(np.array(actual_tail) - np.array(pred_tail)))) if pred_tail else None
        row = {
            "": topic_id, "Topic": topic_id, "TopicName": "",
            "Trendy": bool(trendy), "RankSum": rank_sum, "ModelMAE": mae,
        }
        for i, p in enumerate(preds):
            row[f"Pred_M{i}"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Orchestration
# ==========================================================================

def latest_pub_date(papers_df):
    if papers_df is None or papers_df.empty:
        return None
    dates = pd.to_datetime(papers_df["PubDate"], errors="coerce").dropna()
    return dates.max().date() if not dates.empty else None


def run_pipeline(data_dir, since=None, skip_fetch=False, skip_topic_model=False,
                 fetch_only=False, topic_model_only=False):
    os.makedirs(data_dir, exist_ok=True)
    papers_path = os.path.join(data_dir, "papers.csv")
    topics_path = os.path.join(data_dir, "topics.csv")
    monthly_path = os.path.join(data_dir, "monthly_mentions.csv")
    predictions_path = os.path.join(data_dir, "trendy_predictions.csv")
    paper_topics_path = os.path.join(data_dir, "paper_topic_assignments.csv")

    existing_papers = pd.read_csv(papers_path) if os.path.exists(papers_path) else pd.DataFrame()

    print("[1/6] Fetching new papers from OpenAlex...")
    if skip_fetch:
        print("  --skip-fetch set; using existing papers only")
        papers_df = existing_papers
    else:
        since_date = since or latest_pub_date(existing_papers) or (date.today() - timedelta(days=365 * 10))
        print(f"  fetching papers published on/after {since_date}")
        new_rows, resume_date, fetch_complete = fetch_new_papers(since_date)
        print(f"  fetched {len(new_rows)} new paper(s) this run")
        papers_df = merge_papers(existing_papers, new_rows)
        print(f"  corpus now has {len(papers_df)} papers total")
        papers_df.to_csv(papers_path, index=False)  # save progress BEFORE deciding whether to continue

        if not fetch_complete:
            print()
            print(f"  Fetch was interrupted by a network error. Progress saved to {papers_path}.")
            print(f"  Re-run the exact same command to continue -- it will automatically resume from "
                  f"{resume_date} (found via the newest PubDate now in papers.csv). Stopping here rather "
                  "than running the topic/trend model on a corpus that's still mid-fetch.")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "incomplete_fetch": True,
            }

        if fetch_only:
            print()
            print(f"  --fetch-only set; wrote {len(papers_df)} papers to {papers_path}. "
                  "Stopping before the topic/trend model.")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "fetch_only": True,
            }

        if skip_topic_model and new_rows:
            print(f"  WARNING: {len(new_rows)} new paper(s) were fetched but --skip-topic-model means "
                  "they won't be topic-modelled or counted below, since the paper-topic matrix is being "
                  "reloaded from a previous run instead of rebuilt. Use --skip-fetch alongside "
                  "--skip-topic-model when you just want to rerun the trend model on unchanged data.")

    if skip_fetch:
        print(f"  corpus now has {len(papers_df)} papers total")
        if not topic_model_only:  # --topic-model-only doesn't touch the corpus file
            papers_df.to_csv(papers_path, index=False)

    if skip_topic_model:
        print("[2-4/6] --skip-topic-model set; reloading topics and paper-topic matrix from --data-dir...")
        if not (os.path.exists(topics_path) and os.path.exists(paper_topics_path)):
            raise SystemExit(
                f"--skip-topic-model requires an existing {topics_path} and {paper_topics_path} "
                "from a previous full run."
            )
        topics_df = pd.read_csv(topics_path)
        paper_topic_matrix = pd.read_csv(paper_topics_path)
        n_topic_cols = len([c for c in paper_topic_matrix.columns if re.match(r"^Topic\d+$", c)])
        print(f"  loaded {len(topics_df)} topics, {len(paper_topic_matrix)} paper-topic rows "
              f"({n_topic_cols} topic columns)")
    else:
        print("[2/6] Refitting the topic model on the full corpus...")
        topic_model, docs, topic_distr = fit_topic_model(papers_df)
        n_topics = topic_distr.shape[1]
        print(f"  found {n_topics} topics across {len(docs)} documents")
        summarize_topic_model(topic_model)

        print("[3/6] Naming topics...")
        names_and_keywords = name_topics(topic_model)
        topics_df = pd.DataFrame(
            [
                {"Topic": tid, "Name": name, "Words": keywords}
                for tid, (name, keywords) in sorted(names_and_keywords.items())
            ]
        )
        topics_df.to_csv(topics_path, index=False)

        print("[4/6] Building the binary paper-topic assignment matrix...")
        paper_topic_matrix = build_paper_topic_matrix(papers_df, topic_distr)
        paper_topic_matrix.to_csv(paper_topics_path, index=False)

        if topic_model_only:
            print()
            print(f"  --topic-model-only set; wrote {topics_path} ({len(topics_df)} topics) "
                  f"and {paper_topics_path}. Stopped before monthly mentions, the trend "
                  "model and the DB reload.")
            print(f"  To finish from here without refitting: "
                  f"python scripts/update_data_monthly.py --data-dir {data_dir} "
                  "--skip-fetch --skip-topic-model")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "topic_model_only": True,
            }

    print("[5/6] Building monthly mention counts...")
    monthly_df = build_monthly_mentions(paper_topic_matrix)
    monthly_df.to_csv(monthly_path, index=False)

    print("[6/6] Fitting the trend model and computing trendiness...")
    predictions_df = fit_trend_model(monthly_df, checkpoint_path=predictions_path)
    predictions_df.to_csv(predictions_path, index=False)

    n_trendy = int(predictions_df["Trendy"].sum()) if not predictions_df.empty else 0
    print(f"  {n_trendy} of {len(predictions_df)} topics flagged trendy")

    return {
        "papers": papers_path,
        "topics": topics_path,
        "monthly": monthly_path,
        "predictions": predictions_path,
        "paper_topics": paper_topics_path,
    }


def reload_database(paths):
    """Full reset + reimport, since a full-corpus topic model refit means
    topic ids from the previous run aren't meaningful anymore."""
    import subprocess

    scripts_dir = os.path.join(BASE_DIR, "scripts")
    print("Reloading the live database...")
    subprocess.run(["python3", os.path.join(scripts_dir, "init_db.py"), "--reset"], check=True)
    subprocess.run(
        [
            "python3", os.path.join(scripts_dir, "import_notebook_outputs.py"),
            "--topics", paths["topics"],
            "--monthly", paths["monthly"],
            "--predictions", paths["predictions"],
            "--papers", paths["papers"],
            "--paper-topics", paths["paper_topics"],
        ],
        check=True,
    )
    subprocess.run(["python3", os.path.join(scripts_dir, "compute_embeddings.py")], check=True)
    print("Database reloaded.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "pipeline_data"),
                         help="where the pipeline's CSVs are read from/written to")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; defaults to the day after the "
                         "newest paper already on file, or 10 years ago if there's no existing corpus")
    parser.add_argument("--skip-fetch", action="store_true",
                         help="skip the OpenAlex fetch and just re-run modelling on the existing corpus "
                              "(useful for testing, or if you fetched papers separately)")
    parser.add_argument("--fetch-only", action="store_true",
                         help="run only the OpenAlex fetch + dedupe, write papers.csv, then stop "
                              "before the topic/trend model (implies --skip-db-reload)")
    parser.add_argument("--skip-topic-model", action="store_true",
                         help="skip refitting BERTopic and naming topics; reload topics.csv and "
                              "paper_topic_assignments.csv from --data-dir instead (from a previous full "
                              "run), then just rebuild monthly mentions and rerun the trend model. Combine "
                              "with --skip-fetch when debugging the trend model on unchanged data.")
    parser.add_argument("--topic-model-only", action="store_true",
                         help="run just the topic stage on the existing corpus: refit BERTopic, "
                              "print a topic-count / size summary, LLM-name the topics, and write "
                              "topics.csv + paper_topic_assignments.csv -- then stop before the "
                              "monthly mentions, trend model and DB reload. Resume with "
                              "--skip-fetch --skip-topic-model.")
    parser.add_argument("--skip-db-reload", action="store_true",
                         help="write the CSVs but don't touch data/galenos.db")
    args = parser.parse_args()

    if args.fetch_only and args.skip_fetch:
        parser.error("--fetch-only and --skip-fetch are mutually exclusive")
    if args.topic_model_only and (args.fetch_only or args.skip_topic_model):
        parser.error("--topic-model-only can't be combined with --fetch-only or --skip-topic-model")

    since = datetime.strptime(args.since, "%Y-%m-%d").date() if args.since else None
    paths = run_pipeline(args.data_dir, since=since,
                          skip_fetch=args.skip_fetch or args.topic_model_only,
                          skip_topic_model=args.skip_topic_model, fetch_only=args.fetch_only,
                          topic_model_only=args.topic_model_only)

    if paths.get("fetch_only") or paths.get("incomplete_fetch") or paths.get("topic_model_only"):
        print("Skipping database reload.")
    elif not args.skip_db_reload:
        reload_database(paths)
    else:
        print("--skip-db-reload set; run scripts/import_notebook_outputs.py yourself when ready.")




if __name__ == "__main__":
    main()
