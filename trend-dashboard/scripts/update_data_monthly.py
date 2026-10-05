"""Monthly batch update: the full literature-mining pipeline in one script.

Replaces running the original notebooks by hand each month:

  1. fetch_papers()               -- full OpenAlex refetch of the rolling corpus window
  2. fit_topic_model()            -- BERTopic refit on the FULL corpus
  3. build_paper_topic_matrix()   -- binary Topic0..N columns, thresholded
  4. map_to_previous_run()        -- link topics to the previous run's (history/), inheriting names
  5. name_topics()                -- LLM-based short names from keywords, for topics not carried over
  6. build_monthly_mentions()     -- wide PubDate,Topic0,Topic1,... counts
  7. fit_trend_model()            -- GRU walk-forward prediction + Trendy/RankSum
  8. archive_run()                -- topics + assignments saved to history/<YYYY-MM>/

The corpus is a rolling window of the CORPUS_WINDOW_MONTHS (120) complete
calendar months before the current one, and it is refetched in full each
run rather than topped up: OpenAlex keeps adding papers to past months
(indexing lag) and updating existing records, and neither shows up in a
fetch of only newly published papers (filtering on OpenAlex's
updated/created dates needs a paid plan). Papers that fall out of the
window are dropped.

...then writes all five CSVs to --data-dir and does a full
init_db --reset + import_notebook_outputs.py + compute_embeddings.py refresh
of the live database (topic ids aren't stable across a full-corpus refit, so
this is a full rebuild each run, not an incremental patch; continuity across
runs is carried by the LineageIds archived under --data-dir/history/).

Usage:
    python scripts/update_data_monthly.py --data-dir pipeline_data
    python scripts/update_data_monthly.py --data-dir pipeline_data --fetch-only  # refetch + dedupe papers.csv, then stop
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


def load_dotenv(path=os.path.join(BASE_DIR, ".env")):
    """Minimal KEY=value loader for trend-dashboard/.env (gitignored), so API
    keys needn't be exported in every shell. Variables already set in the
    environment take precedence."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key.strip(), value)


load_dotenv()

search_term = '("mental" OR "psychological" OR "behavioural" OR "psychology" OR "psychiatry" OR "neurological" OR "mind" OR "brain" OR "behaviour" OR "psychiatric") \
              AND ("anxiety" OR "depression" OR "psychosis") AND ("treatment" OR "therapy" OR "therapeutic" OR "mechanism" OR "intervention" OR "early" OR "diagnosis" OR "diagnostic" OR "translation")'
OPENALEX_FILTER = f"has_abstract:true,title_and_abstract.search:{search_term},language:en" 
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "user.name@example.com")  # OpenAlex's "polite pool"
CORPUS_WINDOW_MONTHS = 120      # the corpus is a rolling window of this many complete calendar months
MAX_CORPUS_SHRINK = 0.05        # refuse to replace papers.csv if a refetch comes back this much smaller

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")  # optional; unlocks OpenAlex's premium rate limits
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")      # DeepInfra token (used with the base_url below)


ASSIGNMENT_THRESHOLD = 0.08     # per the paper's methodology
UMAP_RANDOM_STATE = 73          # fixed so re-running the topic stage on the same corpus gives the same topics
                                # (UMAP is otherwise nondeterministic; fixing it also makes UMAP single-threaded)
TOPIC_MATCH_MIN_KEYWORDS = 10       # a topic continues one from the previous run if at least this many of the
TOPIC_MATCH_MIN_PAPER_OVERLAP = 0.8 # 15 keywords are shared AND the shared papers (among papers in both runs'
                                    # corpora) make up at least this share of BOTH topics
SLIDING_WINDOW_MONTHS = 6       # per the paper's methodology
TREND_N_MONTHS = 4              # of the last...
TREND_M_MONTHS = 6               # ...months, actual must exceed predicted to count as trendy
TREND_HIDDEN_SIZE = 10          # GRU hidden units (both stacked layers)
TREND_EPOCHS = 10
TREND_BATCH_SIZE = 32
TREND_LEAVE_K_OUT = 10         # topics held out (and predicted) per training run; 1 == exact leave-one-topic-out
TREND_SIZE_NORM_EXPONENT = 0.55 # RankSum denominator is mean_actual ** this. 1.0 = original (÷mean, biased
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


def corpus_window(today=None):
    """(first_day, last_day) of the rolling corpus window: the
    CORPUS_WINDOW_MONTHS complete calendar months before the current one.
    E.g. on any day in Oct 2026 -> (2016-10-01, 2026-09-30)."""
    today = today or date.today()
    last_day = today.replace(day=1) - timedelta(days=1)
    first_month_index = today.year * 12 + (today.month - 1) - CORPUS_WINDOW_MONTHS
    first_day = date(first_month_index // 12, first_month_index % 12 + 1, 1)
    return first_day, last_day


def restrict_to_window(df, window):
    """Drop rows whose PubDate falls outside `window` (or is unparseable)."""
    if df is None or df.empty:
        return df
    pub = pd.to_datetime(df["PubDate"], errors="coerce")
    in_window = (pub >= pd.Timestamp(window[0])) & (pub <= pd.Timestamp(window[1]))
    n_dropped = int((~in_window).sum())
    if n_dropped:
        print(f"  dropped {n_dropped} paper(s) published outside {window[0]} .. {window[1]}")
    return df[in_window].reset_index(drop=True)


def normalize_work(work):
    paper_id = work["id"].rsplit("/", 1)[-1]
    title = work.get("title") or work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    authors = "; ".join(
        a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")
    )
    concepts = sorted(work.get("concepts") or [], key=lambda c: -(c.get("score") or 0))[:3]
    fields_of_study = "; ".join(c["display_name"] for c in concepts)

    return {
        "PaperId": paper_id,
        "PaperTitle": title,
        "CitedByCount": work.get("cited_by_count") or 0,
        "coFoS": fields_of_study,
        "Authors": authors,
        "Abstract": abstract,
        "Lang": work.get("language") or "",
        "PubYear": work.get("publication_year"),
        "PubDate": work.get("publication_date") or "",
    }


def fetch_papers(since_date, until_date, filter_str=OPENALEX_FILTER, mailto=OPENALEX_MAILTO,
                 api_key=OPENALEX_API_KEY, max_pages=None):
    """Page through OpenAlex works matching `filter_str` published between
    `since_date` and `until_date` inclusive (dates or 'YYYY-MM-DD' strings).

    Returns (rows, resume_date, complete):
      - rows: list of dicts matching the papers.csv schema (whatever was
        successfully fetched -- possibly not all of it, see below).
      - resume_date: the publication date to resume from on the next
        run to continue from here. Deliberately the same date as the last
        paper actually fetched (not the day after), since OpenAlex's
        from_publication_date filter is inclusive: if a failure happens
        mid-page, there could be other papers with that exact same date
        still unfetched. Re-including that one day means a handful of
        already-fetched papers get refetched (harmless -- they're deduped
        by PaperId), which is a small price for not silently missing
        same-day papers.
      - complete: False if a network error cut the fetch short (results
        are sorted oldest-first, so `rows` is everything from `since_date`
        up to `resume_date`, not a scattered partial sample); True if
        pagination ran to completion normally.

    Network errors (timeouts, connection resets, HTTP errors) are retried
    a few times with backoff before being treated as a real failure, so a
    single transient blip doesn't lose an otherwise-long fetch run.
    """
    if isinstance(since_date, date):
        since_date = since_date.isoformat()
    if isinstance(until_date, date):
        until_date = until_date.isoformat()

    base_url = "https://api.openalex.org/works"
    full_filter = f"{filter_str},from_publication_date:{since_date},to_publication_date:{until_date}"
    cursor = "*"
    rows = []
    resume_date = since_date
    page = 0
    max_retries = 3
    retry_backoff_seconds = (5, 15, 30)

    while cursor:
        data = None
        last_error = None
        params = {
            "filter": full_filter, "per-page": 200, "cursor": cursor,
            "sort": "publication_date:asc",  # so "resume from the latest date fetched" is valid
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
            return rows, resume_date, False

        if page == 0:
            print(f"  OpenAlex reports {data.get('meta', {}).get('count')} matching paper(s)")
        results = data.get("results", [])
        for work in results:
            row = normalize_work(work)
            rows.append(row)
            if row["PubDate"]:
                resume_date = max(resume_date, row["PubDate"])

        cursor = data.get("meta", {}).get("next_cursor")
        page += 1
        if page % 100 == 0:
            print(f"  ...{len(rows)} paper(s) fetched, up to {resume_date}")
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

      1. by PaperId -- new rows win on conflict (e.g. the overlapping
         day refetched when resuming an interrupted fetch);
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
    umap_model = UMAP(n_neighbors=18, n_components=12, metric='cosine', random_state=UMAP_RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=22, min_samples=15, metric='euclidean')

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
# 3. Binary paper-topic assignment matrix
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
# 4. Topic history: map this run's topics onto the previous run's
#
# Topic ids aren't stable across full-corpus refits, so continuity is
# tracked separately: every topic carries a LineageId, inherited from the
# previous run's topic it continues (and with it, that topic's name), or
# newly allocated if it doesn't continue any. Each completed run is
# archived under <data-dir>/history/<YYYY-MM of the run>/ -- topics.csv
# (lineage, name, keywords, trendiness, match scores) plus
# assignments.csv.gz (PaperId,Topic; Topic -1 for papers in no topic) --
# which is what the next run maps against, and what the dashboard's
# topic History panel is built from.
# ==========================================================================

def topic_keywords(topic_model, n_topics):
    return {t: [w for w, _ in topic_model.get_topic(t)[:15]] for t in range(n_topics)}


def history_label(today=None):
    """History folders are named after the month the pipeline ran in."""
    return (today or date.today()).strftime("%Y-%m")


def _archived_labels(history_dir, before):
    if not os.path.isdir(history_dir):
        return []
    return sorted(d for d in os.listdir(history_dir) if re.match(r"^\d{4}-\d{2}$", d) and d < before)


def load_previous_run(history_dir, label):
    """(label, topics_df, assignments_df) for the newest archived run before
    `label`, or None. Re-running in the same month therefore maps against
    the previous month again, not against the run being replaced."""
    earlier = _archived_labels(history_dir, label)
    if not earlier:
        return None
    prev = earlier[-1]
    topics = pd.read_csv(os.path.join(history_dir, prev, "topics.csv"))
    assignments = pd.read_csv(os.path.join(history_dir, prev, "assignments.csv.gz"))
    return prev, topics, assignments


def next_lineage_id(history_dir, label):
    """One more than the highest LineageId in any archived run before `label`."""
    ids = [pd.read_csv(os.path.join(history_dir, d, "topics.csv"), usecols=["LineageId"])["LineageId"].max()
           for d in _archived_labels(history_dir, label)]
    return int(max(ids)) + 1 if ids else 0


def map_to_previous_run(keywords, paper_topic_matrix, previous, first_new_lineage_id,
                        min_keywords=TOPIC_MATCH_MIN_KEYWORDS, min_paper_overlap=TOPIC_MATCH_MIN_PAPER_OVERLAP):
    """Match this run's topics 1-to-1 onto the previous run's.

    For every (previous, current) topic pair:
      - KeywordOverlap: how many of the 15 keywords they share;
      - PaperOverlap: among papers present in both runs' corpora, the
        shared papers as a share of the previous topic and of the current
        topic -- whichever is smaller, so a topic that merely absorbed
        another (or was split off one) doesn't count as a continuation.
    Pairs meeting both thresholds are matched greedily, best first (by
    PaperOverlap + KeywordOverlap/15), so each topic continues at most one.

    Returns one row per current topic: Topic, LineageId, InheritedName
    (None unless matched), Matched, and PrevTopic/KeywordOverlap/
    PaperOverlap -- for the matched topic, or for the closest previous
    topic if unmatched, so thresholds can be tuned from the output.
    """
    import ast
    import scipy.sparse as sp

    n_cur = len(keywords)
    out = pd.DataFrame({"Topic": range(n_cur), "LineageId": pd.array([pd.NA] * n_cur, dtype="Int64"),
                        "InheritedName": None, "Matched": False,
                        "PrevTopic": pd.array([pd.NA] * n_cur, dtype="Int64"),
                        "KeywordOverlap": pd.array([pd.NA] * n_cur, dtype="Int64"), "PaperOverlap": np.nan})
    if previous is None:
        print("  no previous run archived; every topic starts a new lineage")
        out["LineageId"] = first_new_lineage_id + out["Topic"]
        return out

    prev_label, prev_topics, prev_assign = previous
    prev_topics = prev_topics.sort_values("Topic").reset_index(drop=True)
    n_prev = len(prev_topics)

    # Paper overlap, restricted to papers in both corpora
    topic_cols = [f"Topic{t}" for t in range(n_cur)]
    cur_ids = paper_topic_matrix["PaperId"].to_numpy()
    common = np.intersect1d(cur_ids, prev_assign["PaperId"].unique())
    cur = sp.csr_matrix(paper_topic_matrix[topic_cols].to_numpy(dtype=np.int8)[pd.Index(cur_ids).get_indexer(common)],
                        dtype=np.int32)
    pa = prev_assign[(prev_assign["Topic"] >= 0) & prev_assign["PaperId"].isin(common)]
    prev = sp.csr_matrix((np.ones(len(pa), dtype=np.int32),
                          (pd.Index(common).get_indexer(pa["PaperId"]), pa["Topic"].to_numpy())),
                         shape=(len(common), n_prev))
    shared = (prev.T @ cur).toarray().astype(float)
    prev_sizes, cur_sizes = prev.sum(axis=0).A1, cur.sum(axis=0).A1
    with np.errstate(divide="ignore", invalid="ignore"):
        paper_overlap = np.nan_to_num(np.minimum(shared / prev_sizes[:, None], shared / cur_sizes[None, :]))

    prev_kw = [set(ast.literal_eval(w)) for w in prev_topics["Words"]]
    cur_kw = [set(keywords[t]) for t in range(n_cur)]
    kw_overlap = np.array([[len(p & c) for c in cur_kw] for p in prev_kw])

    score = paper_overlap + kw_overlap / 15
    passing = (kw_overlap >= min_keywords) & (paper_overlap >= min_paper_overlap)
    candidates = sorted(zip(*np.nonzero(passing)), key=lambda ij: -score[ij])
    used_prev, matched = set(), {}
    for i, j in candidates:
        if i not in used_prev and j not in matched:
            used_prev.add(i)
            matched[j] = i

    next_id = first_new_lineage_id
    for j in range(n_cur):
        i = matched.get(j, int(score[:, j].argmax()))
        out.loc[j, ["PrevTopic", "KeywordOverlap", "PaperOverlap"]] = [
            int(prev_topics.loc[i, "Topic"]), int(kw_overlap[i, j]), round(float(paper_overlap[i, j]), 3)]
        if j in matched:
            out.loc[j, ["LineageId", "InheritedName", "Matched"]] = [
                int(prev_topics.loc[i, "LineageId"]), prev_topics.loc[i, "Name"], True]
        else:
            out.loc[j, "LineageId"] = next_id
            next_id += 1

    print(f"  {len(common)} papers in both this corpus and the {prev_label} run's")
    print(f"  {len(matched)} of {n_cur} topics continue one of the {n_prev} topics from {prev_label} "
          f"(>= {min_keywords}/15 keywords and >= {min_paper_overlap:.0%} shared papers); "
          f"{n_cur - len(matched)} new; {n_prev - len(matched)} {prev_label} topic(s) not continued")
    return out


def archive_run(history_dir, label, topics_df, paper_topic_matrix, predictions_df):
    """Write this run's topics (with trendiness) and compact assignments to
    history/<label>/, replacing any earlier archive of the same month."""
    folder = os.path.join(history_dir, label)
    os.makedirs(folder, exist_ok=True)
    topic_cols = [f"Topic{t}" for t in topics_df["Topic"]]
    binary = paper_topic_matrix[topic_cols].to_numpy(dtype=np.int8)
    paper_ids = paper_topic_matrix["PaperId"].to_numpy()
    rows, cols = np.nonzero(binary)
    unassigned = paper_ids[binary.sum(axis=1) == 0]
    assignments = pd.concat([
        pd.DataFrame({"PaperId": paper_ids[rows], "Topic": topics_df["Topic"].to_numpy()[cols]}),
        pd.DataFrame({"PaperId": unassigned, "Topic": -1}),
    ], ignore_index=True)
    assignments.to_csv(os.path.join(folder, "assignments.csv.gz"), index=False)

    archived = topics_df.merge(predictions_df[["Topic", "Trendy", "RankSum"]], on="Topic", how="left")
    archived["NPapers"] = binary.sum(axis=0)
    archived.to_csv(os.path.join(folder, "topics.csv"), index=False)
    print(f"  archived {len(archived)} topics and {len(assignments)} paper assignments to {folder}")


# ==========================================================================
# 5. Topic naming
# ==========================================================================

def name_topics(keywords, inherited_names):
    """Topics that continue one from the previous run keep its name
    (`inherited_names`: topic id -> name); the rest are named by the LLM
    from their keywords, falling back to the top three keywords if the
    call fails."""
    to_name = sorted(t for t in keywords if t not in inherited_names)
    print(f"  {len(keywords) - len(to_name)} topic name(s) carried over; {len(to_name)} to name")
    results = dict(inherited_names)
    if not to_name:
        return results

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

    for topic_id in to_name:
        try:
            name = generateFromPrompt(f"Please give a concise phrase to describe the main research topic within the field of mental health that unifies the following words and indicates the mental health relevance: {keywords[topic_id]}. Please return only the word or phrase with no explanation. Topic: ").strip()
        except Exception as e:
            print(f"  LLM naming failed for topic {topic_id} ({e}); using its top keywords instead")
            name = ", ".join(keywords[topic_id][:3])
        results[topic_id] = name
    return results


# ==========================================================================
# 6. Monthly mention counts (wide format)
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
# 7. Trend prediction model
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
#   - TRENDY FLAG: the paper's rule -- actual > predicted in >=
#     TREND_N_MONTHS of the last TREND_M_MONTHS. See _trend_rank_and_flag().
#   - NON-NEGATIVE PREDICTIONS: the GRU's output head and the inverse
#     MinMax transform are both unconstrained affine maps, so a predicted
#     mention count can come out negative (typically for a near-flat,
#     near-zero series read through a scale fit on other, larger topics).
#     Both train_one_model() and predict_topic() clip predictions to >= 0
#     before they're used anywhere -- ModelMAE, Pred_M*, and the Trendy/
#     RankSum comparison below all see only non-negative predictions.
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


def _trend_rank_and_flag(actual_tail, predicted_tail, mean_actual, n=TREND_N_MONTHS, m=TREND_M_MONTHS,size_norm_exponent=TREND_SIZE_NORM_EXPONENT):
    """Trendiness scoring.  A topic is
    flagged trendy only if actual > predicted in >= n of the last m months. 

    RankSum returned for every topic regardless of
    the flag: it sums the positive monthly excesses (actual - predicted),
    each weighted by e^(1/(m-i)) (i=0 oldest of the m, i=m-1 most recent),
    divided by `max(mean_actual, 1) ** size_norm_exponent`. The original
    used exponent 1.0 (plain mean), which over-normalizes -- monthly counts
    are ~Poisson, so noise scales as sqrt(mean), and dividing an absolute
    excess by the mean leaves small topics with an inflated score from
    noise alone. 0.5 divides by the Poisson standard deviation instead
    (~size-neutral under the null); 0.0 disables it."""
    exceed_count = sum(1 for a, p in zip(actual_tail, predicted_tail) if a > p)

    trendy = exceed_count >= n

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
        # The GRU's output head and the inverse MinMax transform are both
        # unconstrained affine maps, so a predicted mention count can come
        # out negative (typically for a near-flat, near-zero series read
        # through a scale fit on other, larger topics). Mentions can't be
        # negative, so clip -- this keeps MAE from being inflated by an
        # impossible prediction, and predict_topic() below applies the same
        # clip before Pred_M/Trendy/RankSum ever see these values.
        pred_actual = np.clip(scaler.inverse_transform(pred_scaled), 0.0, None)
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
        # See the matching clip + comment in train_one_model() above: a
        # predicted mention count is never allowed to be negative.
        y_pred = np.clip(scaler.inverse_transform(pred_scaled).flatten(), 0.0, None)
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


def report_corpus_changes(old_df, new_df):
    """Print how the refetched corpus differs from the previous papers.csv."""
    if old_df is None or old_df.empty:
        return
    old_ids, new_ids = set(old_df["PaperId"]), set(new_df["PaperId"])
    common = old_df[old_df["PaperId"].isin(new_ids)].set_index("PaperId")
    updated = new_df[new_df["PaperId"].isin(old_ids)].set_index("PaperId").loc[common.index]
    content_cols = ["PaperTitle", "Abstract", "PubDate"]
    changed = (common[content_cols].astype(str) != updated[content_cols].astype(str)).any(axis=1)
    print(f"  vs previous corpus: +{len(new_ids - old_ids)} added, -{len(old_ids - new_ids)} removed, "
          f"{int(changed.sum())} with a changed title/abstract/date")


def run_pipeline(data_dir, since=None, skip_fetch=False, skip_topic_model=False,
                 fetch_only=False, topic_model_only=False):
    os.makedirs(data_dir, exist_ok=True)
    papers_path = os.path.join(data_dir, "papers.csv")
    topics_path = os.path.join(data_dir, "topics.csv")
    monthly_path = os.path.join(data_dir, "monthly_mentions.csv")
    predictions_path = os.path.join(data_dir, "trendy_predictions.csv")
    paper_topics_path = os.path.join(data_dir, "paper_topic_assignments.csv")
    staging_path = os.path.join(data_dir, "papers_fetching.csv")
    history_dir = os.path.join(data_dir, "history")
    run_label = history_label()

    existing_papers = pd.read_csv(papers_path) if os.path.exists(papers_path) else pd.DataFrame()

    window = corpus_window()
    print(f"[1/8] Fetching papers published {window[0]} .. {window[1]} from OpenAlex...")
    if skip_fetch:
        print("  --skip-fetch set; using existing papers only")
        papers_df = existing_papers
    else:
        # Fetch into a staging file and only replace papers.csv once the
        # whole window has been fetched, so an interrupted run never leaves
        # a half-refreshed corpus behind -- re-running resumes from the
        # staging file's newest PubDate instead.
        staged = pd.read_csv(staging_path) if os.path.exists(staging_path) else pd.DataFrame()
        if not staged.empty:
            print(f"  resuming the interrupted fetch in {staging_path} ({len(staged)} papers so far)")
        since_date = since or latest_pub_date(staged) or window[0]
        print(f"  fetching papers published on/after {since_date}")
        new_rows, resume_date, fetch_complete = fetch_papers(since_date, window[1])
        print(f"  fetched {len(new_rows)} paper(s) this run")
        staged = merge_papers(staged, new_rows)
        staged.to_csv(staging_path, index=False)  # save progress BEFORE deciding whether to continue

        if not fetch_complete:
            print()
            print(f"  Fetch was interrupted by a network error. Progress saved to {staging_path}; "
                  f"{papers_path} is unchanged.")
            print(f"  Re-run the exact same command to continue -- it will automatically resume from "
                  f"{resume_date} (the newest PubDate in the staging file). Stopping here rather "
                  "than running the topic/trend model on a corpus that's still mid-fetch.")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "incomplete_fetch": True,
            }

        papers_df = restrict_to_window(staged, window)
        previous = restrict_to_window(existing_papers, window)
        report_corpus_changes(previous, papers_df)
        if previous is not None and len(papers_df) < (1 - MAX_CORPUS_SHRINK) * len(previous):
            raise SystemExit(
                f"Refetched corpus has {len(papers_df)} papers vs {len(previous)} previously in the same "
                f"window -- a drop of more than {MAX_CORPUS_SHRINK:.0%}, which suggests a problem on "
                f"OpenAlex's side or with the query. {papers_path} was NOT replaced; the refetch is in "
                f"{staging_path}. Inspect it, then either delete it and re-run, or move it over "
                f"{papers_path} and re-run with --skip-fetch if the drop is genuine."
            )
        papers_df.to_csv(papers_path, index=False)
        os.remove(staging_path)
        print(f"  corpus now has {len(papers_df)} papers total")

        if fetch_only:
            print()
            print(f"  --fetch-only set; wrote {len(papers_df)} papers to {papers_path}. "
                  "Stopping before the topic/trend model.")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "fetch_only": True,
            }

        if skip_topic_model:
            print("  WARNING: the corpus was refetched but --skip-topic-model means the changes "
                  "won't be topic-modelled or counted below, since the paper-topic matrix is being "
                  "reloaded from a previous run instead of rebuilt. Use --skip-fetch alongside "
                  "--skip-topic-model when you just want to rerun the trend model on unchanged data.")

    if skip_fetch:
        papers_df = restrict_to_window(papers_df, window)
        print(f"  corpus now has {len(papers_df)} papers total")
        if not topic_model_only:  # --topic-model-only doesn't touch the corpus file
            papers_df.to_csv(papers_path, index=False)

    if skip_topic_model:
        print("[2-5/8] --skip-topic-model set; reloading topics and paper-topic matrix from --data-dir...")
        if not (os.path.exists(topics_path) and os.path.exists(paper_topics_path)):
            raise SystemExit(
                f"--skip-topic-model requires an existing {topics_path} and {paper_topics_path} "
                "from a previous full run."
            )
        topics_df = pd.read_csv(topics_path)
        paper_topic_matrix = restrict_to_window(pd.read_csv(paper_topics_path), window)
        n_topic_cols = len([c for c in paper_topic_matrix.columns if re.match(r"^Topic\d+$", c)])
        print(f"  loaded {len(topics_df)} topics, {len(paper_topic_matrix)} paper-topic rows "
              f"({n_topic_cols} topic columns)")
    else:
        print("[2/8] Refitting the topic model on the full corpus...")
        topic_model, docs, topic_distr = fit_topic_model(papers_df)
        n_topics = topic_distr.shape[1]
        print(f"  found {n_topics} topics across {len(docs)} documents")
        summarize_topic_model(topic_model)

        print("[3/8] Building the binary paper-topic assignment matrix...")
        paper_topic_matrix = build_paper_topic_matrix(papers_df, topic_distr)
        paper_topic_matrix.to_csv(paper_topics_path, index=False)

        print(f"[4/8] Mapping topics onto the previous run's ({history_dir})...")
        keywords = topic_keywords(topic_model, n_topics)
        mapping = map_to_previous_run(keywords, paper_topic_matrix,
                                      load_previous_run(history_dir, run_label),
                                      next_lineage_id(history_dir, run_label))

        print("[5/8] Naming topics...")
        inherited = {int(r.Topic): r.InheritedName for r in mapping.itertuples() if r.Matched}
        names = name_topics(keywords, inherited)
        topics_df = pd.DataFrame({"Topic": range(n_topics),
                                  "Name": [names[t] for t in range(n_topics)],
                                  "Words": [keywords[t] for t in range(n_topics)]})
        topics_df = topics_df.merge(mapping.drop(columns="InheritedName"), on="Topic")
        topics_df.to_csv(topics_path, index=False)

        if topic_model_only:
            print()
            print(f"  --topic-model-only set; wrote {topics_path} ({len(topics_df)} topics) "
                  f"and {paper_topics_path}. Stopped before monthly mentions, the trend "
                  "model, the history archive and the DB reload.")
            print(f"  To finish from here without refitting: "
                  f"python scripts/update_data_monthly.py --data-dir {data_dir} "
                  "--skip-fetch --skip-topic-model")
            return {
                "papers": papers_path, "topics": topics_path, "monthly": monthly_path,
                "predictions": predictions_path, "paper_topics": paper_topics_path,
                "topic_model_only": True,
            }

    print("[6/8] Building monthly mention counts...")
    monthly_df = build_monthly_mentions(paper_topic_matrix)
    monthly_df.to_csv(monthly_path, index=False)

    print("[7/8] Fitting the trend model and computing trendiness...")
    predictions_df = fit_trend_model(monthly_df, checkpoint_path=predictions_path)
    predictions_df.to_csv(predictions_path, index=False)

    n_trendy = int(predictions_df["Trendy"].sum()) if not predictions_df.empty else 0
    print(f"  {n_trendy} of {len(predictions_df)} topics flagged trendy")

    print(f"[8/8] Archiving this run as {run_label}...")
    if "LineageId" in topics_df.columns:
        archive_run(history_dir, run_label, topics_df, paper_topic_matrix, predictions_df)
    else:
        print(f"  {topics_path} has no LineageId column (it predates topic history); not archiving")

    return {
        "papers": papers_path,
        "topics": topics_path,
        "monthly": monthly_path,
        "predictions": predictions_path,
        "paper_topics": paper_topics_path,
        "history": history_dir,
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
            "--history-dir", paths["history"],
        ],
        check=True,
    )
    subprocess.run(["python3", os.path.join(scripts_dir, "compute_embeddings.py")], check=True)
    print("Database reloaded.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "pipeline_data"),
                         help="where the pipeline's CSVs are read from/written to")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; override where an interrupted fetch "
                         "resumes from. Normally not needed: each run refetches the whole rolling "
                         f"{CORPUS_WINDOW_MONTHS}-month window, resuming automatically from "
                         "papers_fetching.csv if the previous run was interrupted")
    parser.add_argument("--skip-fetch", action="store_true",
                         help="skip the OpenAlex fetch and just re-run modelling on the existing corpus "
                              "(useful for testing, or if you fetched papers separately)")
    parser.add_argument("--fetch-only", action="store_true",
                         help="run only the OpenAlex refetch + dedupe, write papers.csv, then stop "
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
