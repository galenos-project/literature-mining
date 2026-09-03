"""Import real pipeline output files into galenos.db.

Expected files (columns exactly as produced by the literature-mining pipeline):

  --papers PAPERS.csv
      PaperId,PaperTitle,Citations,coFoS,Authors,Abstract,Lang,PubYear,PubDate

  --topics TOPICS.csv
      Topic,Name,Words
      (Topic = integer topic id; Words = the topic's characteristic keywords,
      however they were serialised — a Python-list-looking string, a
      comma-separated string, or a plain space-separated string are all
      handled)

  --monthly MONTHLY_MENTIONS.csv
      PubDate,Topic0,Topic1,Topic2,...
      One row per month; each TopicN column is that topic's actual mention
      count in that month.

  --predictions TRENDY_PREDICTIONS.csv
      ,Topic,TopicName,Trendy,RankSum,ModelMAE,Pred_M0,Pred_M1,...,Pred_M115
      One row per topic. The Pred_M* columns are predicted monthly mention
      counts with no dates attached, so they're aligned here to the LAST
      len(Pred_M*) months of the monthly-mentions file (i.e. the most recent
      prediction lines up with the most recent actual month, working
      backwards from there). This mirrors the paper's design, where
      predictions run from 6 months into the series up to the present, but
      if your export uses a different convention, adjust `align_predicted_months()`.

  --paper-topics PAPER_TOPICS.csv
      Same columns as the papers file, plus binary Topic0,Topic1,... columns
      (1 = this paper is assigned to this topic, already thresholded — no
      continuous probability is expected). If --papers is omitted, this
      file's own metadata columns are used to populate the papers table too,
      since it's a superset of the papers file.

Usage:

    python scripts/init_db.py --reset
    python scripts/import_notebook_outputs.py \\
        --papers papers.csv \\
        --topics topics.csv \\
        --monthly monthly_mentions.csv \\
        --predictions trendy_predictions.csv \\
        --paper-topics paper_topic_assignments.csv
"""
import argparse
import ast
import csv
import json
import os
import re
import sqlite3
from datetime import date

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "galenos.db")

TOPIC_COL_RE = re.compile(r"^Topic(\d+)$")
TRUE_STRINGS = {"1", "true", "yes", "y"}


def topic_columns(fieldnames):
    """Return {column_name: topic_id} for every TopicN column in a header."""
    out = {}
    for col in fieldnames:
        m = TOPIC_COL_RE.match(col.strip())
        if m:
            out[col] = int(m.group(1))
    return out


def parse_words(raw):
    """Words column may be a Python-list repr, JSON, comma-separated, or
    plain whitespace-separated text — handle all of them."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                val = parser(raw)
                if isinstance(val, (list, tuple)):
                    return [str(w).strip().strip("'\"") for w in val]
            except (ValueError, SyntaxError):
                continue
    if "," in raw:
        return [w.strip() for w in raw.split(",") if w.strip()]
    return [w.strip() for w in raw.split() if w.strip()]


def parse_bool(raw):
    return str(raw).strip().lower() in TRUE_STRINGS


def parse_float(raw):
    raw = (raw or "").strip()
    if raw == "" or raw.lower() in ("none", "nan"):
        return None
    return float(raw)


def parse_int(raw):
    raw = (raw or "").strip()
    if raw == "" or raw.lower() in ("none", "nan"):
        return None
    return int(float(raw))  # tolerate "12.0"-style ints


def parse_citation_count(raw):
    """The Citations column holds a '|'-delimited list of the IDs of papers
    that cite this one (e.g. '1972772384|1983609344|...'), not a plain
    count — so the citation count is the number of IDs listed, including
    the single-ID and empty (0 citations) cases."""
    raw = (raw or "").strip()
    if not raw:
        return 0
    return len([p for p in raw.split("|") if p.strip()])


def normalize_month(raw):
    """Coerce a month value into 'YYYY-MM-01'. Accepts 'YYYY-MM',
    'YYYY-MM-DD', or a full timestamp string — anything starting with a
    4-digit year and 1-2 digit month."""
    raw = str(raw).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})", raw)
    if not m:
        raise ValueError(f"Can't parse month from {raw!r}")
    year, month = int(m.group(1)), int(m.group(2))
    return f"{year:04d}-{month:02d}-01"


def normalize_pub_date(pubdate, pubyear):
    """Normalise a paper's publication date to ISO 'YYYY-MM-DD'. Falls back
    to Jan 1 of PubYear if PubDate is missing/unparsable."""
    pubdate = (pubdate or "").strip()
    for pattern in (
        r"^(\d{4})-(\d{1,2})-(\d{1,2})",   # YYYY-MM-DD
        r"^(\d{4})-(\d{1,2})$",             # YYYY-MM
        r"^(\d{1,2})/(\d{1,2})/(\d{4})",    # MM/DD/YYYY
    ):
        m = re.match(pattern, pubdate)
        if m:
            groups = m.groups()
            if pattern.startswith(r"^(\d{1,2})/"):
                mo, d, y = groups
                return date(int(y), int(mo), int(d)).isoformat()
            elif len(groups) == 3:
                y, mo, d = groups
                return date(int(y), int(mo), int(d)).isoformat()
            else:
                y, mo = groups
                return date(int(y), int(mo), 1).isoformat()
    if pubyear and str(pubyear).strip():
        try:
            return date(int(float(pubyear)), 1, 1).isoformat()
        except ValueError:
            pass
    return None


def guess_openalex_url(paper_id):
    pid = (paper_id or "").strip()
    if re.match(r"^W\d+$", pid):
        return f"https://openalex.org/{pid}"
    if pid.startswith("https://openalex.org/"):
        return pid
    return None


def align_predicted_months(actual_months_sorted, pred_values):
    """Zip predicted values against calendar months by aligning the END of
    both series (see module docstring for rationale)."""
    n_pred = len(pred_values)
    months_for_pred = actual_months_sorted[-n_pred:] if n_pred <= len(actual_months_sorted) else actual_months_sorted
    return list(zip(months_for_pred[-len(pred_values):], pred_values[-len(months_for_pred):]))


# ---------------------------------------------------------------------------

def load_topics(conn, path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_id = parse_int(row["Topic"])
            conn.execute(
                "INSERT OR REPLACE INTO topics (topic_id, name, keywords, n_papers, is_trendy, trend_rank, trend_mae) "
                "VALUES (?, ?, ?, COALESCE((SELECT n_papers FROM topics WHERE topic_id = ?), 0), "
                "        COALESCE((SELECT is_trendy FROM topics WHERE topic_id = ?), 0), "
                "        (SELECT trend_rank FROM topics WHERE topic_id = ?), "
                "        (SELECT trend_mae FROM topics WHERE topic_id = ?))",
                (topic_id, row["Name"], json.dumps(parse_words(row.get("Words"))),
                 topic_id, topic_id, topic_id, topic_id),
            )
    print(f"  topics loaded from {path}")


def load_monthly_mentions(conn, path):
    """Wide format: PubDate,Topic0,Topic1,... -> one monthly_counts row per
    (topic, month), and updates topics.n_papers as the column totals."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        topic_cols = topic_columns(reader.fieldnames)
        totals = {tid: 0 for tid in topic_cols.values()}
        rows_by_month = []
        for row in reader:
            month = normalize_month(row["PubDate"])
            for col, topic_id in topic_cols.items():
                count = parse_int(row.get(col)) or 0
                rows_by_month.append((topic_id, month, count))
                totals[topic_id] += count

    for topic_id, month, count in rows_by_month:
        conn.execute(
            "INSERT INTO monthly_counts (topic_id, month, actual_count, predicted_count) "
            "VALUES (?, ?, ?, NULL) "
            "ON CONFLICT(topic_id, month) DO UPDATE SET actual_count = excluded.actual_count",
            (topic_id, month, count),
        )
    for topic_id, total in totals.items():
        conn.execute("UPDATE topics SET n_papers = ? WHERE topic_id = ?", (total, topic_id))

    months_sorted = sorted({m for _, m, _ in rows_by_month})
    print(f"  monthly mentions loaded from {path} ({len(topic_cols)} topics x {len(months_sorted)} months)")
    return months_sorted


def load_predictions(conn, path, actual_months_sorted):
    pred_col_re = re.compile(r"^Pred_M(\d+)$")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        pred_cols = sorted(
            [c for c in reader.fieldnames if pred_col_re.match(c)],
            key=lambda c: int(pred_col_re.match(c).group(1)),
        )
        n_topics = 0
        n_trendy_in_file = 0
        missing_topic_ids = []
        for row in reader:
            topic_id = parse_int(row["Topic"])
            is_trendy = parse_bool(row.get("Trendy"))
            trend_rank = parse_float(row.get("RankSum"))
            trend_mae = parse_float(row.get("ModelMAE"))
            if is_trendy:
                n_trendy_in_file += 1

            cur = conn.execute(
                "UPDATE topics SET is_trendy = ?, trend_rank = ?, trend_mae = ? WHERE topic_id = ?",
                (int(is_trendy), trend_rank, trend_mae, topic_id),
            )
            if cur.rowcount == 0:
                missing_topic_ids.append(topic_id)

            pred_values = [parse_float(row.get(c)) for c in pred_cols]
            for month, value in align_predicted_months(actual_months_sorted, pred_values):
                if value is None:
                    continue
                conn.execute(
                    "INSERT INTO monthly_counts (topic_id, month, actual_count, predicted_count) "
                    "VALUES (?, ?, COALESCE((SELECT actual_count FROM monthly_counts WHERE topic_id = ? AND month = ?), 0), ?) "
                    "ON CONFLICT(topic_id, month) DO UPDATE SET predicted_count = excluded.predicted_count",
                    (topic_id, month, topic_id, month, value),
                )
            n_topics += 1

    n_trendy_in_db = conn.execute("SELECT COUNT(*) AS c FROM topics WHERE is_trendy = 1").fetchone()[0]
    print(f"  predictions loaded from {path} ({n_topics} topics, {len(pred_cols)} predicted months each)")
    print(f"  {n_trendy_in_file} row(s) had Trendy=True in the file; {n_trendy_in_db} topic(s) are now marked "
          f"is_trendy=1 in the database.")
    if missing_topic_ids:
        preview = missing_topic_ids[:10]
        more = f" (+{len(missing_topic_ids) - 10} more)" if len(missing_topic_ids) > 10 else ""
        print(f"  WARNING: {len(missing_topic_ids)} topic id(s) from this file don't exist in the topics table yet, "
              f"so their Trendy/rank data was NOT applied: {preview}{more}")
        print("  -> Make sure --topics is loaded (in this run or an earlier one, without an intervening --reset) "
              "before --predictions.")


def _insert_paper_row(conn, row):
    paper_id = row["PaperId"].strip()
    pub_date = normalize_pub_date(row.get("PubDate"), row.get("PubYear"))
    conn.execute(
        "INSERT OR REPLACE INTO papers "
        "(paper_id, title, abstract, authors, publication_date, citations, fields_of_study, language, openalex_url) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paper_id,
            (row.get("PaperTitle") or "").strip(),
            row.get("Abstract"),
            row.get("Authors"),
            pub_date,
            parse_citation_count(row.get("Citations")),
            row.get("coFoS"),
            row.get("Lang"),
            guess_openalex_url(paper_id),
        ),
    )


def load_papers(conn, path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        n = 0
        for row in csv.DictReader(f):
            _insert_paper_row(conn, row)
            n += 1
    print(f"  papers loaded from {path} ({n} rows)")


def load_paper_topics(conn, path, populate_papers=False):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        topic_cols = topic_columns(reader.fieldnames)
        n_links = 0
        n_papers = 0
        for row in reader:
            if populate_papers:
                _insert_paper_row(conn, row)
                n_papers += 1
            paper_id = row["PaperId"].strip()
            for col, topic_id in topic_cols.items():
                val = (row.get(col) or "").strip()
                if val == "1" or parse_bool(val):
                    conn.execute(
                        "INSERT OR REPLACE INTO paper_topics (paper_id, topic_id, probability) VALUES (?, ?, 1.0)",
                        (paper_id, topic_id),
                    )
                    n_links += 1
    extra = f", populated {n_papers} papers from this file" if populate_papers else ""
    print(f"  paper-topic links loaded from {path} ({n_links} links{extra})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", help="search-results CSV: PaperId,PaperTitle,Citations,coFoS,Authors,Abstract,Lang,PubYear,PubDate")
    parser.add_argument("--topics", help="topics CSV: Topic,Name,Words")
    parser.add_argument("--monthly", help="wide monthly mentions CSV: PubDate,Topic0,Topic1,...")
    parser.add_argument("--predictions", help="trendy predictions CSV: Topic,TopicName,Trendy,RankSum,ModelMAE,Pred_M0..Pred_MN")
    parser.add_argument("--paper-topics", dest="paper_topics", help="binary paper-topic assignment CSV")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")  # allow flexible load order

    print("Importing...")
    if args.topics:
        load_topics(conn, args.topics)

    actual_months_sorted = []
    if args.monthly:
        actual_months_sorted = load_monthly_mentions(conn, args.monthly)
        conn.commit()

    if args.predictions:
        if not actual_months_sorted:
            actual_months_sorted = sorted(
                r["month"] for r in conn.execute("SELECT DISTINCT month FROM monthly_counts").fetchall()
            )
        load_predictions(conn, args.predictions, actual_months_sorted)

    if args.papers:
        load_papers(conn, args.papers)

    if args.paper_topics:
        load_paper_topics(conn, args.paper_topics, populate_papers=not args.papers)

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
