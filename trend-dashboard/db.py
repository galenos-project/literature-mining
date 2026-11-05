import json
import sqlite3

from flask import current_app, g


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)


# ---------------------------------------------------------------------------
# Query helpers. Kept as plain functions (rather than an ORM) since the app
# is read-mostly and the schema is small and stable.
# ---------------------------------------------------------------------------

def list_topics(db, q=None, trendy_only=False, limit=None):
    sql = "SELECT topic_id, name, keywords, n_papers, is_trendy, trend_rank, trend_mae FROM topics"
    clauses, params = [], []
    if q:
        # keywords is a JSON array stored as text, so a substring LIKE on it
        # matches individual keywords well enough for a free-text search.
        clauses.append("(name LIKE ? OR keywords LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if trendy_only:
        clauses.append("is_trendy = 1")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY (trend_rank IS NULL), trend_rank DESC, n_papers DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = db.execute(sql, params).fetchall()
    return [_topic_row_to_dict(r) for r in rows]


def get_topic(db, topic_id):
    row = db.execute(
        "SELECT topic_id, name, keywords, n_papers, is_trendy, trend_rank, trend_mae, lineage_id "
        "FROM topics WHERE topic_id = ?",
        (topic_id,),
    ).fetchone()
    return _topic_row_to_dict(row) if row else None


def get_topic_history(db, lineage_id):
    """One row per monthly run this topic's lineage appears in, newest first,
    with the number of trendy topics in that run for context."""
    if lineage_id is None:
        return []
    rows = db.execute(
        """
        SELECT h.run_month, h.name, h.keywords, h.n_papers, h.is_trendy, h.trend_rank,
               h.trendy_position, h.keyword_overlap, h.paper_overlap,
               (SELECT COUNT(*) FROM topic_history t
                 WHERE t.run_month = h.run_month AND t.is_trendy = 1) AS n_trendy_in_run
        FROM topic_history h
        WHERE h.lineage_id = ?
        ORDER BY h.run_month DESC
        """,
        (lineage_id,),
    ).fetchall()
    return [_topic_row_to_dict(r) for r in rows]


def get_topic_timeline(db, topic_id):
    rows = db.execute(
        "SELECT month, actual_count, predicted_count FROM monthly_counts "
        "WHERE topic_id = ? ORDER BY month",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_papers_for_topic_month(db, topic_id, year, month):
    month_str = f"{int(year):04d}-{int(month):02d}"
    rows = db.execute(
        """
        SELECT p.paper_id, p.title, p.abstract, p.authors, p.publication_date,
               p.citations, p.fields_of_study, p.language, p.openalex_url, pt.probability
        FROM papers p
        JOIN paper_topics pt ON pt.paper_id = p.paper_id
        WHERE pt.topic_id = ?
          AND strftime('%Y-%m', p.publication_date) = ?
        ORDER BY p.citations DESC, p.publication_date DESC
        """,
        (topic_id, month_str),
    ).fetchall()
    return [dict(r) for r in rows]


def get_paper_scatter(db):
    rows = db.execute(
        """
        SELECT tps.topic_id, t.name AS topic_name, t.is_trendy,
               tps.paper_id, p.title, tps.embed_x, tps.embed_y
        FROM topic_paper_samples tps
        JOIN topics t ON t.topic_id = tps.topic_id
        JOIN papers p ON p.paper_id = tps.paper_id
        """
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["is_trendy"] = bool(d["is_trendy"])
        out.append(d)
    return out


# Study types assigned by the pipeline (StudyType codes), in display order,
# with their display label and family. Codes not listed here (e.g. from an
# older category list) are counted as "other".
STUDY_TYPES = [
    ("interventional", "Interventional (trials)", "Human research"),
    ("observational", "Observational", "Human research"),
    ("qualitative", "Qualitative", "Human research"),
    ("case_report", "Case report / series", "Human research"),
    ("animal", "Animal", "Pre-clinical"),
    ("in_vitro", "In vitro / molecular", "Pre-clinical"),
    ("systematic_review", "Systematic review / meta-analysis", "Reviews"),
    ("other_review", "Other review", "Reviews"),
    ("protocol", "Study protocol", "Other"),
    ("other", "Other / non-research", "Other"),
]


def get_study_type_counts(db, topic_id=None):
    """Paper counts per study type -- for the whole corpus, or for the papers
    assigned to one topic -- each split into published / grey literature /
    not recorded. One dict per study type in STUDY_TYPES order, plus a
    trailing "Not yet classified" row only if some papers have no label."""
    if topic_id is None:
        rows = db.execute(
            "SELECT study_type, grey_literature, COUNT(*) AS n FROM papers GROUP BY study_type, grey_literature"
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT p.study_type, p.grey_literature, COUNT(*) AS n
            FROM paper_topics pt JOIN papers p ON p.paper_id = pt.paper_id
            WHERE pt.topic_id = ?
            GROUP BY p.study_type, p.grey_literature
            """,
            (topic_id,),
        ).fetchall()

    known = {code for code, _, _ in STUDY_TYPES}
    counts = {}
    for r in rows:
        code = r["study_type"]
        if code is not None and code not in known:
            code = "other"
        split = counts.setdefault(code, {"published": 0, "grey": 0, "unrecorded": 0})
        key = {0: "published", 1: "grey"}.get(r["grey_literature"], "unrecorded")
        split[key] += r["n"]

    rows_out = list(STUDY_TYPES)
    if None in counts:
        rows_out.append((None, "Not yet classified", "Unclassified"))
    out = []
    for code, label, family in rows_out:
        split = counts.get(code, {"published": 0, "grey": 0, "unrecorded": 0})
        out.append({"code": code, "label": label, "family": family, **split, "total": sum(split.values())})
    return out


def get_corpus_stats(db):
    n_topics = db.execute("SELECT COUNT(*) AS c FROM topics").fetchone()["c"]
    n_trendy = db.execute("SELECT COUNT(*) AS c FROM topics WHERE is_trendy = 1").fetchone()["c"]
    n_papers = db.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]
    # Corpus window: the span of complete months in monthly_counts. The
    # aggregated series already has the in-progress current month dropped
    # (see build_monthly_mentions in the pipeline), so month_hi is the last
    # month the dashboard actually has full data for. Fall back to raw
    # paper dates if no monthly data has been loaded yet.
    span = db.execute(
        "SELECT MIN(month) AS lo, MAX(month) AS hi FROM monthly_counts"
    ).fetchone()
    if span["lo"] is None:
        span = db.execute(
            "SELECT MIN(publication_date) AS lo, MAX(publication_date) AS hi FROM papers"
        ).fetchone()
    return {
        "n_topics": n_topics,
        "n_trendy": n_trendy,
        "n_papers": n_papers,
        "month_lo": span["lo"],
        "month_hi": span["hi"],
    }


def _topic_row_to_dict(row):
    d = dict(row)
    if d.get("keywords"):
        try:
            d["keywords"] = json.loads(d["keywords"])
        except (TypeError, json.JSONDecodeError):
            d["keywords"] = []
    else:
        d["keywords"] = []
    d["is_trendy"] = bool(d.get("is_trendy"))
    return d
