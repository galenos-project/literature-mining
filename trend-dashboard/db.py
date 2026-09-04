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
        "SELECT topic_id, name, keywords, n_papers, is_trendy, trend_rank, trend_mae "
        "FROM topics WHERE topic_id = ?",
        (topic_id,),
    ).fetchone()
    return _topic_row_to_dict(row) if row else None


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
