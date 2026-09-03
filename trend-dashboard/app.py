from flask import Flask, abort, jsonify, render_template, request

import db
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    # -- Pages -------------------------------------------------------------

    @app.route("/")
    def index():
        conn = db.get_db()
        stats = db.get_corpus_stats(conn)
        topics = db.list_topics(conn)
        trendy_topics = [t for t in topics if t["is_trendy"]]
        default_ids = [t["topic_id"] for t in trendy_topics[: app.config["DEFAULT_TREND_TOPIC_COUNT"]]]
        if not default_ids:
            default_ids = [t["topic_id"] for t in topics[: app.config["DEFAULT_TREND_TOPIC_COUNT"]]]
        return render_template(
            "index.html",
            stats=stats,
            topics=topics,
            trendy_topics=trendy_topics[:12],
            default_topic_ids=default_ids,
        )

    @app.route("/topic/<int:topic_id>")
    def topic_detail(topic_id):
        conn = db.get_db()
        topic = db.get_topic(conn, topic_id)
        if topic is None:
            abort(404)
        timeline = db.get_topic_timeline(conn, topic_id)
        return render_template("topic.html", topic=topic, timeline=timeline)

    @app.route("/topic/<int:topic_id>/month/<int:year>/<int:month>")
    def topic_month(topic_id, year, month):
        conn = db.get_db()
        topic = db.get_topic(conn, topic_id)
        if topic is None:
            abort(404)
        if not (1 <= month <= 12):
            abort(404)
        papers = db.get_papers_for_topic_month(conn, topic_id, year, month)
        return render_template(
            "month.html", topic=topic, year=year, month=month, papers=papers
        )

    # -- JSON API (consumed by static/js/main.js for the interactive charts) --

    @app.route("/api/topics")
    def api_topics():
        conn = db.get_db()
        q = request.args.get("q")
        trendy_only = request.args.get("trendy") == "1"
        limit = request.args.get("limit", type=int)
        return jsonify(db.list_topics(conn, q=q, trendy_only=trendy_only, limit=limit))

    @app.route("/api/topics/<int:topic_id>/timeline")
    def api_topic_timeline(topic_id):
        conn = db.get_db()
        topic = db.get_topic(conn, topic_id)
        if topic is None:
            abort(404)
        return jsonify({"topic": topic, "timeline": db.get_topic_timeline(conn, topic_id)})

    @app.route("/api/paper-scatter")
    def api_paper_scatter():
        conn = db.get_db()
        return jsonify(db.get_paper_scatter(conn))

    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(port=5000)
