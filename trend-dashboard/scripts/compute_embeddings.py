"""Compute a 2D scatter-plot layout for a sample of papers per topic.

Usage:
    python scripts/compute_embeddings.py [--samples-per-topic 5] [--max-features 5000]
"""
import argparse
import os
import random
import sqlite3
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "galenos.db")

random.seed(42)


def sample_papers_per_topic(conn, samples_per_topic):
    """Return {topic_id: [paper_id, ...]}, sampling up to `samples_per_topic`
    distinct papers per topic from among those with usable title/abstract
    text."""
    topic_ids = [r[0] for r in conn.execute("SELECT topic_id FROM topics").fetchall()]
    samples = {}
    for topic_id in topic_ids:
        rows = conn.execute(
            """
            SELECT p.paper_id
            FROM papers p
            JOIN paper_topics pt ON pt.paper_id = p.paper_id
            WHERE pt.topic_id = ?
              AND (p.title IS NOT NULL AND p.title != '' OR p.abstract IS NOT NULL AND p.abstract != '')
            """,
            (topic_id,),
        ).fetchall()
        paper_ids = [r[0] for r in rows]
        if not paper_ids:
            continue
        k = min(samples_per_topic, len(paper_ids))
        samples[topic_id] = random.sample(paper_ids, k)
    return samples


def fetch_texts(conn, paper_ids):
    """Return {paper_id: 'title. abstract'} for the given papers."""
    texts = {}
    # SQLite has a default limit on the number of host parameters (999);
    # chunk the IN clause to stay well under that.
    paper_ids = list(paper_ids)
    for i in range(0, len(paper_ids), 500):
        chunk = paper_ids[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT paper_id, title, abstract FROM papers WHERE paper_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for paper_id, title, abstract in rows:
            texts[paper_id] = f"{title or ''}. {abstract or ''}".strip()
    return texts


def embed_texts(texts_by_paper_id):
    """Fit sentence-transformer embeddings + TSNE(2) across all given texts
    and return {paper_id: (x, y)}. """

    paper_ids = list(texts_by_paper_id.keys())
    corpus = [texts_by_paper_id[pid] for pid in paper_ids]

    n_components = 2
    if len(corpus) <= n_components:
        return {pid: (0.0, 0.0) for pid in paper_ids}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, show_progress_bar=True, normalize_embeddings=True)

    # Step 3: Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    coords = tsne.fit_transform(embeddings)

    return {pid: (float(coords[i, 0]), float(coords[i, 1])) for i, pid in enumerate(paper_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-topic", type=int, default=5)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")

    print("Sampling papers per topic...")
    samples = sample_papers_per_topic(conn, args.samples_per_topic)
    n_topics_with_samples = len(samples)
    n_total_samples = sum(len(v) for v in samples.values())
    print(f"  {n_topics_with_samples} topics have at least one usable paper "
          f"({n_total_samples} (topic, paper) pairs to place)")

    if n_total_samples == 0:
        print("No papers with title/abstract text found via paper_topics — nothing to embed.")
        return

    unique_paper_ids = sorted({pid for pids in samples.values() for pid in pids})
    print(f"Fetching text for {len(unique_paper_ids)} unique sampled papers...")
    texts = fetch_texts(conn, unique_paper_ids)
    # Preserve a stable order for embed_texts.
    texts = {pid: texts.get(pid, "") for pid in unique_paper_ids}

    print(f"Embedding {len(texts)} documents...")
    coords_by_paper = embed_texts(texts)

    conn.execute("DELETE FROM topic_paper_samples")
    n_written = 0
    for topic_id, paper_ids in samples.items():
        for paper_id in paper_ids:
            x, y = coords_by_paper[paper_id]
            conn.execute(
                "INSERT OR REPLACE INTO topic_paper_samples (topic_id, paper_id, embed_x, embed_y) "
                "VALUES (?, ?, ?, ?)",
                (topic_id, paper_id, x, y),
            )
            n_written += 1
    conn.commit()
    conn.close()
    print(f"Done. Wrote {n_written} scatter points across {n_topics_with_samples} topics.")


if __name__ == "__main__":
    main()
