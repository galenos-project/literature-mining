"""Create (or recreate) the SQLite database from schema.sql.

Usage:
    python scripts/init_db.py [--reset]

--reset drops existing tables first. Without it, CREATE TABLE IF NOT EXISTS
statements just make sure the tables exist.
"""
import argparse
import os
import sqlite3

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "galenos.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")

TABLES = ["paper_topics", "papers", "monthly_counts", "topics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop existing tables first")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if args.reset:
        for t in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    main()
