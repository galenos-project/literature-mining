import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    DATABASE_PATH = os.path.join(BASE_DIR, "data", "galenos.db")
    JSON_SORT_KEYS = False
    # Number of topics shown by default on the front-page trend chart.
    DEFAULT_TREND_TOPIC_COUNT = 8
