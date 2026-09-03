"""Populate galenos.db with SYNTHETIC sample data.

This is fake data shaped like the real pipeline's output (see schema.sql /
README.md), so the dashboard is fully interactive out of the box. None of
the topic names, papers, authors or abstracts are real — swap this out with
scripts/import_notebook_outputs.py once you have real exports.

Usage:
    python scripts/init_db.py --reset
    python scripts/generate_sample_data.py
"""
import json
import math
import os
import random
import sqlite3
from datetime import date

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "galenos.db")

random.seed(42)

N_MONTHS = 120  # 10 years
END_MONTH = (2026, 6)  # dataset window ends the month before "now"

CONDITIONS = [
    "depression", "anxiety", "psychosis", "PTSD", "bipolar disorder", "OCD",
    "eating disorders", "insomnia", "substance use disorder", "ADHD",
    "schizophrenia", "panic disorder", "social anxiety", "postpartum depression",
    "early-onset psychosis", "treatment-resistant depression", "suicidal behaviour",
]
INTERVENTIONS = [
    "cognitive behavioural therapy", "ketamine", "mindfulness training", "exercise",
    "digital therapeutics", "psilocybin-assisted therapy", "transcranial magnetic stimulation",
    "virtual reality exposure", "peer support programmes", "telehealth delivery",
    "school-based prevention", "family therapy", "antipsychotic dose reduction",
    "sleep intervention", "compassion-focused therapy", "app-based self-guided therapy",
]
POPULATIONS = [
    "adolescents", "university students", "older adults", "refugees",
    "postpartum women", "frontline healthcare workers", "veterans", "LGBTQ+ youth",
    "rural communities", "incarcerated individuals", "children in foster care",
    "new parents", "autistic adults", "low-income families",
]
MECHANISMS = [
    "neuroinflammation", "the gut-brain axis", "default mode network connectivity",
    "HPA axis dysregulation", "glutamatergic signalling", "sleep architecture",
    "emotion regulation circuitry", "epigenetic markers", "microbiome composition",
    "interoception", "reward processing", "cortical thinning",
]

TEMPLATES = [
    ("{i} for {c}", "intervention"),
    ("{c} in {p}", "population"),
    ("{m} in {c}", "mechanism"),
    ("artificial intelligence for {c} screening", "ai"),
    ("digital biomarkers of {c}", "digital"),
    ("{i} in {p} with {c}", "combo"),
]

FIELDS_OF_STUDY = [
    "Psychiatry", "Clinical Psychology", "Psychology", "Neuroscience",
    "Public Health", "Medicine", "Cognitive Science",
]

FIRST_NAMES = ["A.", "J.", "M.", "S.", "R.", "L.", "K.", "T.", "N.", "C."]
LAST_NAMES = [
    "Nakamura", "Okafor", "Fernandez", "Kowalski", "Singh", "Bianchi", "Larsen",
    "Haddad", "Petrova", "Kim", "Dubois", "Osei", "Almeida", "Reyes", "Novak",
]


def add_months(year, month, delta):
    total = (year * 12 + (month - 1)) + delta
    return total // 12, total % 12 + 1


def month_range(end_year, end_month, n):
    months = []
    for delta in range(-(n - 1), 1):
        y, m = add_months(end_year, end_month, delta)
        months.append(f"{y:04d}-{m:02d}-01")
    return months


def make_topic_name(used):
    for _ in range(50):
        tmpl, kind = random.choice(TEMPLATES)
        c = random.choice(CONDITIONS)
        i = random.choice(INTERVENTIONS)
        p = random.choice(POPULATIONS)
        m = random.choice(MECHANISMS)
        name = tmpl.format(c=c, i=i, p=p, m=m)
        name = name[0].upper() + name[1:]
        if name not in used:
            used.add(name)
            keywords = list({*name.lower().replace(",", "").split(" ")})
            filler = ["risk", "outcomes", "intervention", "cohort", "symptoms",
                      "clinical", "longitudinal", "biomarker", "screening", "trial"]
            random.shuffle(filler)
            keywords = (keywords + filler)[:15]
            return name, keywords
    raise RuntimeError("ran out of unique topic name combinations")


def simulate_series(n_months, baseline, growth_recent, noisy=True):
    """Return (actual, predicted) monthly counts. `predicted` is None for the
    first 6 months, mirroring the paper's 6-month sliding-window warm-up."""
    actual = []
    for t in range(n_months):
        trend = baseline
        if growth_recent and t >= n_months - 30:
            recent_t = t - (n_months - 30)
            trend += growth_recent * (recent_t / 30) ** 1.5
        noise = random.gauss(0, max(1.0, trend * 0.18)) if noisy else 0
        val = max(0, round(trend + noise))
        actual.append(val)

    predicted = [None] * min(6, n_months)
    for t in range(6, n_months):
        window = actual[max(0, t - 6):t]
        pred = sum(window) / len(window) if window else 0
        pred += random.gauss(0, 0.6)
        predicted.append(max(0.0, round(pred, 2)))
    return actual, predicted


def compute_trend_rank(actual, predicted, n=4, m=6):
    """Loosely mirrors the paper's trendiness rank: decayed sum of
    (actual - predicted) over the last m months where actual > predicted,
    normalised by mean actual (or 1 if that mean < 1)."""
    tail_a = actual[-m:]
    tail_p = predicted[-m:]
    exceed_months = sum(1 for a, p in zip(tail_a, tail_p) if p is not None and a > p)
    if exceed_months < n:
        return False, None
    score = 0.0
    for idx, (a, p) in enumerate(zip(tail_a, tail_p)):
        if p is None:
            continue
        decay = 0.7 ** (m - 1 - idx)  # more recent months weigh more
        score += max(0, a - p) * decay
    mean_actual = sum(tail_a) / len(tail_a) if tail_a else 0
    norm = mean_actual if mean_actual >= 1 else 1
    return True, round(score / norm, 3)


def make_abstract(topic_name, keywords):
    kw_sample = random.sample(keywords, k=min(5, len(keywords)))
    return (
        f"This study examines {topic_name.lower()}, drawing on a cohort analysed for "
        f"{', '.join(kw_sample[:-1])} and {kw_sample[-1]}. We report associations "
        f"between key clinical variables and discuss implications for future research "
        f"and practice in this area. (Synthetic sample abstract for demo purposes.)"
    )


def make_authors():
    n = random.randint(2, 5)
    return "; ".join(f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n))


def make_mae(actual, predicted):
    errs = [abs(a - p) for a, p in zip(actual, predicted) if p is not None]
    return round(sum(errs) / len(errs), 3) if errs else None


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    months = month_range(END_MONTH[0], END_MONTH[1], N_MONTHS)
    used_names = set()

    n_topics = 60
    paper_seq = 1
    for topic_id in range(1, n_topics + 1):
        name, keywords = make_topic_name(used_names)

        baseline = random.choice([1, 2, 3, 5, 8, 15, 25])
        is_growth_candidate = random.random() < 0.35
        growth_recent = random.uniform(3, 20) if is_growth_candidate else 0

        actual, predicted = simulate_series(N_MONTHS, baseline, growth_recent)
        is_trendy, trend_rank = compute_trend_rank(actual, predicted)
        trend_mae = make_mae(actual, predicted)

        conn.execute(
            "INSERT INTO topics (topic_id, name, keywords, n_papers, is_trendy, trend_rank, trend_mae) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (topic_id, name, json.dumps(keywords), sum(actual), int(is_trendy), trend_rank, trend_mae),
        )

        for month_str, a, p in zip(months, actual, predicted):
            conn.execute(
                "INSERT INTO monthly_counts (topic_id, month, actual_count, predicted_count) VALUES (?, ?, ?, ?)",
                (topic_id, month_str, a, p),
            )

            year, mo = int(month_str[:4]), int(month_str[5:7])
            # Cap synthetic papers per topic-month to keep dataset a
            # reasonable demo size while still populating every drill-down.
            n_papers_this_month = min(a, 12)
            for _ in range(n_papers_this_month):
                paper_id = f"W{900000000 + paper_seq}"
                paper_seq += 1
                day = random.randint(1, 28)
                pub_date = date(year, mo, day).isoformat()
                title = f"{name}: a study of clinical and research patterns ({year})"
                conn.execute(
                    "INSERT INTO papers (paper_id, title, abstract, authors, publication_date, "
                    "citations, fields_of_study, language, openalex_url) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper_id,
                        title,
                        make_abstract(name, keywords),
                        make_authors(),
                        pub_date,
                        max(0, round(random.gauss(6, 8))),
                        random.choice(FIELDS_OF_STUDY),
                        "en",
                        f"https://openalex.org/{paper_id}",
                    ),
                )
                conn.execute(
                    "INSERT INTO paper_topics (paper_id, topic_id, probability) VALUES (?, ?, 1.0)",
                    (paper_id, topic_id),
                )

        if topic_id % 10 == 0:
            print(f"...generated {topic_id}/{n_topics} topics")

    conn.commit()
    conn.close()
    print(f"Sample data written to {DB_PATH} ({n_topics} topics, {paper_seq - 1} papers).")


if __name__ == "__main__":
    main()
