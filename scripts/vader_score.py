"""
VADER scoring for the Steam review dataset.

Input : data/cleaned/cleaned_steam_reviews.csv  (must have review_raw, app_name, app_id, recommended)
Output: data/scored/vader_scored_reviews.csv   (every original column + VADER scores and label)
        data/scored/per_game_summary.csv       (one row per game, used by the app and the charts)

Run:  python3 scripts/vader_score.py [input.csv]   (from the project folder)
Needs: pip install pandas vaderSentiment tqdm
"""
import os
import sys
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

INPUT = (
    sys.argv[1] if len(sys.argv) > 1
    else os.path.join(DATA_FOLDER, "cleaned", "cleaned_steam_reviews.csv")
)
SCORED_FOLDER = os.path.join(DATA_FOLDER, "scored")
os.makedirs(SCORED_FOLDER, exist_ok=True)
SCORED_OUT = os.path.join(SCORED_FOLDER, "vader_scored_reviews.csv")
SUMMARY_OUT = os.path.join(SCORED_FOLDER, "per_game_summary.csv")

# Standard VADER cut-offs on the compound score
POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05

df = pd.read_csv(INPUT)
for col in ["review_raw", "app_name", "app_id", "recommended"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

analyzer = SentimentIntensityAnalyzer()

# VADER works best on the ORIGINAL text (capitals, punctuation, emoji), so use review_raw
tqdm.pandas(desc="Scoring with VADER")
scores = df["review_raw"].astype(str).progress_apply(analyzer.polarity_scores)

df["vader_neg"] = scores.apply(lambda s: s["neg"])
df["vader_neu"] = scores.apply(lambda s: s["neu"])
df["vader_pos"] = scores.apply(lambda s: s["pos"])
df["vader_compound"] = scores.apply(lambda s: s["compound"])


def label(compound):
    if compound >= POSITIVE_CUTOFF:
        return "positive"
    if compound <= NEGATIVE_CUTOFF:
        return "negative"
    return "neutral"


df["vader_label"] = df["vader_compound"].apply(label)
df.to_csv(SCORED_OUT, index=False)

# Per-game summary for the app / charts
g = df.groupby(["app_id", "app_name"], sort=False)
summary = g.agg(
    total_reviews=("vader_label", "size"),
    pct_positive=("vader_label", lambda s: (s == "positive").mean() * 100),
    pct_neutral=("vader_label", lambda s: (s == "neutral").mean() * 100),
    pct_negative=("vader_label", lambda s: (s == "negative").mean() * 100),
    avg_compound=("vader_compound", "mean"),
    pct_recommended_by_steam=("recommended", lambda s: s.mean() * 100),
).reset_index()
summary = summary.round({"pct_positive": 1, "pct_neutral": 1, "pct_negative": 1,
                         "avg_compound": 3, "pct_recommended_by_steam": 1})
summary = summary.sort_values("app_name").reset_index(drop=True)

# Fairer comparison with Steam's thumbs up / down: ignore the neutral reviews
summary["pct_positive_excl_neutral"] = (
    summary["pct_positive"] / (summary["pct_positive"] + summary["pct_negative"]) * 100
).round(1)

summary.to_csv(SUMMARY_OUT, index=False)

# Quick sanity check against Steam's own thumbs up / down label
decided = df[df["vader_label"] != "neutral"]
agree = ((decided["vader_label"] == "positive") == decided["recommended"]).mean()
print(f"\nReviews scored: {len(df)}")
print(df["vader_label"].value_counts(normalize=True).mul(100).round(1).to_string())
print(f"Agreement with Steam recommended (non-neutral only): {agree:.1%}")
print(f"Saved {SCORED_OUT} and {SUMMARY_OUT}")
