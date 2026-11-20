"""Quick checks on the cleaned reviews file. Run: python3 scripts/check_cleaned.py"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "data", "cleaned", "cleaned_steam_reviews.csv")
d = pd.read_csv(path)

print("Rows, columns:", d.shape, "| games:", d.app_name.nunique())
print("language column removed:", "language" not in d.columns)
print("'more' kept by cleaning:", d.review_clean.str.contains(r"\bmore\b").mean() > 0)
print("share of reviews under 5 words:", round((d.review_raw.str.split().str.len() < 5).mean(), 3))
print("glued-word check (should be 0):",
      d.review_clean.str.contains("exactlynew|realsequels|skincannot").sum())
print("reviews per game:")
print(d.app_name.value_counts().to_string())
