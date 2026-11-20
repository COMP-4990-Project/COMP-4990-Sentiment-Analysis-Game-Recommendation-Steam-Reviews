"""Steam Review Sentiment - prototype app (VADER scores per game).

Run from the repo root:
    streamlit run app/app.py

Reads data/scored/per_game_summary.csv (one row per game).
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "scored" / "per_game_summary.csv"

# The three sentiment columns from per_game_summary.csv
SENTIMENT_COLS = ["pct_positive", "pct_neutral", "pct_negative"]
LABELS = ["Positive", "Neutral", "Negative"]
COLORS = ["#2e7d32", "#9e9e9e", "#c62828"]

# Column names we try, in order, in case the file uses a slightly different name.
GAME_NAME_CANDIDATES = ["app_name", "game", "game_name", "name", "title"]
COUNT_CANDIDATES = ["total_reviews", "review_count", "n_reviews", "num_reviews", "reviews", "count"]


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the summary CSV. Raises a readable error if it can't be used."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure you pulled the latest main branch "
            "and are running the app from the repo root."
        )
    # A Git LFS pointer is a tiny text file; it means `git lfs pull` was not run.
    with open(path, "r", errors="ignore") as f:
        if f.readline().startswith("version https://git-lfs"):
            raise ValueError(
                "per_game_summary.csv is only a Git LFS pointer. "
                "Run `git lfs install` and then `git lfs pull`, then restart the app."
            )
    df = pd.read_csv(path)
    missing = [c for c in SENTIMENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def as_percent(series: pd.Series) -> pd.Series:
    """Make sure values are on a 0-100 scale (handles a 0-1 file too)."""
    return series * 100 if series.dropna().max() <= 1.0 else series


def sentiment_chart(values: list[float]):
    data = pd.DataFrame({"Sentiment": LABELS, "Percent": values})
    base = alt.Chart(data).encode(
        x=alt.X("Sentiment:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Percent:Q", scale=alt.Scale(domain=[0, 100]), title="Share of reviews (%)"),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=LABELS, range=COLORS), legend=None),
        tooltip=["Sentiment", alt.Tooltip("Percent:Q", format=".1f")],
    )
    labels = base.mark_text(dy=-8).encode(text=alt.Text("Percent:Q", format=".1f"))
    return (bars + labels).properties(height=320)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Steam Review Sentiment", page_icon="🎮", layout="centered")
st.title("Steam Review Sentiment")
st.caption("VADER sentiment on English Steam reviews for 20 games (prototype).")

try:
    df = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

game_col = find_col(df, GAME_NAME_CANDIDATES)
if game_col is None:  # fall back to the first text column
    text_cols = df.select_dtypes(exclude="number").columns
    game_col = text_cols[0] if len(text_cols) else df.columns[0]
count_col = find_col(df, COUNT_CANDIDATES)

for col in SENTIMENT_COLS + ["pct_positive_excl_neutral", "pct_recommended_by_steam"]:
    if col in df.columns:
        df[col] = as_percent(df[col])

# --- Game selection: type to search, then pick from the matches -------------
query = st.text_input("Search for a game", placeholder="e.g. DARK SOULS III")
all_games = sorted(df[game_col].astype(str).unique(), key=str.lower)
matches = [g for g in all_games if query.strip().lower() in g.lower()] if query.strip() else all_games

if not matches:
    st.warning(
        f"No game found matching “{query}”. This prototype covers {len(all_games)} games. "
        "Check the spelling or clear the search box to see the full list."
    )
    with st.expander("Show all available games"):
        st.write(all_games)
    st.stop()

game = st.selectbox("Choose a game", matches)
row = df[df[game_col].astype(str) == game].iloc[0]

# --- Sentiment percentages + bar chart ---------------------------------------
st.subheader(game)
if count_col:
    st.caption(f"Based on {int(row[count_col]):,} English reviews.")

cols = st.columns(3)
for c, label, key in zip(cols, LABELS, SENTIMENT_COLS):
    c.metric(label, f"{row[key]:.1f}%")

st.altair_chart(sentiment_chart([row[k] for k in SENTIMENT_COLS]))

# --- Steam comparison --------------------------------------------------------
st.subheader("Compared with Steam's own rating")
if "pct_recommended_by_steam" in df.columns and pd.notna(row["pct_recommended_by_steam"]):
    steam = row["pct_recommended_by_steam"]
    vader_pos = row["pct_positive"]
    c1, c2 = st.columns(2)
    c1.metric("VADER positive", f"{vader_pos:.1f}%", f"{vader_pos - steam:+.1f} pts vs Steam")
    c2.metric("Steam recommended", f"{steam:.1f}%")
    if "pct_positive_excl_neutral" in df.columns and pd.notna(row["pct_positive_excl_neutral"]):
        ex = row["pct_positive_excl_neutral"]
        st.caption(
            f"Ignoring neutral reviews, VADER's positive share is {ex:.1f}% "
            f"({ex - steam:+.1f} pts vs Steam)."
        )
    st.caption(
        "Steam's rating is the share of reviewers who clicked “Recommended”. "
        "VADER reads the review text, so the two can differ."
    )
else:
    st.info("No Steam rating is available for this game.")
