# Steam Game Review Sentiment Analysis (COMP-4990A)

Sentiment analysis of Steam game reviews with a game recommendation feature.
This repo holds the data cleaning, VADER scoring, and the Streamlit app.

## Folder structure

```
scripts/
  main.py            cleans the raw reviews and keeps the 20 chosen games
  vader_score.py     scores reviews with VADER and builds the per-game summary
  check_cleaned.py   quick checks on the cleaned file
data/
  raw/                       steam_reviews.csv goes here (about 8 GB, NOT in GitHub)
  steam_game_list.csv        the 20 games used
  steam_app_type_cache.json  cache of Steam app types (game / software)
  cleaned/                   cleaned_steam_reviews.csv, steam_review_sample.csv
  scored/                    vader_scored_reviews.csv, per_game_summary.csv
app/                         Streamlit app (reads data/scored/per_game_summary.csv)
```

## Setup

1. Install Python 3.9 or newer and Git LFS (`brew install git-lfs`, then `git lfs install`).
2. After cloning, run `git lfs pull` to download the large CSV files.
3. Install the packages: `pip3 install -r requirements.txt`

## Run order (from the project folder)

```
python3 scripts/main.py [path/to/steam_reviews.csv]
python3 scripts/check_cleaned.py
python3 scripts/vader_score.py
```

The raw dataset is the Kaggle Steam reviews file. Put it in `data/raw/` or pass its path to `main.py`.

## The scored files

`data/scored/per_game_summary.csv` has one row per game:
`pct_positive`, `pct_neutral`, `pct_negative` (VADER labels, compound cut-offs of 0.05 and -0.05),
`pct_positive_excl_neutral` (positive share ignoring neutral), and
`pct_recommended_by_steam` (the real Steam thumbs up rate, for comparison).

## Notes

- VADER is scored on `review_raw` (original text). `review_clean` is for the machine learning models.
- If the NLTK stopwords download fails on a Mac (certificate error), run:
  `python3 -c "import ssl, nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('stopwords')"`
