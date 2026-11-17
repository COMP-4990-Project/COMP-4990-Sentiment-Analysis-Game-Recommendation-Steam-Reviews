
import pandas as pd
import numpy as np
import re
import nltk
import os
import json
import time
import requests

from tqdm import tqdm
from nltk.corpus import stopwords
from fast_langdetect import LangDetectConfig, LangDetector



#initial setup


nltk.download("stopwords")
tqdm.pandas()

print("Loading Steam reviews dataset...")

df = pd.read_csv("steam_reviews.csv")

original_review_count = len(df)

print("\nDataset successfully loaded!")
print("Total original reviews:", original_review_count)



#configuration


MAX_REVIEWS_PER_GAME = 2000
MIN_REVIEWS_PER_GAME = 100
NUMBER_OF_GAMES = 20
SAMPLE_SIZE = 1000

ENGLISH_CONFIDENCE = 0.80


LANGUAGE_BATCH_SIZE = 50000


LANGUAGE_MAX_INPUT_LENGTH = None

# Steam API settings
STEAM_API_URL = "https://store.steampowered.com/api/appdetails/"
STEAM_REQUEST_DELAY = 1.6

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

STEAM_CACHE_PATH = os.path.join(
    script_folder,
    "steam_app_type_cache.json"
)



#remove unnecessary columns


print("\nStep 1: Removing unnecessary columns...")

columns_to_remove = [
    column for column in df.columns
    if column == "Unnamed: 0" or column.startswith("author.")
]

print("Columns being removed:", columns_to_remove)

df.drop(
    columns=columns_to_remove,
    inplace=True
)

print("Remaining columns:")
print(df.columns.tolist())



#original dataset check


print("\n========== ORIGINAL DATASET CHECK ==========")

print("\nNumber of unique apps:")
print(df["app_name"].nunique())

print("\nMost common apps:")
print(df["app_name"].value_counts().head(20))

print("\nAvailable columns:")
print(df.columns.tolist())

print("\n============================================")


# Check required columns

required_columns = [
    "app_id",
    "app_name",
    "review"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"The dataset does not contain the required column: {column}"
        )



#Preserve original reviews


print("\nStep 2: Preserving original reviews...")

df = df.rename(
    columns={"review": "review_raw"}
)

print("Original reviews preserved in review_raw.")



#remove empty reviews


print("\nStep 3: Removing empty reviews...")

df = df.dropna(
    subset=["review_raw", "app_name", "app_id"]
).copy()

df["review_raw"] = df["review_raw"].astype(str)

df = df[
    df["review_raw"].str.strip() != ""
].copy()

print("Reviews remaining:", len(df))



#Remove duplicates


print("\nStep 4: Removing duplicate reviews...")

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["review_raw"],
    keep="first"
).copy()

duplicates_removed = before_duplicates - len(df)

print("Duplicate reviews removed:", duplicates_removed)
print("Reviews remaining:", len(df))



#remove short reviews


print("\nStep 5: Removing very short reviews...")

before_short = len(df)

review_lengths = df["review_raw"].str.strip().str.len()

short_review_mask = (
    (review_lengths < 3) &
    (~df["review_raw"].str.strip().str.lower().isin(["no"]))
)

df = df[~short_review_mask].copy()

short_removed = before_short - len(df)

print("Short reviews removed:", short_removed)
print("Reviews remaining:", len(df))



#remove repeated characters like aaaaaaaaaaaaaaaa


print("\nStep 6: Removing repeated-character spam...")

before_spam = len(df)


def is_repeated_character_spam(text):

    text = text.strip()

    return bool(
        re.fullmatch(r"(.)\1{4,}", text)
    )


df = df[
    ~df["review_raw"].progress_apply(
        is_repeated_character_spam
    )
].copy()

spam_removed = before_spam - len(df)

print("Spam reviews removed:", spam_removed)
print("Reviews remaining:", len(df))



#language detection


print("\nStep 7: Checking review languages")

print("Ignoring the original CSV language column.")
print("English confidence threshold:", ENGLISH_CONFIDENCE)

print("\nLoading Fast-LangDetect language model...")

language_config = LangDetectConfig(
    model="lite",
    max_input_length=LANGUAGE_MAX_INPUT_LENGTH
)

language_detector = LangDetector(
    language_config
)

print("Fast-LangDetect loaded successfully!")


#Prepare reviews for language detection

def prepare_review_for_detection(text):

    return (
        str(text)
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


#Detect whether an individual review is English

def is_english_review(text):

    text = prepare_review_for_detection(text)

    if not text:
        return False

    try:

        prediction = language_detector.detect(
            text,
            k=1
        )

        if not prediction:
            return False

        predicted_language = prediction[0].get("lang")
        confidence = prediction[0].get("score", 0)

        return (
            predicted_language == "en"
            and confidence >= ENGLISH_CONFIDENCE
        )

    except ValueError:
        return False


#Identify English reviews in batches

print("\nBeginning language detection...")

before_language = len(df)

english_mask = np.zeros(
    len(df),
    dtype=bool
)

with tqdm(
    total=len(df),
    desc="Detecting review languages"
) as progress:

    for start in range(
        0,
        len(df),
        LANGUAGE_BATCH_SIZE
    ):

        end = min(
            start + LANGUAGE_BATCH_SIZE,
            len(df)
        )

        review_batch = df["review_raw"].iloc[start:end]

        for position, review_text in enumerate(review_batch, start=start):

            english_mask[position] = is_english_review(
                review_text
            )

            progress.update(1)


#apply the language filter

df = df.loc[
    english_mask
].copy()

language_removed = before_language - len(df)

print("\nFast-LangDetect language detection completed!")

print("Non-English/uncertain reviews removed:", language_removed)
print("English reviews remaining:", len(df))
print("English confidence threshold:", ENGLISH_CONFIDENCE)



#make new columns (review_clean and review_raw)

print("\nStep 8: Creating columns...")

df["review_clean"] = df["review_raw"].copy()

df["review_clean"] = df["review_clean"].str.lower()


#Expanding contractions

print("\nStep 9: Expanding contractions...")

contractions = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "needn't": "need not",
    "mightn't": "might not",
    "shan't": "shall not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would"
}


def expand_contractions(text):

    text = text.replace("’", "'")

    for contraction, expansion in contractions.items():

        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text
        )

    return text


df["review_clean"] = df["review_clean"].progress_apply(
    expand_contractions
)



#Remove numbers and punctuation


print("\nStep 10: Removing punctuation and numbers...")

df["review_clean"] = df["review_clean"].progress_apply(
    lambda text: re.sub(r"[^a-z\s]", "", text)
)



#remove stopwords


print("\nStep 11: Removing stopwords...")

stop_words = set(stopwords.words("english"))

keep_words = {
    "not",
    "no",
    "never",
    "but",
    "very",
    "too",
    "nor",
    "neither"
}

stop_words = stop_words - keep_words


def remove_stopwords(text):

    return " ".join(
        word for word in text.split()
        if word not in stop_words
    )


df["review_clean"] = df["review_clean"].progress_apply(
    remove_stopwords
)



# Normalize whitespace


print("\nStep 12: Normalizing whitespace...")

df["review_clean"] = (
    df["review_clean"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)



#Remove empty reviews again

print("\nStep 13: Removing reviews that became empty...")

before_empty = len(df)

df = df[
    df["review_clean"].notna() &
    (df["review_clean"] != "")
].copy()

empty_removed = before_empty - len(df)

print("Empty cleaned reviews removed:", empty_removed)
print("Reviews remaining:", len(df))



# verify steam app types


print("\nStep 14: Verifying Steam app types...")

print("Checking Steam to distinguish games from software, DLC, etc.")


#Load previously cached app types
if os.path.exists(STEAM_CACHE_PATH):

    with open(
        STEAM_CACHE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        steam_app_cache = json.load(file)

    print(
        "Loaded",
        len(steam_app_cache),
        "cached Steam app types."
    )

else:

    steam_app_cache = {}

    print("No existing Steam app cache found.")


#Normalize app IDs

def normalize_app_id(app_id):

    numeric_id = pd.to_numeric(
        app_id,
        errors="coerce"
    )

    if pd.isna(numeric_id):

        raise ValueError(
            f"Invalid Steam app_id encountered: {app_id}"
        )

    return str(int(numeric_id))


#Look up an app's type using Steam's Store API

def get_steam_app_type(app_id):

    app_id = normalize_app_id(app_id)

    if app_id in steam_app_cache:

        return steam_app_cache[app_id]

    print(f"Checking Steam app ID: {app_id}")

    for attempt in range(3):

        try:

            time.sleep(STEAM_REQUEST_DELAY)

            response = requests.get(
                STEAM_API_URL,
                params={
                    "appids": app_id,
                    "cc": "ca",
                    "l": "english"
                },
                headers={
                    "User-Agent": "SteamReviewDatasetCleaner/1.0"
                },
                timeout=20
            )

            response.raise_for_status()

            result = response.json()

            app_result = result.get(app_id)

            if not app_result or not app_result.get("success"):

                raise RuntimeError(
                    f"Steam could not verify app ID {app_id}."
                )

            app_data = app_result.get("data", {})

            app_type = app_data.get("type")

            if not app_type:

                raise RuntimeError(
                    f"Steam did not provide an app type for ID {app_id}."
                )

            app_type = str(app_type).strip().lower()

            steam_app_cache[app_id] = app_type

            with open(
                STEAM_CACHE_PATH,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    steam_app_cache,
                    file,
                    indent=4
                )

            return app_type

        except requests.RequestException as error:

            if attempt == 2:

                raise RuntimeError(
                    f"Unable to contact Steam for app ID {app_id}."
                ) from error

            print(
                f"Request failed. Retrying app {app_id} "
                f"({attempt + 1}/3)..."
            )

            time.sleep(5 * (attempt + 1))

    raise RuntimeError(
        f"Could not determine Steam app type for ID {app_id}."
    )


#Only check apps with enough reviews to qualify

game_review_counts = df["app_name"].value_counts()

candidate_games = game_review_counts[
    game_review_counts >= MIN_REVIEWS_PER_GAME
].index.tolist()

candidate_rows = df[
    df["app_name"].isin(candidate_games)
].copy()

candidate_app_ids = (
    candidate_rows["app_id"]
    .dropna()
    .unique()
)

print("\nApps with at least 100 reviews:", len(candidate_games))
print("Unique Steam app IDs to verify:", len(candidate_app_ids))

if len(candidate_app_ids) == 0:

    raise ValueError(
        "No eligible Steam app IDs were found."
    )




app_type_map = {}

for app_id in tqdm(
    candidate_app_ids,
    desc="Verifying Steam app types"
):

    normalized_id = normalize_app_id(app_id)

    app_type_map[normalized_id] = get_steam_app_type(
        normalized_id
    )


#Add verified types to the dataset

df["app_id_normalized"] = df["app_id"].apply(
    normalize_app_id
)

df["steam_app_type"] = df["app_id_normalized"].map(
    app_type_map
)

before_app_filter = len(df)

df = df[
    df["steam_app_type"].isna() |
    (df["steam_app_type"] == "game")
].copy()

apps_removed = before_app_filter - len(df)

df.drop(
    columns=["app_id_normalized"],
    inplace=True
)

print("\nSteam app type verification completed!")

print("Non-game reviews removed:", apps_removed)

print("\nVerified app types:")
print(df["steam_app_type"].value_counts(dropna=False))

print("\nReviews remaining:", len(df))



#Review columns


print("\n========== REVIEW COLUMN CHECK ==========")

if len(df) > 0:

    print("\nOriginal review:")
    print(df["review_raw"].iloc[0])

    print("\nCleaned review:")
    print(df["review_clean"].iloc[0])

    print("\nBoth review columns:")
    print(
        df[["review_raw", "review_clean"]].head(5)
    )

print("\n==========================================")



#Diagnostic check before games are picked


print("\n========== BEFORE GAME SELECTION ==========")

print("Unique games remaining:", df["app_name"].nunique())

print("\nGame distribution:")
print(df["app_name"].value_counts().head(30))

print("\n===========================================")



#Check available games

print("\nStep 15: Checking available games...")

game_counts = df["app_name"].value_counts()

print("Number of different games available:", len(game_counts))

eligible_games = game_counts[
    game_counts >= MIN_REVIEWS_PER_GAME
].index.tolist()

print("Verified games with at least 100 reviews:", len(eligible_games))

if len(eligible_games) < NUMBER_OF_GAMES:

    raise ValueError(
        "There are fewer than 20 verified games with at least 100 reviews."
    )



#Randomly select 20 games


print("\nStep 16: Selecting 20 different games...")

game_list = pd.Series(eligible_games).sample(
    n=NUMBER_OF_GAMES,
    random_state=42
).tolist()

print("\nSelected games:")

for game in game_list:
    print(game)

df = df[
    df["app_name"].isin(game_list)
].copy()

unique_games = df["app_name"].nunique()

if unique_games != NUMBER_OF_GAMES:

    raise ValueError(
        f"Expected 20 games but found {unique_games}."
    )

print("\nSuccessfully verified all 20 games!")



#Cap reviews for each game


print("\nStep 17: Limiting reviews per game...")

before_cap = len(df)

print("Maximum reviews allowed per game:", MAX_REVIEWS_PER_GAME)

print("\nReview counts BEFORE cap:")
print(df["app_name"].value_counts())


#Sample each game's reviews separately

sampled_groups = []

for game_name, group in df.groupby("app_name", sort=False):

    sampled_group = group.sample(
        n=min(len(group), MAX_REVIEWS_PER_GAME),
        random_state=42
    )

    sampled_groups.append(sampled_group)


#combine all sampled groups into one DataFrame

df = pd.concat(
    sampled_groups,
    ignore_index=True
)


reviews_removed_by_cap = before_cap - len(df)

print("\nReviews removed by per-game cap:", reviews_removed_by_cap)

print("\nReview counts AFTER cap:")
print(df["app_name"].value_counts())

print("\nTotal reviews after cap:", len(df))

final_game_counts = df["app_name"].value_counts()

if (final_game_counts > MAX_REVIEWS_PER_GAME).any():

    raise ValueError(
        "REVIEW CAP FAILED: A game has more than 2,000 reviews."
    )

if df["app_name"].nunique() != NUMBER_OF_GAMES:

    raise ValueError(
        "Expected 20 games after applying the review cap."
    )

print("\nSuccessfully verified the 2,000-review limit!")



#take sample of reviews (1000)

print("\nStep 18: Taking sample of reviews...")

sample_reviews = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
)

print("\nFirst 10 sample reviews:")

print(
    sample_reviews[
        ["app_name", "review_raw", "review_clean"]
    ].head(10)
)



#save files


print("\nStep 19: Saving files...")

output_folder = os.path.join(
    script_folder,
    "cleaned_files"
)

os.makedirs(output_folder, exist_ok=True)

print("\nSaving files to:", output_folder)

cleaned_path = os.path.join(
    output_folder,
    "cleaned_steam_reviews.csv"
)

sample_path = os.path.join(
    output_folder,
    "steam_review_sample.csv"
)

games_path = os.path.join(
    output_folder,
    "steam_game_list.csv"
)

df.to_csv(
    cleaned_path,
    index=False
)

print("Cleaned reviews saved!")

sample_reviews.to_csv(
    sample_path,
    index=False
)

print("Review sample saved!")

pd.DataFrame(
    game_list,
    columns=["game_name"]
).to_csv(
    games_path,
    index=False
)

print("Game list saved!")



# final csv check


print("\n========== FINAL CSV CHECK ==========")

check_df = pd.read_csv(cleaned_path)

print(
    "Unique games in final CSV:",
    check_df["app_name"].nunique()
)

print("\nGames in final CSV:")
print(check_df["app_name"].value_counts())

print("\nReview columns in final CSV:")
print(
    check_df[
        ["app_name", "review_raw", "review_clean"]
    ].head(5)
)

if check_df["app_name"].nunique() != NUMBER_OF_GAMES:

    raise ValueError(
        "FINAL CHECK FAILED: The saved CSV does not contain 20 games."
    )

if "review_raw" not in check_df.columns:
    raise ValueError("Missing review_raw column.")

if "review_clean" not in check_df.columns:
    raise ValueError("Missing review_clean column.")

if (
    check_df["app_name"].value_counts()
    > MAX_REVIEWS_PER_GAME
).any():

    raise ValueError(
        "FINAL CHECK FAILED: A game exceeds 2,000 reviews."
    )

if not (check_df["steam_app_type"] == "game").all():

    raise ValueError(
        "FINAL CHECK FAILED: A non-game app was found."
    )

unwanted_columns = [
    column for column in check_df.columns
    if column == "Unnamed: 0" or column.startswith("author.")
]

if unwanted_columns:

    raise ValueError(
        f"Unwanted columns found in final CSV: {unwanted_columns}"
    )

print("\nUnnecessary columns successfully removed!")
print("Both review columns successfully verified!")
print("All selected apps verified as Steam games!")
print("Per-game review cap successfully verified!")

print("\n=====================================")


print("\n========================================")
print("STEAM REVIEW CLEANING COMPLETED!")
print("========================================")

print("Original reviews:", original_review_count)

print("\nFiltering summary:")
print("Non-game reviews removed:", apps_removed)
print("Exact duplicates removed:", duplicates_removed)
print("Short reviews removed:", short_removed)
print("Repeated-character spam removed:", spam_removed)
print("Non-English/uncertain reviews removed:", language_removed)
print("Empty cleaned reviews removed:", empty_removed)
print("Reviews removed by per-game cap:", reviews_removed_by_cap)

print("\nFinal dataset information:")
print("Total cleaned reviews:", len(df))
print("Total sample reviews:", len(sample_reviews))
print("Total games selected:", len(game_list))
print("Maximum reviews per game:", MAX_REVIEWS_PER_GAME)

print("\nLanguage detection:")
print("Fast-LangDetect (FastText-based language identification)")
print("Language model: lite")
print("English confidence threshold:", ENGLISH_CONFIDENCE)
print("Processing batch size:", LANGUAGE_BATCH_SIZE)
print("Maximum input length:", LANGUAGE_MAX_INPUT_LENGTH)

print("\nAll three CSV files have been saved!")
print("Output folder:", output_folder)

print("Steam app type cache:", STEAM_CACHE_PATH)

print("========================================")
