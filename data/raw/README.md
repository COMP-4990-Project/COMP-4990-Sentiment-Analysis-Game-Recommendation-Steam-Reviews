# data/raw

Put the raw dataset file here:

```
data/raw/steam_reviews.csv
```

The file is **not stored in GitHub**. It is about 8.17 GB, and GitHub LFS rejects any single file over 2 GB. Each team member downloads it once.

## Where to get it

**Steam Reviews Dataset 2021** on Kaggle (by Marko M.)

- Kaggle link: https://www.kaggle.com/datasets/najzeko/steam-reviews-2021/data
- On Kaggle you can also search for "Steam Reviews Dataset 2021".
- File to use: `steam_reviews.csv`
- Size: about 8.17 GB, 23 columns, 21,747,371 reviews from 315 games
- Collected from Steam's reviews API in 2021
- A free Kaggle account is needed to download it. The license is listed as GPL 2 on the sources we found, so check the Kaggle page for the current terms.

## How to use it

1. Download the dataset from Kaggle (the download button on the dataset page) and unzip it.
2. Move `steam_reviews.csv` into this folder (`data/raw/`), or keep it anywhere you like and pass its path:
   ```
   python3 scripts/main.py /path/to/steam_reviews.csv
   ```
3. Run `python3 scripts/main.py`. The first lines of the output should say `Total original reviews: 21747371`. If that number differs, you have a different version of the file.

## Notes

- `main.py` loads the whole file into memory, so close other programs before running it.
- Git ignores everything in this folder except this README, so the 8 GB file can't be committed by accident.
- You only need this file to re-create the cleaned data. The cleaned and scored CSVs in `data/cleaned/` and `data/scored/` are enough to run the app and the VADER step.
