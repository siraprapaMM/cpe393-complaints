"""Step 0 (Role A): load Wongnai reviews, build complaint labels, make the split.

Usage (run from the repo root):
    python src/01_prepare_data.py --input data/raw/w_review_train.csv

Input:  Wongnai training file (semicolon-separated, no header: review;star).
        This is the only Wongnai file that has star ratings.
Output (data/splits/):
    train.csv, val.csv, test.csv   1-2 and 4-5 star reviews, 80/10/10 stratified
    three_star.csv                 3-star reviews, kept out of training (for Role D)
Columns in every file: review_id, text, stars, label
    label = 1 complaint (1-2 stars), 0 not a complaint (4-5 stars), empty for 3 stars
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # the one seed the whole team uses
OUT_DIR = Path("data/splits")


def load_wongnai(path: Path) -> pd.DataFrame:
    """Read the raw file the way the Wongnai corpus README does (sep=';', no header).
    If that fails, fall back to splitting each line on its LAST ';'."""
    try:
        df = pd.read_csv(path, sep=";", header=None, names=["text", "stars"],
                         encoding="utf-8", dtype=str, keep_default_na=False)
    except pd.errors.ParserError:
        print("pandas could not parse the file; splitting each line on its last ';'")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ";" in line:
                    text, star = line.rstrip("\n").rsplit(";", 1)
                    rows.append((text, star))
        df = pd.DataFrame(rows, columns=["text", "stars"])
    df["text"] = df["text"].astype(str).str.strip()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    bad = df["stars"].isna() | ~df["stars"].isin([1, 2, 3, 4, 5])
    if bad.any():
        print(f"Dropped {bad.sum()} rows without a 1-5 star value (e.g. a header line)")
    df = df[~bad].copy()
    df["stars"] = df["stars"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/w_review_train.csv")
    args = ap.parse_args()

    df = load_wongnai(Path(args.input))
    df.insert(0, "review_id", range(len(df)))  # stable id = row order in the raw file
    n_raw = len(df)

    df = df[df["text"].str.strip() != ""]
    n_before = len(df)
    df = df.drop_duplicates(subset="text")  # same text in train and test would leak
    print(f"Raw rows: {n_raw:,} | empty removed: {n_raw - n_before:,} | "
          f"duplicate texts removed: {n_before - len(df):,}")

    three = df[df["stars"] == 3].copy()
    three["label"] = pd.NA
    labeled = df[df["stars"] != 3].copy()
    labeled["label"] = (labeled["stars"] <= 2).astype(int)

    train, rest = train_test_split(labeled, test_size=0.2, stratify=labeled["label"],
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"],
                                 random_state=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = {"train": train, "val": val, "test": test, "three_star": three}
    for name, part in parts.items():
        part.sort_values("review_id").to_csv(OUT_DIR / f"{name}.csv", index=False,
                                             encoding="utf-8")

    # Summary to paste into the team chat
    print("\nStar distribution:", df["stars"].value_counts().sort_index().to_dict())
    print(f"\nFiles in {OUT_DIR}/  (columns: review_id, text, stars, label; seed={SEED})")
    print(f"{'file':<16}{'rows':>8}{'complaints':>12}{'share':>8}")
    for name, part in parts.items():
        if name == "three_star":
            print(f"{name + '.csv':<16}{len(part):>8,}{'-':>12}{'-':>8}")
        else:
            c = int(part["label"].sum())
            print(f"{name + '.csv':<16}{len(part):>8,}{c:>12,}{c / len(part):>8.1%}")


if __name__ == "__main__":
    main()
