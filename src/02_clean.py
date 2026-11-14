"""Step 1: cleaning + tokenization.  Owner: Amm
Reads  data/splits/{train,val,test,three_star}.csv
Writes data/processed/{name}_clean.csv  (adds: text_clean, tokens, n_tokens)
Run:   python src/02_clean.py
(05_sentences.py reuses clean_text / tokenize from this file)
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    tqdm = lambda x, **kw: x

URL = re.compile(r"https?://\S+|www\.\S+")
HTML = re.compile(r"<[^>]+>")
LAUGH = re.compile(r"5{3,}")                                   # 55555 -> 555
REPEAT = re.compile(r"([\u0E01-\u0E2Ea-z])\1{2,}")            # อร่อยยยย -> อร่อย
NOT_ALLOWED = re.compile(r"[^\u0E00-\u0E7Fa-z0-9\s]")          # emoji, punctuation
SPACES = re.compile(r"\s+")

# Stopwords in pythainlp that still carry complaint meaning -> keep them
KEEP = {
    "ไม่", "ไม่ได้", "ไม่ค่อย", "ไม่มี", "ไม่เคย", "ไม่ใช่",
    "ยัง", "น้อย", "เกิน", "หมด", "ช้า", "นาน", "มาก", "แค่", "เลย",
}
STOPWORDS = set(thai_stopwords()) - KEEP

NAMES = ["train", "val", "test", "three_star"]


def clean_text(text) -> str:
    """Raw review -> cleaned string (still readable, good for BERTopic)."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL.sub(" ", t)
    t = HTML.sub(" ", t)
    t = normalize(t)                 # fix duplicated vowels / tone marks
    t = t.replace("ๆ", "")
    t = LAUGH.sub("555", t)
    t = REPEAT.sub(r"\1", t)
    t = NOT_ALLOWED.sub(" ", t)
    return SPACES.sub(" ", t).strip()


def tokenize(clean: str) -> list:
    """Cleaned string -> token list (stopwords removed, negations kept)."""
    toks = word_tokenize(clean, engine="newmm", keep_whitespace=False)
    return [w for w in toks if len(w) > 1 and w not in STOPWORDS]


def main(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        src = in_dir / f"{name}.csv"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        df = pd.read_csv(src)
        df["text_clean"] = [clean_text(t) for t in tqdm(df["text"], desc=name)]
        toks = [tokenize(t) for t in tqdm(df["text_clean"], desc=f"{name} tokens")]
        df["tokens"] = [" ".join(t) for t in toks]      # space-separated
        df["n_tokens"] = [len(t) for t in toks]
        dst = out_dir / f"{name}_clean.csv"
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"[ok] {name}: {len(df)} rows -> {dst} "
              f"(empty after cleaning: {(df['n_tokens'] == 0).sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/splits")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    main(Path(a.in_dir), Path(a.out_dir))