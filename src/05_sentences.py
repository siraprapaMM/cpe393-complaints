"""Step 4: split 1-2 star (complaint) reviews into sentences.  Owner: Amm
Reads  data/processed/{train,val,test}_clean.csv   (rows with label == 1)
Writes data/processed/complaint_sentences.csv
       columns: sentence_id, review_id, sent_idx, split, stars,
                sentence_raw, sentence, tokens
         sentence_raw = original sentence (for the hand-labeling sheet)
         sentence     = cleaned text (for BERTopic)
         tokens       = space-separated tokens (for LDA)
Positive sentences inside low-star reviews are KEPT: step 6 labels them
"Not a complaint". Only sentences with too few tokens are dropped.
Run:   python src/05_sentences.py
"""
import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd
from pythainlp.tokenize import sent_tokenize

# reuse clean_text / tokenize from 02_clean.py (file name starts with a digit,
# so it can't be imported normally)
_spec = importlib.util.spec_from_file_location(
    "clean02", Path(__file__).with_name("02_clean.py"))
_clean02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_clean02)
clean_text, tokenize = _clean02.clean_text, _clean02.tokenize

# split on newlines, after ! or ?, and on 2+ spaces BEFORE cleaning
# (cleaning removes the very punctuation/newlines that mark sentence ends)
COARSE = re.compile(r"[\r\n]+|(?<=[!?])|\s{2,}")


def split_sentences(raw: str):
    for chunk in COARSE.split(raw or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        for s in sent_tokenize(chunk, engine="crfcut"):
            yield s.strip()


def main(proc: Path, min_tokens: int):
    parts = []
    for name in ["train", "val", "test"]:
        f = proc / f"{name}_clean.csv"
        if f.exists():
            d = pd.read_csv(f)
            parts.append(d[d["label"] == 1].assign(split=name))
        else:
            print(f"[skip] {f} not found")
    reviews = pd.concat(parts, ignore_index=True)

    rows = []
    for r in reviews.itertuples():
        raw = r.text if isinstance(r.text, str) else ""
        for i, s in enumerate(split_sentences(raw)):
            clean = clean_text(s)
            toks = tokenize(clean)
            if len(toks) < min_tokens:
                continue
            rows.append((r.review_id, i, r.split, r.stars,
                         re.sub(r"\s+", " ", s), clean, " ".join(toks)))

    out = pd.DataFrame(rows, columns=["review_id", "sent_idx", "split", "stars",
                                      "sentence_raw", "sentence", "tokens"])
    out.insert(0, "sentence_id", range(len(out)))
    dst = proc / "complaint_sentences.csv"
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"[ok] {len(reviews)} complaint reviews -> {len(out)} sentences -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--min-tokens", type=int, default=2)
    a = ap.parse_args()
    main(Path(a.proc_dir), a.min_tokens)
