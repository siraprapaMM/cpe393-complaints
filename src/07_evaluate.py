"""Step 7 (Role F): hand-label evaluation of the root-cause grouping.

Run from the repo root.

1) sample - draw 120 sentences (20 per member) and build the labeling sheet
    python src/07_evaluate.py sample [--strategy stratified|random] [--pool test]

   Input : outputs/topics/sentence_assignments.csv   (Role E, step 06)
   Output: outputs/evaluation/labeling_sheet.xlsx    one tab per member + "All"; NO model answer
           outputs/evaluation/labeling_key.csv       model category + weight per sample (keep private!)

   stratified (default): about equal number of sentences per model category, so rare categories
   (Cleanliness is 0.4% of sentences) still get checked. A pure random sample would be ~75% "Not a
   complaint". Each row carries a weight = (category size in pool) / (number sampled), so score can
   also report a population-weighted accuracy that corrects for the over-sampling.
   random: plain random sample (what the guide literally says).

2) score - compare the filled sheet with the model's categories
    python src/07_evaluate.py score

   Output: outputs/evaluation/metrics.json, per_class.csv, confusion_matrix.csv,
           disagreements.csv, labels_merged.csv

   demo runs both modes on made-up data in a temp folder to check the code.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SEED = 42
MEMBERS = ["MM", "Amm", "Pang", "Van", "Gim", "Toffy"]
PER_MEMBER = 20
EXTRA_LABELS = ["Not a complaint", "Other"]  # always available to labelers (guide section 6)
MIN_CHARS = 8  # 4-7 character fragments cannot be labeled sensibly

ASSIGN_FILE = Path("outputs/topics/sentence_assignments.csv")
EVAL_DIR = Path("outputs/evaluation")


def allocate(n: int, caps: dict) -> dict:
    """Split n as evenly as possible over the keys, never above a key's cap."""
    quota = {k: 0 for k in caps}
    left = n
    while left > 0:
        open_ = [k for k in caps if quota[k] < caps[k]]
        if not open_:
            break
        share = max(1, left // len(open_))
        for k in open_:
            add = min(share, caps[k] - quota[k], left)
            quota[k] += add
            left -= add
            if left == 0:
                break
    return quota


def build_sample(a: pd.DataFrame, pool: str, n_total: int, strategy: str) -> pd.DataFrame:
    s = a.dropna(subset=["sentence_raw", "category"]).copy()
    s["sentence_raw"] = s["sentence_raw"].astype(str).str.strip()
    s = s[s["sentence_raw"].str.len() >= MIN_CHARS].drop_duplicates(subset="sentence_raw")
    if pool != "all":
        s = s[s["split"] == pool]
    if len(s) < n_total:
        sys.exit(f"Only {len(s)} usable sentences in pool '{pool}', need {n_total}.")
    pop = s["category"].value_counts()
    if strategy == "random":
        pick = s.sample(n=n_total, random_state=SEED).copy()
        pick["weight"] = 1.0
    else:
        quota = allocate(n_total, pop.to_dict())
        parts = []
        for c, q in quota.items():
            if q:
                g = s[s["category"] == c].sample(n=q, random_state=SEED).copy()
                g["weight"] = pop[c] / q  # how many pool sentences each sampled one stands for
                parts.append(g)
        pick = pd.concat(parts)
        print("Sampled per category:", quota)
    pick = pick.sample(frac=1, random_state=SEED).reset_index(drop=True)  # mix before splitting
    pick.insert(0, "sample_id", [f"{i + 1:03d}" for i in range(len(pick))])
    pick["annotator"] = [MEMBERS[i // PER_MEMBER] for i in range(len(pick))]
    return pick


def mode_sample(args, assign_path: Path, out_dir: Path):
    a = pd.read_csv(assign_path)
    missing = {"sentence_id", "split", "sentence_raw", "category"} - set(a.columns)
    if missing:
        sys.exit(f"{assign_path} is missing columns: {sorted(missing)}")
    s = build_sample(a, args.pool, PER_MEMBER * len(MEMBERS), args.strategy)

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        cats = list(a["category"].value_counts().index)
    cats = [c for c in cats if c not in EXTRA_LABELS] + EXTRA_LABELS

    out_dir.mkdir(parents=True, exist_ok=True)
    s[["sample_id", "annotator", "sentence_id", "review_id", "split", "sentence_raw",
       "category", "weight"]].rename(columns={"category": "model_category"}).to_csv(
        out_dir / "labeling_key.csv", index=False, encoding="utf-8-sig")

    sheet = s[["sample_id", "annotator", "sentence_raw"]].copy()
    sheet.columns = ["ID", "Annotator", "Sentence"]
    sheet["Your label"] = ""

    from openpyxl.worksheet.datavalidation import DataValidation
    xlsx = out_dir / "labeling_sheet.xlsx"
    # Inline list: Google Sheets drops dropdowns that point at another (hidden) sheet.
    # Excel limits an inline list to 255 characters and items cannot contain commas.
    inline = '"' + ",".join(cats) + '"'
    if len(inline) > 255 or any("," in c for c in cats):
        sys.exit("Category list too long for a dropdown (max 255 chars, no commas in names).")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        tabs = [("All", sheet)] + [(m, sheet[sheet["Annotator"] == m]) for m in MEMBERS]
        for name, df in tabs:
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            dv = DataValidation(type="list", formula1=inline, allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"D2:D{len(df) + 1}")
            for col, w in zip("ABCD", (6, 10, 80, 28)):
                ws.column_dimensions[col].width = w
    print(f"Sample of {len(s)} sentences ({PER_MEMBER} per member), pool={args.pool}, "
          f"strategy={args.strategy}")
    print(f"Dropdown categories: {cats}")
    print(f"Sheet: {xlsx}")
    print(f"Key (model answers - do not share): {out_dir / 'labeling_key.csv'}")


def read_filled_sheet(xlsx: Path) -> pd.DataFrame:
    """Labels from the per-member tabs (the 'All' tab is only a convenience copy)."""
    tabs = pd.read_excel(xlsx, sheet_name=None, dtype=str)
    # Any tab except "All" that has the sheet's columns (members may rename their tab)
    parts = [df for name, df in tabs.items()
             if name != "All" and {"ID", "Your label"} <= set(df.columns)] or [tabs["All"]]
    df = pd.concat(parts, ignore_index=True).rename(columns={"ID": "sample_id",
                                                             "Your label": "human_label"})
    df["sample_id"] = df["sample_id"].astype(str).str.zfill(3)
    df["human_label"] = df["human_label"].fillna("").str.strip()
    return df[["sample_id", "human_label"]]


def mode_score(args, out_dir: Path):
    key = pd.read_csv(out_dir / "labeling_key.csv", dtype={"sample_id": str})
    m = key.merge(read_filled_sheet(out_dir / "labeling_sheet.xlsx"), on="sample_id", how="left")
    blank = m["human_label"].fillna("") == ""
    if blank.any():
        print(f"WARNING: {blank.sum()} rows have no human label yet (ignored): "
              f"{sorted(m.loc[blank, 'annotator'].unique())}")
    m = m[~blank].copy()
    if m.empty:
        sys.exit("No labeled rows to score.")
    m.to_csv(out_dir / "labels_merged.csv", index=False, encoding="utf-8-sig")

    y_true, y_pred, w = m["human_label"], m["model_category"], m["weight"]
    classes = sorted(set(y_true) | set(y_pred))
    metrics = {
        "n_scored": int(len(m)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=classes,
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=classes,
                                            zero_division=0)), 4),
        # corrects for stratified over-sampling of rare categories (equals accuracy if random)
        "population_weighted_accuracy": round(
            float(accuracy_score(y_true, y_pred, sample_weight=w)), 4),
        "population_weighted_macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", labels=classes, sample_weight=w,
                           zero_division=0)), 4),
        "per_annotator_accuracy": {
            a: round(float((g["human_label"] == g["model_category"]).mean()), 4)
            for a, g in m.groupby("annotator")},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    rep = classification_report(y_true, y_pred, labels=classes, zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.round(3).to_csv(out_dir / "per_class.csv", encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes),
                 index=[f"human: {c}" for c in classes],
                 columns=[f"model: {c}" for c in classes]).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    bad = m[m["human_label"] != m["model_category"]]
    bad.to_csv(out_dir / "disagreements.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print()
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"Disagreements saved: {len(bad)} rows")


def run_demo():
    import random
    random.seed(SEED)
    cats = ["Slow service or long wait", "Staff behavior", "Food taste or quality",
            "Price or value", "Cleanliness"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = [0.05, 0.05, 0.2, 0.1, 0.02]  # skewed on purpose, like the real data
        n = 3000
        a = pd.DataFrame({
            "sentence_id": range(n), "review_id": [i // 3 for i in range(n)], "split": "test",
            "sentence_raw": [f"ประโยคทดสอบหมายเลข {i}" for i in range(n)],
            "category": random.choices(cats + ["Not a complaint"], w + [1 - sum(w)], k=n)})
        ap, out = tmp / "assign.csv", tmp / "eval"
        a.to_csv(ap, index=False)
        ns = argparse.Namespace(pool="test", strategy="stratified", categories="")
        mode_sample(ns, ap, out)
        key = pd.read_csv(out / "labeling_key.csv", dtype={"sample_id": str})
        fake = {r.sample_id: (r.model_category if random.random() < 0.7
                              else random.choice(cats + EXTRA_LABELS)) for r in key.itertuples()}
        tabs = pd.read_excel(out / "labeling_sheet.xlsx", sheet_name=None, dtype=str)
        with pd.ExcelWriter(out / "labeling_sheet.xlsx") as xw:
            for name, df in tabs.items():
                if "ID" in df.columns:
                    df["Your label"] = df["ID"].map(fake)
                df.to_excel(xw, sheet_name=name, index=False)
        mode_score(ns, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["sample", "score", "demo"])
    ap.add_argument("--pool", default="test", choices=["train", "val", "test", "all"],
                    help="which split's sentences to sample from (default: test)")
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "random"])
    ap.add_argument("--categories", default="",
                    help="optional comma-separated dropdown list (default: Role E's categories)")
    args = ap.parse_args()
    if args.mode == "sample":
        mode_sample(args, ASSIGN_FILE, EVAL_DIR)
    elif args.mode == "score":
        mode_score(args, EVAL_DIR)
    else:
        run_demo()


if __name__ == "__main__":
    main()
