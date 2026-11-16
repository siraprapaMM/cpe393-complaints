"""Step 3: complaint classifier (TF-IDF word + char n-grams, LR vs Linear SVM).  Owner: Van
Reads  data/processed/{train,val,test,three_star}_clean.csv
Writes outputs/classifier/
         val_tuning.csv                  every config tried on val (default threshold)
         threshold_tuning.csv            chosen threshold per model (tuned on val)
         model_comparison.csv            LR vs SVM on val and test
         classification_report_{lr,svm}.csv
         confusion_matrix_{lr,svm}.csv / .png   (test)
         top_features_{lr,svm}.csv       most predictive word / char n-grams per class
         three_star_predictions.csv      review_id + score + pred, scored by the selected model
         three_star_summary.csv
         metrics_summary.md              numbers ready for the report
Run:   python src/04_classifier.py

Protocol
- Features and models are fit on train only.
- n-gram ranges, C and the decision threshold are chosen on val only (macro-F1).
- The better model is chosen on val macro-F1, not on test.
- Test is scored once per model, with the settings fixed on val.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.svm import LinearSVC

SEED = 42
NAMES = ["train", "val", "test", "three_star"]

WORD_NGRAMS = [(1, 1), (1, 2), (1, 3)]      # on `tokens` (already tokenized by 02_clean)
CHAR_NGRAMS = [(2, 4), (3, 5)]              # on `text_clean`, analyzer="char_wb"
C_GRID = {
    "lr": [0.1, 0.3, 1, 3, 10, 30, 100],
    "svm": [0.01, 0.03, 0.1, 0.3, 1],
}
MODEL_NAMES = {"lr": "Logistic Regression", "svm": "Linear SVM"}
TOP_K = 25


# ---------------------------------------------------------------- data
def load(in_dir: Path) -> dict:
    data = {}
    for name in NAMES:
        df = pd.read_csv(in_dir / f"{name}_clean.csv")
        # n_tokens == 0 -> tokens is NaN after read_csv; keep the row, use "".
        df["tokens"] = df["tokens"].fillna("").astype(str)
        df["text_clean"] = df["text_clean"].fillna("").astype(str)
        data[name] = df
        print(f"[load] {name}: {len(df)} rows, empty tokens: {(df['tokens'] == '').sum()}")
    return data


def word_vectorizer(ngram):
    # tokens are space-separated -> split on spaces only, no re-tokenization
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           preprocessor=None, ngram_range=ngram, min_df=2,
                           sublinear_tf=True, dtype=np.float32)


def char_vectorizer(ngram):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=3,
                           lowercase=False, sublinear_tf=True, dtype=np.float32)


def make_model(kind, C):
    if kind == "lr":
        return LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                  max_iter=2000, random_state=SEED)
    return LinearSVC(C=C, class_weight="balanced", max_iter=10000, random_state=SEED)


# ---------------------------------------------------------------- metrics
def metrics(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1,
                                                    zero_division=0)
    return {"macro_f1": f, "macro_precision": p, "macro_recall": r,
            "complaint_f1": fc, "complaint_precision": pc, "complaint_recall": rc,
            "accuracy": accuracy_score(y, pred)}


def best_threshold(y, scores, default):
    """Threshold on the decision score that maximizes macro-F1 (val only)."""
    cands = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, 400)))
    cands = np.append(cands, default)
    f1s = [f1_score(y, (scores >= t).astype(int), average="macro") for t in cands]
    i = int(np.argmax(f1s))
    return float(cands[i]), float(f1s[i])


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Not complaint (0)", "Complaint (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    row = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / row[i, 0]:.1%})", ha="center",
                    va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(in_dir: Path, out_dir: Path):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(in_dir)
    y_tr = data["train"]["label"].astype(int).values
    y_va = data["val"]["label"].astype(int).values
    y_te = data["test"]["label"].astype(int).values

    # Fit each vectorizer once on train, reuse across the grid.
    word_X, char_X = {}, {}
    for ng in WORD_NGRAMS:
        v = word_vectorizer(ng)
        word_X[ng] = (v, v.fit_transform(data["train"]["tokens"]),
                      v.transform(data["val"]["tokens"]))
        print(f"[feat] word {ng}: {len(v.vocabulary_):,} features")
    for ng in CHAR_NGRAMS:
        v = char_vectorizer(ng)
        char_X[ng] = (v, v.fit_transform(data["train"]["text_clean"]),
                      v.transform(data["val"]["text_clean"]))
        print(f"[feat] char {ng}: {len(v.vocabulary_):,} features")

    # ---- grid search on val (default threshold) ----
    rows = []
    best = {}            # kind -> dict(config, model, val_macro_f1)
    for wng in WORD_NGRAMS:
        for cng in CHAR_NGRAMS:
            X_tr = hstack([word_X[wng][1], char_X[cng][1]]).tocsr()
            X_va = hstack([word_X[wng][2], char_X[cng][2]]).tocsr()
            for kind, cs in C_GRID.items():
                for C in cs:
                    m = make_model(kind, C).fit(X_tr, y_tr)
                    res = metrics(y_va, m.predict(X_va))
                    rows.append({"model": kind, "word_ngram": str(wng), "char_ngram": str(cng),
                                 "C": C, "n_features": X_tr.shape[1], **res})
                    print(f"[val] {kind:3s} word{wng} char{cng} C={C:<5} "
                          f"macro-F1={res['macro_f1']:.4f}")
                    if kind not in best or res["macro_f1"] > best[kind]["val_macro_f1"]:
                        best[kind] = {"word": wng, "char": cng, "C": C, "model": m,
                                      "val_macro_f1": res["macro_f1"]}
    tuning = pd.DataFrame(rows).sort_values(["model", "macro_f1"], ascending=[True, False])
    tuning.round(4).to_csv(out_dir / "val_tuning.csv", index=False)

    # ---- per model: threshold on val, then test once ----
    comp_rows, thr_rows, summary = [], [], {}
    for kind, b in best.items():
        wv, cv = word_X[b["word"]][0], char_X[b["char"]][0]

        def feats(df, wv=wv, cv=cv):
            return hstack([wv.transform(df["tokens"]), cv.transform(df["text_clean"])]).tocsr()

        model = b["model"]                                 # already fit on train only
        X_va, X_te = feats(data["val"]), feats(data["test"])
        s_va = model.decision_function(X_va)
        thr, thr_f1 = best_threshold(y_va, s_va, default=0.0)
        thr_rows.append({"model": kind, "word_ngram": str(b["word"]),
                         "char_ngram": str(b["char"]), "C": b["C"],
                         "val_macro_f1_default_thr": b["val_macro_f1"],
                         "threshold": thr, "val_macro_f1_tuned_thr": thr_f1})

        s_te = model.decision_function(X_te)
        for split, y, s in [("val", y_va, s_va), ("test", y_te, s_te)]:
            for thr_name, t in [("default (0)", 0.0), ("tuned on val", thr)]:
                comp_rows.append({"model": MODEL_NAMES[kind], "split": split,
                                  "threshold": thr_name, "threshold_value": t,
                                  **metrics(y, (s >= t).astype(int))})

        pred_te = (s_te >= thr).astype(int)
        rep = classification_report(y_te, pred_te, target_names=["not_complaint", "complaint"],
                                    output_dict=True, digits=4)
        pd.DataFrame(rep).T.round(4).to_csv(out_dir / f"classification_report_{kind}.csv")
        cm = confusion_matrix(y_te, pred_te, labels=[0, 1])
        pd.DataFrame(cm, index=["true_0_not", "true_1_complaint"],
                     columns=["pred_0_not", "pred_1_complaint"]).to_csv(
            out_dir / f"confusion_matrix_{kind}.csv")
        plot_cm(cm, f"{MODEL_NAMES[kind]} — test (thr={thr:.3f})",
                out_dir / f"confusion_matrix_{kind}.png")

        # top features per class (word and char separately)
        coef = model.coef_.ravel()
        n_w = len(wv.vocabulary_)
        names = np.concatenate([wv.get_feature_names_out(), cv.get_feature_names_out()])
        ftype = np.array(["word"] * n_w + ["char"] * (len(names) - n_w))
        feat_rows = []
        for t in ["word", "char"]:
            idx = np.where(ftype == t)[0]
            order = idx[np.argsort(coef[idx])]
            for cls, sel in [("complaint", order[::-1][:TOP_K]), ("not_complaint", order[:TOP_K])]:
                for rank, i in enumerate(sel, 1):
                    feat_rows.append({"feature_type": t, "class": cls, "rank": rank,
                                      "feature": names[i], "coef": round(float(coef[i]), 4)})
        pd.DataFrame(feat_rows).to_csv(out_dir / f"top_features_{kind}.csv", index=False,
                                       encoding="utf-8-sig")
        summary[kind] = {"thr": thr, "feats": feats, "model": model, "cm": cm,
                         "feat_rows": feat_rows}

    comp = pd.DataFrame(comp_rows)
    comp.round(4).to_csv(out_dir / "model_comparison.csv", index=False)
    thr_df = pd.DataFrame(thr_rows)
    thr_df.round(4).to_csv(out_dir / "threshold_tuning.csv", index=False)

    # ---- choose the better model on VAL macro-F1 (tuned threshold) ----
    sel = thr_df.sort_values("val_macro_f1_tuned_thr", ascending=False).iloc[0]["model"]
    s = summary[sel]
    print(f"[select] {MODEL_NAMES[sel]} (by val macro-F1)")

    # ---- score 3-star reviews ----
    ts = data["three_star"]
    score = s["model"].decision_function(s["feats"](ts))
    out = pd.DataFrame({"review_id": ts["review_id"], "stars": ts["stars"],
                        "score": np.round(score, 4),
                        "pred_complaint": (score >= s["thr"]).astype(int)})
    if sel == "lr":
        out["prob_complaint"] = np.round(1 / (1 + np.exp(-score)), 4)
    out["model"] = MODEL_NAMES[sel]
    out["threshold"] = round(s["thr"], 4)
    out.to_csv(out_dir / "three_star_predictions.csv", index=False)
    n_c, n = int(out["pred_complaint"].sum()), len(out)
    ts_sum = pd.DataFrame([{"model": MODEL_NAMES[sel], "threshold": round(s["thr"], 4),
                            "n_three_star": n, "n_pred_complaint": n_c,
                            "pct_pred_complaint": round(100 * n_c / n, 2),
                            "n_pred_not_complaint": n - n_c}])
    ts_sum.to_csv(out_dir / "three_star_summary.csv", index=False)
    print(f"[3star] {n_c:,} / {n:,} ({100 * n_c / n:.1f}%) predicted complaint")

    write_md(out_dir, comp, thr_df, sel, summary, ts_sum.iloc[0])
    print(f"[done] {time.time() - t0:.0f}s -> {out_dir}")


def write_md(out_dir, comp, thr_df, sel, summary, ts):
    f = lambda x: f"{x:.4f}"
    L = ["# Step 3 — Complaint classifier: results", "",
         "Features: TF-IDF word n-grams on `tokens` + char_wb n-grams on `text_clean` "
         "(sublinear tf). Models use `class_weight=\"balanced\"`, `random_state=42`.",
         "n-grams, C and threshold tuned on **val** (macro-F1); fit on **train** only; "
         "**test** scored once.", "", "## Selected settings (val)", "",
         "| model | word n-gram | char n-gram | C | val macro-F1 (thr 0) | threshold | val macro-F1 (tuned thr) |",
         "|---|---|---|---|---|---|---|"]
    for _, r in thr_df.iterrows():
        L.append(f"| {MODEL_NAMES[r['model']]} | {r['word_ngram']} | {r['char_ngram']} | {r['C']} | "
                 f"{f(r['val_macro_f1_default_thr'])} | {r['threshold']:.4f} | {f(r['val_macro_f1_tuned_thr'])} |")
    L += ["", "## Test results (threshold tuned on val)", "",
          "| model | macro-F1 | macro-P | macro-R | complaint P | complaint R | complaint F1 | accuracy |",
          "|---|---|---|---|---|---|---|---|"]
    t = comp[(comp["split"] == "test") & (comp["threshold"] == "tuned on val")]
    for _, r in t.iterrows():
        L.append(f"| {r['model']} | **{f(r['macro_f1'])}** | {f(r['macro_precision'])} | "
                 f"{f(r['macro_recall'])} | {f(r['complaint_precision'])} | "
                 f"{f(r['complaint_recall'])} | {f(r['complaint_f1'])} | {f(r['accuracy'])} |")
    L += ["", "Test at the default threshold (0) is in `model_comparison.csv`.", "",
          "## Confusion matrices (test)", ""]
    for kind, s in summary.items():
        cm = s["cm"]
        L += [f"**{MODEL_NAMES[kind]}** — TN {cm[0,0]:,} · FP {cm[0,1]:,} · "
              f"FN {cm[1,0]:,} · TP {cm[1,1]:,}", ""]
    L += [f"## Selected model: {MODEL_NAMES[sel]} (higher val macro-F1)", "",
          "Top word features:", "", "| rank | complaint | not complaint |", "|---|---|---|"]
    fr = pd.DataFrame(summary[sel]["feat_rows"])
    w = fr[fr["feature_type"] == "word"]
    c1 = w[w["class"] == "complaint"]["feature"].tolist()
    c0 = w[w["class"] == "not_complaint"]["feature"].tolist()
    for i in range(min(15, len(c1), len(c0))):
        L.append(f"| {i + 1} | {c1[i]} | {c0[i]} |")
    L += ["", "## 3-star reviews", "",
          f"{ts['n_pred_complaint']:,} of {ts['n_three_star']:,} three-star reviews "
          f"(**{ts['pct_pred_complaint']}%**) are predicted as complaints "
          f"({ts['model']}, threshold {ts['threshold']}). "
          "Per-review results with `review_id`: `three_star_predictions.csv`.", ""]
    (out_dir / "metrics_summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs/classifier")
    a = ap.parse_args()
    main(Path(a.in_dir), Path(a.out_dir))
