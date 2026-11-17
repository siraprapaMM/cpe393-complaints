"""Step 8 (Role F, with E): per-category summaries and insights.

Usage (run from the repo root):
    python src/08_summaries.py

Input : outputs/topics/sentence_assignments.csv   one row per sentence, with `category` (step 06)
        outputs/topics/categories.csv              distinctive terms and example quotes (step 06)
        outputs/evaluation/labels_merged.csv       hand labels vs model (step 07), optional
Output (outputs/summaries/):
    category_summary.csv   one row per complaint category: sentence and review shares,
                           1-star vs 2-star comparison, hand-label precision, terms, 3 quotes
    star_comparison.csv    1-star vs 2-star, per category, at sentence and review level
    summary.md             the same numbers as tables + recommendations, ready for the report

Two levels are reported because they answer different questions:
    sentence level - "what share of complaint sentences is about X?"
    review level   - "what share of unhappy customers (1-2 star reviews) mention X at least once?"
"""
from pathlib import Path

import pandas as pd

TOPIC_DIR = Path("outputs/topics")
EVAL_FILE = Path("outputs/evaluation/labels_merged.csv")
OUT_DIR = Path("outputs/summaries")
NOT_COMPLAINT = "Not a complaint"

# Hand-label precision below this is flagged: the category's numbers should not be trusted alone
LOW_PRECISION = 0.5

# One action per category for the restaurant owner (report section 7)
RECOMMENDATIONS = {
    "Food taste or quality": "Largest cause. Check recipe consistency and ingredient freshness; "
                             "track which dishes are named most often in these sentences.",
    "Price or value": "Review portion size against price and state extra charges (service, VAT) "
                      "up front; customers complain about value, not price alone.",
    "Staff behavior": "Train staff on attentiveness and tone; these complaints are short and "
                      "strongly worded, and are more common in 1-star reviews.",
    "Slow service or long wait": "Tell customers the expected waiting time and serve in order of "
                                 "arrival; waiting with no information is what is criticised.",
    "Atmosphere or facilities": "Check air-conditioning, seating and parking at peak hours. "
                                "Low hand-label precision: read the sentences before acting.",
    "Order or billing error": "Double-check the bill before presenting it. Low hand-label "
                              "precision: most of these sentences only mention paying.",
    "Cleanliness": "Rare but reliable signal (insects, dirty tables or toilets); treat each "
                   "mention as urgent because it is also a food-safety issue.",
    "Other": "General dissatisfaction with no stated cause ('will not come back'). Follow up "
             "with these customers to learn the reason; the text alone does not say.",
}


def pct(x) -> str:
    return "-" if pd.isna(x) else f"{x:.1%}"


def main():
    a = pd.read_csv(TOPIC_DIR / "sentence_assignments.csv")
    cats = pd.read_csv(TOPIC_DIR / "categories.csv").set_index("category")
    comp = a[a["category"] != NOT_COMPLAINT]  # sentences the model scores as complaints

    n_reviews = a["review_id"].nunique()
    reviews_by_star = a.groupby("stars")["review_id"].nunique()
    sent_by_star = comp["stars"].value_counts()

    # Hand-label precision per model category (step 07): of the sampled sentences the model put
    # in this category, the share where the human chose the same one.
    prec = pd.Series(dtype=float)
    n_lab = pd.Series(dtype=float)
    if EVAL_FILE.exists():
        ev = pd.read_csv(EVAL_FILE)
        g = (ev["human_label"] == ev["model_category"]).groupby(ev["model_category"])
        prec, n_lab = g.mean(), g.size()
    else:
        print(f"{EVAL_FILE} not found: precision columns left empty (run step 07 first)")

    rows, stars = [], []
    for c, n in comp["category"].value_counts().items():
        sub = comp[comp["category"] == c]
        rev_ids = sub["review_id"].nunique()
        row = {
            "category": c,
            "n_sentences": int(n),
            "share_of_complaint_sentences": n / len(comp),
            "n_reviews": int(rev_ids),
            "share_of_complaint_reviews": rev_ids / n_reviews,
        }
        for s in (1, 2):
            ss = sub[sub["stars"] == s]
            row[f"sent_share_{s}star"] = len(ss) / sent_by_star[s]
            row[f"review_share_{s}star"] = ss["review_id"].nunique() / reviews_by_star[s]
        # > 1 means the cause is more typical of 1-star (angrier) reviews
        row["ratio_1star_vs_2star"] = row["review_share_1star"] / row["review_share_2star"]
        row["handlabel_precision"] = prec.get(c)
        row["handlabel_n"] = n_lab.get(c)
        row["reliable"] = None if pd.isna(prec.get(c)) else bool(prec.get(c) >= LOW_PRECISION)
        row["top_terms"] = cats.at[c, "top_terms"] if c in cats.index else ""
        row["examples"] = cats.at[c, "examples"] if c in cats.index else ""
        row["recommendation"] = RECOMMENDATIONS.get(c, "")
        rows.append(row)
        stars.append({k: row[k] for k in ("category", "sent_share_1star", "sent_share_2star",
                                          "review_share_1star", "review_share_2star",
                                          "ratio_1star_vs_2star")})
    summ = pd.DataFrame(rows)
    star_cmp = pd.DataFrame(stars).sort_values("ratio_1star_vs_2star", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(OUT_DIR / "category_summary.csv", index=False, encoding="utf-8-sig")
    star_cmp.round(4).to_csv(OUT_DIR / "star_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- summary.md ----
    with_complaint = comp["review_id"].nunique()
    L = ["# Step 8 - Category summaries and insights", "",
         "Generated by `src/08_summaries.py`. Tables: `category_summary.csv`, `star_comparison.csv`.",
         "", "## Data", "",
         f"- {len(a):,} sentences from {n_reviews:,} complaint (1-2 star) reviews "
         f"({reviews_by_star[1]:,} one-star, {reviews_by_star[2]:,} two-star).",
         f"- {len(comp):,} sentences ({len(comp) / len(a):.1%}) are scored as complaints; the rest "
         f"are \"{NOT_COMPLAINT}\".",
         f"- {with_complaint:,} reviews ({with_complaint / n_reviews:.1%}) have at least one "
         f"complaint sentence; {n_reviews - with_complaint:,} low-star reviews have none.",
         "", "## How often each cause occurs", "",
         "Sentence share = share of complaint sentences. Review share = share of all 1-2 star "
         "reviews that mention the cause at least once (a review can mention several, so the "
         "column does not sum to 100%). Precision = hand-label check from step 7.", "",
         "| category | sentences | sentence share | reviews | review share | hand-label precision |",
         "|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        p = "-" if pd.isna(r.handlabel_precision) else \
            f"{r.handlabel_precision:.0%} (n={int(r.handlabel_n)})" + ("" if r.reliable else " ⚠")
        L.append(f"| {r.category} | {r.n_sentences:,} | {pct(r.share_of_complaint_sentences)} | "
                 f"{r.n_reviews:,} | {pct(r.share_of_complaint_reviews)} | {p} |")
    L += ["", f"⚠ = precision below {LOW_PRECISION:.0%}: most sampled sentences in this category "
              "were labeled differently by a human, so its counts overstate the real cause.",
          "", "## 1-star vs 2-star", "",
          "Share of reviews at each star level that mention the cause. Ratio > 1 means the cause "
          "is more typical of 1-star reviews.", "",
          "| category | 1-star reviews | 2-star reviews | ratio |", "|---|---|---|---|"]
    for r in star_cmp.itertuples():
        L.append(f"| {r.category} | {pct(r.review_share_1star)} | {pct(r.review_share_2star)} | "
                 f"{r.ratio_1star_vs_2star:.2f} |")
    L += ["", "## Per category", ""]
    for r in summ.itertuples():
        L += [f"### {r.category}",
              f"- Share: {pct(r.share_of_complaint_sentences)} of complaint sentences, "
              f"{pct(r.share_of_complaint_reviews)} of low-star reviews.",
              f"- Top keywords: {r.top_terms}",
              "- Representative sentences:"]
        L += [f"  - {x.strip()}" for x in str(r.examples).split(" || ") if x.strip() and x != "nan"]
        L += [f"- 1-star vs 2-star: {pct(r.review_share_1star)} vs {pct(r.review_share_2star)} "
              f"of reviews (ratio {r.ratio_1star_vs_2star:.2f}).",
              f"- Recommendation: {r.recommendation}", ""]
    (OUT_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")

    show = summ[["category", "n_sentences", "share_of_complaint_sentences", "n_reviews",
                 "share_of_complaint_reviews", "ratio_1star_vs_2star", "handlabel_precision"]]
    print(show.round(3).to_string(index=False))
    print(f"\nWrote {OUT_DIR}/category_summary.csv, star_comparison.csv, summary.md")


if __name__ == "__main__":
    main()
