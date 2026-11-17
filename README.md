# CPE393 — Thai Restaurant Complaint Analysis (Wongnai)

Detect complaints in Wongnai reviews, group them into root causes, report how often each occurs.
Full plan: see the project guide. Deadline: **Wed 18 Nov 2026**.

## Setup
```bash
git clone <repo-url>
cd cpe393-complaints
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then download `data/splits/*.csv` from the shared Drive folder into `data/splits/`.

## Folders
| Folder | What goes there | In git? |
|---|---|---|
| `src/` | Scripts, one per step, numbered | yes |
| `notebooks/` | Exploration notebooks (clear outputs before commit) | yes |
| `data/raw/` | Original Wongnai file | no (Drive) |
| `data/splits/` | train / val / test / three_star CSVs | no (Drive) |
| `data/processed/` | Cleaned and tokenized data, sentences | no (Drive) |
| `outputs/` | Charts, metrics, tables, models | no (Drive) |
| `report/` | Report draft and figures | yes |

## Team rules
1. **Paths are relative to the repo root** (`data/splits/train.csv`), never `C:\Users\...`.
2. **Every script runs on its own**: reads its input files, writes its output files.
3. **Seed = 42** everywhere (`random_state=42`).
4. **Edit only your own files.** Need a change in someone else's? Ask them.
5. **Commit small, push often** to `main`: `git pull` → work → `git add` → `git commit` → `git push`.
6. Don't commit data or big outputs — put them in the Drive folder.

## Data contract
`data/splits/{train,val,test,three_star}.csv` — columns:

| column | meaning |
|---|---|
| `review_id` | stable id (row number in the raw file) — use it to join |
| `text` | raw review text |
| `stars` | 1–5 |
| `label` | 1 = complaint (1–2★), 0 = not (4–5★), empty for 3★ |

Later steps **add columns, keep `review_id`**.

## Pipeline and owners
| Script | Step | Owner | Reads | Writes |
|---|---|---|---|---|
| `src/01_prepare_data.py` | Labels + split | A — MM | `data/raw/w_review_train.csv` | `data/splits/*.csv` |
| `src/02_clean.py` | Cleaning, tokenization (step 1) | B — Amm | `data/splits/*.csv` | `data/processed/*_clean.csv` (+ `tokens` column) |
| `src/03_explore.py` | Charts (step 2) | C — Pang | `data/processed/` | `outputs/figures/` |
| `src/04_classifier.py` | TF-IDF + LR / SVM, score 3★ (step 3) | D — Van | `data/processed/` | `outputs/classifier/` |
| `src/05_sentences.py` | Split complaints into sentences (step 4) | B — Amm | `data/processed/` | `data/processed/complaint_sentences.csv` |
| `src/06_topics.py` | LDA vs BERTopic, categories (steps 5–6) | E — Gim | `complaint_sentences.csv` | `outputs/topics/` |
| `src/07_evaluate.py` | Hand-label evaluation (step 7) | F — Toffy | `outputs/topics/sentence_assignments.csv` + filled labeling sheet | `outputs/evaluation/` |
| `src/08_summaries.py` | Per-category summaries (step 8) | F — Toffy, with E | `outputs/topics/`, `outputs/evaluation/labels_merged.csv` | `outputs/summaries/` |

## Preprocessing outputs (B — Amm)

### `src/02_clean.py` → `data/processed/{train,val,test,three_star}_clean.csv`
Same columns as `data/splits/*.csv` (`review_id, text, stars, label`), plus:

| column | meaning |
|---|---|
| `text_clean` | cleaned text, not tokenized (use for BERTopic) |
| `tokens` | space-separated tokens, stopwords removed, negations kept (use for TF-IDF / LDA) |
| `n_tokens` | number of tokens (0 = empty after cleaning; rows are kept so `review_id` and the test set size do not change) |

Cleaning: lowercase, remove URLs / HTML / emoji / punctuation, PyThaiNLP `normalize`, collapse repeated letters (`อร่อยยยย` → `อร่อย`), `55555` → `555`, tokenize with `newmm`, remove stopwords **except** negation and complaint words (`ไม่, ไม่ได้, ไม่ค่อย, ไม่มี, ไม่เคย, ไม่ใช่, ยัง, น้อย, เกิน, หมด, ช้า, นาน, มาก, แค่, เลย`).

### `src/05_sentences.py` → `data/processed/complaint_sentences.csv`
One row per sentence from 1–2★ reviews (`label == 1`) of train, val and test.
Positive sentences inside low-star reviews are **kept** (they are labeled "Not a complaint" in step 6).

| column | meaning |
|---|---|
| `sentence_id` | unique sentence id (use as ID in the labeling sheet) |
| `review_id` | source review (join key) |
| `sent_idx` | position of the sentence inside its review (starts at 0) |
| `split` | train / val / test |
| `stars` | stars of the source review (1 or 2) |
| `sentence_raw` | original sentence (for human labelers) |
| `sentence` | cleaned sentence (use for BERTopic) |
| `tokens` | space-separated tokens (use for LDA) |

Sentences are split from the **raw** text (before cleaning, because cleaning removes `!`, `?` and line breaks), first by regex, then with PyThaiNLP `crfcut`. Sentences with fewer than 2 tokens are dropped (`--min-tokens`).

### Notes
- `review_id` is not in order in these files: it is the row number in the raw file and the split is random. Do not renumber it.
- `05_sentences.py` needs one extra package: `pip install python-crfsuite`.
- `05_sentences.py` reuses `clean_text` and `tokenize` from `02_clean.py`, so keep both files in `src/`.
- `05_sentences.py` does not depend on the classifier (04); it can run right after `02`.

## Exploration outputs (C — Pang)

### `src/03_explore.py` → `outputs/figures/`
Reads `data/processed/{train,val,test,three_star}_clean.csv`, writes six PNGs plus
`exploration_summary.md` (all the numbers, ready to paste into the report).

| file | what it shows | data used |
|---|---|---|
| `fig01_star_distribution.png` | star ratings, grouped into the two classes | all four files |
| `fig02_review_length.png` | length by class, and by star rating | train / all four |
| `fig03_top_words.png` | most common words per class, as % of reviews | train |
| `fig04_distinctive_words.png` | words leaning to one class (log-odds ratio) | train |
| `fig05_wordcloud_complaint.png` | word cloud, complaints | train |
| `fig06_wordcloud_not_complaint.png` | word cloud, non-complaints | train |

Word-level figures use **train only**, so val and test stay unseen and nothing we
notice here can leak into how the later steps are tuned.

Options: `--font` (path to a Thai .ttf), `--top-n` (fig03, default 20),
`--top-n-distinct` (fig04, per side, default 12), `--min-count` (fig04 cut-off,
default 50), `--max-words` (word clouds, default 120).

### What the charts show
- Complaints are **8.1%** of the labeled data (5.7% of all reviews). Step 3 should
  use class weighting and report macro-F1, not accuracy.
- 3★ is **30.4%** of the dataset, all held out of training.
- Complaints are longer: median **63** tokens against **52**.
- Positive words fill the complaint reviews: อร่อย is in **50.2%** of them, and the
  top-20 word lists of the two classes are nearly the same. This is why step 6 works
  on sentences and needs a "Not a complaint" category.
- fig04 already previews the root causes: แย่มาก, เซ็ง, แมลงวัน, เช็คบิล, ปรับปรุง.

### Notes
- Thai text needs a Thai font. The script looks for one on Windows, macOS and Linux;
  if the charts come out as empty boxes, pass your own:
  `python src/03_explore.py --font "C:/Windows/Fonts/leelawui.ttf"`.
- It reads only the `tokens` column, so it does not need PyThaiNLP. pandas,
  matplotlib and wordcloud are enough, and all three are already in requirements.txt.
- Negation is a separate token, so "ไม่สนิท" shows up as สนิท on its own in fig04.
  Worth a line in the report's limitations.
- `outputs/` is gitignored: the figures go to the Drive folder, not into git.

## Classifier outputs (D — Van)

### `src/04_classifier.py` → `outputs/classifier/`
Reads `data/processed/{train,val,test,three_star}_clean.csv`. Runs on its own in about 13 minutes.

**Features**: TF-IDF word n-grams on `tokens` (split on spaces, no re-tokenizing) +
TF-IDF `char_wb` n-grams on `text_clean`, both with `sublinear_tf=True`.
**Models**: Logistic Regression (`liblinear`) and Linear SVM, both `class_weight="balanced"`, `random_state=42`.

**Protocol**: everything is fit on **train only**. Word n-grams (1,1)/(1,2)/(1,3), char n-grams
(2,4)/(3,5), C and the decision threshold are chosen on **val** by macro-F1. The better model is
chosen on **val**, not test. **Test** is scored once per model.

| file | what it holds |
|---|---|
| `metrics_summary.md` | all the numbers below, ready to paste into the report |
| `val_tuning.csv` | every n-gram / C combination tried, scored on val |
| `threshold_tuning.csv` | chosen settings and threshold for each model |
| `model_comparison.csv` | LR vs SVM on val and test, at threshold 0 and at the tuned threshold |
| `classification_report_{lr,svm}.csv` | per-class precision / recall / F1 on test |
| `confusion_matrix_{lr,svm}.png` / `.csv` | test confusion matrices |
| `top_features_{lr,svm}.csv` | top 25 word and char n-grams per class (by coefficient) |
| `three_star_predictions.csv` | `review_id, stars, score, pred_complaint` for every 3★ review |
| `three_star_summary.csv` | how many 3★ reviews are predicted as complaints |

### Results
| model | settings (val) | test macro-F1 | complaint P | complaint R | complaint F1 |
|---|---|---|---|---|---|
| **Linear SVM** (selected) | word (1,2) + char (3,5), C=0.3 | **0.8724** | 0.818 | 0.717 | 0.764 |
| Logistic Regression | word (1,3) + char (3,5), C=100 | 0.8781 | 0.805 | 0.748 | 0.775 |

- **SVM is the selected model** because its val macro-F1 is higher (0.8643 vs 0.8626). LR scores
  slightly higher on test, but switching models after seeing test would leak test into the choice.
  The two are within about 0.006 of each other, so treat them as equal in the report.
- Accuracy is about 96% for both, but it is not the main metric: always predicting "not complaint"
  already gives 92%.
- Top complaint words: ไม่ อร่อย, แย่, จืด, รสชาติ ธรรมดา, ไม่ ประทับใจ, ไม่ สด, ไม่ คุ้ม.
  Top non-complaint words: อร่อย, ชอบ, ไม่ ผิดหวัง, อร่อย มาก, เด็ด.
  Word bigrams catch negation (`ไม่ อร่อย`), which single words miss (see the fig04 note above).
- **3★ reviews**: **1,845 of 12,171 (15.2%)** are predicted as complaints. Join on `review_id`.

### Notes
- It needs only pandas, scikit-learn and matplotlib (no PyThaiNLP), all in requirements.txt.
- Empty `tokens` (`n_tokens = 0`) are filled with `""`; no rows are dropped.
- For LR, the best C is the top of the grid (100), but val macro-F1 only moves by about 0.001
  between C = 10 and 100, so going higher is not worth the extra run time.
- `outputs/classifier/` is gitignored: upload the whole folder to the Drive.

## Topic and category outputs (E — Gim)

### `src/06_topics.py` → `outputs/topics/`
Reads `data/processed/complaint_sentences.csv`, plus `train_clean.csv` and
`outputs/classifier/threshold_tuning.csv` to re-fit D's selected classifier.
First run about 20 min on a laptop CPU (embeddings, LDA grid), later runs about 3 min (cached).

**Handoff files**

| file | what it holds |
|---|---|
| `categories.csv` | final category list: `category_id`, `category`, `description` (for the labeling sheet), counts, share of complaint sentences, 1★ / 2★ shares, distinctive terms, 3 example quotes |
| `sentence_assignments.csv` | one row per sentence of `complaint_sentences.csv`: `sentence_id`, `review_id`, `sent_idx`, `split`, `stars`, `sentence_raw`, `sentence`, `topic`, `topic_label`, `bertopic_outlier`, `aspect`, `complaint_score`, `is_complaint`, **`category`** |

Other files: `topics_summary.md` (all numbers for report section 6, ready to paste),
`topic_to_category.csv` (each raw topic → category, with matched keywords and examples),
`bertopic_topics.csv`, `lda_topics.csv`, `lda_grid.csv`, `model_comparison.csv`.

**Method**
1. Topic models use B's `tokens` minus negations, intensifiers, pronouns and particles
   (they carry polarity, not the cause), and only words in ≥ 5 sentences.
2. **LDA** (gensim) for K = 5–25, best by c_v → K = 8.
   **BERTopic**: `paraphrase-multilingual-mpnet-base-v2` embeddings of `sentence` → UMAP → HDBSCAN
   (min cluster 60) → 100 topics; noise sentences go to their nearest topic.
   Both are scored with the same gensim coherence on the same corpus.
3. **BERTopic is chosen.** LDA has higher coherence (c_v 0.443 vs 0.378), but its 8 topics each mix
   several causes; BERTopic separates waiting, staff, price, portion, insects, toilets, billing, heat, ...
4. **Category** = the topic's category (keyword lists in `CATEGORIES`, written after reading the
   topics; matched on keywords, not topic numbers) **if** D's classifier scores the sentence above
   D's threshold; otherwise **Not a complaint**. Clusters group by subject, not sentiment, so this
   is what separates "อร่อย" from "ไม่อร่อย".

### Results (33,246 sentences, 9,450 scored as complaints)
| category | share of complaint sentences |
|---|---|
| Food taste or quality | 41.9% |
| Other (dissatisfaction with no stated cause, or unlisted) | 32.9% |
| Price or value | 9.6% |
| Staff behavior | 5.4% |
| Slow service or long wait | 5.1% |
| Atmosphere or facilities | 2.4% |
| Order or billing error | 1.4% |
| Cleanliness | 1.2% |

The other 23,796 sentences (71.6%) are **Not a complaint**.

### Changes to the planned category list (guide section 6)
- **Order or billing error** replaces "Wrong or missing order": no wrong-order topic emerged; a payment / bill topic did.
- **Atmosphere or facilities** added: heat, air-con, seating, parking and noise form clear topics.

### Notes for F (step 7)
- Sample the 120 rows from `complaint_sentences.csv`; use `sentence_id` as the sheet ID and the names and
  descriptions from `categories.csv` as the label options. Compare with `category` in
  `sentence_assignments.csv`. Do not tune `06_topics.py` on those labels.
- `bertopic_outlier = 1` marks sentences HDBSCAN left unclustered (35%); useful for error analysis.

### Notes
- Uses scikit-learn's `HDBSCAN`, not the `hdbscan` package, so nothing needs compiling.
- Tokenizer issue found: newmm splits "ชามละ" (per bowl) into ชา | มละ (same for ชิมละ, ขนมละ), so the
  junk token "มละ" topped the price topic. `06_topics.py` repairs it for the topic models only
  (`fix_mla`: ชา + มละ → ชาม). It also occurs 239 times in `train_clean.csv`; a proper fix would be a
  custom dictionary in `02_clean.py` (B).
- Runs are deterministic on one machine (seed 42). On another machine UMAP can give slightly
  different topics and topic numbers; categories are matched by keyword, not topic number, so the
  category list stays the same, but a few sentence assignments may change. Use the Drive copy of
  `sentence_assignments.csv` for steps 7–8 so everyone works from the same file.
- Upload `outputs/topics/` to the Drive **without `cache/`** (≈100 MB of embeddings; it is rebuilt).

## Evaluation and summary outputs (F — Toffy)

### `src/07_evaluate.py` → `outputs/evaluation/`
Two modes. `sample` builds the labeling sheet, `score` compares the filled sheet with the model.
`python src/07_evaluate.py demo` runs both on made-up data to check the code.

| file | what it holds |
|---|---|
| `labeling_sheet.xlsx` | 120 sentences, one tab per member (20 each) with a category dropdown; no model answer |
| `labeling_key.csv` | `sample_id` → `sentence_id`, `model_category`, `weight` (kept off the Drive until labeling was done) |
| `labels_merged.csv` | key + `human_label`, one row per labeled sentence |
| `metrics.json` | accuracy, macro-F1, population-weighted accuracy, accuracy per annotator |
| `per_class.csv`, `confusion_matrix.csv` | per-category precision / recall / F1; human (rows) vs model (columns) |
| `disagreements.csv` | the sentences where human and model differ (examples for the limitations section) |

**Sampling: stratified, not plain random (differs from the guide).** 72% of sentences are
"Not a complaint" and Cleanliness is 0.4%, so a random 120 would hold about 90 "Not a complaint"
and maybe no Cleanliness sentence. The sample takes about the same number per **model** category
(14 each; 12 Order or billing error and 10 Cleanliness, all that the pool has), from **test-split
reviews only**, seed 42, sentences of at least 8 characters, no duplicates. Each row has a
`weight` = category size in the pool / number sampled, used to weight the accuracy back to the
real category mix. `--strategy random` gives the plain random sample.

### Results (120 sentences, 6 labelers, one label per sentence)
| metric | value |
|---|---|
| Accuracy on the stratified sample | 54.2% |
| Macro-F1 | 0.52 |
| Population-weighted accuracy | 74.7% |
| Agreement on complaint vs not a complaint | 89.2% |

Precision per model category (share of sampled sentences where the human chose the same category):
Cleanliness 90%, Food taste or quality 86%, Not a complaint 79%, Staff behavior 71%,
Price or value 64%, Slow service or long wait 43%, Other 36%, Atmosphere or facilities 14%,
Order or billing error 8%.

The model separates complaints from non-complaints well; the weak point is naming the cause for
the small keyword-matched categories. Order or billing error catches sentences that only mention
paying (เงิน, จ่าย, เช็คบิล), and Atmosphere or facilities catches โต๊ะ / ร้อน / อุ่น used about food.

### `src/08_summaries.py` → `outputs/summaries/`
Reads `sentence_assignments.csv`, `categories.csv` and `labels_merged.csv`.

| file | what it holds |
|---|---|
| `category_summary.csv` | per complaint category: sentence share, review share, 1★ vs 2★, hand-label precision, top terms, 3 quotes, recommendation |
| `star_comparison.csv` | share of 1★ and 2★ reviews (and sentences) mentioning each category, and the ratio |
| `summary.md` | the same as tables and per-category text, ready for report sections 6–7 |

Review share = share of the 2,260 low-star reviews with at least one sentence in the category
(a review can mention several causes, so it does not sum to 100%).

- Food taste or quality is in 76.8% of low-star reviews, Price or value in 30.2%.
- Staff behavior (25.1% of 1★ reviews vs 14.5% of 2★) and Slow service (23.6% vs 15.5%) are more
  typical of 1★ reviews; food and price are about equal at both levels.

### Notes
- Per-category precision rests on 10–14 sentences each: read it as a tendency, not an exact value.
- Sampling by model category measures precision directly; recall and macro-F1 are approximate,
  because the human categories were not sampled evenly.
- Each sentence has one labeler, so agreement between labelers cannot be measured.
- Top terms and example quotes in step 8 are E's (`categories.csv`), not recomputed. The
  recommendations are written by hand in `RECOMMENDATIONS`; categories with precision below 50%
  are flagged ⚠ in `summary.md`.
- The dropdown is an inline list: Google Sheets drops dropdowns that point at a hidden sheet.
- `score` reads every tab except `All`, so a renamed member tab still counts.
- 07 and 08 need only pandas, scikit-learn and openpyxl.

## Run everything
```bash
python src/01_prepare_data.py --input data/raw/w_review_train.csv
python src/02_clean.py
python src/03_explore.py
python src/04_classifier.py
python src/05_sentences.py
python src/06_topics.py
python src/07_evaluate.py sample   # then everyone fills outputs/evaluation/labeling_sheet.xlsx
python src/07_evaluate.py score
python src/08_summaries.py
```
