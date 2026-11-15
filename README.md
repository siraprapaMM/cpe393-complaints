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
| `src/07_evaluate.py` | Hand-label evaluation (step 7) | F | labeling sheet + topics | `outputs/evaluation/` |
| `src/08_summaries.py` | Per-category summaries (step 8) | F + E | `outputs/topics/` | `outputs/summaries/` |

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

## Run everything
```bash
python src/01_prepare_data.py --input data/raw/w_review_train.csv
python src/02_clean.py
python src/03_explore.py
# ... 04 ...
python src/05_sentences.py
# ... 06 to 08 in order
```
