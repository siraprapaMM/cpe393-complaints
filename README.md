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

## Run everything
```bash
python src/01_prepare_data.py --input data/raw/w_review_train.csv
python src/02_clean.py
# ... 03 to 08 in order
```
