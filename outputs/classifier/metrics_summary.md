# Step 3 — Complaint classifier: results

Features: TF-IDF word n-grams on `tokens` + char_wb n-grams on `text_clean` (sublinear tf). Models use `class_weight="balanced"`, `random_state=42`.
n-grams, C and threshold tuned on **val** (macro-F1); fit on **train** only; **test** scored once.

## Selected settings (val)

| model | word n-gram | char n-gram | C | val macro-F1 (thr 0) | threshold | val macro-F1 (tuned thr) |
|---|---|---|---|---|---|---|
| Logistic Regression | (1, 3) | (3, 5) | 100.0 | 0.8588 | -0.6997 | 0.8626 |
| Linear SVM | (1, 2) | (3, 5) | 0.3 | 0.8631 | -0.0118 | 0.8643 |

## Test results (threshold tuned on val)

| model | macro-F1 | macro-P | macro-R | complaint P | complaint R | complaint F1 | accuracy |
|---|---|---|---|---|---|---|---|
| Logistic Regression | **0.8781** | 0.8913 | 0.8659 | 0.8048 | 0.7478 | 0.7752 | 0.9648 |
| Linear SVM | **0.8724** | 0.8967 | 0.8514 | 0.8182 | 0.7168 | 0.7642 | 0.9641 |

Test at the default threshold (0) is in `model_comparison.csv`.

## Confusion matrices (test)

**Logistic Regression** — TN 2,516 · FP 41 · FN 57 · TP 169

**Linear SVM** — TN 2,521 · FP 36 · FN 64 · TP 162

## Selected model: Linear SVM (higher val macro-F1)

Top word features:

| rank | complaint | not complaint |
|---|---|---|
| 1 | ไม่ อร่อย | อร่อย |
| 2 | แย่ | ชอบ |
| 3 | ไม่ | ไม่ ผิดหวัง |
| 4 | จืด | อร่อย มาก |
| 5 | รสชาติ ธรรมดา | เด็ด |
| 6 | ไม่ค่อย | ดีมาก |
| 7 | ไม่ ประทับใจ | 555 |
| 8 | แข็ง | นุ่ม |
| 9 | ไม่ค่อย อร่อย | ฟิน |
| 10 | ไม่ สด | รสชาติ ดี |
| 11 | แย่มาก | ไม่ แพง |
| 12 | พอใช้ได้ | รสชาติ อร่อย |
| 13 | ปรับปรุง | หอม |
| 14 | น้อย มาก | กำลังดี |
| 15 | ไม่ คุ้ม | ชอบ มาก |

## 3-star reviews

1,845 of 12,171 three-star reviews (**15.16%**) are predicted as complaints (Linear SVM, threshold -0.0118). Per-review results with `review_id`: `three_star_predictions.csv`.
