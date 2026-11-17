"""Steps 5-6: topic modeling (LDA vs BERTopic) and category naming.  Owner: Gim
Reads  data/processed/complaint_sentences.csv          (from 05_sentences.py)
       data/processed/train_clean.csv                  (to re-fit the step-3 classifier)
       outputs/classifier/threshold_tuning.csv         (step-3 selected settings)
Writes outputs/topics/
         lda_grid.csv               LDA coherence for every K tried
         lda_topics.csv             topics of the best LDA (keywords, size)
         bertopic_topics.csv        raw BERTopic topics (keywords, size, coherence)
         model_comparison.csv       LDA vs BERTopic: coherence (c_v, c_npmi), diversity
         topic_to_category.csv      every raw topic of the chosen model -> named category
         categories.csv             HANDOFF: final category list (+ description for labelers)
         sentence_assignments.csv   HANDOFF: one row per sentence -> topic, category
         topics_summary.md          numbers ready to paste into the report
         cache/                     sentence embeddings + complaint scores (re-used on re-runs)
Run:   python src/06_topics.py        (first run ~20 min on a laptop CPU, cached re-runs ~3 min)

Method
1. Aspect tokens. Topic models should group sentences by WHAT they are about
   (food, staff, price, ...). Negations, intensifiers, pronouns and particles carry
   polarity or nothing, so they are removed from `tokens` for the topic models only.
2. LDA baseline (gensim) on aspect tokens; K chosen by c_v coherence.
3. BERTopic: multilingual sentence embeddings of `sentence` -> UMAP -> HDBSCAN.
   Topic keywords come from c-TF-IDF over the SAME aspect tokens as LDA, so both
   models are scored with the same gensim coherence on the same corpus. Sentences
   HDBSCAN leaves as noise are then given their nearest topic (embedding similarity).
   The model used for the categories is BERTopic (--model; reasons in CHOICE_REASON).
4. Category naming (step 6). Every raw topic of the chosen model is mapped to one
   named category by scoring its top keywords against a short hand-written keyword
   list per category (CATEGORIES below, written after reading the raw topics). The
   mapping uses keywords, not topic numbers, so it gives the same result even if
   topic numbering changes on another machine.
5. "Not a complaint". Clusters group by aspect, not by sentiment: "อร่อย" and
   "ไม่อร่อย" fall into the same food topic. So every sentence is also scored by the
   step-3 complaint classifier (same model, n-grams, C and threshold as
   04_classifier.py, re-fitted on train). A sentence at or below the threshold is
   "Not a complaint" (positive or neutral); otherwise it gets its topic's category.
"""
import argparse
import ast
import copy
import importlib.util
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# =============================================================== aspect tokens
# B keeps these in `tokens` because they carry complaint meaning (02_clean.KEEP).
# For the topic models they are polarity / intensity, not aspect, so they go.
# ช้า, นาน, น้อย, เกิน, หมด are kept: they name the problem itself.
POLARITY_WORDS = {"ไม่", "ไม่ได้", "ไม่ค่อย", "ไม่มี", "ไม่เคย", "ไม่ใช่",
                  "ยัง", "มาก", "แค่", "เลย"}
# Pronouns, particles and filler words that survive the PyThaiNLP stop-word list.
FILLER_WORDS = {"ผม", "เรา", "เค้า", "เขา", "นะคะ", "นะครับ", "ค่ะ", "คะ", "ครับ",
                "จ้า", "อ่ะ", "อะ", "นะ", "บอ", "ดู", "ทำ", "ตัว", "รู้", "รู้สึก",
                "เหมือน", "สำหรับ", "ตอน", "ลอง", "เจอ", "ร้าน", "555"}
ASPECT_STOP = POLARITY_WORDS | FILLER_WORDS
NUMBER = re.compile(r"^[0-9]+$")


# Tokenizer fix: "มละ".
# Found: the meaningless token "มละ" was a top keyword of the price topic, and "ชา"
# (tea) showed up as a typical word of "Price or value". Traced back: newmm splits a
# word ending in "ม" followed by the particle "ละ" ("per" / "then") in the wrong
# place, e.g. "ชามละ 60 บาท" (60 baht per bowl) -> ชา | มละ instead of ชาม | ละ.
# In complaint_sentences.csv, 52 of the 57 "มละ" tokens come from "ชามละ"; the rest
# follow the same pattern (ชิมละ -> ชิ|มละ, ขนมละ -> ขน|มละ, หนองขามละ -> ขา|มละ).
# Fix (topic models only): move "ม" back onto the previous token (ชา -> ชาม) and drop
# "ละ", a particle with no subject. B's `tokens` column and the step-3 complaint
# score are left unchanged. The proper fix belongs in 02_clean.py (custom dictionary).
SPLIT_MLA = "มละ"


def fix_mla(words: list) -> list:
    """['ชา', 'มละ', '60', 'บาท'] -> ['ชาม', '60', 'บาท'] (see the note above)."""
    out = []
    for w in words:
        if w == SPLIT_MLA:
            if out:
                out[-1] += "ม"            # give "ม" back to the word it belongs to
            continue                      # "ละ" is dropped
        out.append(w)
    return out


def aspect_tokens(tokens) -> list:
    """B's space-separated `tokens` -> token list used by both topic models."""
    if not isinstance(tokens, str):
        return []
    return [w for w in fix_mla(tokens.split())
            if w not in ASPECT_STOP and not NUMBER.match(w)]


# =============================================================== categories
# Final category list (step 6), written after reading the ~100 raw BERTopic topics.
# Names follow the hand-labeling guide (section 6), with two changes the topics
# called for: no "wrong order" topic emerged, but a billing / payment topic did
# (-> "Order or billing error"), and a clear facilities group did (heat, air-con,
# seats, parking, noise -> "Atmosphere or facilities").
#
# `keywords` are matched EXACTLY against a topic's top keywords (newmm tokens, as
# they appear in bertopic_topics.csv). "Other" also has keywords: topics of general
# dissatisfaction with no cause ("ผิดหวัง", "คงไม่มาอีก", "ปรับปรุงด่วน") must win
# over a stray aspect word, otherwise e.g. "ลืม" would pull them into a named category.
# Left out on purpose after reading the topics: ถูกปาก, หิว, อิ่ม, ท้องเสีย, สนใจ
# (they top clusters of short generic fragments such as "รับไม่ได้เลย"), ความคุ้มค่า and
# วิจารณ์ (they top a "regarding the taste ..." cluster), โทร (tops a junk cluster).
NOT_COMPLAINT = "Not a complaint"
OTHER = "Other"
CATEGORIES = [
    {"name": "Slow service or long wait",
     "description": "Waiting too long for a table, for food or for staff; slow serving.",
     "keywords": ["รอ", "นาน", "ช้า", "นาที", "ชั่วโมง", "ชม", "ใช้เวลา", "ครึ่ง", "คิว",
                  "เวลา", "สักครู่", "สักที", "เสิร์ฟ", "พนักงานเสิร์ฟ", "เด็กเสิร์ฟ",
                  "ทีหลัง", "ครัว", "ทวง", "ความเร็ว"]},
    {"name": "Staff behavior",
     "description": "Rude, careless, inattentive or unfriendly staff or owner.",
     "keywords": ["พนักงาน", "บริการ", "พนักงานบริการ", "หน้าบึ้ง", "ทั่วถึง", "มารยาท",
                  "ตะโกน", "หัวหน้า", "สุภาพ", "ยิ้มแย้ม", "ยิ้มแย้มแจ่มใส",
                  "ใจดี", "ดูแล", "เอาใจใส่", "ทักทาย"]},
    {"name": "Food taste or quality",
     "description": "Taste, texture, freshness, smell or temperature of the food or "
                    "drinks (bland, too sweet, tough, not fresh, smells bad, ...).",
     "keywords": [
         # taste / texture / smell / quality words
         "อร่อย", "รสชาติ", "รส", "ชาด", "ชาต", "รสชาด", "จืด", "เค็ม", "หวาน", "เปรี้ยว",
         "เผ็ด", "เลี่ยน", "กลมกล่อม", "ด้อย", "ห่วย", "โดดเด่น", "เท่าที่ควร",
         "ธรรมดา", "ปานกลาง", "มาตรฐาน", "ใช้ได้", "พอใช้ได้", "พอใช้", "โอเค", "กลาง",
         "หอม", "กลิ่น", "ได้กลิ่น", "กลิ่นคาว", "เหม็น", "เหม็นคาว", "เหม็นหืน",
         "เหม็นเขียว", "เน่า", "เครื่องเทศ", "เข้าเนื้อ", "มีรส", "นุ่ม", "ความนุ่ม",
         "เหนียว", "ความเหนียว", "แข็ง", "แห้ง", "เละ", "ยุ่ย", "เปื่อย", "นิ่ม", "แฉะ",
         "หุง", "สด", "ผงชูรส", "คุณภาพ", "วัตถุดิบ", "น่ากิน", "น่าอร่อย", "การปรุง",
         "ข้น", "เข้มข้น",
         # dishes, ingredients and drinks (topics that are about one kind of food)
         "กุ้ง", "หอย", "ปลาหมึก", "ปลา", "แซลมอน", "หมึก", "หอยนางรม", "หมู", "เนื้อหมู",
         "หมูกรอบ", "หมูแดง", "หมูสับ", "ซี่โครง", "หมูสามชั้น", "เนื้อ", "เนื้อวัว",
         "สเต็ก", "สเต๊ก", "steak", "beef", "เป็ด", "ไก่", "ไก่ทอด", "ไก่ย่าง", "เนื้อไก่",
         "ข้าวมันไก่", "ไข่", "ไข่ต้ม", "ไข่ตุ๋น", "มะม่วง", "มะพร้าว", "ผลไม้", "ส้มตำ",
         "ซุป", "ซุบ", "น้ำแกง", "พริก", "เห็ด", "ต้มยำ", "ผัด", "ผัก", "ผักสด", "ข้าว",
         "rice", "บะหมี่", "ข้าวผัด", "ผัดไทย", "ก๋วยเตี๋ยว", "ซูชิ", "sushi", "ขนมปัง",
         "ไส้กรอก", "สลัด", "แกง", "แป้ง", "พิซซ่า", "pizza", "สปา", "ชีส", "cheese",
         "เค้ก", "ขนม", "ของหวาน", "ไอศครีม", "ไอศกรีม", "ไอติม", "ช็อคโกแลต",
         "ช็อกโกแลต", "น้ำแข็ง", "น้ำแข็งไส", "กาแฟ", "coffee", "ชา", "ชาเขียว", "ชาเย็น",
         "นม", "นมสด", "เครื่องดื่ม", "เบียร์", "หม้อ", "ต้ม", "หม้อไฟ"]},
    {"name": "Price or value",
     "description": "Too expensive, not worth the money, extra charges, or a portion "
                    "too small for the price.",
     "keywords": ["ราคา", "แพง", "ราคาแพง", "บาท", "vat", "charge", "ค่าเสียหาย", "เทียบ",
                  "คุ้ม", "คุ้มค่า", "สมราคา", "ประหยัด", "ปริมาณ", "น้อย",
                  "นิดเดียว", "ตัวเล็ก", "ลดลง"]},
    {"name": "Cleanliness",
     "description": "Dirty shop, table, dishes or toilet; insects, hair or other "
                    "things that should not be there.",
     "keywords": ["สกปรก", "สะอาด", "ความสะอาด", "สะอาดสะอ้าน", "ห้องน้ำ", "ล้าง", "เช็ด",
                  "ทำความสะอาด", "คราบ", "ฝุ่น", "รก", "ยุง", "แมลงวัน", "แมลงสาบ", "แมลง",
                  "ยากันยุง"]},
    {"name": "Order or billing error",
     "description": "Wrong, missing or forgotten order; mistakes or problems with the "
                    "bill and payment.",
     "keywords": ["ออเดอร์", "สั่งผิด", "ผิด", "ตกหล่น", "เช็คบิล", "บิล", "คิดเงิน",
                  "ตังค์", "เก็บเงิน", "ทอน", "บัตรเครดิต"]},
    {"name": "Atmosphere or facilities",
     "description": "Seating, heat or air-conditioning, noise, crowding, smoke, parking "
                    "and other problems with the place itself.",
     "keywords": ["เก้าอี้", "ที่นั่ง", "โต๊ะ", "โซฟา", "นั่ง", "เฟอร์นิเจอร์", "แอร์",
                  "พัดลม", "ห้อง", "อากาศ", "ร้อน", "อบอ้าว", "แดด", "ควัน", "ความร้อน",
                  "ถ่ายเท", "ที่จอดรถ", "จอด", "ลานจอดรถ", "เสียงดัง", "วุ่นวาย", "แออัด",
                  "คับแคบ", "บรรยากาศ", "ตกแต่ง", "ดนตรี", "เพลง", "สูบบุหรี่", "wifi"]},
    {"name": OTHER,
     "description": "A complaint with no specific cause (\"disappointed\", \"won't come "
                    "back\", \"please improve\") or a cause not listed above.",
     "keywords": ["ผิดหวัง", "ประทับใจ", "ถูกใจ", "น่าประทับใจ", "คาดหวัง", "โดยรวม",
                  "อย่า", "อีกแล้ว", "เตือน", "ครั้งสุดท้าย", "ลาก่อน", "เลิก", "ไม่ต้อง",
                  "กล้า", "ปรับปรุง", "การปรับปรุง", "ดีขึ้น", "พัฒนา", "เซ็ง", "โมโห",
                  "ด่า", "หลอก", "น่ารำคาญ", "รำคาญ", "น่ากลัว", "เสียดาย", "เสียใจ",
                  "เสียความรู้สึก", "ขออภัย", "ความไม่พอใจ", "งง", "สอบตก", "disappointed",
                  "ความคิดเห็น", "ความเห็นส่วนตัว", "รีวิว", "ดาว", "คะแนน"]},
]
NOT_COMPLAINT_DESC = ("Positive or neutral sentence: praise, or a description of the "
                      "shop, menu, location or visit, with no problem stated.")

# Why BERTopic is used for the categories (step 5 asks to choose by coherence AND by
# reading the topics). The coherence sentence before this one is written from the
# actual numbers at run time.
CHOICE_REASON = (
    "Reading the topics decides it: LDA's few topics each mix several causes "
    "(e.g. taste, price and portion in one topic; staff, seating and waiting in "
    "another) and none isolates cleanliness or billing, while BERTopic gives "
    "one clean topic per cause (waiting time, rude staff, insects, dirty toilet, "
    "billing, heat / air-con, portion size, ...). Coherence counts how often top words "
    "co-occur, so it rewards LDA's frequent generic words (กิน, ทาน, อาหาร, สั่ง); on "
    "short sentences this is a known weakness of the measure.")


# =============================================================== helpers
def load_module(name: str, filename: str):
    """Import a sibling script whose file name starts with a digit (e.g. 04_...)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topic_diversity(topics: list) -> float:
    """Share of unique words among all top words (1.0 = no topic repeats a word)."""
    words = [w for t in topics for w in t]
    return len(set(words)) / max(len(words), 1)


def coherence(topics: list, texts: list, dictionary) -> dict:
    """Mean and per-topic gensim coherence (c_v and c_npmi) of top-word lists.
    Words missing from the dictionary are skipped; topics with < 2 words get NaN."""
    from gensim.models.coherencemodel import CoherenceModel
    kept = [[w for w in t if w in dictionary.token2id] for t in topics]
    idx = [i for i, t in enumerate(kept) if len(t) >= 2]
    out = {}
    for measure in ["c_v", "c_npmi"]:
        per = np.full(len(topics), np.nan)
        if idx:
            cm = CoherenceModel(topics=[kept[i] for i in idx], texts=texts,
                                dictionary=dictionary, coherence=measure, processes=1)
            per[idx] = cm.get_coherence_per_topic()
        out[measure] = float(np.nanmean(per)) if idx else np.nan
        out[f"{measure}_per_topic"] = per
    return out


# =============================================================== complaint score
def complaint_scores(sent: pd.DataFrame, clean_dir: Path, clf_dir: Path,
                     cache_dir: Path, use_cache: bool):
    """Score every sentence with the step-3 selected classifier.

    The model, n-gram ranges, C and threshold are read from D's
    outputs/classifier/threshold_tuning.csv (the row with the best val macro-F1,
    i.e. the model D selected), the vectorizers and model come from
    04_classifier.py, and the model is re-fitted on train only, exactly as in step 3.
    Returns (scores, threshold, settings-dict). score > threshold = complaint.
    """
    tuning = pd.read_csv(clf_dir / "threshold_tuning.csv")
    best = tuning.loc[tuning["val_macro_f1_tuned_thr"].idxmax()]
    cfg = {"model": best["model"], "word": ast.literal_eval(best["word_ngram"]),
           "char": ast.literal_eval(best["char_ngram"]), "C": float(best["C"]),
           "threshold": float(best["threshold"])}
    tag = (f"{cfg['model']}_w{cfg['word'][0]}{cfg['word'][1]}"
           f"_c{cfg['char'][0]}{cfg['char'][1]}_C{cfg['C']:g}")      # cache key
    cache = cache_dir / f"complaint_scores_{tag}.npy"
    if use_cache and cache.exists():
        s = np.load(cache)
        if len(s) == len(sent):
            log(f"complaint scores: loaded from {cache}")
            return s, cfg["threshold"], cfg

    log(f"complaint scores: re-fitting step-3 model {tag} on train ...")
    from scipy.sparse import hstack
    clf = load_module("clf04", "04_classifier.py")
    tr = pd.read_csv(clean_dir / "train_clean.csv")
    tr["tokens"] = tr["tokens"].fillna("").astype(str)
    tr["text_clean"] = tr["text_clean"].fillna("").astype(str)
    wv, cv = clf.word_vectorizer(cfg["word"]), clf.char_vectorizer(cfg["char"])
    X = hstack([wv.fit_transform(tr["tokens"]), cv.fit_transform(tr["text_clean"])]).tocsr()
    model = clf.make_model(cfg["model"], cfg["C"]).fit(X, tr["label"].astype(int))
    # sentences: full `tokens` (negations kept!) for word n-grams, `sentence` for chars
    Xs = hstack([wv.transform(sent["tokens"].fillna("").astype(str)),
                 cv.transform(sent["sentence"].fillna("").astype(str))]).tocsr()
    s = model.decision_function(Xs).astype(np.float32)
    np.save(cache, s)
    return s, cfg["threshold"], cfg


# =============================================================== embeddings
def sentence_embeddings(texts: list, model_name: str, cache_dir: Path, use_cache: bool):
    """Normalised sentence embeddings, cached as .npy (computing them takes ~5 min)."""
    cache = cache_dir / f"embeddings_{model_name.replace('/', '_')}.npy"
    if use_cache and cache.exists():
        e = np.load(cache)
        if len(e) == len(texts):
            log(f"embeddings: loaded from {cache}")
            return e
    from sentence_transformers import SentenceTransformer
    log(f"embeddings: encoding {len(texts):,} sentences with {model_name} ...")
    model = SentenceTransformer(model_name)
    e = model.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    np.save(cache, e)
    return e


# =============================================================== LDA
def run_lda(texts: list, dictionary, ks: list, passes: int):
    """Fit one LDA per K and score each by coherence.
    Returns (grid df, {K: (model, top words, coherence)}, bag-of-words corpus)."""
    from gensim.models import LdaModel
    corpus = [dictionary.doc2bow(t) for t in texts]
    rows, models = [], {}
    for k in ks:
        t0 = time.time()
        lda = LdaModel(corpus, id2word=dictionary, num_topics=k, passes=passes,
                       iterations=200, chunksize=2000, alpha="auto", eta="auto",
                       random_state=SEED, eval_every=None)
        top = [[w for w, _ in lda.show_topic(i, topn=10)] for i in range(k)]
        coh = coherence(top, texts, dictionary)
        rows.append({"k": k, "c_v": round(coh["c_v"], 4), "c_npmi": round(coh["c_npmi"], 4),
                     "diversity": round(topic_diversity(top), 4)})
        models[k] = (lda, top, coh)
        log(f"LDA K={k:<3} c_v={coh['c_v']:.4f} c_npmi={coh['c_npmi']:.4f} "
            f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), models, corpus


def lda_assign(lda, corpus) -> np.ndarray:
    """Most likely topic per sentence; -1 if no word of the sentence is in the vocabulary."""
    out = np.full(len(corpus), -1)
    for i, bow in enumerate(corpus):
        if bow:
            out[i] = max(lda.get_document_topics(bow, minimum_probability=0.0),
                         key=lambda x: x[1])[0]
    return out


# =============================================================== BERTopic
def build_bertopic(min_topic_size: int, min_samples: int):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.cluster import HDBSCAN          # scikit-learn's HDBSCAN: no extra C build
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    return BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, min_samples=min_samples,
                              metric="euclidean", cluster_selection_method="eom"),
        # docs are already tokenized and filtered to the shared vocabulary -> just split.
        # (c-TF-IDF fits this on ONE document per topic, so min_df must stay 1 here.)
        vectorizer_model=CountVectorizer(tokenizer=str.split, token_pattern=None,
                                         lowercase=False, min_df=1),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        # "english" (the default) strips every non-ASCII character, i.e. all Thai,
        # before computing keywords. Embeddings are passed in, so no model is loaded.
        language="multilingual",
        top_n_words=10, calculate_probabilities=False, verbose=False)


def bertopic_words(tm, topn=10) -> dict:
    """topic id -> list of top keywords (outlier topic -1 excluded)."""
    return {t: [w for w, _ in tm.get_topic(t)[:topn] if w]
            for t in sorted(tm.get_topics()) if t != -1}


# =============================================================== categories
def map_topics(topic_words: dict, topn: int, min_score: int) -> dict:
    """topic id -> (category, score, matched keywords).

    Each of a topic's top-`topn` keywords votes for every category whose keyword
    list contains it, weighted by rank (1st keyword = topn points, last = 1).
    The category with the most points wins. A topic whose best score is below
    `min_score` (e.g. one weak match on the 6th keyword of a mixed topic) -> "Other".
    """
    out = {}
    for t, words in topic_words.items():
        scores, hits = Counter(), {}
        for rank, w in enumerate(words[:topn]):
            for c in CATEGORIES:
                if w in c["keywords"]:
                    scores[c["name"]] += topn - rank
                    hits.setdefault(c["name"], []).append(w)
        name, sc = scores.most_common(1)[0] if scores else (OTHER, 0)
        out[t] = (name if sc >= min_score else OTHER, sc, " ".join(hits.get(name, [])))
    return out


def distinctive_terms(docs: list, mask: np.ndarray, n: int = 10, min_count: int = 5) -> str:
    """Words over-represented in docs[mask] vs all docs (count x log-lift)."""
    inside = Counter(w for d, m in zip(docs, mask) if m for w in set(d))
    total = Counter(w for d in docs for w in set(d))
    n_in, n_all = max(mask.sum(), 1), len(docs)
    score = {w: c * np.log((c / n_in) / (total[w] / n_all))
             for w, c in inside.items() if c >= min_count}
    return " ".join(w for w, s in sorted(score.items(), key=lambda x: -x[1])[:n] if s > 0)


def representative(emb: np.ndarray, idx: np.ndarray, raw: pd.Series, k: int = 3,
                   score: np.ndarray = None, pool: int = 50, max_len: int = 150) -> list:
    """k typical sentences for a group of rows.

    Takes the `pool` sentences closest to the group's mean embedding (typical of the
    group); if `score` is given, returns the k with the highest score among them
    (e.g. the clearest complaints), otherwise the k closest. Sentences longer than
    `max_len` characters are skipped when enough shorter ones exist (quotes for the
    report); exact duplicates are skipped.
    """
    if len(idx) == 0:
        return []
    short = idx[raw.iloc[idx].str.len().values <= max_len]
    if len(short) >= k:
        idx = short
    c = emb[idx].mean(0)
    order = idx[np.argsort(-(emb[idx] @ c))]
    if score is not None:
        order = order[:pool]
        order = order[np.argsort(-score[order], kind="stable")]
    seen, out = set(), []
    for i in order:                       # skip exact duplicates
        s = raw.iloc[i]
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == k:
            break
    return out


# =============================================================== main
def main(a):
    out_dir, cache_dir = Path(a.out_dir), Path(a.out_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not a.no_cache

    sent = pd.read_csv(a.sentences)
    texts = [aspect_tokens(t) for t in sent["tokens"]]

    # shared vocabulary: LDA, BERTopic keywords and both coherence scores use it
    from gensim.corpora import Dictionary
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
    vocab = dictionary.token2id
    docs = [" ".join(w for w in t if w in vocab) for t in texts]   # BERTopic keyword docs
    log(f"{len(sent):,} sentences from {sent['review_id'].nunique():,} complaint reviews; "
        f"vocabulary {len(vocab):,} aspect tokens (in >= 5 sentences); "
        f"{sum(not d for d in docs):,} sentences have none of them")

    # ---------------------------------------------------- step 5a: LDA baseline
    lda_grid, lda_models, corpus = run_lda(texts, dictionary, a.lda_k, a.lda_passes)
    best_k = int(lda_grid.loc[lda_grid["c_v"].idxmax(), "k"])
    lda, lda_top, lda_coh = lda_models[best_k]
    lda_topic = lda_assign(lda, corpus)
    lda_grid.to_csv(out_dir / "lda_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "topic": range(best_k),
        "n_sentences": [(lda_topic == t).sum() for t in range(best_k)],
        "keywords": [" ".join(w) for w in lda_top],
        "c_v": np.round(lda_coh["c_v_per_topic"], 4),
        "c_npmi": np.round(lda_coh["c_npmi_per_topic"], 4),
    }).to_csv(out_dir / "lda_topics.csv", index=False, encoding="utf-8-sig")
    log(f"LDA: best K = {best_k}")

    # ---------------------------------------------------- step 5b: BERTopic
    emb = sentence_embeddings(sent["sentence"].fillna("").astype(str).tolist(),
                              a.embed_model, cache_dir, use_cache)
    log(f"BERTopic: UMAP + HDBSCAN (min_topic_size={a.min_topic_size}, "
        f"min_samples={a.min_samples}) ...")
    tm = build_bertopic(a.min_topic_size, a.min_samples)
    topics, _ = tm.fit_transform(docs, embeddings=emb)
    topics = np.asarray(topics)
    bt_words = bertopic_words(tm)
    bt_coh = coherence(list(bt_words.values()), texts, dictionary)
    n_outliers = int((topics == -1).sum())
    pd.DataFrame({                         # raw topics as fitted (before outlier step)
        "topic": list(bt_words),
        "n_sentences": [(topics == t).sum() for t in bt_words],
        "keywords": [" ".join(w) for w in bt_words.values()],
        "c_v": np.round(bt_coh["c_v_per_topic"], 4),
        "c_npmi": np.round(bt_coh["c_npmi_per_topic"], 4),
    }).to_csv(out_dir / "bertopic_topics.csv", index=False, encoding="utf-8-sig")
    log(f"BERTopic: {len(bt_words)} topics, {n_outliers:,} outliers "
        f"({n_outliers / len(topics):.1%}), c_v={bt_coh['c_v']:.4f} "
        f"c_npmi={bt_coh['c_npmi']:.4f}")

    # same model merged down to LDA's K, for a like-for-like coherence comparison
    tm_k = copy.deepcopy(tm)
    tm_k.reduce_topics(docs, nr_topics=best_k + 1)        # +1: the outlier topic -1
    bt_k_words = bertopic_words(tm_k)
    bt_k_coh = coherence(list(bt_k_words.values()), texts, dictionary)
    del tm_k
    log(f"BERTopic reduced to {len(bt_k_words)} topics: c_v={bt_k_coh['c_v']:.4f} "
        f"c_npmi={bt_k_coh['c_npmi']:.4f}")

    comparison = pd.DataFrame([
        {"model": f"LDA (K={best_k})", "n_topics": best_k,
         "c_v": lda_coh["c_v"], "c_npmi": lda_coh["c_npmi"],
         "diversity": topic_diversity(lda_top), "unassigned_share": (lda_topic == -1).mean()},
        {"model": "BERTopic", "n_topics": len(bt_words),
         "c_v": bt_coh["c_v"], "c_npmi": bt_coh["c_npmi"],
         "diversity": topic_diversity(list(bt_words.values())),
         "unassigned_share": n_outliers / len(topics)},
        {"model": f"BERTopic reduced to K={best_k}", "n_topics": len(bt_k_words),
         "c_v": bt_k_coh["c_v"], "c_npmi": bt_k_coh["c_npmi"],
         "diversity": topic_diversity(list(bt_k_words.values())),
         "unassigned_share": n_outliers / len(topics)},
    ]).round(4)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # outliers -> nearest topic by embedding similarity (if similar enough)
    topics_final = np.asarray(tm.reduce_outliers(docs, topics, strategy="embeddings",
                                                 embeddings=emb,
                                                 threshold=a.outlier_threshold))
    tm.update_topics(docs, topics=topics_final.tolist(),
                     vectorizer_model=tm.vectorizer_model, ctfidf_model=tm.ctfidf_model)
    bt_words = bertopic_words(tm)
    log(f"BERTopic: outliers {n_outliers:,} -> {(topics_final == -1).sum():,} after "
        f"assigning to the nearest topic (cosine >= {a.outlier_threshold})")

    # ---------------------------------------------------- choose the model
    if a.model == "lda":
        chosen, topic_of, topic_words = f"LDA (K={best_k})", lda_topic, \
            {t: lda_top[t] for t in range(best_k)}
    else:
        chosen, topic_of, topic_words = "BERTopic", topics_final, bt_words

    # ---------------------------------------------------- step 6: categories
    score, thr, clf_cfg = complaint_scores(sent, Path(a.clean_dir), Path(a.classifier_dir),
                                           cache_dir, use_cache)
    if a.complaint_threshold is not None:            # optional manual override
        thr = clf_cfg["threshold"] = a.complaint_threshold
    is_complaint = score > thr
    log(f"complaint sentences: {is_complaint.sum():,} of {len(sent):,} "
        f"({is_complaint.mean():.1%}) score above the step-3 threshold {thr:.4f}")

    mapping = map_topics(topic_words, a.map_topn, a.map_min_score)
    aspect = np.array([mapping[t][0] if t in mapping else OTHER for t in topic_of])
    category = np.where(is_complaint, aspect, NOT_COMPLAINT)

    raw = sent["sentence_raw"].astype(str)
    cat_names = [c["name"] for c in CATEGORIES] + [NOT_COMPLAINT]      # Other is last of CATEGORIES

    # topic table of the chosen model, with its category
    trows = []
    for t, words in topic_words.items():
        idx = np.flatnonzero(topic_of == t)
        trows.append({
            "topic": t, "n_sentences": len(idx),
            "complaint_share": round(is_complaint[idx].mean(), 4) if len(idx) else np.nan,
            "keywords": " ".join(words),
            "category": mapping[t][0], "map_score": mapping[t][1],
            "matched_keywords": mapping[t][2],
            "examples": " || ".join(representative(emb, idx, raw)),
        })
    ttable = pd.DataFrame(trows)
    ttable.to_csv(out_dir / "topic_to_category.csv", index=False, encoding="utf-8-sig")

    # per-sentence handoff
    topic_label = {t: "_".join([str(t)] + w[:4]) for t, w in topic_words.items()}
    assign = sent[["sentence_id", "review_id", "sent_idx", "split", "stars",
                   "sentence_raw", "sentence"]].copy()
    assign["topic"] = topic_of
    assign["topic_label"] = [topic_label.get(t, "-1_outlier") for t in topic_of]
    # 1 = HDBSCAN left it unclustered and it was given its nearest topic afterwards
    assign["bertopic_outlier"] = (topics == -1).astype(int) if a.model != "lda" else 0
    assign["aspect"] = aspect
    assign["complaint_score"] = np.round(score, 4)
    assign["is_complaint"] = is_complaint.astype(int)
    assign["category"] = category
    assign.to_csv(out_dir / "sentence_assignments.csv", index=False, encoding="utf-8-sig")

    # final category list
    desc = {c["name"]: c["description"] for c in CATEGORIES}
    desc[NOT_COMPLAINT] = NOT_COMPLAINT_DESC
    kw = {c["name"]: " ".join(c["keywords"][:12]) for c in CATEGORIES}
    n_compl = max(is_complaint.sum(), 1)
    crows = []
    for i, name in enumerate(cat_names, start=1):
        m = category == name
        crows.append({
            "category_id": i, "category": name, "description": desc[name],
            "n_topics": sum(v[0] == name for v in mapping.values()),
            "n_sentences": int(m.sum()),
            "share_of_sentences": round(m.mean(), 4),
            "share_of_complaints": (round(m.sum() / n_compl, 4)
                                    if name != NOT_COMPLAINT else np.nan),
            "share_1star": round(m[sent["stars"].values == 1].mean(), 4),
            "share_2star": round(m[sent["stars"].values == 2].mean(), 4),
            "mapping_keywords": kw.get(name, ""),
            "top_terms": distinctive_terms(texts, m),
            # clearest complaints for complaint categories, clearest non-complaints else
            "examples": " || ".join(representative(
                emb, np.flatnonzero(m), raw,
                score=-score if name == NOT_COMPLAINT else score)),
        })
    cats = pd.DataFrame(crows)
    cats.to_csv(out_dir / "categories.csv", index=False, encoding="utf-8-sig")

    write_md(out_dir, a, sent, lda_grid, best_k, lda_top, comparison, chosen, ttable,
             cats, clf_cfg, is_complaint, n_outliers, topics_final)
    log(f"done -> {out_dir}")


# =============================================================== report
def write_md(out_dir, a, sent, lda_grid, best_k, lda_top, comparison, chosen, ttable,
             cats, clf_cfg, is_complaint, n_outliers, topics_final):
    """Write topics_summary.md: every number and table needed for report section 6."""
    f4 = lambda x: f"{x:.4f}"
    lda_row, bt_row = comparison.iloc[0], comparison.iloc[1]
    better = lambda m: "LDA" if lda_row[m] > bt_row[m] else "BERTopic"
    coh_text = (f"On coherence, {better('c_v')} scores higher on c_v ({f4(lda_row.c_v)} LDA "
                f"vs {f4(bt_row.c_v)} BERTopic) and {better('c_npmi')} on c_npmi "
                f"({f4(lda_row.c_npmi)} vs {f4(bt_row.c_npmi)}).")
    L = ["# Steps 5-6 — Topic modeling and complaint categories", "",
         "Generated by `src/06_topics.py`. Tables are in the same folder.", "",
         "## Data", "",
         f"- {len(sent):,} sentences from {sent['review_id'].nunique():,} complaint "
         f"(1-2★) reviews (train + val + test), from `complaint_sentences.csv`.",
         f"- Topic models use aspect tokens: B's `tokens` minus negations, intensifiers, "
         f"pronouns and particles ({len(ASPECT_STOP)} words, `ASPECT_STOP`), and only "
         f"words found in at least 5 sentences. Polarity is handled separately (step 6).", "",
         "## Step 5 — LDA vs BERTopic", "",
         f"**LDA** (gensim, {a.lda_passes} passes, alpha/eta learned, seed {SEED}); K chosen "
         f"by c_v:", "",
         "| K | c_v | c_npmi | diversity |", "|---|---|---|---|"]
    L += [f"| {r.k} | {f4(r.c_v)} | {f4(r.c_npmi)} | {f4(r.diversity)} |"
          for r in lda_grid.itertuples()]
    L += ["", f"Best LDA (K={best_k}) topics:", "", "| topic | keywords |", "|---|---|"]
    L += [f"| {i} | {' '.join(w)} |" for i, w in enumerate(lda_top)]
    L += ["", f"**BERTopic**: `{a.embed_model}` embeddings of `sentence` -> UMAP (5 dims, "
          f"cosine, seed {SEED}) -> scikit-learn HDBSCAN (min cluster size "
          f"{a.min_topic_size}, min samples {a.min_samples}); keywords by c-TF-IDF on the "
          f"same aspect tokens. Raw topics: `bertopic_topics.csv`.", "",
          "Both models scored with the same gensim coherence (top 10 words, same corpus "
          "and dictionary). The third row merges BERTopic down to LDA's K for a "
          "like-for-like comparison:", "",
          "| model | topics | c_v | c_npmi | diversity | unassigned |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r.model} | {r.n_topics} | {f4(r.c_v)} | {f4(r.c_npmi)} | "
          f"{f4(r.diversity)} | {r.unassigned_share:.1%} |" for r in comparison.itertuples()]
    L += ["", f"**Chosen model: {chosen}.** {coh_text} "
          + (CHOICE_REASON if chosen == "BERTopic" else ""), "",
          f"BERTopic outliers (HDBSCAN noise): {n_outliers:,} sentences "
          f"({n_outliers / len(sent):.1%}); {(topics_final == -1).sum():,} remain after "
          f"assigning each to the nearest topic (cosine similarity >= "
          f"{a.outlier_threshold}).", "",
          "## Step 6 — Categories", "",
          "Two decisions per sentence:", "",
          f"1. **Aspect** — each of the {len(ttable)} raw topics is mapped to one category by "
          f"its top {a.map_topn} keywords (rank-weighted match against the keyword lists in "
          f"`CATEGORIES`; best score below {a.map_min_score} -> Other). Full mapping with "
          f"matched keywords and examples: `topic_to_category.csv`.",
          f"2. **Complaint or not** — the step-3 model ({clf_cfg['model'].upper()}, word "
          f"{clf_cfg['word']} + char {clf_cfg['char']}, C={clf_cfg['C']:g}), re-fitted on "
          f"train, threshold {clf_cfg['threshold']:.4f} (tuned on val by D). "
          f"{is_complaint.sum():,} of {len(sent):,} sentences ({is_complaint.mean():.1%}) "
          f"score as complaints and get their topic's category; the rest are "
          f"\"{NOT_COMPLAINT}\".", "",
          "| # | category | topics | sentences | share of all sentences | share of "
          "complaint sentences | share of 1★ sentences | share of 2★ sentences |",
          "|---|---|---|---|---|---|---|---|"]
    for r in cats.itertuples():
        nc = r.category == NOT_COMPLAINT
        sc = "-" if nc else f"{r.share_of_complaints:.1%}"
        L.append(f"| {r.category_id} | {r.category} | {'-' if nc else r.n_topics} | "
                 f"{r.n_sentences:,} | {r.share_of_sentences:.1%} | {sc} | "
                 f"{r.share_1star:.1%} | {r.share_2star:.1%} |")
    L += ["", "### Which raw topics were merged into each category", "",
          "Largest topics first (topic id: first keywords). All topics are in "
          "`topic_to_category.csv`.", ""]
    for name in [c["name"] for c in CATEGORIES]:
        g = ttable[ttable["category"] == name].sort_values("n_sentences", ascending=False)
        shown = "; ".join(f"{r.topic}: {' '.join(r.keywords.split()[:4])}"
                          for r in g.head(8).itertuples())
        more = f"; ... (+{len(g) - 8} more)" if len(g) > 8 else ""
        L.append(f"- **{name}** ({len(g)} topics) — {shown}{more}")
    L += ["", "### Category descriptions (for the labeling sheet)", ""]
    L += [f"- **{r.category}** — {r.description}" for r in cats.itertuples()]
    L += ["", "### Examples", "",
          "Among the 50 sentences closest to the category centre, the 3 clearest "
          "(highest complaint score; lowest for \"Not a complaint\"), at most 150 "
          "characters when possible.", ""]
    for r in cats.itertuples():
        L.append(f"**{r.category}** — distinctive terms: {r.top_terms}")
        L += [f"- {x}" for x in str(r.examples).split(" || ") if x and x != "nan"]
        L.append("")
    L += ["## Using the handoff files", "",
          "- **Step 7 (F)**: sample the 120 labeling rows from `complaint_sentences.csv`, "
          "show `sentence_raw`, and offer the categories in `categories.csv` (names + "
          "descriptions). Join the human labels to `sentence_assignments.csv` on "
          "`sentence_id` and compare with `category`. Do not tune anything in this script "
          "on those 120 labels.",
          "- **Step 8 (F + E)**: `sentence_assignments.csv` has `stars`, `review_id` and "
          "`category` for per-category shares, 1★ vs 2★ and example quotes; "
          "`categories.csv` already has counts, distinctive terms and 3 examples per "
          "category.", "",
          "## Limitations to mention", "",
          "- Complaint vs not is decided by a review-level classifier applied to single "
          "sentences: short or implicit complaints (\"ไม่บอกว่าจองโต๊ะไหนไว้ให้\") can score "
          "below the threshold and become \"Not a complaint\".",
          "- Categories are assigned per topic, so a sentence that mentions two problems "
          "gets the category of the cluster it is closest to.",
          "- Coherence favours LDA; the choice of BERTopic rests on reading the topics and "
          "on the hand-label evaluation (step 7).",
          "- LDA is unstable here: fixing one tokenizer error (57 \"มละ\" tokens, see "
          "`fix_mla`) moved its best K from 5 to 8, while BERTopic's topics and "
          "categories did not change.",
          "- \"Other\" is large: many complaint sentences state dissatisfaction with no "
          "cause (\"ไม่ประทับใจ\", \"คงไม่มาอีก\").",
          "- No topic for wrong or missing orders emerged; \"Order or billing error\" "
          "comes from one payment topic, which also holds sentences that only mention "
          "asking for the bill (\"ขอเช็คบิลดีกว่า\").",
          "- 35% of sentences were HDBSCAN noise and were given their nearest topic "
          "(`bertopic_outlier` = 1); expect more errors among them.", "",
          "## Files", "",
          "| file | what it holds |", "|---|---|",
          "| `categories.csv` | **handoff** — final category list, descriptions, counts, "
          "1★/2★ shares, distinctive terms, examples |",
          "| `sentence_assignments.csv` | **handoff** — every sentence: `sentence_id`, "
          "`review_id`, topic, aspect, complaint score, final `category` |",
          "| `topic_to_category.csv` | every raw topic of the chosen model -> category, with "
          "matched keywords and examples |",
          "| `bertopic_topics.csv` / `lda_topics.csv` | raw topics with keywords and "
          "per-topic coherence |",
          "| `lda_grid.csv` / `model_comparison.csv` | coherence numbers |"]
    (out_dir / "topics_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sentences", default="data/processed/complaint_sentences.csv")
    ap.add_argument("--clean-dir", default="data/processed")
    ap.add_argument("--classifier-dir", default="outputs/classifier")
    ap.add_argument("--out-dir", default="outputs/topics")
    ap.add_argument("--embed-model", default="paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--lda-k", type=int, nargs="+", default=[5, 8, 10, 12, 15, 20, 25])
    ap.add_argument("--lda-passes", type=int, default=10)
    ap.add_argument("--min-topic-size", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--outlier-threshold", type=float, default=0.3)
    ap.add_argument("--map-topn", type=int, default=10,
                    help="topic keywords used to pick its category")
    ap.add_argument("--map-min-score", type=int, default=8,
                    help="below this keyword score a topic goes to Other")
    ap.add_argument("--model", choices=["bertopic", "lda"], default="bertopic",
                    help="topic model used for the categories")
    ap.add_argument("--complaint-threshold", type=float, default=None,
                    help="override the step-3 threshold (default: read from "
                         "threshold_tuning.csv; do NOT tune it on the hand labels)")
    ap.add_argument("--no-cache", action="store_true", help="recompute embeddings and scores")
    main(ap.parse_args())
