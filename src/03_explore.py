"""Step 2: exploration charts.  Owner: Pang
Reads  data/processed/{train,val,test,three_star}_clean.csv   (from 02_clean.py)
Writes outputs/figures/*.png  +  outputs/figures/exploration_summary.md
Run:   python src/03_explore.py

Which data each figure uses
    fig01 star distribution      all four files  (describes the whole dataset)
    fig02 review length          all four files  (length by star) + train (by class)
    fig03 top words              train only
    fig04 distinctive words      train only
    fig05/06 word clouds         train only
Word-level figures use TRAIN ONLY on purpose: val and test stay unseen so that
nothing we notice here can leak into how the later steps are tuned.

Thai text needs a Thai font. The script looks for one (Windows, macOS, Linux);
if your charts come out as empty boxes, pass your own: --font "C:/Windows/Fonts/leelawui.ttf"
"""
import argparse
import glob
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window needed, just write PNG files
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

SPLITS = ["train", "val", "test", "three_star"]
DPI = 150
# complaint = warm/red, not a complaint = cool/blue, 3 stars = grey
C_COMPLAINT, C_NOT, C_NEUTRAL = "#C0392B", "#2E6DA4", "#9E9E9E"

FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\leelawui.ttf", r"C:\Windows\Fonts\Leelawadee.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
    "/System/Library/Fonts/Thonburi.ttc",
    # Linux
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
]
FONT_GLOBS = [
    "/usr/share/fonts/**/*Thai*.ttf", "/usr/share/fonts/**/Sarabun*.ttf",
    os.path.expanduser("~/fonts/*.ttf"), os.path.expanduser("~/.fonts/*.ttf"),
]


# --------------------------------------------------------------------------- font
def find_thai_font(explicit: str | None) -> str | None:
    """Return a path to a font that can draw Thai, or None."""
    if explicit:
        if Path(explicit).exists():
            return explicit
        sys.exit(f"--font {explicit} does not exist")
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    for pattern in FONT_GLOBS:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    return None


def use_font(path: str | None) -> str | None:
    """Make matplotlib draw Thai with this font. Returns the family name."""
    if not path:
        print("[warn] no Thai font found - Thai text will show as boxes.\n"
              "       pass one with --font, e.g. --font C:/Windows/Fonts/leelawui.ttf")
        return None
    try:
        font_manager.fontManager.addfont(path)
        name = font_manager.FontProperties(fname=path).get_name()
    except Exception as e:                                   # .ttc and odd fonts
        print(f"[warn] matplotlib could not load {path} ({e}); using the default font")
        return None
    # Thai fonts often lack ★ → ≥ , so fall back to a font that has them
    plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[font] {name}  ({path})")
    return name


# --------------------------------------------------------------------------- data
def load(proc: Path) -> dict:
    """Read the *_clean.csv files that exist. Keys: train, val, test, three_star."""
    data = {}
    for name in SPLITS:
        f = proc / f"{name}_clean.csv"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")          # 02_clean.py writes a BOM
        missing = {"review_id", "stars", "tokens", "n_tokens"} - set(df.columns)
        if missing:
            sys.exit(f"{f} is missing columns: {sorted(missing)}")
        df["tokens"] = df["tokens"].fillna("")
        data[name] = df
        print(f"[ok] {name:<11} {len(df):>7,} rows")
    if "train" not in data:
        sys.exit("train_clean.csv is required - run src/02_clean.py first")
    return data


def token_lists(df: pd.DataFrame) -> list:
    return [t.split() for t in df["tokens"]]


def doc_freq(docs: list) -> Counter:
    """How many reviews contain each word (not how many times in total)."""
    c = Counter()
    for toks in docs:
        c.update(set(toks))
    return c


# ------------------------------------------------------------------------ figures
def fig_star_distribution(data: dict, out: Path):
    counts = Counter()
    for df in data.values():
        counts.update(df["stars"].tolist())
    stars = [1, 2, 3, 4, 5]
    vals = [counts.get(s, 0) for s in stars]
    total = sum(vals)
    colors = [C_COMPLAINT, C_COMPLAINT, C_NEUTRAL, C_NOT, C_NOT]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([str(s) for s in stars], vals, color=colors, width=0.68)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v / total:.1%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Star rating distribution (whole dataset)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_COMPLAINT, C_NEUTRAL, C_NOT)]
    ax.legend(handles, ["1-2★ = complaint", "3★ = held out", "4-5★ = not a complaint"],
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig01_star_distribution.png", dpi=DPI)
    plt.close(fig)
    return counts


def fig_review_length(data: dict, out: Path):
    train = data["train"]
    comp = train.loc[train["label"] == 1, "n_tokens"]
    noncomp = train.loc[train["label"] == 0, "n_tokens"]
    cap = int(pd.concat([comp, noncomp]).quantile(0.99))     # ignore the long tail

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # outline + light fill, so the two classes stay readable where they overlap.
    # Reviews longer than the cap are DROPPED from this panel, not clipped into it:
    # clipping would pile the whole long tail into one misleading bar at the edge.
    style = dict(bins=40, range=(0, cap), density=True, histtype="stepfilled",
                 alpha=0.30, lw=1.8)
    ax1.hist(noncomp[noncomp <= cap], color=C_NOT, edgecolor=C_NOT, **style,
             label=f"not a complaint (median {noncomp.median():.0f})")
    ax1.hist(comp[comp <= cap], color=C_COMPLAINT, edgecolor=C_COMPLAINT, **style,
             label=f"complaint (median {comp.median():.0f})")
    ax1.set_xlabel(f"Tokens per review (train; longest 1% above {cap} not shown)")
    ax1.set_ylabel("Share of reviews")
    ax1.set_title("Review length by class")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    by_star, labels, colors = [], [], []
    for s in [1, 2, 3, 4, 5]:
        vals = pd.concat([df.loc[df["stars"] == s, "n_tokens"] for df in data.values()])
        if len(vals):
            by_star.append(vals)          # showfliers=False already hides the tail
            labels.append(f"{s}★")
            colors.append(C_COMPLAINT if s <= 2 else C_NEUTRAL if s == 3 else C_NOT)
    # set tick labels separately: the boxplot 'labels' kwarg was renamed in mpl 3.9
    bp = ax2.boxplot(by_star, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black"))
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax2.set_xlabel("Star rating")
    ax2.set_ylabel("Tokens per review")
    ax2.set_title("Review length by star rating (whole dataset)")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "fig02_review_length.png", dpi=DPI)
    plt.close(fig)
    return {"complaint_median": float(comp.median()), "not_median": float(noncomp.median()),
            "cap": cap}


def fig_top_words(df_comp, df_non, out: Path, top_n: int):
    dfc, dfn = doc_freq(df_comp), doc_freq(df_non)
    nc, nn = len(df_comp), len(df_non)

    # sharex: equal scales, so the two panels can honestly be compared
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True)
    for ax, dfreq, n, title, color in [
        (axes[0], dfc, nc, f"Complaints (1-2★, n={nc:,})", C_COMPLAINT),
        (axes[1], dfn, nn, f"Not a complaint (4-5★, n={nn:,})", C_NOT),
    ]:
        words = dfreq.most_common(top_n)[::-1]
        ax.barh([w for w, _ in words], [c / n * 100 for _, c in words],
                color=color, alpha=0.85)
        ax.set_xlabel("% of reviews containing the word")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle(f"Top {top_n} words by share of reviews (train)", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "fig03_top_words.png", dpi=DPI)
    plt.close(fig)
    return dfc, dfn, nc, nn


def fig_distinctive_words(dfc, dfn, nc, nn, out: Path, top_n: int, min_count: int):
    """Words that lean strongly to one class, by smoothed log-odds ratio."""
    vocab = {w for w in set(dfc) | set(dfn) if dfc[w] + dfn[w] >= min_count}
    if not vocab:
        print(f"[warn] no word reaches --min-count {min_count}; skipping fig04")
        return []
    a, v = 1.0, len(vocab)
    scored = []
    for w in vocab:
        p1 = (dfc[w] + a) / (nc + a * v)
        p0 = (dfn[w] + a) / (nn + a * v)
        scored.append((w, math.log(p1 / p0), dfc[w] + dfn[w]))
    scored.sort(key=lambda t: t[1])
    low, high = scored[:top_n], scored[-top_n:]
    rows = high[::-1] + low                                  # complaint side first

    words = [w for w, _, _ in rows][::-1]
    vals = [s for _, s, _ in rows][::-1]
    colors = [C_COMPLAINT if s > 0 else C_NOT for s in vals]

    fig, ax = plt.subplots(figsize=(8, 0.30 * len(words) + 1.6))
    ax.barh(words, vals, color=colors, alpha=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("← more typical of 4-5★       log-odds ratio       more typical of 1-2★ →")
    ax.set_title(f"Most distinctive words (train, word in ≥{min_count} reviews)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig04_distinctive_words.png", dpi=DPI)
    plt.close(fig)
    return high[::-1]


def shade_picker(cmap_name: str, lo=0.45, hi=0.95, seed=42):
    """Colour words from the darker half of a colormap.
    wordcloud's default picks anywhere in the map, and the pale end of Reds/Blues
    is nearly white - large words then become unreadable on a white background."""
    import random
    cmap = matplotlib.colormaps[cmap_name]
    rng = random.Random(seed)

    def color(*args, **kwargs):
        r, g, b, _ = cmap(rng.uniform(lo, hi))
        return int(r * 255), int(g * 255), int(b * 255)
    return color


def fig_wordclouds(dfc, dfn, out: Path, font: str | None, max_words: int):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[warn] wordcloud is not installed - skipping fig05/fig06 "
              "(pip install wordcloud)")
        return False
    if not font:
        print("[warn] no Thai font - skipping the word clouds (they would be boxes)")
        return False
    for freq, name, title, cmap in [
        (dfc, "fig05_wordcloud_complaint.png", "Complaints (1-2★)", "Reds"),
        (dfn, "fig06_wordcloud_not_complaint.png", "Not a complaint (4-5★)", "Blues"),
    ]:
        # max_font_size + margin: without them the top word is scaled to fill the
        # canvas and Thai vowel marks get clipped at the edges
        wc = WordCloud(font_path=font, width=1200, height=650, background_color="white",
                       max_words=max_words, color_func=shade_picker(cmap),
                       prefer_horizontal=0.95, random_state=42,
                       max_font_size=120, margin=6, relative_scaling=0.5,
                       ).generate_from_frequencies(dict(freq))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / name, dpi=DPI)
        plt.close(fig)
    return True


# ------------------------------------------------------------------------- report
def write_summary(out: Path, data: dict, stars: Counter, length: dict,
                  top_comp, top_non, distinctive, nc, nn, empty, clouds):
    total = sum(stars.values())
    lines = [
        "# Exploration summary (step 2)",
        "",
        "Generated by `src/03_explore.py`. Figures are in the same folder.",
        "",
        "## Dataset",
        "",
        "| split | rows | complaints | share |",
        "|---|---|---|---|",
    ]
    for name, df in data.items():
        if "label" in df and df["label"].notna().any():
            c = int(df["label"].sum())
            lines.append(f"| {name} | {len(df):,} | {c:,} | {c / len(df):.1%} |")
        else:
            lines.append(f"| {name} | {len(df):,} | - | - |")
    lines += [
        "",
        f"Whole dataset: {total:,} reviews. "
        f"1-2★ {stars[1] + stars[2]:,} ({(stars[1] + stars[2]) / total:.1%}), "
        f"3★ {stars[3]:,} ({stars[3] / total:.1%}), "
        f"4-5★ {stars[4] + stars[5]:,} ({(stars[4] + stars[5]) / total:.1%}).",
        "",
        f"Complaints are the minority class, so the classifier (step 3) should use "
        f"class weighting and report macro-F1, not accuracy.",
        "",
        f"Reviews that became empty after cleaning: {empty:,} in train "
        f"(kept, so review_id and the split sizes do not change).",
        "",
        "## Review length",
        "",
        f"- Median tokens, complaints: **{length['complaint_median']:.0f}**",
        f"- Median tokens, not a complaint: **{length['not_median']:.0f}**",
        f"- Histogram capped at the 99th percentile ({length['cap']} tokens).",
        "",
        "## Words",
        "",
        f"Train only: {nc:,} complaint reviews, {nn:,} non-complaint reviews.",
        "",
        "Most distinctive complaint words (log-odds ratio):",
        "",
        "| word | log-odds | reviews |",
        "|---|---|---|",
    ]
    for w, s, n in distinctive[:15]:
        lines.append(f"| {w} | {s:+.2f} | {n:,} |")
    lines += [
        "",
        "Most frequent complaint words (share of complaint reviews):",
        "",
        "| word | share |",
        "|---|---|",
    ]
    for w, c in top_comp.most_common(15):
        lines.append(f"| {w} | {c / nc:.1%} |")
    lines += [
        "",
        "## Figures",
        "",
        "| file | what it shows |",
        "|---|---|",
        "| `fig01_star_distribution.png` | star ratings across the whole dataset, grouped into the two classes |",
        "| `fig02_review_length.png` | length by class (train) and by star rating (all data) |",
        "| `fig03_top_words.png` | most common words in each class (train) |",
        "| `fig04_distinctive_words.png` | words that lean most to one class (train) |",
    ]
    if clouds:
        lines += [
            "| `fig05_wordcloud_complaint.png` | word cloud, complaints |",
            "| `fig06_wordcloud_not_complaint.png` | word cloud, non-complaints |",
        ]
    (out / "exploration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Exploration charts for step 2")
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--font", default=None, help="path to a Thai .ttf font")
    ap.add_argument("--top-n", type=int, default=20, help="fig03: words per class")
    ap.add_argument("--top-n-distinct", type=int, default=12,
                    help="fig04: words per side")
    ap.add_argument("--min-count", type=int, default=50,
                    help="fig04: ignore words appearing in fewer reviews than this")
    ap.add_argument("--max-words", type=int, default=120, help="word cloud size")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    font = find_thai_font(a.font)
    use_font(font)
    print(f"[info] {platform.system()}, writing to {out}/")

    data = load(Path(a.proc_dir))
    train = data["train"]
    empty = int((train["n_tokens"] == 0).sum())

    stars = fig_star_distribution(data, out)
    length = fig_review_length(data, out)
    docs_c = token_lists(train[train["label"] == 1])
    docs_n = token_lists(train[train["label"] == 0])
    top_comp, top_non, nc, nn = fig_top_words(docs_c, docs_n, out, a.top_n)
    distinctive = fig_distinctive_words(top_comp, top_non, nc, nn, out,
                                        a.top_n_distinct, a.min_count)
    clouds = fig_wordclouds(top_comp, top_non, out, font, a.max_words)
    write_summary(out, data, stars, length, top_comp, top_non, distinctive,
                  nc, nn, empty, clouds)

    made = sorted(p.name for p in out.glob("*.png"))
    print(f"\n[done] {len(made)} figures + exploration_summary.md in {out}/")
    for m in made:
        print(f"       {m}")


if __name__ == "__main__":
    main()
