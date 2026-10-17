
import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from mcare.data import ROOT  # noqa: E402
from mcare.taxonomy import CATEGORIES  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": SURF})
PRETTY = {c: c.replace("_", " ") for c in CATEGORIES}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def hbar(ax, labels, values, color=BLUE, fmt="{:.0f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62, edgecolor=SURF, linewidth=1)
    ax.set_yticks(y, labels)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7.5, color=INK2)


# ------------------------------------------------------------------ diagrams
def box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=color, lw=1))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.4); ax.axis("off")
    box(ax, 0.05, 1.55, 2.0, 1.25, "MOTHER V4\n(Harvard Dataverse)\n503 intents\n698 questions, 701 answers\nMD5-verified", "#f4f4f2", INK2, 5.8)
    box(ax, 2.45, 1.55, 2.0, 1.25, "Preprocessing\nNFKC, 120-term\nabbreviation lexicon,\nselective stop-words", fs=6.3)
    box(ax, 4.85, 3.05, 2.1, 1.05, "Descriptive analysis\nlength, readability,\nterms, TF-IDF", fs=6.3)
    box(ax, 4.85, 1.7, 2.1, 0.95, "Sentiment\nVADER + domain\nre-weighting", fs=6.3)
    box(ax, 4.85, 0.3, 2.1, 1.05, "Topic model\nLDA, k = 4-12\nNPMI / C_v", fs=6.3)
    box(ax, 7.35, 3.05, 2.0, 1.05, "Category-term\nknowledge graph\n(descriptive)", fs=6.3)
    box(ax, 7.35, 0.95, 2.0, 1.55, "Author-derived\n12-category labels\n(rules on tag and\ncontext_set only)", "#fdf0ea", ORANGE, 6.3)
    box(ax, 9.75, 0.95, 1.8, 1.55, "Leakage-free\nbenchmark:\nGraphSAGE vs\nbaselines and\nablations\n(text + sentiment\nas inputs)", "#e8f7f1", AQUA, 6.3)
    arrow(ax, 2.05, 2.17, 2.45, 2.17)
    for yy in (3.57, 2.17, 0.82):
        arrow(ax, 4.45, 2.17, 4.85, yy)
    arrow(ax, 6.95, 3.57, 7.35, 3.57)
    arrow(ax, 9.35, 1.72, 9.75, 1.72)
    save(fig, "fig01_pipeline")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(-0.5, 4.2); ax.axis("off")
    rng = np.random.default_rng(3)
    qx, tx = 0.6, 2.4
    qs = [(qx, y) for y in np.linspace(0.5, 3.5, 6)]
    ts = [(tx, y) for y in np.linspace(0.3, 3.7, 8)]
    for i, (x, y) in enumerate(qs):
        held = i >= 4
        for j in rng.choice(8, 3, replace=False):
            ax.annotate("", xy=(x + 0.13, y), xytext=(ts[j][0] - 0.11, ts[j][1]),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, alpha=0.7, mutation_scale=7))
            if not held:
                ax.annotate("", xy=(ts[j][0] - 0.11, ts[j][1]), xytext=(x + 0.13, y),
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.6, alpha=0.6, mutation_scale=7,
                                            connectionstyle="arc3,rad=0.15"))
        ax.add_patch(plt.Circle((x, y), 0.12, fc=ORANGE if held else BLUE, ec=SURF, lw=1.5, zorder=3))
    for (x, y) in ts:
        ax.add_patch(plt.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, fc=AQUA, ec=SURF, lw=1.5, zorder=3))
    ax.text(qx, 3.8, "question nodes", ha="center", fontsize=7)
    ax.text(tx, 3.95, "term nodes", ha="center", fontsize=7)
    ax.scatter([0.15, 0.15], [-0.05, -0.35], c=[BLUE, ORANGE], s=25)
    ax.text(0.3, -0.05, "training question (sends and receives)", fontsize=6.3, va="center", color=INK2)
    ax.text(0.3, -0.35, "held-out question (receives only, never sends)", fontsize=6.3, va="center", color=INK2)
    x0, w, gap = 3.4, 1.7, 1.95
    steps = [("Node features\nSVD(TF-IDF) 128\n+ sentiment 1\n+ log #terms 1\n+ node-type flag", "#f4f4f2", INK2),
             ("SAGEConv 1\n131 -> h\nReLU, dropout", "#eef4fc", BLUE),
             ("SAGEConv 2\nh -> h\nReLU, dropout", "#eef4fc", BLUE),
             ("Linear\nh -> 12\nsoftmax", "#e8f7f1", AQUA)]
    for i, (st, fc, ec) in enumerate(steps):
        box(ax, x0 + i * gap, 1.3, w, 1.5, st, fc, ec, 6.5)
        if i:
            arrow(ax, x0 + i * gap - (gap - w), 2.05, x0 + i * gap, 2.05)
    ax.text(x0, 0.95, "h in {128, 256}; aggregation in {mean, max}; dropout in {0.3, 0.5};\n"
                      "AdamW (lr 5e-3, weight decay 1e-4); class-weighted cross-entropy;\n"
                      "early stopping on validation macro-F1 (patience 30, max 200 epochs)",
            fontsize=6.3, color=INK2, va="top")
    save(fig, "fig02_graphsage_architecture")


def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 3.1); ax.axis("off")
    box(ax, 0.05, 0.85, 2.2, 1.35, "Leakage groups\n(intents sharing a tag,\nquestion or answer text\nare merged)", fs=6.3)
    box(ax, 2.65, 0.85, 2.3, 1.35, "5 repetitions of\nStratifiedGroupKFold\n(k = 5; seeds\n1000-1004)", fs=6.3)
    box(ax, 5.35, 1.8, 2.7, 1.1, "Training part:\ninner grouped split\n7/8 train, 1/8 validation", fs=6.3)
    box(ax, 5.35, 0.2, 2.7, 1.1, "Test fold (about 20%):\nlabels never used for graph,\nfeatures or tuning", "#fdf0ea", ORANGE, 6.3)
    box(ax, 8.45, 0.85, 3.1, 1.35, "25 held-out folds per model:\nmean, SD, corrected\nresampled-t 95% CI;\npooled out-of-fold confusion matrix", "#e8f7f1", AQUA, 6.3)
    arrow(ax, 2.25, 1.52, 2.65, 1.52); arrow(ax, 4.95, 1.62, 5.35, 2.3); arrow(ax, 4.95, 1.42, 5.35, 0.75)
    arrow(ax, 8.05, 2.3, 8.45, 1.65); arrow(ax, 8.05, 0.75, 8.45, 1.4)
    save(fig, "fig03_evaluation_protocol")


# ------------------------------------------------------------------ descriptive
def descriptive_figs():
    d = json.loads((RES / "descriptive.json").read_text())
    cc = d["category_counts"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    order = sorted(CATEGORIES, key=lambda c: -cc["intent_records"][c])
    y = np.arange(len(order))[::-1]
    w = 0.27
    for k, (key, col, lab) in enumerate([("intent_records", BLUE, "intent records (n=503)"),
                                         ("questions", ORANGE, "questions (n=698)"),
                                         ("responses", AQUA, "responses (n=701)")]):
        ax.barh(y + (1 - k) * w, [cc[key][c] for c in order], height=w, color=col, label=lab,
                edgecolor=SURF, linewidth=0.8)
    ax.set_yticks(y, [PRETTY[c] for c in order])
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("count"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig04_category_distribution")

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    terms = d["top_terms"]
    hbar(ax, [t for t, _ in terms], [v for _, v in terms])
    ax.set_xlabel("occurrences in 701 expert responses")
    save(fig, "fig05_term_frequency")

    from wordcloud import WordCloud
    from collections import Counter
    from mcare.preprocess import analysis_tokens
    a = pd.read_csv(ROOT / "data" / "derived" / "responses_labelled.csv")
    cnt = Counter(t for x in a.text for t in analysis_tokens(x))
    wc = WordCloud(width=1600, height=800, background_color="white", colormap="Blues",
                   random_state=42, max_words=150).generate_from_frequencies(cnt)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    save(fig, "fig06_wordcloud")

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    tf = d["tfidf_top20"]
    hbar(ax, [t for t, _ in tf], [v for _, v in tf], fmt="{:.3f}")
    ax.set_xlabel("mean TF-IDF weight")
    save(fig, "fig07_tfidf_top20")

    s = pd.read_csv(RES / "descriptive" / "sentiment_scores.csv")
    from scipy.stats import gaussian_kde
    xs = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for col, c, lab in [("compound_default", BLUE, "VADER default lexicon"),
                        ("compound_domain", ORANGE, "VADER + domain re-weighting")]:
        ax.plot(xs, gaussian_kde(s[col], bw_method=0.15)(xs), color=c, lw=2, label=lab)
    ax.axvspan(-0.05, 0.05, color=GRID, alpha=0.8, lw=0)
    ax.text(0.07, 0.12, "neutral band (|score| <= 0.05)", ha="left", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlabel("VADER compound score"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(0, None)
    save(fig, "fig08_sentiment_kde")

    sw = pd.DataFrame(d["lda_sweep"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, col, lab, c in [(axs[0], "npmi", "mean NPMI (top-10 words)", BLUE), (axs[1], "c_v", "C_v coherence", ORANGE)]:
        ax.plot(sw.k, sw[col], "-o", color=c, lw=2, ms=5, mec=SURF)
        ax.set_xlabel("number of topics k"); ax.set_ylabel(lab)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        kb = d["lda_best_k_by_npmi"]
        ax.plot([kb], [sw.loc[sw.k == kb, col].iloc[0]], "o", ms=10, mfc="none", mec=INK)
    axs[0].axhline(0, color=INK2, lw=0.6)
    save(fig, "fig09_lda_coherence")

    tp = d["lda_topics"]
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(tp) + 0.6))
    labels = [f"Topic {t['topic']}: " + ", ".join(t["top_words"][:6]) for t in tp]
    hbar(ax, labels, [t["pct_responses_dominant"] for t in tp], fmt="{:.1f}%")
    ax.set_xlabel("% of responses for which the topic is dominant")
    save(fig, "fig10_lda_topics")

    kg = d["knowledge_graph"]
    P = nx.Graph()
    P.add_nodes_from(CATEGORIES)
    for u, v, w in kg["similarity_edges_list"]:
        P.add_edge(u, v, weight=w)
    pos = nx.circular_layout(sorted(P.nodes, key=CATEGORIES.index))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ws = np.array([P[u][v]["weight"] for u, v in P.edges])
    nx.draw_networkx_edges(P, pos, ax=ax, width=1 + 12 * (ws - ws.min()), edge_color=INK2, alpha=0.6)
    sizes = [120 + 6 * d["category_counts"]["intent_records"][n] for n in P.nodes]
    nx.draw_networkx_nodes(P, pos, ax=ax, node_color=BLUE, node_size=sizes, edgecolors=SURF, linewidths=2)
    for n, (x, y) in pos.items():
        ax.text(x * 1.33, y * 1.24, "\n".join(textwrap.wrap(PRETTY[n], 16)), ha="center", va="center", fontsize=7.5)
    ax.text(0, -1.42, f"edge width proportional to cosine similarity ({ws.min():.2f}-{ws.max():.2f}); "
                      "node area proportional to number of intent records", ha="center", fontsize=6.5, color=INK2)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.4); ax.axis("off")
    save(fig, "fig11_category_similarity_graph")


# ------------------------------------------------------------------ models
def model_figs(task="question"):
    out = RES / task
    summ = pd.read_csv(out / "summary.csv").sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(5.0, 0.36 * len(summ) + 0.9))
    y = np.arange(len(summ))
    cols = [ORANGE if m == "GraphSAGE (full)" else (AQUA if m.startswith("GraphSAGE without") or m.startswith("MLP") else BLUE)
            for m in summ.model]
    ax.errorbar(summ.macro_f1_mean, y, xerr=[summ.macro_f1_mean - summ.macro_f1_ci_lo, summ.macro_f1_ci_hi - summ.macro_f1_mean],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.scatter(summ.macro_f1_mean, y, c=cols, s=60, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.set_yticks(y, summ.model)
    for yi, m in zip(y, summ.macro_f1_mean):
        ax.text(m, yi + 0.28, f"{m:.3f}", ha="center", fontsize=7, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("macro-F1 (mean of 25 held-out folds, 95% CI)")
    save(fig, f"fig12_model_comparison_{task}")

    cm = pd.read_csv(out / "confusion_matrix_graphsage_rep0.csv", index_col=0).loc[CATEGORIES, CATEGORIES].to_numpy()
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    lab = [PRETTY[c] for c in CATEGORIES]
    ax.set_xticks(range(len(lab)), lab, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    ax.set_xlabel("predicted category"); ax.set_ylabel("true category")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("row-normalised recall", fontsize=7)
    save(fig, f"fig13_confusion_matrix_{task}")

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for v, c, lab in [("full", ORANGE, "GraphSAGE (full)"), ("no_graph", BLUE, "MLP, same features (no graph)"),
                      ("no_lexicon", AQUA, "GraphSAGE without domain lexicon")]:
        curves = []
        for f in range(5):
            lg = pd.read_csv(out / "training_logs" / f"{v}_r0_f{f}.csv")
            curves.append(lg.set_index("epoch").val_macro_f1)
        M = pd.concat(curves, axis=1)
        n_alive = M.notna().sum(1)
        M = M.ffill()
        m = M.mean(1)[n_alive >= 1]
        ax.plot(m.index, m.values, color=c, lw=2, label=lab)
    ax.set_xlabel("training epoch"); ax.set_ylabel("validation macro-F1")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, f"fig14_learning_curves_{task}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    if a.only in ("all", "diagrams"):
        fig_pipeline(); fig_architecture(); fig_protocol()
    if a.only in ("all", "descriptive"):
        descriptive_figs()
    if a.only in ("all", "models"):
        for task in ("question", "response"):
            if (RES / task / "summary.csv").exists():
                model_figs(task)
    print("figures written to", FIG)
