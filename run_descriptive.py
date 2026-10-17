"""Descriptive / unsupervised analyses (Sections 2.1, 3.1-3.4) -> results/descriptive.json"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mcare import descriptive as D  # noqa: E402
from mcare.data import DATAVERSE_DOI, DATAVERSE_VERSION, ROOT, dataset_inventory, load_tables, verify_checksums  # noqa: E402
from mcare.frames import build_frames  # noqa: E402
from mcare.taxonomy import CATEGORIES  # noqa: E402

OUT = ROOT / "results" / "descriptive"
OUT.mkdir(parents=True, exist_ok=True)

t = load_tables()
mapping, q, a = build_frames(t)
res = {"source": {"doi": DATAVERSE_DOI, "version": DATAVERSE_VERSION, "checksums": verify_checksums()},
       "inventory": dataset_inventory(t)}

# Category distribution at intent, question and response level
res["category_counts"] = {
    "intent_records": mapping.category.value_counts().reindex(CATEGORIES).astype(int).to_dict(),
    "questions": q.category.value_counts().reindex(CATEGORIES).astype(int).to_dict(),
    "responses": a.category.value_counts().reindex(CATEGORIES).astype(int).to_dict(),
}
res["category_pct_intent_records"] = {k: 100 * v / len(mapping) for k, v in res["category_counts"]["intent_records"].items()}
res["n_rule_overrides"] = int((mapping.rule != "section_range").sum())
res["n_leakage_groups"] = int(mapping.leakage_group.nunique())

texts = a.text.tolist()
res["length_readability"] = D.length_and_readability(a)
uni, bi = D.term_frequencies(texts)
res["top_terms"], res["top_bigrams"] = uni, bi
res["surface_bigrams"] = D.raw_bigrams(texts)
res["tfidf_top20"] = D.tfidf_top_terms(texts)

sent_df, sent_sum = D.sentiment(texts)
sent_df.insert(0, "response_id", a.row_id.values)
sent_df.to_csv(OUT / "sentiment_scores.csv", index=False)
res["sentiment"] = sent_sum
res["sentiment_by_category_domain_mean"] = (
    pd.DataFrame({"cat": a.category, "s": sent_df.compound_domain}).groupby("cat").s.mean().reindex(CATEGORIES).to_dict())

sweep, models, dic, corpus, docs = D.lda_sweep(texts)
sweep.to_csv(OUT / "lda_sweep.csv", index=False)
best_k = int(sweep.sort_values("npmi", ascending=False).iloc[0].k)
res["lda_sweep"] = sweep.to_dict(orient="records")
res["lda_best_k_by_npmi"] = best_k
res["lda_stability"] = D.lda_seed_stability(texts, best_k)
lda = models[best_k]
dom = D.dominant_topics(lda, corpus)
res["lda_topics"] = [{"topic": i + 1, "top_words": [w for w, _ in lda.show_topic(i, topn=10)],
                      "pct_responses_dominant": float((dom == i).mean() * 100)} for i in range(best_k)]
# Topic-category alignment (normalised mutual information)
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402
res["lda_category_nmi"] = float(normalized_mutual_info_score(a.category, dom))
pd.crosstab(a.category, dom + 1).to_csv(OUT / "lda_topic_by_category.csv")
pd.DataFrame({"response_id": a.row_id, "dominant_topic": dom + 1}).to_csv(OUT / "lda_dominant_topic.csv", index=False)

G, P, kg = D.category_term_graph(texts, a.category.tolist())
res["knowledge_graph"] = kg
import networkx as nx  # noqa: E402
nx.write_graphml(G, OUT / "category_term_graph.graphml")
nx.write_graphml(nx.Graph((u, v, {"weight": d["weight"]}) for u, v, d in P.edges(data=True)), OUT / "category_projection.graphml")

(ROOT / "results" / "descriptive.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: res[k] for k in ("inventory", "length_readability", "sentiment", "lda_best_k_by_npmi", "lda_stability", "lda_category_nmi")}, indent=1))
print(sweep)
for tpc in res["lda_topics"]:
    print(tpc)
print(kg)
print(res["tfidf_top20"][:10], res["top_bigrams"][:5], res["surface_bigrams"])
