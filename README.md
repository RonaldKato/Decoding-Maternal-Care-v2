# Decoding Maternal Care - reproducible analysis code (v2.0.0)
<p align="center">
  <img
    src="https://github.com/user-attachments/assets/5086fa15-d346-449b-b734-43d6aee1309e"
    alt="fig01_pipeline"
    width="100%"
  />
</p>


This repository reproduces **every number, table and figure** in the revised manuscript
*"Decoding Maternal Care: A Graph Neural Network Approach for Intent Classification and
Linguistic Analysis of Expert-Authored Maternal Health Dialogue in Low-Resource Uganda"*
from the public MOTHER dataset.

> **Version 2.0.0 replaces the earlier release in full.** The v1 code base contained
> simulation modules (hard-coded classification metrics, ablation scores derived by fixed
> multipliers, a hard-coded LLM comparison table, and randomly generated clinical-validation,
> fairness and robustness scores). Those modules have been deleted, and the corresponding
> results have been removed from the manuscript. See `CHANGELOG.md`.

## 1. Data (frozen inputs)

| item | value |
|---|---|
| Dataset | MOTHER: A Dataset For Maternal Online Technology for Health Care (Eyobu et al.) |
| DOI | https://doi.org/10.7910/DVN/EZLCH3 |
| Version | Harvard Dataverse **V4** (released 2024-12-11T21:12:48Z), licence CC0 1.0 |
| Files used | `intents.csv` (503 rows), `patterns.csv` (698 rows), `responses.csv` (701 rows) |
| Integrity | MD5 checked against the Dataverse-published checksums on every run (`data/MD5SUMS`, `data/SHA256SUMS`) |

The raw files are redistributed unchanged in `data/raw/` (CC0). Row-level identifiers are the
`id` / `intent_id` columns of the Dataverse files and are carried through every output.

**Labels.** MOTHER contains 503 fine-grained intent records (495 distinct tags) and **no coarse
intent label**. The 12 care-continuum categories used for the supervised benchmark are
*author-derived*: `mcare/taxonomy.py` assigns each record by deterministic rules that read only
the record id, `tag` and `context_set` (section ranges plus 56 documented record-level
overrides). The full mapping is in `data/derived/intent_category_mapping.csv`, with the rule
that produced each label. These labels were **not** assigned by clinicians;
`scripts/make_annotation_sheet.py` exports a blinded sheet for independent clinical coding and
scores agreement (Cohen's κ) once completed.

## 2. Leakage-free evaluation protocol

* Unit: one user question (`patterns.csv`, n = 698); secondary task: one expert answer
  (`responses.csv`, n = 701).
* Grouping: all rows of an intent record, and of any records sharing a normalised tag,
  question or answer text, are kept in the same fold (`leakage_groups`).
* Splits: 5 repetitions × 5-fold `StratifiedGroupKFold` (seeds 1000–1004); inside each training
  part a grouped 1/8 validation split is used for model selection and early stopping. Every
  assignment is saved in `results/<task>/splits.csv`.
* Graph: question and term nodes only (no label nodes). TF-IDF vocabulary, IDF and SVD are fit
  on training questions only. Held-out questions receive messages from their own terms but never
  send messages, so each held-out prediction is inductive. Tested in `tests/test_leakage.py`.
* Reporting: mean ± SD over the 25 held-out folds, Nadeau–Bengio corrected resampled-t 95% CIs,
  paired corrected-t tests against GraphSAGE, pooled out-of-fold confusion matrices.

## 3. Reproduce

```bash
python -m pip install -r requirements.txt     # Python 3.13, CPU is sufficient
make test           # checksum, grouping and leakage tests
make descriptive    # Sections 3.1-3.4  -> results/descriptive.json
make experiments    # Section 3.5       -> results/question, results/response (~1 h on 2 CPU cores)
make figures        # all manuscript figures -> figures/
```

## 4. Outputs

```
results/descriptive.json                    corpus statistics, sentiment, LDA sweep, knowledge graph
results/<task>/splits.csv                   repeat, fold, row_id, role (train/val/test)
results/<task>/predictions.csv              every held-out prediction of every model
results/<task>/fold_metrics.csv             per-fold metrics (25 folds x 11 models)
results/<task>/summary.csv                  means, SDs, 95% CIs, paired tests vs GraphSAGE
results/<task>/per_class_graphsage_rep0.csv per-class P/R/F1 + true-positive counts
results/<task>/confusion_*.csv              confusion matrices (pooled out-of-fold, repetition 0)
results/<task>/selected_hyperparameters.csv hyper-parameters chosen on validation, per fold
results/<task>/training_logs/*.csv          epoch-level loss and validation macro-F1
results/<task>/checkpoints/*.pt             GraphSAGE state_dicts for all 25 folds
results/run_log_<task>.txt                  console log of the run
```

## 5. Layout

```
mcare/data.py          checksum-verified loading of the Dataverse V4 files
mcare/taxonomy.py      12-category author-derived mapping (rules + overrides + provenance)
mcare/preprocess.py    NFKC, 120-term abbreviation lexicon, selective stop-words
mcare/lexicon_abbrev.py the 120-term lexicon (Appendix A)
mcare/descriptive.py   length/readability, terms, TF-IDF, VADER, LDA, knowledge graph
mcare/graph.py         fold-wise leakage-free question-term graph
mcare/models.py        GraphSAGE (PyTorch Geometric SAGEConv), MLP ablation, training loop
mcare/experiment.py    grouping, splits, baselines, Node2Vec, ablations, statistics
scripts/               entry points; tests/ leakage and integrity tests
```

## 6. What this repository does not contain

No dialogue system, retrieval-augmented generator or large language model was implemented or
evaluated, and no expert ratings of generated text exist. The response-generation comparison
(former Tables 3–4) has therefore been removed from the manuscript.

Licence: MIT (code); the MOTHER data are CC0 (Eyobu et al., Harvard Dataverse V4).
