# Changelog

## 2.0.0 — 2026-10-17 (revision in response to Academia comments, set 2)

### Removed (simulated or unsupported outputs)
* `intent_classifier.py` — returned the per-class table (former Table 6) as literals.
* `model_comparer.py` — trained a Random Forest on a synthetic binary target, derived the
  "GNN ablation" by fixed multipliers, hard-coded the LLM comparison (Med-PaLM 2,
  Llama-3-70B-Med, ClinicalBERT, ...) and generated a "fairness audit" from random numbers.
* `clinical_validator.py` — propensity-score, triage-time and QA-alignment results generated
  from random distributions.
* `multilingual_robustness.py` — noise-robustness and zero-shot scores generated from formulas
  plus random noise.
* `ontology.py` synthetic clinical scenarios and `data_processor.py` synthetic-data generator
  and keyword-based random intent inference.
* `visualizer.py` — figures built from the values above.

### Added
* Frozen Harvard Dataverse V4 files with MD5/SHA-256 verification.
* Author-derived 12-category taxonomy with per-record provenance and a clinician annotation
  sheet + κ scorer.
* Leakage-free inductive GraphSAGE (PyTorch Geometric), MLP / lexicon / sentiment / TF-IDF
  ablations, Logistic Regression, Linear SVM, Random Forest, Node2Vec+MLP and majority baselines.
* Grouped, repeated, stratified cross-validation with saved split IDs, predictions, confusion
  matrices, training logs, checkpoints and corrected resampled-t confidence intervals.
* Unit tests for checksums, grouping and graph leakage.
