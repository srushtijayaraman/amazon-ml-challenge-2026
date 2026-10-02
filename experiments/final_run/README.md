# Production Model Run & Training Metadata

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution
**Official Competition Leaderboard Score:** **0.967915** Macro $F_{0.5}$

---

## 1. Overview

This directory documents the final production model configuration that produced our competitive leaderboard result. The model incorporates 61 pairwise, decoy-contrastive, and within-record relative tie-break features trained on 18.98 million candidate pairs.

### Within-Record Relative Tiebreaks
Error analysis of prior iterations indicated that records with missing addresses constituted a significant portion of false negative misses. Because generic commercial names are often shared across independent catalog entries in the same jurisdiction, conventional pairwise string similarities alone could not distinguish between competing master entities.

We introduced 13 label-free within-record tiebreak features:
- `raw_n_qgap`, `raw_a_qgap`: Difference in raw lowercase similarity between the candidate and the query's best-matching candidate.
- `raw_n_qbest`, `raw_a_qbest`: Boolean indicator of whether this candidate is uniquely top-ranked for the query.
- `n_ratio_qgap`, `a_tset_qgap`: Token set and ratio margins against the best alternative.
- `raw_n_qties`, `raw_a_qties`: Count of competing candidates tied for the top score.

These features account for ~7.8% of total model feature gain (led by `a_tset_qgap` at 5.1% and `a_tset_qbest` at 1.1%).

---

## 2. Model Hyperparameters & Configuration

- **Algorithm:** LightGBM Booster (449 trees)
- **Objective:** `binary` (logistic loss)
- **Learning Rate:** `0.05`
- **Num Leaves:** `511`
- **Min Data in Leaf:** `50`
- **Feature Fraction:** `0.80`
- **Bagging Fraction:** `0.80`
- **Bagging Frequency:** `1`
- **Trained Pairs:** `18,984,346` pairs (~311 seconds training time on 32 threads)

---

## 3. Directory Artifacts

- [`metrics.json`](file:///experiments/final_run/metrics.json): Machine-readable record of training hyperparameters, feature gain rankings, and test dataset volume statistics.
- **Note on Excluded Artifacts:** In accordance with competition data policies, raw training logs containing entity text (`errors.tsv`) and pre-trained model weights (`model.txt`) are excluded from public redistribution. Model weights can be reproduced locally via `python src/match.py fit models` when training data is placed in `dataset/train`.
