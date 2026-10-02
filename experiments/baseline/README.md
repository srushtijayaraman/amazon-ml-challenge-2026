# Historical Baseline Pipeline

**Official Leaderboard Score:** **0.753000** Macro $F_{0.5}$
**Status:** Archived (Replaced by Production Reverse Retrieval Architecture)

---

## 1. Overview

This directory preserves the initial baseline pipeline used during the early phase of the competition.

### Architecture
- **Retrieval Direction:** Forward candidate retrieval ($S_1 \to S_2/S_3$). Master catalog entities queried an inverted token/n-gram index over the target records.
- **Candidate Pruning:** Per-source FIFO candidate caps (`MAX_CANDS_PER_SOURCE = 15`). Candidates were streamed sequentially and truncated upon reaching the quota.
- **Feature Space:** 24–28 pairwise similarity metrics (Levenshtein, Jaro-Winkler, token overlap, prefix/suffix matching).
- **Models:** LightGBM + XGBoost ensemble ($\alpha = 0.50$, 100 trees each).
- **Decision Layer:** Fixed probability threshold ($\tau = 0.55$).

---

## 2. Forensic Autopsy & Findings

1. **Candidate Recall Collapse:** Forward search over 10.32 million target records caused high-frequency false positives to exhaust the candidate quota. True candidate recall plummeted from 77.54% (unconstrained) to 19.28% post-cap, dropping 5,043 of 8,656 true links before model scoring.
2. **Cross-Entity Collisions:** Because each $S_1$ generated candidates independently, over 152,000 target records were assigned to multiple master entities simultaneously, incurring severe precision penalties.
3. **Historical Validation Leakage:** Early validation code had restricted the target search space to known ground-truth targets plus samples, artificially masking the hub congestion problem.

For full details on how these limitations were resolved, refer to [`docs/methodology.md`](file:///docs/methodology.md) and [`docs/architecture.md`](file:///docs/architecture.md).
