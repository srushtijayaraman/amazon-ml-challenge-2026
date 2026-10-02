# Evaluation Metric & Mathematical Optimization

**Challenge Metric:** Entity-Level Macro-Averaged $F_{0.5}$
**Official Competition Leaderboard Score:** **0.967915** Macro $F_{0.5}$

---

## 1. Metric Formulation

The official evaluation metric for the Amazon ML Challenge 2026 is **Macro-Averaged $F_{0.5}$**, computed across all $N$ entities in Source 1:

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}(S_i)$$

Where for each individual entity $S_i$:
$$F_{0.5}(S_i) = \frac{(1 + \beta^2) \cdot \text{Precision}(S_i) \cdot \text{Recall}(S_i)}{\beta^2 \cdot \text{Precision}(S_i) + \text{Recall}(S_i)} \quad \text{with } \beta = 0.5$$

Expressing precision and recall in terms of True Positives ($\text{TP}$), False Positives ($\text{FP}$), and False Negatives ($\text{FN}$):
$$F_{0.5}(S_i) = \frac{1.25 \cdot \text{TP}}{1.25 \cdot \text{TP} + \text{FP} + 0.25 \cdot \text{FN}}$$

---

## 2. Key Optimization Properties

### 1. Asymmetric Precision Weighting (4:1 Penalty Ratio)
In the denominator:
- Each **False Positive** (false merge) contributes $+1.0 \cdot \text{FP}$.
- Each **False Negative** (missed match) contributes $+0.25 \cdot \text{FN}$.

Consequently, a False Positive is penalized **$4\times$ as severely** as a False Negative in the per-entity objective function. Predicting a link with even moderate uncertainty degrades the macro score.

### 2. The Singleton All-or-Nothing Cliff
Entities with zero true matches in Source 2 or Source 3 (singletons) constitute $\approx 5.8\% - 6.3\%$ of the master catalog:
$$\text{If } |G_i| = 0 \text{ (True Singleton):} \quad F_{0.5}(S_i) = \begin{cases} 1.0 & \text{if } |P_i| = 0 \text{ (Predicted Empty)} \\ 0.0 & \text{if } |P_i| \ge 1 \text{ (Any False Link)} \end{cases}$$

Predicting a single spurious candidate link for a true singleton drops its score from **1.0 to 0.0**. In a test set of 1.73M master entities, over 100,000 entities are singletons; maintaining high precision on singletons is mandatory to achieve competitive macro scores.

---

## 3. Threshold Optimization Dynamics

Because Macro $F_{0.5}$ weights precision heavily, conventional balanced classification thresholds ($\tau = 0.50$) are sub-optimal:

- **Low Thresholds ($\tau \le 0.50$):** High candidate recall is achieved, but false positive links degrade both multi-match entity scores and convert singletons from 1.0 to 0.0.
- **Moderate Thresholds ($\tau \approx 0.60 - 0.70$):** Reduces false merges significantly while capturing clear name and address matches.
- **Calibrated High Threshold ($\tau^* = 0.75$):** Optimizes the 4:1 precision-recall trade-off against the high density of unlinked distractors and synthetic decoys present in large-scale target pools.
- **Overly Conservative Thresholds ($\tau \ge 0.85$):** Severe recall degradation outpaces precision gains, as legitimate matches with slight spelling variations are discarded.

---

## 4. Official Submission Verification

The repository includes the official verification suite (`scripts/validate_submission.py`). A submission is valid if and only if:
1. `matching_results.tsv` contains exactly $1,732,544$ rows (header: `source1_entity_id\tmatched_entity_ids`).
2. `candidate_pairs.tsv` contains exactly $1,732,544$ rows (header: `source1_entity_id\tcandidate_entity_ids`).
3. For every entity, the predicted `matched_entity_ids` must form a strict subset of `candidate_entity_ids`.
4. All entity IDs match the official test set keys with no malformed quote marks or stray delimiters.
