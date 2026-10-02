# Error Analysis & Failure Modes

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution
**Official Competition Leaderboard Score:** **0.967915** Macro $F_{0.5}$

---

## 1. Executive Summary & Error Taxonomy

In large-scale business entity resolution evaluated under the precision-weighted Macro $F_{0.5}$ metric, remaining errors categorize into four structural failure modes:

| Error Category | Impact on Objective | Core Root Cause | Synthetic Illustrative Pattern |
| :--- | :---: | :--- | :--- |
| **Never Retrieved (Blocking Misses)** | Minor Recall Loss | High lexical divergence or extreme transliteration shifts | Brand acronym vs. legal business name |
| **Retrieved but Rejected (False Negatives)** | Moderate Recall Loss | Empty or missing address fields | Identical trade name with null street address |
| **False Merges (False Positives)** | Severe Precision Loss | Near-duplicate decoys with shifted door numbers | Shared street name but differing building units |
| **Target Preemption (Assigned Elsewhere)** | Precision & Recall Loss | Target record assigned to a competing candidate with higher $p$ | Chain stores or franchise hubs sharing names |

---

## 2. Deep-Dive Taxonomy of Error Modes

### 2.1 Near-Duplicate & Decoy False Merges (House-Number Shifts)
The dataset includes synthetic distractors that mimic legitimate commercial entities but differ by slight building number shifts or appended generic tokens.

* **Pattern A: Shifted House Number with Repetitive Suffix**
  * *Catalog Entity:* `Example Energy Holdings LLC` — `1 North Avenue, Springfield, IL`
  * *Query Distractor:* `Example Energy Holdings Holdings` — `82 North Ave, Springfield, IL`
  * *Mechanism:* The query duplicates the suffix "Holdings" while shifting the door number from `1` to `82`. High token set similarity can induce false merges unless numerical token differentials (`num_first_diff`) are heavily weighted.

* **Pattern B: Adjacent Commercial Unit Shift**
  * *Catalog Entity:* `Commercial Plaza Corp` — `105 Business Lane, Suite A`
  * *Query Distractor:* `Commercial Plaza Corporation` — `107 Business Lane, Suite B`
  * *Mechanism:* In multi-tenant corporate parks, adjacent door numbers frequently designate distinct legal entities despite near-identical commercial names.

---

### 2.2 The "Empty Address" False Negative Dilemma
Records lacking address information represent a substantial fraction of false negative misses:
- When query records have missing or empty address fields, address similarity features (`a_ratio`, `a_tset`) evaluate to null or zero.
- In nationwide registries, common corporate names (e.g., "Apex Logistics", "National Trust", "Modern Healthcare") are often shared across independent firms in different cities.
- Without address tokens to disambiguate candidates, the model cannot distinguish between competing master entities, suppressing the predicted probability below the decision threshold ($\tau = 0.75$).

---

### 2.3 Address-Only Matches with Masked Trade Names
Certain query records exhibit synthetic trade name alterations while preserving valid multi-line addresses:
- *Pattern:* A synthetic pseudo-word (e.g. `ACMEXYZ`) is substituted as the business name, while the multi-line street address, postal code, and jurisdiction match the catalog entity exactly.
- *Mechanism:* If name similarity is zero, the model must rely entirely on address token overlap (`a_tset`). While address match signals can successfully bridge the gap for rare street addresses, highly ambiguous commercial addresses (e.g., large commercial towers with dozens of tenants) are conservatively rejected to protect precision.

---

### 2.4 Multi-Match vs. Singleton Error Profiles

| Entity Type | Characteristics | Primary Vulnerability under $F_{0.5}$ |
| :--- | :--- | :--- |
| **Singletons (0 true links)** | Entities with zero observations in observation sources | **False Merge (Catastrophic):** Predicting any false link drops entity score from $1.0 \to 0.0$. |
| **Single-Match Entities** | Entities with exactly 1 true observation | **Full Miss or Decoy Swap:** Missing the single link yields entity score $0.0$. |
| **Multi-Match Entities** | Entities with 2 to 11 true observations | **Partial Miss:** Missing 1 out of 4 links degrades score gracefully ($F_{0.5} \approx 0.80$). |

---

### 2.5 Jurisdiction & Cross-Region Challenges

#### United States:
- Highly structured address grids (`123 Main St, Suite 400`). House number distance (`num_first_diff`) provides the most reliable discrimination signal.

#### India:
- Multi-script presence (Devanagari / Latin transliteration) and complex nested address structures (e.g., care-of notations, plot numbers, ward divisions). First-number extraction frequently encounters ward or plot identifiers rather than postal door numbers.

#### France (Unseen in Training Data):
- High density of commercial registry hubs in metropolitan areas where multiple corporate entities share identical building addresses. Within-record relative tiebreaks (`raw_n_qgap`, `a_tset_qgap`) are necessary to resolve competing co-located entities.
