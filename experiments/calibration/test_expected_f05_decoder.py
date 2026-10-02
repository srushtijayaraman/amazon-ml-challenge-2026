"""
Validation and Unit Test for Expected-F0.5 Decoder (Task 10)
Verifies DP-based Top-k Expected F0.5 Decoder against Brute-Force Exhaustive Enumeration
over all 2^n subsets on toy candidate sets.
"""

import itertools
import numpy as np


BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25


def f05_metric(true_set, pred_set):
    """Official competition metric for a single entity."""
    n_true = len(true_set)
    n_pred = len(pred_set)
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    p = tp / n_pred
    r = tp / n_true
    return (ONE_PLUS_BETA2 * p * r) / (BETA2 * p + r)


def brute_force_expected_f05(probs):
    """
    Exhaustively enumerates all 2^n outcome configurations y in {0, 1}^n
    and all 2^n candidate prediction subsets S in P({0..n-1}).
    Computes exact E[F0.5(S)] = sum_{y} P(y) * F0.5(y, S).
    Returns: best_subset (tuple of indices), best_expected_f05.
    """
    n = len(probs)
    best_subset = ()
    best_ev = -1.0

    # Precompute probability of each ground truth outcome y in {0, 1}^n
    outcomes = []
    for y_vec in itertools.product([0, 1], repeat=n):
        prob_y = 1.0
        for i, yi in enumerate(y_vec):
            prob_y *= probs[i] if yi == 1 else (1.0 - probs[i])
        true_indices = set(i for i, yi in enumerate(y_vec) if yi == 1)
        outcomes.append((true_indices, prob_y))

    # Evaluate every candidate prediction subset S
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            pred_set = set(subset)
            ev = 0.0
            for true_indices, prob_y in outcomes:
                score = f05_metric(true_indices, pred_set)
                ev += prob_y * score
            if ev > best_ev + 1e-12:
                best_ev = ev
                best_subset = tuple(sorted(subset))

    return best_subset, best_ev


def _pb_add(pmf, p):
    """Convolve a count pmf with one Bernoulli(p)."""
    out = np.zeros(pmf.size + 1)
    out[:-1] += pmf * (1.0 - p)
    out[1:] += pmf * p
    return out


def dp_expected_f05(probs):
    """
    Fast O(n^2) Dynamic Programming Expected F0.5 Decoder.
    Evaluates top-k sets for k = 0..n.
    """
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    n = len(ps)

    prefix = [np.ones(1)]
    for pi in ps:
        prefix.append(_pb_add(prefix[-1], pi))

    suffix = [None] * (n + 1)
    suffix[n] = np.ones(1)  # assuming lam_missed = 0
    for j in range(n - 1, -1, -1):
        suffix[j] = _pb_add(suffix[j + 1], ps[j])

    ev = np.empty(n + 1)
    ev[0] = suffix[0][0]  # P(T = 0)

    for k in range(1, n + 1):
        x = np.arange(k + 1, dtype=np.float64)[:, None]
        z = np.arange(suffix[k].size, dtype=np.float64)[None, :]
        # Util = 1.25 * x / (k + 0.25 * (x + z))
        util = ONE_PLUS_BETA2 * x / (k + BETA2 * (x + z))
        ev[k] = prefix[k] @ util @ suffix[k]

    best_k = int(np.argmax(ev))
    chosen_indices = tuple(sorted(order[:best_k]))
    return chosen_indices, float(ev[best_k]), ev


def test_decoder_equivalence():
    print("=" * 60)
    print("RUNNING DECODER EQUIVALENCE TESTS (EXHAUSTIVE vs DP)")
    print("=" * 60)

    test_cases = [
        # (name, probs)
        ("Single high-confidence candidate", [0.95]),
        ("Single low-confidence candidate", [0.20]),
        ("Single borderline candidate (near threshold)", [0.45]),
        ("Two candidates: one dominant, one weak", [0.88, 0.15]),
        ("Two candidates: both strong", [0.92, 0.85]),
        ("Two candidates: both weak (should reject both)", [0.30, 0.25]),
        ("Three candidates: mixed probabilities", [0.90, 0.65, 0.10]),
        ("Three candidates: equal medium probabilities", [0.55, 0.55, 0.50]),
        ("Four candidates: diverse spectrum", [0.95, 0.70, 0.40, 0.05]),
        ("Five candidates: complex spectrum", [0.89, 0.75, 0.60, 0.35, 0.08])
    ]

    all_passed = True
    for name, probs in test_cases:
        bf_sub, bf_ev = brute_force_expected_f05(probs)
        dp_sub, dp_ev, ev_all = dp_expected_f05(probs)

        diff = abs(bf_ev - dp_ev)
        subset_match = (bf_sub == dp_sub)
        passed = (diff < 1e-9) and subset_match

        status = "PASS" if passed else "FAIL"
        print(f"\nTest: {name}")
        print(f"  Input probs: {probs}")
        print(f"  Brute-Force Optimal Subset : {bf_sub} (E[F0.5] = {bf_ev:.8f})")
        print(f"  DP Decoder Optimal Subset  : {dp_sub} (E[F0.5] = {dp_ev:.8f})")
        print(f"  Absolute Delta             : {diff:.2e} -> {status}")

        if not passed:
            all_passed = False
            print(f"  All DP Top-k Expected Values: {ev_all}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED: DP Decoder is MATHEMATICALLY EXACT to brute force!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test_decoder_equivalence()
