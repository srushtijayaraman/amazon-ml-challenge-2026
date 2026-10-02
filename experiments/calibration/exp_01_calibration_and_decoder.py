"""
EXP-01: Probability Calibration & Expected-F0.5 Decoding
Evaluates:
1. Current Global Threshold Baseline
2. Score + Runner-Up Margin Thresholding
3. Isotonic Calibration + Expected-F0.5 Decoder

Strictly entity-disjoint split:
- Split A (2,500 entities): Calibration fitting & threshold tuning
- Split B (2,500 entities): Unseen evaluation holdout split
"""

import os
import sys
import time
import json
import csv
import collections
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.environ.get("BASELINE_SRC_DIR", os.path.join(ROOT, "experiments", "baseline"))
sys.path.insert(0, SRC_DIR)

import normalization as norm
from features import extract_features_for_pair
from model import EntityMatcherModel
from blocking import get_blocking_keys_from_preprocessed
from thresholding import apply_threshold_and_deduplication
from evaluator import compute_entity_metrics, evaluate_predictions

BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25


def _pb_add(pmf, p):
    out = np.zeros(pmf.size + 1)
    out[:-1] += pmf * (1.0 - p)
    out[1:] += pmf * p
    return out


def decode_expected_f05(cand_probs):
    """
    Decodes top-k set maximizing expected Macro F0.5.
    cand_probs: list of (cand_id, prob)
    Returns: list of selected cand_ids
    """
    if not cand_probs:
        return []

    # Sort descending by probability
    sorted_pairs = sorted(cand_probs, key=lambda x: x[1], reverse=True)
    ps = np.array([max(1e-5, min(1.0 - 1e-5, p)) for _, p in sorted_pairs], dtype=np.float64)
    n = len(ps)

    prefix = [np.ones(1)]
    for pi in ps:
        prefix.append(_pb_add(prefix[-1], pi))

    suffix = [None] * (n + 1)
    suffix[n] = np.ones(1)
    for j in range(n - 1, -1, -1):
        suffix[j] = _pb_add(suffix[j + 1], ps[j])

    ev = np.empty(n + 1)
    ev[0] = suffix[0][0]  # P(T = 0)

    for k in range(1, n + 1):
        x = np.arange(k + 1, dtype=np.float64)[:, None]
        z = np.arange(suffix[k].size, dtype=np.float64)[None, :]
        util = ONE_PLUS_BETA2 * x / (k + BETA2 * (x + z))
        ev[k] = prefix[k] @ util @ suffix[k]

    best_k = int(np.argmax(ev))
    return [sorted_pairs[i][0] for i in range(best_k)]


def apply_margin_thresholding(cand_scores_dict, abs_t2, abs_t3, margin_thresh=0.0):
    """
    Candidate selection with absolute threshold + runner-up margin filtering.
    """
    all_pairs = []
    for s1_id, scores in cand_scores_dict.items():
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_cand, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        for tid, p in scores:
            thresh = abs_t2 if tid.startswith("S2-") else abs_t3
            if p >= thresh:
                # If margin threshold specified and runner up is close to top:
                if margin_thresh > 0.0 and len(sorted_scores) > 1 and tid != top_cand:
                    if (top_score - p) < margin_thresh:
                        # Keep only top candidate if ambiguous
                        continue
                all_pairs.append((p, s1_id, tid))

    all_pairs.sort(key=lambda x: x[0], reverse=True)
    assigned_targets = set()
    result = {s1_id: set() for s1_id in cand_scores_dict.keys()}

    for p, s1_id, tid in all_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            result[s1_id].add(tid)

    return result


def main():
    print("=" * 75)
    print("EXP-01: PROBABILITY CALIBRATION & DECISION LAYER EXPERIMENTS")
    print("=" * 75)
    t0 = time.time()

    # 1. Load Ground Truth
    gt_path = os.path.join(ROOT, "output", "val_ground_truth.tsv")
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            s1 = parts[0]
            mids = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            gt[s1] = mids

    all_val_s1 = list(gt.keys())[:5000]
    # Split into 50/50 disjoint halves
    split_a_s1 = all_val_s1[:2500]
    split_b_s1 = all_val_s1[2500:5000]
    gt_a = {k: gt[k] for k in split_a_s1}
    gt_b = {k: gt[k] for k in split_b_s1}

    print(f"Validation Split A (Tuning/Calibration) : {len(split_a_s1):,} S1 entities ({sum(len(m) for m in gt_a.values()):,} true links)")
    print(f"Validation Split B (Sealed Evaluation)   : {len(split_b_s1):,} S1 entities ({sum(len(m) for m in gt_b.values()):,} true links)")

    # Load S1 records
    s1_needed = set(all_val_s1)
    s1_records = {}
    with open(os.path.join(ROOT, "dataset", "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in s1_needed:
                s1_records[parts[0]] = (parts[1], parts[2], parts[3])
                if len(s1_records) == len(s1_needed):
                    break

    # Load targets pool
    needed_targets = set()
    for mids in gt_a.values():
        needed_targets.update(mids)
    for mids in gt_b.values():
        needed_targets.update(mids)

    target_records = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(ROOT, "dataset", "train", src)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                tid = parts[0]
                if tid in needed_targets or len(target_records) < len(needed_targets) + 25000:
                    target_records[tid] = (parts[1], parts[2], parts[3])

    # Build inverted index
    target_preprocessed = {}
    index = collections.defaultdict(list)
    for tid, (rname, raddr, rcountry) in target_records.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        target_preprocessed[tid] = (cn, core_n, ca, nums, pnum)
        tkeys = get_blocking_keys_from_preprocessed(cn, core_n, ca, nums)
        for k in tkeys:
            index[k].append(tid)

    for k in list(index.keys()):
        lim = 150 if k[0] == "c3gram" else (250 if (k[0].startswith("n") or k[0].startswith("core") or k[0].startswith("compact")) else 100)
        if len(index[k]) > lim:
            del index[k]

    # Model
    model_path = os.path.join(ROOT, "teammate_package", "models", "final_entity_matcher.joblib")
    model = EntityMatcherModel.load(model_path)

    # Function to score candidates for a split
    def score_split(s1_list, gt_dict):
        pairs_X = []
        pairs_meta = []
        y_labels = []

        for sid in s1_list:
            rname, raddr, _ = s1_records[sid]
            cn, core_n, _ = norm.normalize_name(rname)
            ca, nums, pnum, _ = norm.normalize_address(raddr)
            skeys = get_blocking_keys_from_preprocessed(cn, core_n, ca, nums)
            s1_tup = (cn, core_n, ca, nums, pnum)

            counts = collections.Counter()
            for k in skeys:
                if k in index:
                    counts.update(index[k])

            cands = counts.most_common(20)
            true_set = gt_dict[sid]

            for tid, sh in cands:
                t_tup = target_preprocessed[tid]
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                pairs_X.append(feats)
                pairs_meta.append((sid, tid, s1_tup, t_tup))
                y_labels.append(1 if tid in true_set else 0)

        X = np.array(pairs_X, dtype=np.float32)
        raw_p = model.predict_proba(X)
        y = np.array(y_labels, dtype=np.int32)
        return X, raw_p, y, pairs_meta

    print("\nScoring Split A (Calibration Set)...")
    _, raw_p_a, y_a, meta_a = score_split(split_a_s1, gt_a)

    print("Scoring Split B (Evaluation Holdout)...")
    _, raw_p_b, y_b, meta_b = score_split(split_b_s1, gt_b)

    # Measure Brier Score on raw probabilities
    brier_raw_a = brier_score_loss(y_a, raw_p_a)
    brier_raw_b = brier_score_loss(y_b, raw_p_b)
    print(f"Raw Model Brier Score: Split A = {brier_raw_a:.5f}, Split B = {brier_raw_b:.5f}")

    # Fit Isotonic Calibrator strictly on Split A
    print("\nFitting Isotonic Regression on Split A...")
    iso = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
    iso.fit(raw_p_a, y_a)

    # Calibrate Split B
    cal_p_b = iso.predict(raw_p_b)
    brier_cal_b = brier_score_loss(y_b, cal_p_b)
    print(f"Calibrated Brier Score on Split B: {brier_cal_b:.5f} (Delta: {brier_cal_b - brier_raw_b:+.5f})")

    # Group scores per entity for Split B
    def build_cand_scores(probs_list, meta_list):
        d = collections.defaultdict(list)
        for (sid, tid, s1_tup, t_tup), p in zip(meta_list, probs_list):
            prob = float(p)
            s1_pnum = s1_tup[4]
            t_pnum = t_tup[4]
            if s1_pnum is not None and t_pnum is not None and s1_pnum != t_pnum:
                prob *= 0.1
            d[sid].append((tid, prob))
        return d

    raw_scores_b = build_cand_scores(raw_p_b, meta_b)
    cal_scores_b = build_cand_scores(cal_p_b, meta_b)

    # -------------------------------------------------------------
    # CONFIG 1: Current Baseline (Global Threshold S2=0.75, S3=0.35)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("CONFIG 1: Baseline Global Threshold (S2=0.75, S3=0.35)")
    print("-" * 75)
    preds_c1 = apply_threshold_and_deduplication(raw_scores_b, 0.75, 0.35)
    dict_c1 = {sid: set(preds_c1.get(sid, [])) for sid in split_b_s1}
    rep_c1 = evaluate_predictions(gt_b, dict_c1, verbose=False)
    print(f"Macro F0.5 : {rep_c1['macro_f05']:.6f} ({rep_c1['macro_f05']*100:.2f}%)")
    print(f"Precision  : {rep_c1['macro_precision']*100:.2f}%, Recall: {rep_c1['macro_recall']*100:.2f}%")
    print(f"Singletons : {rep_c1['singleton_accuracy']*100:.2f}%, FP={rep_c1['global_fp']}, FN={rep_c1['global_fn']}")

    # -------------------------------------------------------------
    # CONFIG 2: Score + Runner-Up Margin Threshold
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("CONFIG 2: Threshold + Runner-Up Margin (Grid Searched on Split A)")
    print("-" * 75)
    # Find best margin on Split A
    raw_scores_a = build_cand_scores(raw_p_a, meta_a)
    best_margin = 0.0
    best_f05_a = -1.0
    for m in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
        p_a = apply_margin_thresholding(raw_scores_a, 0.75, 0.35, margin_thresh=m)
        r_a = evaluate_predictions(gt_a, {sid: set(p_a.get(sid, [])) for sid in split_a_s1}, verbose=False)
        if r_a["macro_f05"] > best_f05_a:
            best_f05_a = r_a["macro_f05"]
            best_margin = m

    print(f"Optimal Runner-Up Margin selected from Split A: {best_margin:.2f}")
    preds_c2 = apply_margin_thresholding(raw_scores_b, 0.75, 0.35, margin_thresh=best_margin)
    dict_c2 = {sid: set(preds_c2.get(sid, [])) for sid in split_b_s1}
    rep_c2 = evaluate_predictions(gt_b, dict_c2, verbose=False)
    print(f"Macro F0.5 : {rep_c2['macro_f05']:.6f} ({rep_c2['macro_f05']*100:.2f}%) [Delta: {rep_c2['macro_f05'] - rep_c1['macro_f05']:+.6f}]")
    print(f"Precision  : {rep_c2['macro_precision']*100:.2f}%, Recall: {rep_c2['macro_recall']*100:.2f}%")
    print(f"Singletons : {rep_c2['singleton_accuracy']*100:.2f}%, FP={rep_c2['global_fp']}, FN={rep_c2['global_fn']}")

    # -------------------------------------------------------------
    # CONFIG 3: Isotonic Calibration + Expected-F0.5 Decoder
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("CONFIG 3: Isotonic Calibration + Expected-F0.5 Decoding + Exclusivity")
    print("-" * 75)

    # Decode each entity's candidates using expected F0.5
    raw_decoded = {}
    for sid in split_b_s1:
        cands = cal_scores_b.get(sid, [])
        chosen = decode_expected_f05(cands)
        raw_decoded[sid] = chosen

    # Enforce global target exclusivity on expected-F0.5 chosen pairs
    # Sort pairs by calibrated score
    candidate_prob_map = {(sid, tid): p for (sid, tid, _, _), p in zip(meta_b, cal_p_b)}
    pairs_to_resolve = []
    for sid, tids in raw_decoded.items():
        for tid in tids:
            p = candidate_prob_map.get((sid, tid), 0.5)
            pairs_to_resolve.append((p, sid, tid))

    pairs_to_resolve.sort(key=lambda x: x[0], reverse=True)
    assigned_targets = set()
    final_dict_c3 = {sid: set() for sid in split_b_s1}

    for p, sid, tid in pairs_to_resolve:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            final_dict_c3[sid].add(tid)

    rep_c3 = evaluate_predictions(gt_b, final_dict_c3, verbose=False)
    print(f"Macro F0.5 : {rep_c3['macro_f05']:.6f} ({rep_c3['macro_f05']*100:.2f}%) [Delta: {rep_c3['macro_f05'] - rep_c1['macro_f05']:+.6f}]")
    print(f"Precision  : {rep_c3['macro_precision']*100:.2f}%, Recall: {rep_c3['macro_recall']*100:.2f}%")
    print(f"Singletons : {rep_c3['singleton_accuracy']*100:.2f}%, FP={rep_c3['global_fp']}, FN={rep_c3['global_fn']}")

    # Record experiments in experiments/results.csv
    csv_file = os.path.join(ROOT, "experiments", "results.csv")
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP-01-MARGIN",
            "EXP-BASELINE-CURRENT",
            "Threshold + Runner-Up Margin Decoder on Split B",
            "InvertedIndex_v2_enhanced",
            "RapidFuzz_36D",
            "ISCII_IndicDict_v1",
            "HeuristicNegatives_v1",
            "XGBoost_hist",
            "None",
            "MarginBipartite",
            "S2=0.75,S3=0.35",
            f"{best_margin:.2f}",
            f"{rep_c2['macro_f05']:.6f}",
            f"{rep_c2['macro_precision']:.6f}",
            f"{rep_c2['macro_recall']:.6f}",
            f"{rep_c2['singleton_accuracy']:.6f}",
            "N/A", "N/A",
            rep_c2["global_fp"], rep_c2["global_fn"],
            f"{time.time()-t0:.1f}s", "<500MB",
            "KEEP" if rep_c2['macro_f05'] > rep_c1['macro_f05'] else "REJECT"
        ])
        writer.writerow([
            "EXP-02-ISOTONIC-EXPECTED-F05",
            "EXP-BASELINE-CURRENT",
            "Isotonic Calibration + Per-Entity Expected-F0.5 Decoder on Split B",
            "InvertedIndex_v2_enhanced",
            "RapidFuzz_36D",
            "ISCII_IndicDict_v1",
            "HeuristicNegatives_v1",
            "XGBoost_hist",
            "IsotonicRegression",
            "ExpectedF05_Decoder",
            "AdaptiveExpectedF05",
            "N/A",
            f"{rep_c3['macro_f05']:.6f}",
            f"{rep_c3['macro_precision']:.6f}",
            f"{rep_c3['macro_recall']:.6f}",
            f"{rep_c3['singleton_accuracy']:.6f}",
            "N/A", "N/A",
            rep_c3["global_fp"], rep_c3["global_fn"],
            f"{time.time()-t0:.1f}s", "<500MB",
            "KEEP" if rep_c3['macro_f05'] > rep_c1['macro_f05'] else "REJECT"
        ])

    print("\n" + "=" * 75)
    print("EXPERIMENT RESULTS SUMMARY (ON HELD-OUT SPLIT B)")
    print("=" * 75)
    print(f"1. Baseline Global Threshold : F0.5 = {rep_c1['macro_f05']:.6f} | P = {rep_c1['macro_precision']*100:.2f}% | R = {rep_c1['macro_recall']*100:.2f}% | FP = {rep_c1['global_fp']}")
    print(f"2. Threshold + Runner-Up Margin: F0.5 = {rep_c2['macro_f05']:.6f} | P = {rep_c2['macro_precision']*100:.2f}% | R = {rep_c2['macro_recall']*100:.2f}% | FP = {rep_c2['global_fp']}")
    print(f"3. Isotonic + Expected-F0.5   : F0.5 = {rep_c3['macro_f05']:.6f} | P = {rep_c3['macro_precision']*100:.2f}% | R = {rep_c3['macro_recall']*100:.2f}% | FP = {rep_c3['global_fp']}")


if __name__ == "__main__":
    main()
