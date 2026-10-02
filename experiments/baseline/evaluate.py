#!/usr/bin/env python3
"""
ML Challenge 2026 — Official Entity-Level Evaluation Engine (Phase 3 & Phase 6)

Strictly reproduces the official evaluation logic:
- Macro-averaged F_0.5 score per Source 1 entity:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
- Singleton handling: True singleton with 0 predicted matches receives 1.0.
  Any false prediction on a true singleton receives 0.0.
- All Source 1 entities in the evaluation set are included in the macro average.
"""

from collections import defaultdict
from typing import Dict, Set, List, Tuple


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    # Case 1: True Singleton (0 true matches)
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0  # Correct singleton identification
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    # Case 2: True Entity with Matches (n_true > 0)
    if n_pred == 0:
        return 0.0, 0.0, 0.0  # Missed all matches

    # True Positives
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return f05, precision, recall


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> dict:
    """
    Computes macro-averaged entity-level evaluation metrics across all Source 1 entities.
    """
    all_s1_ids = sorted(ground_truth.keys())
    n_entities = len(all_s1_ids)

    if n_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0
        }

    f05_scores = []
    precision_scores = []
    recall_scores = []

    singleton_correct = 0
    singleton_total = 0

    multi_f05_scores = []

    for s1_id in all_s1_ids:
        true_mids = ground_truth.get(s1_id, set())
        pred_mids = predictions.get(s1_id, set())

        f05, p, r = compute_entity_f05(true_mids, pred_mids)

        f05_scores.append(f05)
        precision_scores.append(p)
        recall_scores.append(r)

        if len(true_mids) == 0:
            singleton_total += 1
            if len(pred_mids) == 0:
                singleton_correct += 1
        else:
            multi_f05_scores.append(f05)

    macro_f05 = sum(f05_scores) / n_entities
    macro_precision = sum(precision_scores) / n_entities
    macro_recall = sum(recall_scores) / n_entities
    singleton_acc = (singleton_correct / singleton_total * 100) if singleton_total > 0 else 100.0
    matched_entity_f05 = (sum(multi_f05_scores) / len(multi_f05_scores)) if multi_f05_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_accuracy": singleton_acc,
        "singleton_correct": singleton_correct,
        "singleton_total": singleton_total,
        "matched_entity_f05": matched_entity_f05,
        "total_entities": n_entities
    }
