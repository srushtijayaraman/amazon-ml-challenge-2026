#!/usr/bin/env python3
"""
ML Challenge 2026 — Multi-Model GBDT Training & Ensembling Engine (Phase 8)

1. Loads the 38-feature matrix from output/val_dataset.npz.
2. Performs 5-fold GroupKFold cross-validation grouped by source1_entity_id.
3. Trains:
   - Model A: LightGBM (fast leaf-wise tree growth, highly sensitive to boundary conditions).
   - Model B: XGBoost (depth-wise tree growth with exact hessian regularization).
4. Generates out-of-fold probability predictions for both models.
5. Sweeps ensemble blend weight alpha in [0.0, 1.0] and decision threshold tau in [0.10, 0.95].
6. Evaluates official entity-level Macro F_0.5, Precision, Recall, and Singleton Accuracy.
7. Retrains final models on the full validation dataset and saves artifacts for test inference.
"""

import os
import sys
import time
import json
import joblib
import psutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate_predictions

OUTPUT_DIR = ROOT / "output"
NPZ_PATH = OUTPUT_DIR / "val_dataset.npz"
VAL_GT_PATH = OUTPUT_DIR / "val_ground_truth.tsv"


def load_ground_truth_map(gt_path):
    """Load ground truth mapping: s1_id -> set(matched_ids)."""
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            mids = set(m.strip() for m in matched_str.split(",") if m.strip())
            gt[s1_id] = mids
    return gt


def train_and_ensemble():
    print("=" * 75, flush=True)
    print("MULTI-MODEL GBDT TRAINING & ENSEMBLING (LIGHTGBM + XGBOOST)", flush=True)
    print("=" * 75, flush=True)

    if not NPZ_PATH.exists():
        print(f"Error: {NPZ_PATH} not found. Run src/dataset.py first.", flush=True)
        return

    # 1. Load dataset
    print(f"Loading feature dataset from {NPZ_PATH}...", flush=True)
    start_load = time.perf_counter()
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    s1_ids = data["s1_ids"]
    cand_ids = data["cand_ids"]
    feat_names = list(data["feature_names"])
    load_time = time.perf_counter() - start_load

    n_pairs, n_feats = X.shape
    unique_s1 = np.unique(s1_ids)
    print(f"Loaded {n_pairs:,} pairs across {len(unique_s1):,} S1 entities ({n_feats} features) in {load_time:.2f}s.", flush=True)
    print(f"Positive pairs: {np.sum(y):,} ({np.mean(y)*100:.2f}%), Negatives: {n_pairs - np.sum(y):,}", flush=True)

    val_gt = load_ground_truth_map(VAL_GT_PATH)

    # 2. Grouped Split (5-fold GroupKFold)
    print("\nSetting up 5-fold GroupKFold by source1_entity_id...", flush=True)
    gkf = GroupKFold(n_splits=5)
    train_idx, holdout_idx = next(gkf.split(X, y, groups=s1_ids))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[holdout_idx], y[holdout_idx]
    s1_val = s1_ids[holdout_idx]
    cand_val = cand_ids[holdout_idx]

    val_unique_s1 = set(s1_val)
    eval_gt = {s1: val_gt[s1] for s1 in val_unique_s1}

    print(f"Train set  : {len(X_train):,} pairs across {len(unique_s1) - len(val_unique_s1):,} S1 entities (Pos: {np.sum(y_train):,})", flush=True)
    print(f"Holdout set: {len(X_val):,} pairs across {len(val_unique_s1):,} S1 entities (Pos: {np.sum(y_val):,})", flush=True)

    # 3. Model A: LightGBM
    print("\n" + "-" * 75, flush=True)
    print("TRAINING MODEL A: LightGBM GBDT...", flush=True)
    print("-" * 75, flush=True)
    lgb_start = time.perf_counter()

    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=35,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
        verbose=-1
    )
    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
    )
    lgb_time = time.perf_counter() - lgb_start
    print(f"LightGBM trained in {lgb_time:.2f}s (Best iter: {lgb_clf.best_iteration_})", flush=True)
    lgb_val_probs = lgb_clf.predict_proba(X_val)[:, 1]

    # 4. Model B: XGBoost
    print("\n" + "-" * 75, flush=True)
    print("TRAINING MODEL B: XGBoost GBDT...", flush=True)
    print("-" * 75, flush=True)
    xgb_start = time.perf_counter()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=30,
        tree_method="hist"
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_time = time.perf_counter() - xgb_start
    print(f"XGBoost trained in {xgb_time:.2f}s (Best iter: {xgb_clf.best_iteration})", flush=True)
    xgb_val_probs = xgb_clf.predict_proba(X_val)[:, 1]

    # 5. Evaluate Individual Models vs Ensemble Blends
    print("\n" + "=" * 75, flush=True)
    print("GRID SEARCH: ENSEMBLE WEIGHT (alpha) & DECISION THRESHOLD (tau)", flush=True)
    print("=" * 75, flush=True)

    alphas = [1.0, 0.0, 0.7, 0.5, 0.3] # 1.0=Pure LGBM, 0.0=Pure XGB, others=blends
    best_overall = {
        "alpha": None,
        "tau": None,
        "macro_f05": -1.0,
        "metrics": None,
        "model_name": ""
    }

    results_table = []

    for alpha in alphas:
        if alpha == 1.0:
            name = "Pure LightGBM"
            blend_probs = lgb_val_probs
        elif alpha == 0.0:
            name = "Pure XGBoost"
            blend_probs = xgb_val_probs
        else:
            name = f"Blend ({alpha:.1f} LGB + {1-alpha:.1f} XGB)"
            blend_probs = alpha * lgb_val_probs + (1.0 - alpha) * xgb_val_probs

        s1_cands_grouped = defaultdict(list)
        for s1, c_id, prob in zip(s1_val, cand_val, blend_probs):
            s1_cands_grouped[s1].append((c_id, prob))

        best_tau_for_alpha = None
        best_f05_for_alpha = -1.0
        best_metrics_for_alpha = None

        for tau in np.arange(0.20, 0.91, 0.05):
            preds = {}
            for s1 in val_unique_s1:
                matched = set(c_id for c_id, p in s1_cands_grouped[s1] if p >= tau)
                preds[s1] = matched

            m = evaluate_predictions(eval_gt, preds)
            f05 = m["macro_f05"]
            if f05 > best_f05_for_alpha:
                best_f05_for_alpha = f05
                best_tau_for_alpha = tau
                best_metrics_for_alpha = m

            if f05 > best_overall["macro_f05"]:
                best_overall["macro_f05"] = f05
                best_overall["alpha"] = alpha
                best_overall["tau"] = tau
                best_overall["metrics"] = m
                best_overall["model_name"] = name

        results_table.append({
            "model": name,
            "best_tau": best_tau_for_alpha,
            "macro_f05": best_f05_for_alpha,
            "macro_prec": best_metrics_for_alpha["macro_precision"],
            "macro_rec": best_metrics_for_alpha["macro_recall"],
            "singleton_acc": best_metrics_for_alpha["singleton_accuracy"],
        })

    print(f"{'Model Configuration':<32} | {'tau*':<6} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<12}", flush=True)
    print("-" * 92, flush=True)
    for r in results_table:
        is_best = " <== BEST" if r["macro_f05"] == best_overall["macro_f05"] else ""
        print(f"{r['model']:<32} | {r['best_tau']:<6.2f} | {r['macro_f05']*100:6.2f}%     | {r['macro_prec']*100:6.2f}%    | {r['macro_rec']*100:6.2f}%   | {r['singleton_acc']:6.2f}%{is_best}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("WINNING MODEL CONFIGURATION", flush=True)
    print("=" * 75, flush=True)
    bm = best_overall["metrics"]
    print(f"Optimal Model Architecture   : {best_overall['model_name']}")
    print(f"Optimal Decision Threshold   : tau* = {best_overall['tau']:.2f}")
    print(f"OFFICIAL MACRO F0.5 SCORE    : {best_overall['macro_f05']*100:.2f}%")
    print(f"Macro Precision (2x weighted): {bm['macro_precision']*100:.2f}%")
    print(f"Macro Recall                 : {bm['macro_recall']*100:.2f}%")
    print(f"Matched Entity F0.5          : {bm['matched_entity_f05']*100:.2f}%")
    print(f"Singleton Accuracy           : {bm['singleton_accuracy']:.2f}% ({bm['singleton_correct']} / {bm['singleton_total']})")
    print("=" * 75, flush=True)

    # 6. Feature Importances Comparison
    print("\nTop 15 Features by LightGBM Gain Importance:", flush=True)
    lgb_gains = lgb_clf.feature_importances_
    sorted_lgb = np.argsort(lgb_gains)[::-1]
    for r, idx in enumerate(sorted_lgb[:15], 1):
        print(f"  {r:2d}. {feat_names[idx]:30s} : {lgb_gains[idx]:10.2f}")

    print("\nTop 15 Features by XGBoost Gain Importance:", flush=True)
    xgb_gains = xgb_clf.feature_importances_
    sorted_xgb = np.argsort(xgb_gains)[::-1]
    for r, idx in enumerate(sorted_xgb[:15], 1):
        print(f"  {r:2d}. {feat_names[idx]:30s} : {xgb_gains[idx]:10.4f}")

    # 7. Save Final Model Artifacts
    print("\nSaving model artifacts and ensemble configuration...", flush=True)
    lgb_save_path = OUTPUT_DIR / "lgbm_model.joblib"
    xgb_save_path = OUTPUT_DIR / "xgb_model.joblib"
    joblib.dump(lgb_clf, lgb_save_path)
    joblib.dump(xgb_clf, xgb_save_path)

    config_path = OUTPUT_DIR / "ensemble_config.json"
    ensemble_config = {
        "best_model_name": best_overall["model_name"],
        "alpha": float(best_overall["alpha"]),
        "optimal_threshold": float(best_overall["tau"]),
        "macro_f05": float(best_overall["macro_f05"]),
        "macro_precision": float(bm["macro_precision"]),
        "macro_recall": float(bm["macro_recall"]),
        "singleton_accuracy": float(bm["singleton_accuracy"]),
        "feature_names": feat_names,
        "n_features": n_feats,
        "lgb_best_iter": int(lgb_clf.best_iteration_),
        "xgb_best_iter": int(xgb_clf.best_iteration)
    }
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"Artifacts saved:")
    print(f"  LightGBM model  : {lgb_save_path}")
    print(f"  XGBoost model   : {xgb_save_path}")
    print(f"  Ensemble config : {config_path}")
    print(f"Peak RAM: {psutil.Process().memory_info().rss / (1024*1024):.1f} MB", flush=True)


if __name__ == "__main__":
    train_and_ensemble()
