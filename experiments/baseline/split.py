#!/usr/bin/env python3
"""
ML Challenge 2026 — Grouped Validation Split Generator (Phase 2 / Phase 3)

Creates a leak-free, reproducible validation split grouped strictly by source1_entity_id.
Stratified across country and match-count categories (singletons, 1-to-1, multi-match).
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

RANDOM_SEED = 42
VAL_SIZE = 25000  # 25,000 S1 entities for validation evaluation


def create_validation_split():
    print("=" * 70)
    print(f"CREATING GROUPED VALIDATION SPLIT (Size: {VAL_SIZE:,} S1 entities)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)

    # 1. Read S1 countries
    print("Loading S1 countries...")
    s1_countries = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_countries[parts[0]] = parts[3].strip()

    # 2. Read Ground Truth and categorize each S1 entity into strata:
    # Strata key: (country, match_bucket)
    # Buckets: '0' (singleton), '1' (single match), '2-4' (typical multi), '5+' (heavy multi)
    print("Loading Ground Truth and categorizing strata...")
    strata = defaultdict(list)
    gt_map = {}

    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
            gt_map[s1_id] = matched_ids

            country = s1_countries.get(s1_id, "Unknown")
            n = len(matched_ids)
            if n == 0:
                bucket = "0"
            elif n == 1:
                bucket = "1"
            elif 2 <= n <= 4:
                bucket = "2-4"
            else:
                bucket = "5+"

            strata[(country, bucket)].append(s1_id)

    total_s1 = len(gt_map)
    print(f"Total S1 entities available: {total_s1:,}")

    # 3. Stratified sampling
    rng = random.Random(RANDOM_SEED)
    val_s1_set = set()

    print("\nStratified Allocation:")
    for stratum_key, entity_list in sorted(strata.items()):
        stratum_total = len(entity_list)
        fraction = stratum_total / total_s1
        target_val = int(round(fraction * VAL_SIZE))
        sampled = rng.sample(entity_list, target_val)
        val_s1_set.update(sampled)
        print(f"  Stratum {stratum_key}: Total={stratum_total:7,d} ({fraction*100:5.2f}%) -> Val={len(sampled):5,d}")

    # Adjust if slight rounding discrepancy
    actual_val = len(val_s1_set)
    print(f"\nTotal Selected Validation S1 Entities: {actual_val:,}")

    # 4. Save validation entity IDs and validation ground truth
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_ids_file = OUTPUT_DIR / "val_s1_ids.txt"
    val_gt_file = OUTPUT_DIR / "val_ground_truth.tsv"

    with open(val_ids_file, "w", encoding="utf-8") as f:
        for s1_id in sorted(val_s1_set):
            f.write(f"{s1_id}\n")

    val_links_count = 0
    with open(val_gt_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(val_s1_set):
            mids = gt_map[s1_id]
            val_links_count += len(mids)
            f.write(f"{s1_id}\t{','.join(mids)}\n")

    print(f"Validation ground truth written: {val_gt_file}")
    print(f"Validation total true links: {val_links_count:,} ({val_links_count / actual_val:.2f} links/entity)")
    print(f"Validation singletons: {sum(1 for s in val_s1_set if len(gt_map[s]) == 0):,} ({sum(1 for s in val_s1_set if len(gt_map[s]) == 0)/actual_val*100:.2f}%)")


if __name__ == "__main__":
    create_validation_split()
