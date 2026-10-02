#!/usr/bin/env python3
"""
ML Challenge 2026 — Memory-Safe Dataset Autopsy (Phase 0)
Streams all datasets sequentially with O(1) memory footprint (< 150 MB RAM).
Captures exact ground truth distributions, missingness, noise patterns, and samples pairs.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import argparse
import psutil
from pathlib import Path
from collections import Counter, defaultdict

DELIM = "\t"
ROOT = Path(__file__).resolve().parent.parent


def get_mem_mb():
    """Return current process Resident Set Size (RSS) in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def analyze_ground_truth(gt_path):
    print("\n" + "=" * 70)
    print(f"ANALYZING GROUND TRUTH: {gt_path.name}")
    print(f"Memory before GT analysis: {get_mem_mb():.1f} MB")
    print("=" * 70)

    total_s1 = 0
    zero_matches = 0
    match_count_dist = Counter()
    s2_matches_total = 0
    s3_matches_total = 0

    s1_matched_s2_only = 0
    s1_matched_s3_only = 0
    s1_matched_both = 0

    max_matches = 0
    max_matches_s1 = None

    # We will pick 40 sample S1 entities with their matched IDs to inspect name/address variations
    # 20 single-match, 20 multi-match
    sample_targets = {}
    samples_single = 0
    samples_multi = 0

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            total_s1 += 1

            if not matched_str.strip():
                zero_matches += 1
                match_count_dist[0] += 1
                continue

            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
            num_matches = len(matched_ids)
            match_count_dist[num_matches] += 1

            if num_matches > max_matches:
                max_matches = num_matches
                max_matches_s1 = s1_id

            has_s2 = any(m.startswith("S2-") for m in matched_ids)
            has_s3 = any(m.startswith("S3-") for m in matched_ids)

            s2_cnt = sum(1 for m in matched_ids if m.startswith("S2-"))
            s3_cnt = sum(1 for m in matched_ids if m.startswith("S3-"))
            s2_matches_total += s2_cnt
            s3_matches_total += s3_cnt

            if has_s2 and has_s3:
                s1_matched_both += 1
            elif has_s2:
                s1_matched_s2_only += 1
            elif has_s3:
                s1_matched_s3_only += 1

            # Collect a balanced sample of entities for pair inspection
            if num_matches == 1 and samples_single < 20:
                sample_targets[s1_id] = matched_ids
                samples_single += 1
            elif num_matches >= 2 and samples_multi < 20:
                sample_targets[s1_id] = matched_ids
                samples_multi += 1

    total_positive_links = s2_matches_total + s3_matches_total
    entities_with_matches = total_s1 - zero_matches

    print(f"Total S1 entities in Ground Truth : {total_s1:,}")
    print(f"Zero-match entities (singletons)  : {zero_matches:,} ({zero_matches / total_s1 * 100:.2f}%)")
    print(f"Entities with >= 1 match          : {entities_with_matches:,} ({entities_with_matches / total_s1 * 100:.2f}%)")
    print(f"  - Matches S2 only               : {s1_matched_s2_only:,} ({s1_matched_s2_only / total_s1 * 100:.2f}%)")
    print(f"  - Matches S3 only               : {s1_matched_s3_only:,} ({s1_matched_s3_only / total_s1 * 100:.2f}%)")
    print(f"  - Matches BOTH S2 and S3        : {s1_matched_both:,} ({s1_matched_both / total_s1 * 100:.2f}%)")
    print(f"\nTotal positive candidate pairs    : {total_positive_links:,}")
    print(f"  - S1 -> S2 positive pairs       : {s2_matches_total:,} ({s2_matches_total / total_positive_links * 100:.2f}%)")
    print(f"  - S1 -> S3 positive pairs       : {s3_matches_total:,} ({s3_matches_total / total_positive_links * 100:.2f}%)")
    print(f"Max matches for a single entity   : {max_matches} (e.g. {max_matches_s1})")

    print("\nMatch Count Frequency (Matches per S1 entity):")
    for k in sorted(match_count_dist.keys()):
        if k <= 10 or k == max_matches:
            cnt = match_count_dist[k]
            pct = cnt / total_s1 * 100
            print(f"  {k:2d} matches : {cnt:10,d} ({pct:6.2f}%)")
        elif k == 11:
            cnt_over_10 = sum(match_count_dist[x] for x in match_count_dist if x > 10)
            print(f" >10 matches : {cnt_over_10:10,d} ({cnt_over_10 / total_s1 * 100:6.2f}%)")

    gt_stats = {
        "total_s1": total_s1,
        "zero_matches": zero_matches,
        "zero_matches_pct": zero_matches / total_s1 * 100,
        "entities_with_matches": entities_with_matches,
        "s1_matched_s2_only": s1_matched_s2_only,
        "s1_matched_s3_only": s1_matched_s3_only,
        "s1_matched_both": s1_matched_both,
        "total_positive_links": total_positive_links,
        "s2_matches_total": s2_matches_total,
        "s3_matches_total": s3_matches_total,
        "max_matches": max_matches,
        "match_count_dist": {str(k): v for k, v in match_count_dist.items()}
    }

    print(f"Memory after GT analysis: {get_mem_mb():.1f} MB")
    return gt_stats, sample_targets


def analyze_source_file(file_path, target_ids_to_collect=None):
    """
    Stream a single TSV file line by line.
    Computes summary metrics without storing full rows in RAM.
    If target_ids_to_collect is provided, returns dict of collected records.
    """
    print("\n" + "=" * 70)
    print(f"STREAMING PROFILE: {file_path.name}")
    print(f"Memory before file: {get_mem_mb():.1f} MB")
    print("=" * 70)

    total_rows = 0
    missing_counts = Counter()
    country_counts = Counter()

    name_len_sum = 0
    name_min_len = 10**9
    name_max_len = 0

    addr_len_sum = 0
    addr_min_len = 10**9
    addr_max_len = 0

    first_5_rows = []
    collected_records = {}

    target_set = set(target_ids_to_collect) if target_ids_to_collect else set()

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return {}, {}
        cols = [c.strip().lower() for c in header_line.rstrip("\n").split(DELIM)]

        # Expected cols: entity_id, business_name, business_address, country
        try:
            id_idx = cols.index("entity_id")
            name_idx = cols.index("business_name")
            addr_idx = cols.index("business_address")
            country_idx = cols.index("country")
        except ValueError as e:
            print(f"Error parsing columns in {file_path.name}: {cols} - {e}")
            return {}, {}

        for line in f:
            total_rows += 1
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            # Safe unpack
            ent_id = parts[id_idx] if id_idx < len(parts) else ""
            b_name = parts[name_idx] if name_idx < len(parts) else ""
            b_addr = parts[addr_idx] if addr_idx < len(parts) else ""
            country = parts[country_idx] if country_idx < len(parts) else ""

            # Check missing
            if not ent_id.strip():
                missing_counts["entity_id"] += 1
            if not b_name.strip():
                missing_counts["business_name"] += 1
            if not b_addr.strip():
                missing_counts["business_address"] += 1
            if not country.strip():
                missing_counts["country"] += 1

            country_counts[country] += 1

            # Length stats
            n_len = len(b_name)
            name_len_sum += n_len
            if n_len < name_min_len:
                name_min_len = n_len
            if n_len > name_max_len:
                name_max_len = n_len

            a_len = len(b_addr)
            addr_len_sum += a_len
            if a_len < addr_min_len:
                addr_min_len = a_len
            if a_len > addr_max_len:
                addr_max_len = a_len

            if len(first_5_rows) < 5:
                first_5_rows.append({
                    "entity_id": ent_id,
                    "business_name": b_name,
                    "business_address": b_addr,
                    "country": country
                })

            if ent_id in target_set:
                collected_records[ent_id] = {
                    "entity_id": ent_id,
                    "business_name": b_name,
                    "business_address": b_addr,
                    "country": country
                }

    avg_name_len = name_len_sum / total_rows if total_rows > 0 else 0
    avg_addr_len = addr_len_sum / total_rows if total_rows > 0 else 0

    print(f"Total rows: {total_rows:,}")
    print("\nMissing values:")
    for col in cols:
        cnt = missing_counts[col]
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        print(f"  {col:18s}: {cnt:8,d} ({pct:5.2f}%)")

    print("\nCountry Distribution:")
    for c, cnt in country_counts.most_common():
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        print(f"  {c:10s}: {cnt:10,d} ({pct:6.2f}%)")

    print("\nString Length Statistics:")
    print(f"  Business Name    : min={name_min_len}, max={name_max_len}, avg={avg_name_len:.2f}")
    print(f"  Business Address : min={addr_min_len}, max={addr_max_len}, avg={avg_addr_len:.2f}")

    print("\nFirst 3 sample rows:")
    for r in first_5_rows[:3]:
        print(f"  ID: {r['entity_id']} | Country: {r['country']}")
        print(f"    Name: {r['business_name']}")
        print(f"    Addr: {r['business_address']}")

    print(f"Memory after file: {get_mem_mb():.1f} MB")

    stats = {
        "file": file_path.name,
        "total_rows": total_rows,
        "missing": dict(missing_counts),
        "countries": dict(country_counts),
        "name_len": {"min": name_min_len, "max": name_max_len, "avg": avg_name_len},
        "addr_len": {"min": addr_min_len, "max": addr_max_len, "avg": avg_addr_len},
        "sample_rows": first_5_rows
    }
    return stats, collected_records


def print_noise_and_corruption_deepdive(sample_targets, s1_records, s2_records, s3_records):
    print("\n" + "=" * 70)
    print("EMPIRICAL TRUE-MATCH CORRUPTION & NOISE ANALYSIS")
    print("Direct inspection of actual positive match pairs across sources")
    print("=" * 70)

    inspected_count = 0
    for s1_id, matched_ids in sample_targets.items():
        if s1_id not in s1_records:
            continue
        s1 = s1_records[s1_id]
        print(f"\n--- Entity: {s1_id} ({s1['country']}) ---")
        print(f"  [S1 Reference]")
        print(f"    Name: {s1['business_name']}")
        print(f"    Addr: {s1['business_address']}")

        for m_id in matched_ids:
            target_dict = s2_records if m_id.startswith("S2-") else s3_records
            if m_id in target_dict:
                m = target_dict[m_id]
                source_tag = m_id.split("-")[0]
                print(f"  [{source_tag} Match: {m_id}] (Country: {m['country']})")
                print(f"    Name: {m['business_name']}")
                print(f"    Addr: {m['business_address']}")
            else:
                print(f"  [Match: {m_id}] (Record not found in sample)")

        inspected_count += 1
        if inspected_count >= 15:
            break


def main():
    parser = argparse.ArgumentParser(description="Memory-Safe Streaming Dataset Autopsy for Amazon ML Challenge 2026")
    parser.add_argument("--data-dir", default=str(ROOT / "dataset"), help="Path to dataset directory containing train/ and test/ (default: dataset)")
    parser.add_argument("--output-dir", default=str(ROOT / "output"), help="Path to output directory (default: output)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)

    print("==================================================")
    print("  AMAZON ML CHALLENGE 2026 — DATASET AUTOPSY")
    print("==================================================")
    print(f"Starting memory usage: {get_mem_mb():.1f} MB")

    gt_file = train_dir / "train_ground_truth.tsv"
    if not gt_file.exists():
        print(f"\n[ERROR] Ground truth file not found at: {gt_file}")
        print(f"Please ensure competition dataset files are placed in '{data_dir}' or specify --data-dir.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth
    gt_stats, sample_targets = analyze_ground_truth(gt_file)

    # Collect list of all target IDs needed for pair inspection
    target_s1_ids = set(sample_targets.keys())
    target_s2_ids = set()
    target_s3_ids = set()
    for mids in sample_targets.values():
        for m in mids:
            if m.startswith("S2-"):
                target_s2_ids.add(m)
            elif m.startswith("S3-"):
                target_s3_ids.add(m)

    # 2. Train files
    s1_stats, s1_records = analyze_source_file(train_dir / "train_source1.tsv", target_s1_ids)
    s2_stats, s2_records = analyze_source_file(train_dir / "train_source2.tsv", target_s2_ids)
    s3_stats, s3_records = analyze_source_file(train_dir / "train_source3.tsv", target_s3_ids)

    # 3. Test files
    test_s1_stats, _ = analyze_source_file(test_dir / "test_source1.tsv")
    test_s2_stats, _ = analyze_source_file(test_dir / "test_source2.tsv")
    test_s3_stats, _ = analyze_source_file(test_dir / "test_source3.tsv")

    # 4. Pair inspection
    print_noise_and_corruption_deepdive(sample_targets, s1_records, s2_records, s3_records)

    # Save summary
    summary = {
        "ground_truth": gt_stats,
        "train_source1": s1_stats,
        "train_source2": s2_stats,
        "train_source3": s3_stats,
        "test_source1": test_s1_stats,
        "test_source2": test_s2_stats,
        "test_source3": test_s3_stats,
    }

    out_file = output_dir / "eda_autopsy_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Autopsy summary successfully saved to: {out_file}")
    print(f"Final Peak Process Memory: {get_mem_mb():.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
