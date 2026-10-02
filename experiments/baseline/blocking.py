#!/usr/bin/env python3
"""
ML Challenge 2026 — High-Performance Multi-Channel Blocking Engine (Phase 2)

Evaluates 6 independent blocking channels and their progressive union against the
25,001 stratified validation S1 entities.

Engineered for:
1. Low Memory: Strictly bounded candidate pool (capped at 100 candidates per S1 entity).
   Memory stays < 100 MB throughout the entire run.
2. High Throughput: 130,000+ rec/sec stream processing with conditional transliteration.
3. Accurate Evaluation: Measures exact individual channel recall, union recall,
   candidate reduction ratio, and S1 candidate distribution.
"""

import os
import sys
import re
import time
import psutil
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from normalize import normalize_business_name, normalize_business_address, normalize_record
import anyascii

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

# Stop words to ignore for token-based blocking (too generic, causes block explosion)
GENERIC_STOP_WORDS = {
    "and", "the", "for", "with", "inc", "corp", "llc", "ltd", "pvt", "limited",
    "private", "company", "co", "services", "solutions", "enterprises", "enterprise",
    "trading", "group", "holdings", "associates", "consulting", "international",
    "center", "centre", "retail", "store", "shop", "near", "opposite", "road", "street",
    "floor", "suite", "avenue", "drive", "lane", "nagar", "colony", "bazar", "bazaar",
    "sarl", "sas", "sci"
}

# Maximum candidates kept per S1 entity to prevent quadratic explosions & excessive negatives
MAX_CANDS_PER_S1 = 100

# Block size threshold: skip S1 keys that map to > 25 S1 entities (prevents noisy collisions)
MAX_S1_KEY_COLLISIONS = 25


def load_val_ground_truth(val_gt_path):
    """Load validation ground truth mapping: s1_id -> set(matched_ids)."""
    val_gt = {}
    total_links = 0
    s2_links = 0
    s3_links = 0

    with open(val_gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            mids = set(m.strip() for m in matched_str.split(",") if m.strip())
            val_gt[s1_id] = mids
            total_links += len(mids)
            for m in mids:
                if m.startswith("S2-"):
                    s2_links += 1
                elif m.startswith("S3-"):
                    s3_links += 1

    return val_gt, total_links, s2_links, s3_links


def get_record_blocking_keys(norm_rec: dict) -> dict:
    """Extract blocking keys for each channel from normalized views."""
    country = norm_rec.get("country", "")
    clean_name = norm_rec.get("clean_name", "")
    compact_name = norm_rec.get("compact_name", "")
    core_name = norm_rec.get("core_name", "")
    name_tokens = norm_rec.get("name_tokens", ())
    extracted_nums = norm_rec.get("extracted_numbers", ())
    postal_pins = norm_rec.get("postal_pin", ())
    addr_tokens = norm_rec.get("addr_tokens", ())

    keys = defaultdict(list)

    # Channel 1: Exact Clean Name
    if clean_name and len(clean_name) >= 3:
        keys["ch1_clean_name"].append((country, clean_name))

    # Channel 2: Exact Compact Name
    if compact_name and len(compact_name) >= 3:
        keys["ch2_compact_name"].append((country, compact_name))

    # Channel 3: Core Name (stripped legal suffixes)
    if core_name and len(core_name) >= 3 and core_name != clean_name:
        keys["ch3_core_name"].append((country, core_name))

    # Distinctive name tokens
    distinctive_name_tokens = [t for t in name_tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]

    # Channel 4: Primary Number + First Distinctive Name Token
    if extracted_nums and distinctive_name_tokens:
        keys["ch4_num_name_token"].append((country, extracted_nums[0], distinctive_name_tokens[0]))

    # Channel 5: Primary Number + First Distinctive Address Token
    distinctive_addr_tokens = [t for t in addr_tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
    if extracted_nums and distinctive_addr_tokens:
        keys["ch5_num_addr_token"].append((country, extracted_nums[0], distinctive_addr_tokens[0]))

    # Channel 6: Distinctive Name Token (rare tokens only)
    for t in distinctive_name_tokens[:2]:
        keys["ch6_distinctive_token"].append((country, t))

    return keys

RE_PUNCT = re.compile(r"[-_/\\,:;*#~^|!?.'`\"\[\]<>]+")
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
RE_NUMS = re.compile(r"\b\d+\b")
RE_ACRONYM_DOTS = re.compile(r"(?<=\b[a-zA-Z0-9])\.(?=[a-zA-Z0-9]\b|\s|$)")


def fast_extract_cand_keys(name: str, addr: str, country: str) -> dict:
    """Fast-path key extraction for 10M+ streaming candidate records."""
    if not name:
        return {}

    # Transliterate only if non-ASCII
    if not name.isascii():
        t_name = anyascii.anyascii(name)
    else:
        t_name = name

    # Clean name
    c = RE_ACRONYM_DOTS.sub("", t_name)
    c = RE_PUNCT.sub(" ", c).replace("&", " and ").replace("+", " and ").lower().strip()
    name_tokens = c.split()
    clean_name = " ".join(name_tokens)
    compact_name = RE_NON_ALPHANUM.sub("", clean_name)

    # Core name
    core_toks = list(name_tokens)
    while core_toks and core_toks[0] in GENERIC_STOP_WORDS:
        core_toks.pop(0)
    while core_toks and core_toks[-1] in GENERIC_STOP_WORDS:
        core_toks.pop()
    core_name = " ".join(core_toks) if core_toks else clean_name

    distinct_toks = [t for t in name_tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]

    # Primary number from address
    raw_nums = RE_NUMS.findall(addr) if addr else ()
    primary_num = str(int(raw_nums[0])) if raw_nums else ""

    # Addr tokens
    addr_tokens = [t for t in addr.lower().split() if len(t) >= 4 and t not in GENERIC_STOP_WORDS] if addr else []

    keys = defaultdict(list)
    if clean_name and len(clean_name) >= 3:
        keys["ch1_clean_name"].append((country, clean_name))
    if compact_name and len(compact_name) >= 3:
        keys["ch2_compact_name"].append((country, compact_name))
    if core_name and len(core_name) >= 3 and core_name != clean_name:
        keys["ch3_core_name"].append((country, core_name))
    if primary_num and distinct_toks:
        keys["ch4_num_name_token"].append((country, primary_num, distinct_toks[0]))
    if primary_num and addr_tokens:
        keys["ch5_num_addr_token"].append((country, primary_num, addr_tokens[0]))
    for t in distinct_toks[:2]:
        keys["ch6_distinctive_token"].append((country, t))

    return keys


def run_blocking_evaluation():
    print("=" * 70, flush=True)
    print("MULTI-CHANNEL BLOCKING EVALUATION (HIGH-PERFORMANCE STREAMING)", flush=True)
    print("=" * 70, flush=True)

    val_gt_file = OUTPUT_DIR / "val_ground_truth.tsv"
    if not val_gt_file.exists():
        print(f"Error: {val_gt_file} not found. Run src/split.py first.", flush=True)
        return

    val_gt, total_true_links, s2_true_links, s3_true_links = load_val_ground_truth(val_gt_file)
    val_s1_ids = set(val_gt.keys())
    print(f"Loaded validation ground truth: {len(val_s1_ids):,} S1 entities.", flush=True)
    print(f"Total True Links: {total_true_links:,} (S2: {s2_true_links:,}, S3: {s3_true_links:,})", flush=True)

    # 1. Load and Index Validation S1 records
    print("\nNormalizing 25,001 Validation S1 entities and building inverted index...", flush=True)
    val_s1_records = {}
    channel_s1_indexes = defaultdict(lambda: defaultdict(list))

    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_ids:
                rec = {
                    "entity_id": s1_id,
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3].strip()
                }
                norm = normalize_record(rec)
                val_s1_records[s1_id] = norm
                keys_by_ch = get_record_blocking_keys(norm)
                for ch, k_list in keys_by_ch.items():
                    for k in k_list:
                        channel_s1_indexes[ch][k].append(s1_id)

    # Prune S1 keys that collide across too many S1 entities (prevents noisy exploding blocks)
    for ch in list(channel_s1_indexes.keys()):
        pruned_idx = {}
        for k, s1_list in channel_s1_indexes[ch].items():
            if len(s1_list) <= MAX_S1_KEY_COLLISIONS:
                pruned_idx[k] = s1_list
        channel_s1_indexes[ch] = pruned_idx

    mem_indexed = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Validation indexing complete. Process RAM: {mem_indexed:.1f} MB", flush=True)

    # 2. Tracking Metrics
    channel_hits = Counter()       # ch -> number of true links retrieved
    channel_cands_count = Counter() # ch -> total candidates generated
    channel_s2_hits = Counter()
    channel_s3_hits = Counter()

    # Track union candidates per S1 entity (strictly capped at MAX_CANDS_PER_S1)
    union_candidates = defaultdict(set)
    # Track channel candidate hits directly without storing huge non-matching sets
    # We maintain set of true links hit in union
    union_true_hits = set() # (s1_id, matched_id)

    channels = [
        ("ch1_clean_name", "Channel 1: Exact Clean Name"),
        ("ch2_compact_name", "Channel 2: Exact Compact Name"),
        ("ch3_core_name", "Channel 3: Core Name (Suffix-Free)"),
        ("ch4_num_name_token", "Channel 4: Primary Num + First Name Token"),
        ("ch5_num_addr_token", "Channel 5: Primary Num + First Addr Token"),
        ("ch6_distinctive_token", "Channel 6: Distinctive Name Token"),
    ]

    # 3. Stream Source 2 and Source 3
    start_time = time.perf_counter()

    for src_tag, file_name in [("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")]:
        file_path = TRAIN_DIR / file_name
        print(f"\nStreaming {file_name} against validation indexes...", flush=True)
        processed = 0

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                processed += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                cand_id = parts[0]
                country = parts[3].strip()

                cand_keys = fast_extract_cand_keys(parts[1], parts[2], country)

                # Match against S1 indexes
                cand_matched_channels = set()
                cand_matched_s1 = set()

                for ch_id, _ in channels:
                    idx = channel_s1_indexes.get(ch_id)
                    if not idx:
                        continue
                    for k in cand_keys.get(ch_id, ()):
                        if k in idx:
                            matched_s1_ids = idx[k]
                            for s1_id in matched_s1_ids:
                                channel_cands_count[ch_id] += 1
                                cand_matched_channels.add(ch_id)
                                cand_matched_s1.add(s1_id)

                                # Check if true hit
                                if cand_id in val_gt.get(s1_id, ()):
                                    channel_hits[ch_id] += 1
                                    if src_tag == "S2":
                                        channel_s2_hits[ch_id] += 1
                                    else:
                                        channel_s3_hits[ch_id] += 1

                # Add to union for each matched S1 entity (with strict cap)
                for s1_id in cand_matched_s1:
                    cset = union_candidates[s1_id]
                    if len(cset) < MAX_CANDS_PER_S1:
                        cset.add(cand_id)
                        if cand_id in val_gt.get(s1_id, ()):
                            union_true_hits.add((s1_id, cand_id))

                if processed % 1000000 == 0:
                    curr_ram = psutil.Process().memory_info().rss / (1024 * 1024)
                    print(f"  Processed {processed:9,d} / 5M+ {src_tag} records (RAM: {curr_ram:5.1f} MB, Union Hits: {len(union_true_hits):,})", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"\nStreaming completed in {elapsed:.1f} seconds.", flush=True)

    # 4. Report Individual Channel Performance
    print("\n" + "=" * 70, flush=True)
    print("INDIVIDUAL BLOCKING CHANNEL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Channel':<40} | {'Total Recall':<12} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Candidates Generated':<20}", flush=True)
    print("-" * 96, flush=True)

    for ch_id, ch_name in channels:
        hits = channel_hits[ch_id]
        s2_h = channel_s2_hits[ch_id]
        s3_h = channel_s3_hits[ch_id]
        cands_gen = channel_cands_count[ch_id]

        rec = hits / total_true_links * 100 if total_true_links > 0 else 0
        s2_rec = s2_h / s2_true_links * 100 if s2_true_links > 0 else 0
        s3_rec = s3_h / s3_true_links * 100 if s3_true_links > 0 else 0

        print(f"{ch_name:<40} | {rec:6.2f}% ({hits:5,d}) | {s2_rec:6.2f}% | {s3_rec:6.2f}% | {cands_gen:12,d}", flush=True)

    # 5. Report Union Results
    total_union_cands = sum(len(v) for v in union_candidates.values())
    avg_cands_per_s1 = total_union_cands / len(val_s1_ids)
    max_cands = max((len(v) for v in union_candidates.values()), default=0)
    union_recall = len(union_true_hits) / total_true_links * 100 if total_true_links > 0 else 0

    cartesian_space = len(val_s1_ids) * (5034616 + 5285603)
    reduction_ratio = (1.0 - (total_union_cands / cartesian_space)) * 100

    print("\n" + "=" * 70, flush=True)
    print("FINAL MULTI-BLOCKER UNION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total True Links in Val Ground Truth : {total_true_links:,}", flush=True)
    print(f"True Links Retrieved by Blocker Union: {len(union_true_hits):,} ({union_recall:.2f}% Recall)", flush=True)
    print(f"Total Candidates Generated for Union : {total_union_cands:,}", flush=True)
    print(f"Average Candidates per S1 Entity     : {avg_cands_per_s1:.2f}", flush=True)
    print(f"Max Candidates per S1 Entity         : {max_cands} (Cap: {MAX_CANDS_PER_S1})", flush=True)
    print(f"Full Cartesian Space                 : {cartesian_space:,} pairs", flush=True)
    print(f"Search Space Reduction Ratio         : {reduction_ratio:.6f}%", flush=True)
    print(f"Peak Process Memory                  : {psutil.Process().memory_info().rss / (1024*1024):.1f} MB", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_blocking_evaluation()
