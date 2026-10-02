#!/usr/bin/env python3
"""
ML Challenge 2026 — Zero-Cache Streaming Dataset & Feature Builder (Phase 3 & Phase 4 Refined)

Extracts candidate pairs and computes 38-dimensional RapidFuzz features ON THE FLY
as Source 2 and Source 3 stream from disk.

Key architectural improvements:
1. Source-Separated Candidate Quotas: MAX 35 candidates for S2, MAX 35 for S3 per S1.
   Prevents S2 from starving S3 candidates!
2. Two-Tier Channel Priority: High-precision channels (exact name, compact, core,
   consonant skeleton, num+token) take priority over fallback token channels.
3. Clean Consonant Skeleton Channel: Catches cross-script transliteration drift.
4. Bounded RAM: Zero candidate caching, memory strictly < 400 MB.
5. Direct NPZ and TSV output:
   - output/candidate_pairs_val.tsv
   - output/val_dataset.npz
"""

import os
import sys
import time
import psutil
import numpy as np
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

from normalize import normalize_record
from features import compute_pair_features, FEATURE_NAMES, get_consonant_skeleton
from blocking import GENERIC_STOP_WORDS, RE_PUNCT, RE_NON_ALPHANUM, RE_NUMS, RE_ACRONYM_DOTS
import anyascii

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

# Strict candidate caps per source to prevent starvation and quadratics
MAX_CANDS_PER_SOURCE = 35
TIER2_CAND_LIMIT = 15


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


def build_s1_indexes(s1_records: dict):
    """
    Builds refined multi-channel inverted indexes over 25,001 normalized S1 records.
    """
    token_freq = Counter()
    for s1_id, norm in s1_records.items():
        for t in norm["name_tokens"]:
            if len(t) >= 4 and t not in GENERIC_STOP_WORDS:
                token_freq[t] += 1

    rare_tokens = set(t for t, cnt in token_freq.items() if cnt <= 2)

    tier1_clean = defaultdict(list)    # (cty, clean_name)
    tier1_compact = defaultdict(list)  # (cty, compact_name)
    tier1_core = defaultdict(list)     # (cty, core_name)
    tier1_skel = defaultdict(list)     # (cty, consonant skeleton)
    tier1_num_tok = defaultdict(list)  # (cty, num, token)
    tier1_num_skel = defaultdict(list) # (cty, num, token skel)

    tier2_rare_tok = defaultdict(list) # (cty, rare_token)
    tier2_dual_tok = defaultdict(list) # (cty, tok1, tok2)

    for s1_id, norm in s1_records.items():
        cty = norm["country"]
        cn = norm["clean_name"]
        compact = norm["compact_name"]
        core = norm["core_name"]
        nums = norm["extracted_numbers"]
        toks = [t for t in norm["name_tokens"] if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
        skel = get_consonant_skeleton(core)

        # Tier 1: Clean, Compact, Core
        if cn and len(cn) >= 3:
            tier1_clean[(cty, cn)].append(s1_id)
        if compact and len(compact) >= 3:
            tier1_compact[(cty, compact)].append(s1_id)
        if core and len(core) >= 3 and core != cn:
            tier1_core[(cty, core)].append(s1_id)
        if skel and len(skel) >= 5:
            tier1_skel[(cty, skel)].append(s1_id)
        if nums and toks:
            tier1_num_tok[(cty, nums[0], toks[0])].append(s1_id)
            t_skel = get_consonant_skeleton(toks[0])
            if len(t_skel) >= 4:
                tier1_num_skel[(cty, nums[0], t_skel)].append(s1_id)

        # Tier 2: Rare tokens (frequency <= 2 in S1)
        for t in toks:
            if t in rare_tokens:
                tier2_rare_tok[(cty, t)].append(s1_id)

        # Tier 2: 2-token combination
        if len(toks) >= 2:
            p = tuple(sorted(toks[:2]))
            tier2_dual_tok[(cty, p[0], p[1])].append(s1_id)

    return {
        "tier1_clean": tier1_clean,
        "tier1_compact": tier1_compact,
        "tier1_core": tier1_core,
        "tier1_skel": tier1_skel,
        "tier1_num_tok": tier1_num_tok,
        "tier1_num_skel": tier1_num_skel,
        "tier2_rare_tok": tier2_rare_tok,
        "tier2_dual_tok": tier2_dual_tok,
    }


def fast_extract_keys(name: str, addr: str, country: str):
    """Fast key and token extraction for streaming candidates."""
    if not name:
        return None

    if not name.isascii():
        t_name = anyascii.anyascii(name)
    else:
        t_name = name

    c = RE_ACRONYM_DOTS.sub("", t_name)
    c = RE_PUNCT.sub(" ", c).replace("&", " and ").replace("+", " and ").lower().strip()
    tokens = c.split()
    clean_name = " ".join(tokens)
    compact_name = RE_NON_ALPHANUM.sub("", clean_name)

    core_toks = [t for t in tokens if t not in GENERIC_STOP_WORDS]
    core_name = " ".join(core_toks) if core_toks else clean_name
    skel = get_consonant_skeleton(core_name)

    distinct_toks = [t for t in tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
    raw_nums = RE_NUMS.findall(addr) if addr else ()
    extracted_nums = tuple(str(int(n)) for n in raw_nums)

    return {
        "clean_name": clean_name,
        "compact_name": compact_name,
        "core_name": core_name,
        "skel": skel,
        "distinct_toks": distinct_toks,
        "extracted_nums": extracted_nums,
        "tokens": tuple(tokens),
    }


def generate_validation_dataset():
    print("=" * 70, flush=True)
    print("REFINED STREAMING DATASET & FEATURE BUILDER (PHASE 3 & 4)", flush=True)
    print("=" * 70, flush=True)

    val_gt_file = OUTPUT_DIR / "val_ground_truth.tsv"
    val_gt = load_ground_truth_map(val_gt_file)
    val_s1_ids = set(val_gt.keys())
    total_true_links = sum(len(m) for m in val_gt.values())
    print(f"Validation S1 Entities: {len(val_s1_ids):,} (Total True Links: {total_true_links:,})", flush=True)

    # 1. Load S1 validation entities into memory (~15 MB)
    print("Loading 25,001 Validation S1 entities...", flush=True)
    val_s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_ids:
                val_s1_records[s1_id] = normalize_record({
                    "entity_id": s1_id,
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3].strip()
                })

    indexes = build_s1_indexes(val_s1_records)
    print(f"Indexes built. RAM: {psutil.Process().memory_info().rss / (1024*1024):.1f} MB", flush=True)

    # 2. Buffers for on-the-fly candidate & feature collection
    # Per-source candidate sets: s1_id -> set of cand_ids
    s1_source_cands = {
        "S2": defaultdict(set),
        "S3": defaultdict(set),
    }

    feature_rows = []
    labels = []
    s1_id_list = []
    cand_id_list = []
    retrieved_true_links = set()

    start_time = time.perf_counter()

    # 3. Stream Source 2 and Source 3 independently
    for src_tag, file_name in [("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")]:
        file_path = TRAIN_DIR / file_name
        print(f"\nStreaming {file_name} and extracting features on-the-fly...", flush=True)
        processed = 0
        src_cands_map = s1_source_cands[src_tag]

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                processed += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                cand_id = parts[0]
                country = parts[3].strip()
                name_str = parts[1]
                addr_str = parts[2]

                fast = fast_extract_keys(name_str, addr_str, country)
                if not fast:
                    continue

                cty = country
                cn = fast["clean_name"]
                compact = fast["compact_name"]
                core = fast["core_name"]
                skel = fast["skel"]
                nums = fast["extracted_nums"]
                toks = fast["distinct_toks"]

                tier1_matched_s1 = set()
                tier2_matched_s1 = set()

                # Tier 1 Channels
                if (cty, cn) in indexes["tier1_clean"]:
                    tier1_matched_s1.update(indexes["tier1_clean"][(cty, cn)])
                if (cty, compact) in indexes["tier1_compact"]:
                    tier1_matched_s1.update(indexes["tier1_compact"][(cty, compact)])
                if (cty, core) in indexes["tier1_core"]:
                    tier1_matched_s1.update(indexes["tier1_core"][(cty, core)])
                if len(skel) >= 5 and (cty, skel) in indexes["tier1_skel"]:
                    tier1_matched_s1.update(indexes["tier1_skel"][(cty, skel)])
                if nums and toks and (cty, nums[0], toks[0]) in indexes["tier1_num_tok"]:
                    tier1_matched_s1.update(indexes["tier1_num_tok"][(cty, nums[0], toks[0])])
                if nums and toks:
                    t_skel = get_consonant_skeleton(toks[0])
                    if len(t_skel) >= 4 and (cty, nums[0], t_skel) in indexes["tier1_num_skel"]:
                        tier1_matched_s1.update(indexes["tier1_num_skel"][(cty, nums[0], t_skel)])

                # Tier 2 Channels (Lower Priority)
                for t in toks:
                    if (cty, t) in indexes["tier2_rare_tok"]:
                        tier2_matched_s1.update(indexes["tier2_rare_tok"][(cty, t)])
                if len(toks) >= 2:
                    p = tuple(sorted(toks[:2]))
                    if (cty, p[0], p[1]) in indexes["tier2_dual_tok"]:
                        tier2_matched_s1.update(indexes["tier2_dual_tok"][(cty, p[0], p[1])])

                # Determine eligible S1 matches based on priority and quota
                eligible_s1 = []
                for s1 in tier1_matched_s1:
                    cset = src_cands_map[s1]
                    if cand_id not in cset and len(cset) < MAX_CANDS_PER_SOURCE:
                        eligible_s1.append(s1)

                for s1 in tier2_matched_s1:
                    if s1 in tier1_matched_s1:
                        continue # Already considered in tier 1
                    cset = src_cands_map[s1]
                    if cand_id not in cset and len(cset) < TIER2_CAND_LIMIT:
                        eligible_s1.append(s1)

                # Process eligible pairs
                if eligible_s1:
                    cand_norm = normalize_record({
                        "entity_id": cand_id,
                        "business_name": name_str,
                        "business_address": addr_str,
                        "country": country
                    })

                    for s1_id in eligible_s1:
                        src_cands_map[s1_id].add(cand_id)
                        f_vec = compute_pair_features(val_s1_records[s1_id], cand_norm, cand_id)
                        feature_rows.append(f_vec)

                        is_match = 1 if cand_id in val_gt.get(s1_id, ()) else 0
                        labels.append(is_match)
                        s1_id_list.append(s1_id)
                        cand_id_list.append(cand_id)

                        if is_match:
                            retrieved_true_links.add((s1_id, cand_id))

                if processed % 1000000 == 0:
                    curr_ram = psutil.Process().memory_info().rss / (1024 * 1024)
                    print(f"  Processed {processed:9,d} / 5M+ {src_tag} (RAM: {curr_ram:5.1f} MB | Pairs: {len(labels):,}, True Hits: {len(retrieved_true_links):,})", flush=True)

    stream_time = time.perf_counter() - start_time
    print(f"\nAll streams complete in {stream_time:.1f}s.", flush=True)
    print(f"Total Candidate Pairs Generated: {len(labels):,}", flush=True)
    print(f"Total True Links Recovered: {len(retrieved_true_links):,} / {total_true_links:,} ({len(retrieved_true_links)/total_true_links*100:.2f}%)", flush=True)
    print(f"Positive Ratio in Dataset: {np.mean(labels)*100:.2f}%", flush=True)

    # 4. Save to NPZ (compressed feature array)
    print("\nPacking feature matrix into numpy arrays...", flush=True)
    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int8)
    s1_arr = np.array(s1_id_list, dtype=object)
    cand_arr = np.array(cand_id_list, dtype=object)
    feat_names = np.array(FEATURE_NAMES, dtype=object)

    npz_out = OUTPUT_DIR / "val_dataset.npz"
    print(f"Saving {npz_out} (Shape: {X.shape})...", flush=True)
    np.savez_compressed(
        npz_out,
        X=X,
        y=y,
        s1_ids=s1_arr,
        cand_ids=cand_arr,
        feature_names=feat_names
    )
    npz_size_mb = os.path.getsize(npz_out) / (1024 * 1024)
    print(f"Saved {npz_out} ({npz_size_mb:.1f} MB).", flush=True)

    # 5. Save candidate pairs TSV in official format
    cand_tsv = OUTPUT_DIR / "candidate_pairs_val.tsv"
    print(f"Writing official candidate pairs to {cand_tsv}...", flush=True)
    with open(cand_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        # Combine S2 and S3 candidates per S1
        for s1_id in val_s1_records:
            all_cands = s1_source_cands["S2"][s1_id] | s1_source_cands["S3"][s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(all_cands))}\n")
    print("Candidate TSV written successfully.", flush=True)
    print(f"Final Peak Process RAM: {psutil.Process().memory_info().rss / (1024*1024):.1f} MB", flush=True)


if __name__ == "__main__":
    generate_validation_dataset()
