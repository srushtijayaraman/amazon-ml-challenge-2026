#!/usr/bin/env python3
"""
End-to-End Test Inference & Validation Pipeline
Amazon ML Challenge 2026: Business Entity Resolution

This script coordinates the four pipeline stages to generate the final
submission artifacts and validate them against official competition rules:
  Stage 1: Multi-Pass Text Normalization (src/normalize.py)
  Stage 2: Reverse TF-IDF & Exact-Key Blocking (src/block.py)
  Stage 3: 61-Dimensional Pairwise Feature Engineering (src/features.py)
  Stage 4: LightGBM Inference & Decision Decoding (src/match.py)
  Stage 5: Official Submission Format Validation (scripts/validate_submission.py)

Usage:
  python scripts/run_inference.py --data-dir dataset --model-dir models --output-dir output
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, desc, env=None):
    print(f"\n{'='*70}\n[STEP] {desc}\nCommand: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"Error: Step '{desc}' failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"Completed '{desc}' in {time.time()-t0:.1f}s", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run complete test inference pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory containing test/ and train/")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained model.txt")
    parser.add_argument("--output-dir", default="output", help="Directory where submission TSVs will be saved")
    parser.add_argument("--skip-prep", action="store_true", help="Skip normalization and blocking if already generated in work/")
    args = parser.parse_args()

    env = os.environ.copy()
    env["DATA_DIR"] = str(Path(args.data_dir).resolve())

    # Ensure required model exists
    model_path = Path(args.model_dir) / "model.txt"
    if not model_path.exists():
        print(f"\n[ERROR] Model weights not found at: {model_path}", file=sys.stderr)
        print("Note: In accordance with competition data policies, pre-trained model weights are not", file=sys.stderr)
        print("redistributed in this repository. To run inference, please place training data in", file=sys.stderr)
        print("'dataset/train' and train a model via 'python src/match.py fit models', or provide", file=sys.stderr)
        print("your own trained 'model.txt' under the specified --model-dir.", file=sys.stderr)
        sys.exit(1)

    # Stage 1: Text Normalization
    if not args.skip_prep:
        run_command([sys.executable, "src/normalize.py"], "Stage 1: Multi-Pass Text Normalization", env=env)
        run_command([sys.executable, "src/block.py", "test"], "Stage 2a: Reverse TF-IDF & Exact-Key Candidate Retrieval", env=env)
        run_command([sys.executable, "src/block.py", "test", "--prune"], "Stage 2b: Relative Threshold & Hub Capping Pruning", env=env)
        run_command([sys.executable, "src/features.py", "test"], "Stage 3: 61-Dimensional Pairwise Feature Extraction", env=env)

    # Stage 4: LightGBM Inference & Decision Rule
    run_command([sys.executable, "src/match.py", "predict", args.model_dir], "Stage 4: LightGBM Scoring & Argmax Decision Decoding", env=env)

    # Stage 5: Output Sync & Submission Format Validation
    generated_dir = Path(args.model_dir) / "output"
    dest_dir = Path(args.output_dir)
    generated_matching = generated_dir / "matching_results.tsv"
    generated_cands = generated_dir / "candidate_pairs.tsv"

    matching_path = dest_dir / "matching_results.tsv"
    cands_path = dest_dir / "candidate_pairs.tsv"

    if not generated_matching.exists():
        print(f"\n[ERROR] Expected prediction output not found at: {generated_matching}", file=sys.stderr)
        print("Stage 4 (match.py predict) did not generate matching_results.tsv.", file=sys.stderr)
        sys.exit(1)

    if not generated_cands.exists():
        print(f"\n[ERROR] Expected candidate output not found at: {generated_cands}", file=sys.stderr)
        print("Stage 4 (match.py predict) did not generate candidate_pairs.tsv.", file=sys.stderr)
        sys.exit(1)

    if dest_dir.resolve() != generated_dir.resolve():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_matching, matching_path)
        shutil.copy2(generated_cands, cands_path)
        print(f"\nCopied submission artifacts:\n  {generated_matching} -> {matching_path}\n  {generated_cands} -> {cands_path}", flush=True)

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        print(f"\nNotice: Test directory not found at: {test_dir}. Skipping format validation.", file=sys.stderr)
        print(f"Outputs written to: {dest_dir}", flush=True)
        return

    run_command([
        sys.executable, "scripts/validate_submission.py",
        "--matching", str(matching_path),
        "--candidate", str(cands_path),
        "--test-dir", str(test_dir)
    ], "Stage 5: Official Submission Format Validation")

if __name__ == "__main__":
    main()
