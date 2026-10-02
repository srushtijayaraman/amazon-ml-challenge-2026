# Calibration & Decoder Experiments

**Focus:** Post-Processing, Isotonic Probability Calibration & Dynamic Programming Expected-$F_{0.5}$ Decoding
**Status:** Experimental Exploration (Documented for Research & Completeness)

---

## 1. Overview

This directory contains research experiments evaluating alternative decision layers and calibration techniques:
- **Runner-Up Margin Thresholding:** Enforces a minimum probability gap between the top candidate and runner-up candidates to prevent ambiguous multi-matches.
- **Isotonic Calibration:** Fits a non-parametric isotonic regression model on an entity-disjoint calibration fold to align model confidence scores with true empirical probabilities.
- **Dynamic Programming Expected-$F_{0.5}$ Decoder:** Evaluates optimal top-$k$ candidate selection maximizing expected per-entity $F_{0.5}$ under calibrated probabilities in $O(n^2)$ time.

---

## 2. Included Files

- [`exp_01_calibration_and_decoder.py`](file:///experiments/calibration/exp_01_calibration_and_decoder.py): Script comparing global thresholding, margin thresholding, and calibrated expected-$F_{0.5}$ decoding on disjoint validation splits.
- [`test_expected_f05_decoder.py`](file:///experiments/calibration/test_expected_f05_decoder.py): Unit test verifying the $O(n^2)$ dynamic programming expected-$F_{0.5}$ decoder against exhaustive $O(2^n)$ brute-force enumeration across diverse probability configurations.

---

## 3. Findings

While expected-$F_{0.5}$ decoding provides exact mathematical optimality for single-entity decision problems, global argmax assignment with threshold $\tau^* = 0.75$ combined with within-record relative tie-break features achieved superior empirical performance while executing orders of magnitude faster at scale.
