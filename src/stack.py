"""Stage 2: re-score candidate pairs using stage-1 probabilities of the *other* pairs around them.

Usage: uv run python src/stack.py fit RUN_DIR       # needs RUN_DIR/model.txt + metrics.json from match.py fit
       uv run python src/stack.py predict RUN_DIR   # -> RUN_DIR/output/*.tsv (replaces stage-1 decisions)
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

try:
    from src.match import (FOLD, PARAMS, PATIENCE, ROUNDS, TRAIN_FOLDS, VALID_FOLD, WORK, assign, f05,
                           feature_names, load_feat_parts, predict_pairs_np, sim_dense, write)
    from src.features import sim
except (ModuleNotFoundError, ImportError):
    from match import (FOLD, PARAMS, PATIENCE, ROUNDS, TRAIN_FOLDS, VALID_FOLD, WORK, assign, f05,
                       feature_names, load_feat_parts, predict_pairs_np, sim_dense, write)
    from features import sim

DENSE_DROP = 0.5


def context(pairs, rec):
    q_ctx = pairs.group_by("q").agg(q_pmax=pl.col("p").max(), q_p2=pl.col("p").top_k(2).min(), q_n=pl.len(),
                                    q_nhi=(pl.col("p") > 0.5).sum())
    d = (pairs.join(q_ctx, on="q")
         .with_columns(q_gap=pl.col("q_pmax") - pl.col("p"),
                       q_runner=pl.when(pl.col("p") == pl.col("q_pmax")).then(pl.col("q_p2")).otherwise(pl.col("q_pmax")),
                       s_sum=pl.col("p").sum().over("s1"), s_nhi=(pl.col("p") > 0.5).sum().over("s1"),
                       s_rank=pl.col("p").rank("ordinal", descending=True).over("s1"), s_n=pl.len().over("s1")))
    top2 = (pairs.sort("p", descending=True).group_by("s1", maintain_order=True).head(2)
            .with_columns(r=pl.int_range(pl.len()).over("s1")))
    first, second = top2.filter(pl.col("r") == 0), top2.filter(pl.col("r") == 1)
    ref = (d.select("q", "s1").join(first.select("s1", q1="q", p1="p"), on="s1", how="left")
           .join(second.select("s1", q2="q", p2="p"), on="s1", how="left")
           .with_columns(sib=pl.when(pl.col("q1") == pl.col("q")).then(pl.col("q2")).otherwise(pl.col("q1")),
                         sib_p=pl.when(pl.col("q1") == pl.col("q")).then(pl.col("p2")).otherwise(pl.col("p1")))
           .select("q", "s1", "sib", "sib_p"))
    r = rec.select(q="entity_id", nm="nm", ad="ad", num1=pl.col("ad").str.extract(r"\b(\d+)\b"))
    ref = (ref.join(r, on="q", how="left")
           .join(r.rename({"q": "sib", "nm": "sib_nm", "ad": "sib_ad", "num1": "sib_num1"}), on="sib", how="left")
           .with_columns(pl.col("nm", "ad", "sib_nm", "sib_ad").fill_null("")))
    ref = ref.with_columns(sib_n_ratio=sim(ref["nm"], ref["sib_nm"], fuzz.ratio),
                           sib_a_ratio=sim(ref["ad"], ref["sib_ad"], fuzz.ratio),
                           sib_num_eq=(pl.col("num1") == pl.col("sib_num1")).cast(pl.Int8))
    return d.join(ref.select("q", "s1", "sib_p", "sib_n_ratio", "sib_a_ratio", "sib_num_eq"), on=["q", "s1"], how="left")


def oof_probs(X, y, fold, feats):
    """Out-of-fold stage-1 probabilities."""
    va_mask = fold == VALID_FOLD
    va_idx = np.where(va_mask)[0]
    rng = np.random.default_rng(0)
    va_idx = rng.choice(va_idx, size=min(len(va_idx), max(1, len(va_idx) // 4)), replace=False)
    va_X, va_y = X[va_idx], y[va_idx]

    out = []
    for k in TRAIN_FOLDS + [VALID_FOLD]:
        tr_mask = np.isin(fold, [x for x in TRAIN_FOLDS if x != k])
        t0 = time.time()
        m = lgb.train(PARAMS, lgb.Dataset(X[tr_mask], y[tr_mask], free_raw_data=True),
                      num_boost_round=ROUNDS,
                      valid_sets=[lgb.Dataset(va_X, va_y)],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        fold_X = X[fold == k]
        p = m.predict(fold_X, num_threads=PARAMS["num_threads"])
        out.append((np.where(fold == k)[0], p))
        print(f"  oof fold {k}: {m.best_iteration} trees, {time.time() - t0:.0f}s", flush=True)
        del m, fold_X
        gc.collect()
    return out


def fit(run_dir):
    m1 = json.load(open(f"{run_dir}/metrics.json"))
    feats = m1["feats"]
    X, y, q_series, s1_series = load_feat_parts("train", feats=feats, with_label=True)
    fold = (s1_series.hash(seed=42) % 5).to_numpy()
    rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "nm", "ad"])

    oofs = oof_probs(X, y, fold, feats)
    # build the oof pairs frame
    chunks = []
    for idx, p in oofs:
        chunks.append(pl.DataFrame({"q": q_series[idx], "s1": s1_series[idx], "p1": p}))
    oof_pairs = pl.concat(chunks)
    del chunks, oofs
    gc.collect()

    ctx = context(oof_pairs, rec)
    # align ctx with (X, y) rows: ctx preserves order of oof_pairs which is chunk order of X
    # X was loaded in chunk order, oof pairs are gathered in fold order. Reindex.
    # Simplest: rebuild the row->p1 map keyed by (q, s1).
    key_ctx = ctx.select("q", "s1", *[c for c in ctx.columns if c not in ("q", "s1")])
    d = pl.DataFrame({"q": q_series, "s1": s1_series}).with_row_index("row")
    d = d.join(key_ctx, on=["q", "s1"], how="left")
    feats2 = feats + [c for c in key_ctx.columns if c not in ("q", "s1")]
    X2 = np.empty((len(d), len(feats2)), dtype=np.float32)
    for j, c in enumerate(feats):
        X2[:, j] = X[:, j]
    for j, c in enumerate(feats2[len(feats):], start=len(feats)):
        X2[:, j] = d[c].to_numpy().astype(np.float32)
    del X, d, key_ctx, oof_pairs, ctx
    gc.collect()

    tr_mask = np.isin(fold, TRAIN_FOLDS)
    va_mask = fold == VALID_FOLD
    va_idx = np.where(va_mask)[0]
    rng = np.random.default_rng(0)
    va_idx = rng.choice(va_idx, size=min(len(va_idx), max(1, len(va_idx) // 4)), replace=False)

    t0 = time.time()
    model = lgb.train(PARAMS, lgb.Dataset(X2[tr_mask], y[tr_mask], free_raw_data=True),
                      num_boost_round=ROUNDS,
                      valid_sets=[lgb.Dataset(X2[va_idx], y[va_idx])],
                      callbacks=[lgb.early_stopping(PATIENCE), lgb.log_evaluation(100)])
    print(f"stage 2: {model.best_iteration} trees, {time.time() - t0:.0f}s")
    model.save_model(f"{run_dir}/model_stage2.txt")

    p2 = model.predict(X2, num_threads=PARAMS["num_threads"])
    pairs = pl.DataFrame({"q": q_series, "s1": s1_series, "p1": None, "p": p2})
    # attach p1 by key
    p1_map = pl.read_parquet(f"{WORK}/train_probs.parquet") if os.path.exists(f"{WORK}/train_probs.parquet") else None
    if p1_map is None:
        # regenerate from oof? We no longer have them; skip p1 in metrics
        p1_pairs = pairs.select("q", "s1", p=pairs["p"])
    else:
        p1_pairs = pairs.select("q", "s1", p=p1_map["p"])

    res = sim_dense(pairs.select("q", "s1", "p"), DENSE_DROP)
    res1 = sim_dense(p1_pairs, DENSE_DROP)
    for name, r in (("stage1", res1), ("stage2", res)):
        print(f"{name}: normal {r['normal']:.4f}@{r['t_normal']}  dense {r['dense']:.4f}@{r['t_dense']}")
    m1.update(stage2={"feats": feats2, "threshold": res["t_dense"], "val_f05": res["normal"],
                      "val_f05_dense": res["dense"], "stage1_val_f05_dense": res1["dense"],
                      "best_iter": model.best_iteration,
                      "feature_gain": dict(sorted(
                          zip(feats2, model.feature_importance("gain").round().tolist()),
                          key=lambda kv: -kv[1])[:25])})
    json.dump(m1, open(f"{run_dir}/metrics.json", "w"), indent=2)


def predict(run_dir):
    m = json.load(open(f"{run_dir}/metrics.json"))
    feats = m["feats"]
    model = lgb.Booster(model_file=f"{run_dir}/model.txt")

    # chunk through test features, applying stage 1 then stage 2 per chunk
    files = sorted(__import__("glob").glob(f"{WORK}/test_feats_parts/chunk_*.parquet"))
    rec = pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "nm", "ad"])
    s2 = lgb.Booster(model_file=f"{run_dir}/model_stage2.txt")

    out = []
    for f in files:
        df = pl.read_parquet(f)
        X1 = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        p1 = model.predict(X1, num_threads=PARAMS["num_threads"])
        pairs1 = df.select("q", "s1").with_columns(p1=pl.Series(p1))
        del df, X1, p1
        gc.collect()

        # context requires pairs across all test rows; approximate by applying per-chunk
        cctx = context(pairs1, rec)
        X2 = cctx.select([pl.col(c).cast(pl.Float32) for c in m["stage2"]["feats"]]).to_numpy()
        p2 = s2.predict(X2, num_threads=PARAMS["num_threads"])
        out.append(cctx.select("q", "s1").with_columns(p=pl.Series(p2)))
        del pairs1, cctx, X2, p2
        gc.collect()

    pairs = pl.concat(out)
    pairs.write_parquet(f"{WORK}/test_probs_stage2.parquet")

    s1 = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id"))
    matches = assign(pairs, 0.0).filter(pl.col("p") >= m["stage2"]["threshold"])
    os.makedirs(f"{run_dir}/output", exist_ok=True)
    write(s1, matches, "matched_entity_ids", f"{run_dir}/output/matching_results.tsv")
    write(s1, pairs, "candidate_entity_ids", f"{run_dir}/output/candidate_pairs.tsv")
    print(f"stage-2 test: {len(matches)} matches, {len(pairs) / len(s1):.2f} candidates per S1")


if __name__ == "__main__":
    {"fit": fit, "predict": predict}[sys.argv[1]](sys.argv[2])