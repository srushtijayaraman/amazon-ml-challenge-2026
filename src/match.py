"""Train LightGBM on candidate-pair features, tune the match threshold for macro F0.5 on
held-out S1 entities, and write the submission files. Everything a run produces goes in RUN_DIR.

Reads chunked features from work/{split}_feats_parts/, loads them as float32, and saves the
trained model to disk *before* computing validation metrics so a crash after training is cheap.

Usage: uv run python src/match.py fit RUN_DIR       # -> model.txt, metrics.json, errors.tsv
       uv run python src/match.py predict RUN_DIR   # -> output/matching_results.tsv, output/candidate_pairs.tsv
"""
import gc
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

WORK = "work"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
FOLD = pl.col("s1").hash(seed=42) % 5
VALID_FOLD, TRAIN_FOLDS = 0, [1, 2, 3, 4]
PSEUDO_HI, PSEUDO_LO = 0.97, 0.03
PARAMS = dict(objective="binary", learning_rate=0.20, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
              max_bin=63, force_col_wise=True, min_sum_hessian_in_leaf=1e-3,
              num_threads=THREADS, verbose=-1)


ROUNDS, PATIENCE = 400, 30

# ---------- chunked feature loader ----------

def _chunk_files(split):
    files = sorted(glob.glob(f"{WORK}/{split}_feats_parts/chunk_*.parquet"))
    assert files, f"no feature chunks under {WORK}/{split}_feats_parts/"
    return files


def feature_names(split):
    schema = pl.scan_parquet(_chunk_files(split)[0]).collect_schema()
    return [c for c, dt in schema.items()
            if c not in ("q", "s1", "label") and dt.is_numeric()]

def load_feat_parts(split, feats=None, with_label=True, subsample=1.0, seed=0, folds=None):
    """Load chunked features into one float32 numpy matrix + polars (q, s1) side series.

    If `folds` is given (iterable of ints), only rows whose s1 hash-folds into that set
    are kept. The filter is applied at the parquet scan level so filtered-out rows are
    never materialised in the numpy matrix -- this is how fit() avoids holding two
    copies of the whole dataset at once.
    """
    files = _chunk_files(split)
    if feats is None:
        feats = feature_names(split)
    cols = list(feats) + ["q", "s1"] + (["label"] if with_label else [])
    fold_list = None if folds is None else list(folds)

    def scan(f):
        lf = pl.scan_parquet(f).select(cols)
        if fold_list is not None:
            lf = lf.filter(FOLD.is_in(fold_list))
        return lf

    # pass 1: how many rows (upper bound if subsampling)
    total = 0
    for f in files:
        total += scan(f).select(pl.len()).collect().item()
    if subsample < 1.0:
        total = int(total * min(1.0, subsample) * 1.05) + 1024
    total = max(total, 1024)

    X = np.empty((total, len(feats)), dtype=np.float32)
    y = np.empty(total, dtype=np.int8) if with_label else None
    qs, s1s = [], []
    pos = 0
    for f in files:
        df = scan(f).collect()
        if subsample < 1.0:
            df = df.sample(fraction=subsample, seed=seed)
        n = len(df)
        if n == 0:
            del df
            continue
        if pos + n > total:  # grow
            new_total = max(pos + n, total * 2)
            X2 = np.empty((new_total, len(feats)), dtype=np.float32)
            X2[:pos] = X[:pos]; X = X2
            if with_label:
                y2 = np.empty(new_total, dtype=np.int8)
                y2[:pos] = y[:pos]; y = y2
            total = new_total
        X[pos:pos + n] = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        qs.append(df["q"]); s1s.append(df["s1"])
        if with_label:
            y[pos:pos + n] = df["label"].to_numpy().astype(np.int8)
        pos += n
        del df
        gc.collect()
    X = X[:pos]
    if with_label:
        y = y[:pos]
    q = pl.concat(qs); s1 = pl.concat(s1s)
    return X, y, q, s1

def predict_pairs_np(model, split, feats):
    """Chunked prediction; returns polars (q, s1, p) in the same order as the chunks."""
    qs, s1s, ps = [], [], []
    for f in _chunk_files(split):
        df = pl.read_parquet(f, columns=list(feats) + ["q", "s1"])
        p = model.predict(df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(),
                          num_threads=THREADS)
        qs.append(df["q"]); s1s.append(df["s1"]); ps.append(p)
        del df
        gc.collect()
    return pl.DataFrame({"q": pl.concat(qs), "s1": pl.concat(s1s),
                         "p": pl.Series(np.concatenate(ps))})


# ---------- decision rules (unchanged) ----------

def assign(pairs, t):
    return pairs.sort("p", descending=True).unique("q", keep="first").filter(pl.col("p") >= t)


def expected_f(best):
    b = (best.sort(["s1", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1"), cum=pl.col("p").cum_sum().over("s1"),
                       tot=pl.col("p").sum().over("s1"), none=(1 - pl.col("p")).log().sum().over("s1").exp())
         .with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("k"))))
    keep = (b.group_by("s1").agg(best_k=pl.col("k").get(pl.col("ef").arg_max()), best_ef=pl.col("ef").max(),
                                 none=pl.col("none").first())
            .filter(pl.col("best_ef") > pl.col("none")))
    return b.join(keep, on="s1").filter(pl.col("k") <= pl.col("best_k")).select("q", "s1", "p")


def decide(pairs, rule, t):
    best = assign(pairs, 0.0)
    return expected_f(best) if rule == "expected_f" else best.filter(pl.col("p") >= t)


def f05(pred, gt, s1_ids):
    count = lambda df, name: df.group_by("s1").len(name)
    d = (pl.DataFrame({"s1": s1_ids})
         .join(count(pred.join(gt, on=["s1", "q"]), "tp"), on="s1", how="left")
         .join(count(pred, "n_pred"), on="s1", how="left")
         .join(count(gt, "n_true"), on="s1", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("n_pred"), pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.select(f.mean()).item()


def sim_dense(pairs, drop=0.5):
    gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
    s1_all = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(
        pl.col("src") == 1).select(s1="entity_id")
    # fold derived from s1 only
    s1_ids = s1_all.filter(FOLD == VALID_FOLD)["s1"]
    gt = gt.join(s1_all.rename({"s1": "s1"}), on="s1", how="left")  # ensure s1 column
    # split by fold
    fold_map = s1_all.with_columns(fold=FOLD)
    gt = gt.join(fold_map, on="s1", how="left")
    gt_v = gt.filter(pl.col("fold") == VALID_FOLD).drop("fold")

    dropped = gt_v.sample(fraction=drop, seed=0).select("q")
    out = {}
    for name, P, G in (("normal", pairs, gt_v),
                       ("dense", pairs.join(dropped, on="q", how="anti"),
                        gt_v.join(dropped, on="q", how="anti"))):
        best = assign(P, 0.0).join(fold_map.select("s1"), on="s1", how="semi")
        curve = {round(float(t), 2): f05(best.filter(pl.col("p") >= t), G, s1_ids)
                 for t in np.arange(0.3, 0.96, 0.05)}
        t = max(curve, key=curve.get)
        out.update({name: curve[t], f"t_{name}": t, f"curve_{name}": curve})
    return out


def error_sample(best, pred, gt, n=300):
    rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "business_name", "business_address"])
    txt = lambda side: rec.rename({"entity_id": side, "business_name": f"{side}_name",
                                   "business_address": f"{side}_addr"})
    fp = pred.join(gt, on=["s1", "q"], how="anti").with_columns(kind=pl.lit("false_merge"))
    fn = (gt.join(pred.select("q", "s1"), on=["s1", "q"], how="anti")
          .join(best.select("q", best_s1="s1", best_p="p"), on="q", how="left")
          .with_columns(kind=pl.when(pl.col("best_s1").is_null()).then(pl.lit("miss:not_in_candidates"))
                        .when(pl.col("best_s1") != pl.col("s1")).then(pl.lit("miss:assigned_elsewhere"))
                        .otherwise(pl.lit("miss:not_selected")), p=pl.col("best_p"))
          .drop("best_s1", "best_p"))
    return (pl.concat([fp.sample(min(n, len(fp)), seed=0), fn.sample(min(n, len(fn)), seed=0)], how="diagonal")
            .join(txt("s1"), on="s1", how="left").join(txt("q"), on="q", how="left")
            .select("kind", "p", "s1", "s1_name", "s1_addr", "q", "q_name", "q_addr"))


def pseudo_labels(feats):
    train_c = pl.read_parquet(f"{WORK}/train.parquet", columns=["country"])["country"].unique()
    country = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src", "country"])
               .filter((pl.col("src") != 1) & ~pl.col("country").is_in(train_c.implode())).select(q="entity_id"))
    probs = (pl.read_parquet(f"{WORK}/test_probs.parquet").join(country, on="q")
             .filter((pl.col("p") >= PSEUDO_HI) | (pl.col("p") <= PSEUDO_LO))
             .select("q", "s1", label=(pl.col("p") >= PSEUDO_HI).cast(pl.Int8)))
    ps = (load_feat_parts("test", feats=feats, with_label=False, subsample=1.0)[0:1] and None)
    # Load only the test feats rows that match probs
    ps = (pl.read_parquet(f"{WORK}/test_feats_parts/chunk_0000.parquet", n_rows=0))  # dummy
    # Simpler: read all chunks and semi-join by keys
    dfs = []
    for f in _chunk_files("test"):
        df = pl.read_parquet(f)
        m = probs.join(df.select("q", "s1"), on=["q", "s1"], how="semi")
        if len(m):
            dfs.append(df.join(m, on=["q", "s1"], how="inner").select(*feats, "label"))
        del df, m
        gc.collect()
    ps = pl.concat(dfs)
    print(f"pseudo-labels: {len(ps)} pairs, {ps['label'].mean():.3f} positive", flush=True)
    return ps


def fit(run_dir, pseudo=False):
    os.makedirs(run_dir, exist_ok=True)
    feats = feature_names("train")
    print(f"{len(feats)} features")

    # ---- load features as float32, one fold-group at a time ----
    # Loading everything at once (~5.5 GB) and then splitting with boolean masks
    # allocates a second ~4.4 GB array and OOM-kills the process on a 10 GB box.
    # Instead we ask the loader to filter by fold at the parquet scan level so we
    # only materialise the rows we actually need.
    t0 = time.time()
    X_tr, y_tr, _q_tr, s1_tr = load_feat_parts(
        "train", feats=feats, with_label=True, folds=TRAIN_FOLDS, subsample=0.5)
    X_va, y_va, _q_va, s1_va = load_feat_parts(
        "train", feats=feats, with_label=True, folds=[VALID_FOLD])
    print(f"loaded train-folds {len(X_tr)} + valid-fold {len(X_va)} float32 in "
          f"{time.time() - t0:.0f}s (~{(X_tr.nbytes + X_va.nbytes) / 1e9:.2f} GB)")

    # small validation subsample for early stopping
    rng = np.random.default_rng(0)
    va_idx = rng.choice(len(X_va), size=max(1, len(X_va) // 4), replace=False)
    va_X, va_y = X_va[va_idx], y_va[va_idx]
    del X_va, y_va, va_idx
    gc.collect()

    tr_X, tr_y = X_tr, y_tr
    del X_tr, y_tr
    gc.collect()

    # fold map (needed by the scoring / validation code below)
    fold_df = pl.concat([
        pl.DataFrame({"s1": s1_tr, "fold": (s1_tr.hash(seed=42) % 5)}),
        pl.DataFrame({"s1": s1_va, "fold": (s1_va.hash(seed=42) % 5)}),
    ]).unique("s1")
    del s1_tr, s1_va, _q_tr, _q_va
    gc.collect()

    if pseudo:
        ps = pseudo_labels(feats)
        X_ps = ps.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        y_ps = ps["label"].to_numpy().astype(np.int8)
        tr_X = np.concatenate([tr_X, X_ps])
        tr_y = np.concatenate([tr_y, y_ps])
        del ps, X_ps, y_ps
        gc.collect()

    # ---- train; save model immediately ----
    t0 = time.time()
    model = lgb.train(PARAMS,
                      lgb.Dataset(tr_X, tr_y, free_raw_data=True),
                      num_boost_round=ROUNDS,
                      valid_sets=[lgb.Dataset(va_X, va_y)],
                      callbacks=[lgb.early_stopping(PATIENCE), lgb.log_evaluation(100)])
    train_s, n_train = time.time() - t0, len(tr_X)
    print(f"trained on {n_train} pairs in {train_s:.0f}s, best iter {model.best_iteration}")
    model.save_model(f"{run_dir}/model.txt")
    del tr_X, tr_y, va_X, va_y
    gc.collect()

    # ---- score every pair (records compete across folds) ----
    all_pairs = predict_pairs_np(model, "train", feats)
    all_pairs = all_pairs.join(fold_df, on="s1", how="left")
    best = assign(all_pairs, 0.0).filter(pl.col("fold") == VALID_FOLD)
    s1_ids = fold_df.filter(pl.col("fold") == VALID_FOLD)["s1"]
    gt = (pl.read_parquet(f"{WORK}/train_gt.parquet")
          .join(fold_df, on="s1", how="left").filter(pl.col("fold") == VALID_FOLD).drop("fold"))

    # how many true matches are in candidates at all (blocking recall)
    lab = all_pairs.filter(pl.col("fold") == VALID_FOLD).select("q", "s1")
    found = gt.join(lab, on=["q", "s1"], how="semi")

    curve = {f"{t:.2f}": f05(best.filter(pl.col("p") >= t), gt, s1_ids) for t in np.arange(0.1, 1.0, 0.05)}
    t_best = max(curve, key=curve.get)
    ef = f05(expected_f(best), gt, s1_ids)
    rule = "expected_f" if ef > curve[t_best] else "threshold"
    dense = sim_dense(all_pairs, drop=0.5)
    metrics = {
        "val_f05": max(ef, curve[t_best]), "rule": rule, "val_f05_expected_f": ef,
        "val_f05_threshold": curve[t_best], "threshold": float(dense["t_dense"]),
        "threshold_normal": float(t_best),
        "val_f05_dense": dense["dense"], "f05_dense_by_threshold": dense["curve_dense"],
        "blocking_recall": len(found) / len(gt), "oracle_f05": f05(found, gt, s1_ids),
        "f05_by_threshold": curve, "best_iter": model.best_iteration,
        "n_train_pairs": n_train, "train_seconds": round(train_s),
        "params": PARAMS, "feats": feats, "pseudo": pseudo,
        "feature_gain": dict(sorted(zip(feats, model.feature_importance("gain").round().tolist()),
                                    key=lambda kv: -kv[1])),
    }
    for t, v in curve.items():
        print(f"  t={t}  F0.5={v:.4f}")
    print(f"blocking recall={metrics['blocking_recall']:.4f}  "
          f"oracle F0.5={metrics['oracle_f05']:.4f}  best t={t_best}  F0.5={curve[t_best]:.4f}  "
          f"expected-F rule F0.5={ef:.4f}  -> {rule}  | "
          f"test-like dense F0.5={dense['dense']:.4f} @ t={dense['t_dense']}")
    json.dump(metrics, open(f"{run_dir}/metrics.json", "w"), indent=2)
    pred = expected_f(best) if rule == "expected_f" else best.filter(pl.col("p") >= float(t_best))
    error_sample(best, pred, gt).write_csv(f"{run_dir}/errors.tsv", separator="\t")

def write(s1_ids, pairs, col, path):
    agg = pairs.group_by("s1").agg(pl.col("q").sort().str.join(",").alias(col))
    (s1_ids.join(agg, on="s1", how="left", maintain_order="left").with_columns(pl.col(col).fill_null(""))
     .rename({"s1": "source1_entity_id"}).write_csv(path, separator="\t", quote_style="never"))


def predict(run_dir, unseen_t=None):
    metrics = json.load(open(f"{run_dir}/metrics.json"))
    model = lgb.Booster(model_file=f"{run_dir}/model.txt")
    pairs = predict_pairs_np(model, "test", metrics["feats"])
    pairs.write_parquet(f"{WORK}/test_probs.parquet")
    s1 = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src", "country"])
          .filter(pl.col("src") == 1).select(s1="entity_id", country="country"))
    m = decide(pairs, metrics.get("rule", "threshold"), metrics["threshold"])
    out = f"{run_dir}/output"
    if unseen_t is not None:
        seen = pl.read_parquet(f"{WORK}/train.parquet", columns=["country"])["country"].unique()
        unseen = s1.filter(~pl.col("country").is_in(seen.implode())).select("s1")
        m = pl.concat([m.join(unseen, on="s1", how="anti"),
                       assign(pairs, 0.0).join(unseen, on="s1").filter(pl.col("p") >= unseen_t)])
        out = f"{run_dir}/output_unseen{unseen_t}"
    os.makedirs(out, exist_ok=True)
    write(s1.select("s1"), m, "matched_entity_ids", f"{out}/matching_results.tsv")
    write(s1.select("s1"), pairs, "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
    per = lambda df, name: df.group_by("s1").len(name)
    by_country = (s1.join(per(pairs, "cands"), on="s1", how="left").join(per(m, "matches"), on="s1", how="left")
                  .fill_null(0).group_by("country").agg(
                      n_s1=pl.len(), avg_candidates=pl.col("cands").mean(),
                      avg_matches=pl.col("matches").mean(),
                      pct_empty=(pl.col("matches") == 0).mean() * 100).sort("country"))
    metrics["test" if unseen_t is None else f"test_unseen{unseen_t}"] = {
        "n_s1": len(s1), "n_matches": len(m),
        "avg_candidates_per_s1": len(pairs) / len(s1), "by_country": by_country.to_dicts()}
    json.dump(metrics, open(f"{run_dir}/metrics.json", "w"), indent=2)
    print(by_country)


if __name__ == "__main__":
    # f05 self-test
    P = pl.DataFrame({"s1": ["a", "a", "a", "c"], "q": ["47", "193", "812", "9"]})
    G = pl.DataFrame({"s1": ["a", "a"], "q": ["47", "812"]})
    assert abs(f05(P, G, ["a"]) - 0.7143) < 1e-3 and f05(P, G, ["b"]) == 1.0 and f05(P, G, ["c"]) == 0.0
    B = pl.DataFrame({"q": ["1", "2", "3", "4"], "s1": ["a", "a", "a", "b"], "p": [0.95, 0.9, 0.2, 0.3]})
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python src/match.py fit RUN_DIR [--pseudo]       # Train LightGBM model into RUN_DIR")
        print("       python src/match.py predict RUN_DIR [--unseen-t T] # Generate predictions using RUN_DIR/model.txt")
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Error: Missing RUN_DIR argument.\nUsage: python src/match.py fit|predict RUN_DIR", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    run_dir = sys.argv[2]
    if cmd == "fit":
        fit(run_dir, pseudo="--pseudo" in sys.argv)
    elif cmd == "predict":
        t = sys.argv[sys.argv.index("--unseen-t") + 1] if "--unseen-t" in sys.argv else None
        predict(run_dir, None if t is None else float(t))
    else:
        print(f"Error: Unknown command '{cmd}'. Expected 'fit' or 'predict'.", file=sys.stderr)
        sys.exit(1)