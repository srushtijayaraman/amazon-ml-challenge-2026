"""Blocking: every S2/S3 record looks up its top-K S1 records (same country) by TF-IDF cosine.

Every step is chunked and checkpointed on disk, so peak RAM stays ~1.5-2 GB even for US.

Usage: uv run python src/block.py train|test            # retrieval
       uv run python src/block.py train|test --prune    # prune
"""
import gc
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

WORK = "work"
K = 10
W_NAME = 0.6
CHUNK = 250_000
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
MAX_DF = 0.005
REL = 0.8
CAP = 30
RMAX = 10
KEY_MAX_S1 = 50
KEY_TOP = 2
NUM1 = pl.col("ad").str.extract(r"\b(\d+)\b")
KEYS = {
    1: (pl.col("nm").str.split(" ").list.sort().list.join(" "), "ad"),
    2: (pl.concat_str([NUM1, pl.col("ad").str.extract(
        r"\b\d+\b[^a-z]*?(?:\b[a-z]{1,2}\b\s+)*\b([a-z]{3,})")], separator=" "), "nm"),
    4: (pl.col("nm").str.replace_all(r"(\B)[aeiou]", "").str.split(" ").list.sort().list.join(" "), "ad"),
    8: (pl.col("nm").str.replace_all(" ", ""), "ad"),
    16: (pl.col("nm").str.split(" ").list.head(2).list.join(" "), "ad"),
}


def rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def key_pairs(s1, q, cache_dir, kp_file):
    """Chunked by q-chunk; every step cached under cache_dir. cache_dir is NOT deleted here —
    the caller needs the per-q-chunk files to build the key-only pairs."""
    os.makedirs(cache_dir, exist_ok=True)
    s1_rowidx = s1.select(s1="entity_id").with_row_index("si")

    # ---- per-bit S1 key table, cached once ----
    bit_paths = {}
    for bit, (expr, other) in KEYS.items():
        bp = f"{cache_dir}/s1_bit{bit}.parquet"
        if not os.path.exists(bp):
            sk = s1.select(s1="entity_id", key=expr, o_s=other).filter(
                pl.col("key").is_not_null() & (pl.col("key") != ""))
            sk = sk.filter(pl.len().over("key") <= KEY_MAX_S1)
            sk = sk.join(s1_rowidx, on="s1")
            sk.write_parquet(bp)
            print(f"    s1_bit{bit}: {len(sk)} rows", flush=True)
            del sk
            gc.collect()
        bit_paths[bit] = bp

    # ---- per q-chunk key pairs ----
    n_chunks = (len(q) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        cf = f"{cache_dir}/kp_q{i:05d}.parquet"
        if os.path.exists(cf):
            continue
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, len(q))
        c = q[lo:hi]
        parts = []
        for bit, (expr, other) in KEYS.items():
            pr = (c.select(q="entity_id", key=expr, o_q=other)
                    .filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
                    .join(pl.read_parquet(bit_paths[bit]), on="key"))
            if not len(pr):
                continue
            pr = pr.with_columns(sim=cpdist(pr["o_q"].to_list(), pr["o_s"].to_list(),
                                            scorer=fuzz.ratio, workers=-1))
            pr = (pr.sort("sim", descending=True)
                    .group_by("q", maintain_order=True).head(KEY_TOP)
                    .select("q", "s1", "si", via_key=pl.lit(bit, pl.Int8)))
            parts.append(pr)
            del pr
            gc.collect()
        if parts:
            (pl.concat(parts)
               .group_by("q", "s1")
               .agg(pl.col("via_key").sum().cast(pl.Int8), pl.col("si").first())
               .write_parquet(cf))
        else:
            pl.DataFrame(schema={"q": pl.String, "s1": pl.String,
                                 "si": pl.UInt32, "via_key": pl.Int8}).write_parquet(cf)
        print(f"    kp_q {i+1}/{n_chunks}", flush=True)
        del c, parts
        gc.collect()

    # ---- global kp_file: concatenate per-chunk files ----
    (pl.scan_parquet(f"{cache_dir}/kp_q*.parquet")
       .group_by("q", "s1")
       .agg(pl.col("via_key").sum().cast(pl.Int8), pl.col("si").first())
       .sink_parquet(kp_file))


def block_country(s1, q, country, base_dir):
    out_file = f"{base_dir}/{country}.parquet"
    if os.path.exists(out_file):
        print(f"[ckpt] {country} already assembled", flush=True)
        return out_file

    tf_dir = f"{base_dir}/{country}_tf"
    kp_cache = f"{tf_dir}/kp_cache"
    new_dir = f"{tf_dir}/new"
    new_marker = f"{tf_dir}/new.done"
    final_dir = f"{base_dir}/{country}_final"
    kp_file = f"{base_dir}/{country}_kp.parquet"

    os.makedirs(tf_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # ---- vectorizers (per country) ----
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), dtype=np.float32, max_df=MAX_DF)
    va = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), dtype=np.float32, max_df=MAX_DF)
    key = lambda df: df["nm"].str.replace_all(" ", "").to_list()
    Sn = vn.fit_transform(key(s1)).astype(np.float32).tocsr()
    Sa = va.fit_transform(s1["ad"].to_list()).astype(np.float32).tocsr()
    S_T = sp.hstack([np.sqrt(W_NAME) * Sn, np.sqrt(1 - W_NAME) * Sa]).T.tocsr()
    s1_ids = s1["entity_id"].to_numpy()
    print(f"  vectorizers fit: {len(s1)} S1, {Sn.shape[1]} name feats, {Sa.shape[1]} addr feats",
          flush=True)

    # ---- TF-IDF chunks, checkpointed ----
    n_chunks = (len(q) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        cf = f"{tf_dir}/chunk_{i:05d}.parquet"
        if os.path.exists(cf):
            continue
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, len(q))
        c = q[lo:hi]
        t = time.time()
        Qn = vn.transform(key(c)).astype(np.float32)
        Qa = va.transform(c["ad"].to_list()).astype(np.float32)
        Q = sp.hstack([np.sqrt(W_NAME) * Qn, np.sqrt(1 - W_NAME) * Qa]).tocsr()
        C = sp_matmul_topn(Q, S_T, top_n=K, sort=True, n_threads=THREADS)
        counts = np.diff(C.indptr)
        rows, cols = np.repeat(np.arange(len(c)), counts), C.indices
        pl.DataFrame({
            "q": c["entity_id"].to_numpy()[rows],
            "s1": s1_ids[cols],
            "score": C.data,
            "name_cos": rowdot(Qn[rows], Sn[cols]),
            "addr_cos": rowdot(Qa[rows], Sa[cols]),
        }).write_parquet(cf)
        print(f"    tf {i+1}/{n_chunks}  {time.time() - t:.1f}s", flush=True)
        del c, Qn, Qa, Q, C
        gc.collect()

    # ---- key pairs, per q-chunk cached under kp_cache/ ----
    if not os.path.exists(kp_file):
        t = time.time()
        key_pairs(s1, q, kp_cache, kp_file)
        print(f"  key pairs: done in {time.time() - t:.0f}s", flush=True)

    # ---- key-only pairs: (kp - tf) computed per q-chunk, cosines recomputed per chunk ----
    if not os.path.exists(new_marker):
        os.makedirs(new_dir, exist_ok=True)
        for i in range(n_chunks):
            kpq = f"{kp_cache}/kp_q{i:05d}.parquet"
            tf_i = f"{tf_dir}/chunk_{i:05d}.parquet"
            newf = f"{new_dir}/new_{i:05d}.parquet"
            if os.path.exists(newf) or not os.path.exists(kpq):
                continue
            kpi = pl.read_parquet(kpq)                          # ~500k rows
            tfp = pl.read_parquet(tf_i, columns=["q", "s1"])    # ~2.5M rows
            new_i = kpi.join(tfp, on=["q", "s1"], how="anti")
            del kpi, tfp
            gc.collect()
            if not len(new_i):
                # write empty marker so resume skips
                new_i.write_parquet(newf)
                continue
            lo, hi = i * CHUNK, min((i + 1) * CHUNK, len(q))
            c = q[lo:hi]
            c = c.select(q="entity_id").with_row_index("qri")
            new_i = new_i.join(c, on="q")
            uq = np.unique(new_i["qri"].to_numpy())
            sub = q[lo:hi][pl.Series(uq)]
            Qn = vn.transform(key(sub)).astype(np.float32)
            Qa = va.transform(sub["ad"].to_list()).astype(np.float32)
            r = np.searchsorted(uq, new_i["qri"].to_numpy())
            c_idx = new_i["si"].to_numpy()
            nc, ac = rowdot(Qn[r], Sn[c_idx]), rowdot(Qa[r], Sa[c_idx])
            new_i = (new_i.select("q", "s1", "via_key")
                     .with_columns(name_cos=pl.Series(nc), addr_cos=pl.Series(ac),
                                   score=pl.Series(W_NAME * nc + (1 - W_NAME) * ac)))
            new_i.write_parquet(newf)
            del new_i, Qn, Qa, sub, uq, r, c
            gc.collect()
        open(new_marker, "w").close()

    # ---- per-chunk assembly ----
    kp = pl.read_parquet(kp_file).drop("si")
    print(f"  assembling {n_chunks} chunks...", flush=True)
    for i in range(n_chunks):
        cf = f"{final_dir}/chunk_{i:05d}.parquet"
        if os.path.exists(cf):
            continue
        tf_i = (pl.read_parquet(f"{tf_dir}/chunk_{i:05d}.parquet")
                  .join(kp, on=["q", "s1"], how="left")
                  .with_columns(pl.col("via_key").fill_null(0)))
        parts = [tf_i]
        nf = f"{new_dir}/new_{i:05d}.parquet"
        if os.path.exists(nf):
            parts.append(pl.read_parquet(nf))
        allc = pl.concat(parts, how="diagonal_relaxed")
        allc = allc.with_columns(
            rank=(pl.col("score").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16))
        allc.write_parquet(cf)
        if (i + 1) % 5 == 0 or i == n_chunks - 1:
            print(f"    final {i+1}/{n_chunks}", flush=True)
        del tf_i, parts, allc
        gc.collect()
    del kp
    gc.collect()

    # ---- stream-concat into the per-country output ----
    (pl.scan_parquet(f"{final_dir}/chunk_*.parquet")
       .sink_parquet(out_file))
    print(f"  wrote {out_file}", flush=True)

    # cleanup this country's intermediates
    shutil.rmtree(tf_dir, ignore_errors=True)
    shutil.rmtree(final_dir, ignore_errors=True)
    if os.path.exists(kp_file):
        os.remove(kp_file)
    return out_file


def prune(c, rel=REL, cap=CAP, rmax=RMAX):
    keyed = (pl.col("via_key") > 0) if "via_key" in c.columns else pl.lit(False)
    return (c.filter((pl.col("rank") == 0) | keyed |
                     ((pl.col("rank") < rmax) &
                      (pl.col("score") >= rel * pl.col("score").max().over("q"))))
            .filter((pl.col("rank") == 0) |
                    (pl.col("score").rank("ordinal", descending=True).over("s1") <= cap)))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python src/block.py train|test            # Candidate retrieval")
        print("       python src/block.py train|test --prune    # Candidate pruning")
        sys.exit(0)
    split = sys.argv[1]
    df = pl.read_parquet(f"{WORK}/{split}.parquet",
                         columns=["entity_id", "src", "country", "nm", "ad"])
    n_s1 = df.filter(pl.col("src") == 1).height

    if "--prune" not in sys.argv:
        base_dir = f"{WORK}/{split}_cands_parts"
        os.makedirs(base_dir, exist_ok=True)
        for country in df["country"].unique().sort():
            print(f"=== {country} ===", flush=True)
            s1 = df.filter((pl.col("src") == 1) & (pl.col("country") == country))
            q  = df.filter((pl.col("src") != 1) & (pl.col("country") == country))
            print(f"{country}: {len(s1)} S1, {len(q)} queries", flush=True)
            block_country(s1, q, country, base_dir)
            del s1, q
            gc.collect()
        (pl.scan_parquet(f"{base_dir}/*.parquet")
           .sink_parquet(f"{WORK}/{split}_cands_full.parquet"))
        sys.exit()

        # load as float32 to halve the memory of the three float columns
    full = (pl.read_parquet(f"{WORK}/{split}_cands_full.parquet")
            .with_columns(pl.col("score").cast(pl.Float32),
                          pl.col("name_cos").cast(pl.Float32),
                          pl.col("addr_cos").cast(pl.Float32)))

    cands = prune(full)
    cands.write_parquet(f"{WORK}/{split}_cands.parquet")
    per = cands.group_by("s1").len()["len"]
    print(f"RMAX={RMAX} REL={REL} CAP={CAP}: pairs={len(cands)}  "
          f"candidates per S1: avg {len(cands) / n_s1:.2f}, "
          f"median {per.median()}, p99 {per.quantile(0.99)}, max {per.max()}")
    del full, cands
    gc.collect()

    # single-point recall check (train only)
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
        lab = (pl.read_parquet(f"{WORK}/{split}_cands.parquet")
                 .select("q", "s1")
                 .join(gt.select("q", "s1"), on=["q", "s1"], how="semi"))
        print(f"recall @ RMAX={RMAX} REL={REL} CAP={CAP}: {len(lab) / len(gt):.4f}")