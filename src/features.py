"""Pairwise features for blocked candidates -> work/{split}_feats_parts/chunk_*.parquet (+ label on train).

Every global aggregate is checkpointed to its own parquet, computed with polars' streaming engine
so peak RAM stays under ~2 GB. Per-chunk assembly is checkpointed too.

Usage: uv run python src/features.py train|test
"""
import gc
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

try:
    from src.normalize import name_tokens
except (ModuleNotFoundError, ImportError):
    from normalize import name_tokens

WORK = "work"
CHUNK = 500_000
NUM = r"\d+"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
PROXY_MIN, PROXY_M = 200, 50


def sim(a, b, scorer):
    return cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def string_feats(c):
    nq, ns, aq, as_ = c["nm_q"], c["nm_s"], c["ad_q"], c["ad_s"]
    kq, ks = nq.str.replace_all(" ", ""), ns.str.replace_all(" ", "")
    return c.with_columns(
        n_ratio=sim(nq, ns, fuzz.ratio),
        n_tset=sim(nq, ns, fuzz.token_set_ratio),
        n_tsort=sim(nq, ns, fuzz.token_sort_ratio),
        n_partial=sim(nq, ns, fuzz.partial_ratio),
        n_jw=sim(kq, ks, distance.JaroWinkler.normalized_similarity),
        a_ratio=sim(aq, as_, fuzz.ratio),
        a_tset=sim(aq, as_, fuzz.token_set_ratio),
        a_partial=sim(aq, as_, fuzz.partial_ratio),
        raw_n=sim(c["rq_n"], c["rs_n"], fuzz.ratio),
        raw_a=sim(c["rq_a"], c["rs_a"], fuzz.ratio),
    ).with_columns(
        num_q=pl.col("ad_q").str.extract_all(NUM).list.unique(),
        num_s=pl.col("ad_s").str.extract_all(NUM).list.unique(),
        tok_q=pl.col("nm_q").str.split(" ").list.unique(),
        tok_s=pl.col("nm_s").str.split(" ").list.unique(),
    ).with_columns(
        num_inter=pl.col("num_q").list.set_intersection("num_s").list.len(),
        num_union=pl.col("num_q").list.set_union("num_s").list.len(),
        num_first_eq=(pl.col("num_q").list.first() == pl.col("num_s").list.first()).cast(pl.Int8),
        num_first_diff=(pl.col("num_q").list.first().cast(pl.Int64, strict=False)
                        - pl.col("num_s").list.first().cast(pl.Int64, strict=False)).abs().clip(upper_bound=10**6),
        n_extra_q=pl.col("tok_q").list.set_difference("tok_s").list.len(),
        n_extra_s=pl.col("tok_s").list.set_difference("tok_q").list.len(),
        len_nq=pl.col("nm_q").str.len_chars(), len_ns=pl.col("nm_s").str.len_chars(),
        len_aq=pl.col("ad_q").str.len_chars(), len_as=pl.col("ad_s").str.len_chars(),
    ).with_columns(
        num_jac=pl.col("num_inter") / pl.col("num_union"),
        num_q_in_s=(pl.col("num_inter") == pl.col("num_q").list.len()).cast(pl.Int8),
    ).drop("nm_q", "nm_s", "ad_q", "ad_s", "num_q", "num_s", "tok_q", "tok_s",
           "rq_n", "rs_n", "rq_a", "rs_a")


def full_names(names):
    maps = json.load(open(f"{WORK}/maps.json"))["name"]
    with Pool(THREADS) as p:
        toks = p.map(name_tokens, names.to_list(), chunksize=20_000)
    return (pl.Series(toks).str.split(" ").list.eval(pl.element().filter(pl.element() != "").replace(maps))
            .list.unique())


def decoy_feats_chunk(chunk, q_meta, s_meta, dec, same_num, tok_df):
    d = (chunk.select("q", "s1")
         .join(q_meta.select("q", "country", "full_q", "num1_q", "nums_q"), on="q")
         .join(s_meta.select("s1", "full_s", "num1_s"), on="s1")
         .with_columns(xq=pl.col("full_q").list.set_difference("full_s"),
                       xs=pl.col("full_s").list.set_difference("full_q")))
    key = ["q", "s1"]
    xdec = (d.select(*key, "country", tok="xq").explode("tok").join(dec, on=["country", "tok"])
            .group_by(key).agg(dec_max=pl.col("dec").max(), dec_sum=pl.col("dec").sum()))
    numd = (d.select(*key, n="nums_q", t="num1_s").explode("n")
            .with_columns(diff=(pl.col("n").cast(pl.Int64, strict=False)
                                - pl.col("t").cast(pl.Int64, strict=False)).abs())
            .group_by(key).agg(s_num_mindiff=pl.col("diff").min().clip(upper_bound=10**6)))
    uniq = (d.select(*key, tok="xq").explode("tok").join(tok_df, on=["s1", "tok"])
            .group_by(key).agg(g_uniq_x=(pl.col("df") == 1).sum()))
    return (d.join(same_num, on=["s1", "num1_q"], how="left")
            .select(*key,
                    n_xq=pl.col("xq").list.len(), n_xs=pl.col("xs").list.len(),
                    g_same_num=pl.col("g_same_num").fill_null(1) - 1,
                    g_same_num_frac=(pl.col("g_same_num").fill_null(1) - 1)
                                    / (pl.len().over("s1") - 1).clip(lower_bound=1))
            .join(xdec, on=key, how="left").join(numd, on=key, how="left").join(uniq, on=key, how="left")
            .with_columns(s_num_in_q=(pl.col("s_num_mindiff") == 0).cast(pl.Int8)))


def ambiguity_feats_chunk(chunk, q_meta, s_meta, same_nm, same_ad, known):
    d = (chunk.select("q", "s1")
         .join(q_meta.select("q", "country", "nm_q"), on="q")
         .join(s_meta.select("s1", "nm_s", "ad_s"), on="s1")
         .join(same_nm.rename({"nm": "nm_s", "n": "amb_s_name"}), on=["country", "nm_s"], how="left")
         .join(same_nm.rename({"nm": "nm_q", "n": "amb_q_name"}), on=["country", "nm_q"], how="left")
         .join(same_ad.rename({"ad": "ad_s", "n": "amb_s_addr"}), on=["country", "ad_s"], how="left")
         .with_columns(name_eq=(pl.col("nm_q") == pl.col("nm_s")).cast(pl.Int8)))
    return (d.with_columns(q_name_ties=pl.col("name_eq").sum().over("q"))
            .join(known, on="q", how="left")
            .select(
                "q", "s1",
                amb_s_name=pl.col("amb_s_name"),
                amb_q_name=pl.col("amb_q_name").fill_null(0),
                amb_s_addr=pl.col("amb_s_addr"),
                name_eq=pl.col("name_eq"),
                q_name_ties=pl.col("q_name_ties"),
                q_known_frac=pl.col("q_known_frac"),
            ))

def build(split):
    q_meta_path   = f"{WORK}/{split}_q_meta.parquet"
    s_meta_path   = f"{WORK}/{split}_s_meta.parquet"
    ctx_path      = f"{WORK}/{split}_cands_ctx.parquet"
    dec_path      = f"{WORK}/{split}_decoy_words.parquet"
    same_num_path = f"{WORK}/{split}_same_num.parquet"
    tok_df_path   = f"{WORK}/{split}_tok_df.parquet"
    same_nm_path  = f"{WORK}/{split}_same_nm.parquet"
    same_ad_path  = f"{WORK}/{split}_same_ad.parquet"
    vocab_path    = f"{WORK}/{split}_vocab.parquet"
    known_path    = f"{WORK}/{split}_known.parquet"

    # ---------- Step 1: q_meta, s_meta ----------
    if not (os.path.exists(q_meta_path) and os.path.exists(s_meta_path)):
        rec = pl.read_parquet(f"{WORK}/{split}.parquet")
        rec = rec.with_columns(
            translit=pl.col("business_name").str.contains(r"[^\x00-\x7F]").cast(pl.Int8),
            raw_n=pl.col("business_name").fill_null("").str.to_lowercase(),
            raw_a=pl.col("business_address").fill_null("").str.to_lowercase(),
            nums=pl.col("ad").str.extract_all(NUM).list.unique(),
            num1=pl.col("ad").str.extract(r"\b(\d+)\b"),
        )
        rec = rec.with_columns(full=full_names(rec["business_name"]))
        (rec.filter(pl.col("src") != 1)
            .select(q="entity_id", country="country", full_q="full", num1_q="num1", nums_q="nums",
                    src="src", nm_q="nm", ad_q="ad", translit="translit", rq_n="raw_n", rq_a="raw_a")
            .write_parquet(q_meta_path))
        (rec.filter(pl.col("src") == 1)
            .select(s1="entity_id", country="country", full_s="full", num1_s="num1",
                    nm_s="nm", ad_s="ad", rs_n="raw_n", rs_a="raw_a")
            .write_parquet(s_meta_path))
        del rec
        gc.collect()
        print("  metadata written", flush=True)
    else:
        print("[ckpt] q_meta/s_meta exist", flush=True)

    # ---------- Step 2: decoy_words (skip if already produced) ----------
    if not os.path.exists(dec_path):
        top = (pl.read_parquet(f"{WORK}/{split}_cands_full.parquet",
                                columns=["q", "s1", "rank"])
                 .filter(pl.col("rank") == 0))
        s = pl.read_parquet(s_meta_path, columns=["s1", "full_s", "num1_s"])
        q = pl.read_parquet(q_meta_path, columns=["q", "country", "full_q", "num1_q"])
        t = (q.join(top.select("q", "s1"), on="q").join(s, on="s1")
              .select("country",
                      x=pl.col("full_q").list.set_difference("full_s"),
                      mm=(pl.col("num1_q") != pl.col("num1_s")).cast(pl.Float64))
              .filter(pl.col("mm").is_not_null()).explode("x").filter(pl.col("x").is_not_null()))
        prior = t.group_by("country").agg(prior=pl.col("mm").mean())
        dec = (t.group_by("country", "x").agg(n=pl.len(), mm=pl.col("mm").sum())
                 .filter(pl.col("n") >= PROXY_MIN).join(prior, on="country")
                 .select("country", tok="x",
                         dec=(pl.col("mm") + PROXY_M * pl.col("prior")) / (pl.col("n") + PROXY_M)))
        dec.write_parquet(dec_path)
        print(dec.sort("dec", descending=True).group_by("country").head(15)
              .sort("country", "dec", descending=[False, True]).rows())
        del top, s, q, t, prior, dec
        gc.collect()
    else:
        print("[ckpt] decoy_words exist", flush=True)

    # ---------- Step 3: cands_ctx ----------
    if not os.path.exists(ctx_path):
        c = pl.read_parquet(f"{WORK}/{split}_cands.parquet", columns=["q", "s1", "score", "rank"])
        c = c.with_columns(
            top1=pl.col("score").max().over("q"),
            top2=pl.col("score").top_k(2).min().over("q"),
            s1_n=pl.len().over("s1"),
            s1_n0=(pl.col("rank") == 0).sum().over("s1"),
            s1_rank=pl.col("score").rank("ordinal", descending=True).over("s1"),
        ).with_columns(gap=pl.col("top1") - pl.col("score"), margin=pl.col("top1") - pl.col("top2"))
        c.write_parquet(ctx_path)
        del c
        gc.collect()
        print("  cands_ctx written", flush=True)
    else:
        print("[ckpt] cands_ctx exists", flush=True)

    # ---------- Step 4: same_num (streamed) ----------
    if not os.path.exists(same_num_path):
        (pl.scan_parquet(ctx_path).select("q", "s1")
           .join(pl.scan_parquet(q_meta_path).select("q", "num1_q"), on="q", how="left")
           .filter(pl.col("num1_q").is_not_null())
           .group_by("s1", "num1_q").len("g_same_num")
           .sink_parquet(same_num_path))
        print("  same_num written", flush=True)
    else:
        print("[ckpt] same_num exists", flush=True)

    # ---------- Step 5: tok_df (streamed) ----------
    if not os.path.exists(tok_df_path):
        (pl.scan_parquet(ctx_path).select("q", "s1")
           .join(pl.scan_parquet(q_meta_path).select("q", "full_q"), on="q", how="left")
           .select("s1", tok="full_q")
           .explode("tok")
           .group_by("s1", "tok").len("df")
           .sink_parquet(tok_df_path))
        print("  tok_df written", flush=True)
    else:
        print("[ckpt] tok_df exists", flush=True)

    # ---------- Step 6: same_nm, same_ad, vocab ----------
    if not (os.path.exists(same_nm_path) and os.path.exists(same_ad_path) and os.path.exists(vocab_path)):
        s_meta = pl.read_parquet(s_meta_path, columns=["country", "nm_s", "ad_s"])
        (s_meta.group_by("country", "nm_s").len("n").rename({"nm_s": "nm"})
                .write_parquet(same_nm_path))
        (s_meta.filter(pl.col("ad_s") != "").group_by("country", "ad_s").len("n")
                .rename({"ad_s": "ad"}).write_parquet(same_ad_path))
        (s_meta.select("country", tok=pl.col("nm_s").str.split(" ")).explode("tok")
                .filter(pl.col("tok") != "").group_by("country", "tok").len("df")
                .filter(pl.col("df") >= 2).select("country", "tok")
                .write_parquet(vocab_path))
        del s_meta
        gc.collect()
        print("  ambiguity lookups written", flush=True)
    else:
        print("[ckpt] ambiguity lookups exist", flush=True)

    # ---------- Step 7: known (streamed) ----------
    if not os.path.exists(known_path):
        (pl.scan_parquet(q_meta_path).select("q", "country", tok=pl.col("nm_q").str.split(" "))
           .explode("tok")
           .filter(pl.col("tok") != "")
           .join(pl.scan_parquet(vocab_path).with_columns(k=pl.lit(1)),
                 on=["country", "tok"], how="left")
           .group_by("q").agg(q_known_frac=pl.col("k").fill_null(0).mean())
           .sink_parquet(known_path))
        print("  known written", flush=True)
    else:
        print("[ckpt] known exists", flush=True)

    # ---------- Step 8: chunked assembly ----------
    dec      = pl.read_parquet(dec_path)
    q_meta   = pl.read_parquet(q_meta_path)
    s_meta   = pl.read_parquet(s_meta_path)
    same_num = pl.read_parquet(same_num_path)
    tok_df   = pl.read_parquet(tok_df_path)
    same_nm  = pl.read_parquet(same_nm_path)
    same_ad  = pl.read_parquet(same_ad_path)
    known    = pl.read_parquet(known_path)

    gt = None
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet").with_columns(label=pl.lit(1, pl.Int8))

    total = pl.scan_parquet(ctx_path).select(pl.len()).collect().item()
    out_dir = f"{WORK}/{split}_feats_parts"
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = (total + CHUNK - 1) // CHUNK
    print(f"{split}: {total} pairs -> {n_chunks} chunks", flush=True)
    ctx = pl.read_parquet(ctx_path)

    for i in range(n_chunks):
        cfile = f"{out_dir}/chunk_{i:04d}.parquet"
        if os.path.exists(cfile):
            print(f"[ckpt] chunk {i+1}/{n_chunks} exists", flush=True)
            continue
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, total)
        chunk = ctx[lo:hi]

        df = (chunk
              .join(decoy_feats_chunk(chunk, q_meta, s_meta, dec, same_num, tok_df),
                    on=["q", "s1"], how="left")
              .join(ambiguity_feats_chunk(chunk, q_meta, s_meta, same_nm, same_ad, known),
                    on=["q", "s1"], how="left")
              .join(q_meta, on="q", how="left")
              .join(s_meta, on="s1", how="left"))

        df = string_feats(df)

        for c in ("raw_n", "raw_a", "n_ratio", "a_tset"):
            df = df.with_columns((pl.col(c).max().over("q") - pl.col(c)).alias(f"{c}_qgap"))
            df = df.with_columns((pl.col(f"{c}_qgap") == 0).cast(pl.Int8).alias(f"{c}_qbest"))
        df = df.with_columns(raw_n_qties=pl.col("raw_n_qbest").sum().over("q").cast(pl.Int16),
                             raw_a_qties=pl.col("raw_a_qbest").sum().over("q").cast(pl.Int16))

        drop = [c for c in ("country", "full_q", "num1_q", "nums_q",
                             "full_s", "num1_s", "nm_q", "nm_s", "ad_q", "ad_s")
                if c in df.columns]
        df = df.drop(drop)
        if split == "train":
            df = df.join(gt, on=["q", "s1"], how="left").with_columns(pl.col("label").fill_null(0))

        df.write_parquet(cfile)
        print(f"  wrote chunk {i+1}/{n_chunks}", flush=True)
        del chunk, df
        gc.collect()

    del ctx
    gc.collect()
    print("done", split)


if __name__ == "__main__":
    try:
        from src.normalize import finish
    except (ModuleNotFoundError, ImportError):
        from normalize import finish
    assert name_tokens("SOLOVA FÁCT L.L.C.") == "solova fact lc"
    assert finish(name_tokens("SOLOVA FÁCT L.L.C."), "name", "US") == "solova fact"
    assert name_tokens("Xylozeta Co formerly known as Cervantes Select Mountain LLC") == "cervantes select mountain lc"
    assert name_tokens("Viozeta formerly: New Delhi Communications") == "new delhi comunications"
    assert name_tokens("internationalforteanimation.com") == "internationalforteanimation"
    assert finish(name_tokens("Fire Master (India) Pvt (Ltd) - 9832661323"), "name", "India") == "fire master india"
    assert finish("lc", "name", "US") == "lc"

    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python src/features.py train|test         # Feature extraction")
        sys.exit(0)

    os.makedirs(WORK, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "test"):
        build(sys.argv[1])