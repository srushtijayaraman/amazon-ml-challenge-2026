"""Synthetic decoys for the training split, so training matches the test's decoy density.

Test has ~2.4 decoys per S1 vs ~1.2 in train (5.75 vs 4.67 S2/S3 records per S1; label-free decoy share ~42%
vs 26%). Decoys are near-copies of a real business: a shifted house number and/or an extra marker word or
different legal form. We add ~1.2 more per S1 from the training data itself: half re-shift a real decoy, half
turn a true record into a decoy. They match no S1 (not in train_gt), so they only ever act as negatives.

Usage: uv run python src/synth.py            # replaces any earlier synthetic rows in work/train.parquet
       uv run python src/synth.py --remove   # strip them again
"""
import json
import os
import random
import re
import sys
from multiprocessing import Pool

import polars as pl

try:
    from src import normalize
    from src.normalize import addr_tokens, name_tokens
except (ModuleNotFoundError, ImportError):
    import normalize
    from normalize import addr_tokens, name_tokens

WORK = "work"
PER_S1 = 1.2  # extra decoys per S1
MARKER_MIN_N, MARKER_MIN_SHARE = 500, 0.9  # marker words: seen in >= 500 train records, >= 90% of them decoys
NUM = re.compile(r"\d+")


def shift_number(addr, rng):
    m = NUM.search(addr or "")
    if not m:
        return None
    n = int(m.group())
    d = rng.choice([-1, 1]) * rng.randint(1, 30)
    return addr[:m.start()] + str(max(1, n + d if n + d != n else n + 1)) + addr[m.end():]


def make(args):
    """One synthetic decoy (name, address) from a source record."""
    name, addr, marker, kind, seed = args
    rng = random.Random(seed)
    if kind == "reshift":  # real decoy, new house number
        a = shift_number(addr, rng)
        return (name, a) if a else None
    a = shift_number(addr, rng) if rng.random() < 0.8 else None  # true record -> decoy
    words = name.split()
    if marker and (a is None or rng.random() < 0.6):
        words.insert(rng.randint(1, len(words)) if len(words) > 1 else len(words), marker.title())
    elif a is None:
        return None
    return " ".join(words), a if a is not None else addr


def finish(row):
    nm, ad, country = row
    return normalize.finish(name_tokens(nm), "name", country), normalize.finish(addr_tokens(ad), "addr", country)


if __name__ == "__main__":
    rec = pl.read_parquet(f"{WORK}/train.parquet").filter(~pl.col("entity_id").str.contains("-SYN"))
    if "--remove" in sys.argv:
        rec.write_parquet(f"{WORK}/train.parquet")
        sys.exit(print(f"synthetic rows removed; {len(rec)} records"))
    gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
    q = rec.filter(pl.col("src") != 1).join(gt.select(entity_id="q", s1="s1"), on="entity_id", how="left")
    # marker words: name words that (almost) only ever appear in decoys
    low = q.select("country", decoy=pl.col("s1").is_null(),
                   w=pl.col("business_name").str.to_lowercase().str.extract_all(r"[a-z]+").list.unique()).explode("w")
    markers = (low.group_by("country", "w").agg(n=pl.len(), share=pl.col("decoy").mean())
               .filter((pl.col("n") >= MARKER_MIN_N) & (pl.col("share") >= MARKER_MIN_SHARE)))
    mk = {c: g["w"].to_list() for (c,), g in markers.group_by("country")}
    print({c: len(v) for c, v in mk.items()}, {c: v[:15] for c, v in mk.items()})
    n_new = int(PER_S1 * rec.filter(pl.col("src") == 1).height)
    decoys, trues = q.filter(pl.col("s1").is_null()), q.filter(pl.col("s1").is_not_null())
    pick = pl.concat([decoys.sample(n_new // 2, with_replacement=True, seed=1).with_columns(kind=pl.lit("reshift")),
                      trues.sample(n_new - n_new // 2, with_replacement=True, seed=2).with_columns(kind=pl.lit("mark"))])
    rng = random.Random(0)
    args = [(nm, ad, rng.choice(mk.get(c, [None])) if k == "mark" else None, k, i)
            for i, (nm, ad, c, k) in enumerate(pick.select("business_name", "business_address", "country", "kind").iter_rows())]
    with Pool(int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())) as p:
        out = p.map(make, args, chunksize=20_000)
    keep = [i for i, o in enumerate(out) if o is not None]
    syn = pick[keep].select("country", "src").with_columns(
        business_name=pl.Series([out[i][0] for i in keep]), business_address=pl.Series([out[i][1] for i in keep]))
    syn = syn.with_columns(entity_id=pl.format("S{}-SYN{}", pl.col("src"), pl.int_range(pl.len())))
    normalize.MAPS.update(json.load(open(f"{WORK}/maps.json")))
    noise = json.load(open(f"{WORK}/train_noise.json"))
    normalize.NOISE.update({c: {k: set(v) for k, v in d.items()} for c, d in noise.items()})
    with Pool(int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())) as p:
        res = p.map(finish, zip(syn["business_name"], syn["business_address"], syn["country"]), chunksize=20_000)
    syn = syn.with_columns(nm=pl.Series([a for a, _ in res]), ad=pl.Series([b for _, b in res]))
    pl.concat([rec, syn.select(rec.columns)], how="vertical_relaxed").write_parquet(f"{WORK}/train.parquet")
    print(f"added {len(syn)} synthetic decoys ({len(syn) / rec.filter(pl.col('src') == 1).height:.2f} per S1)")
    print(syn.group_by("country").len().rows())
    for r in syn.sample(12, seed=3).iter_rows(named=True):
        print("  ", r["business_name"], "|", r["business_address"])
