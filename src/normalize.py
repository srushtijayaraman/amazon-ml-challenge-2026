"""Text normalization + prep: raw TSVs -> work/{split}.parquet (and work/train_gt.parquet).

Three layers, all country-agnostic:
  1. rules: alias stripping ("X formerly Y" -> "Y"), transliteration, legal forms, abbreviations;
  2. spelling maps learned from matched training pairs (praivet -> private, sixth -> 6th, ciy -> city);
  3. injected filler words, detected per split and country as tokens over-represented in S2/S3
     relative to S1 (catches France's "participations", "et fils" with no French training data).

Chunked so it fits in ~3 GB:
  PASS 1  read each source TSV, tokenize a 2 M-row chunk at a time, write work/{split}_tok/*.parquet
  PASS 2  (train) learn spelling maps from matched pairs only (small join against train_gt)
  PASS 3  compute per-(country, token) filler counts by streaming over the token chunks
  PASS 4  apply MAPS + stopwords + filler per chunk, write work/{split}_final/*.parquet
  PASS 5  stream-concat the final chunks into work/{split}.parquet

Every pass is checkpointed; re-running skips finished chunks.

Usage: uv run python src/normalize.py
"""
import gc
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

DATA = os.environ.get("DATA_DIR", "dataset")
WORK = "work"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
MAP_MIN, MAP_SHARE = 20, 0.5
NOISE_LIFT = {"nm": 3.0, "ad": 10.0}
NOISE_MIN = 0.0003
CHUNK = 2_000_000      # rows per tokenization chunk

NAME_STOP = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "plc", "lp", "llp", "pc", "pllc", "sas", "sasu", "sarl", "sa", "eurl",
    "sci", "snc", "the", "and", "of", "dba",
}
ALIAS = re.compile(r"^.*\b(?:formerly|also known as|known as|doing business as|trading as|fka|f/k/a|aka|a/k/a"
                   r"|dba|d/b/a)\b[\s:.\-]*", re.I)

ADDR_ABBR = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "lane": "ln", "boulevard": "blvd", "bd": "blvd", "highway": "hwy", "parkway": "pkwy",
    "court": "ct", "place": "pl", "suite": "ste", "apartment": "apt", "floor": "flr",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp",
    "number": "no", "building": "bldg", "sector": "sec", "circle": "cir", "square": "sq",
    "terrace": "ter", "route": "rte", "r": "rue", "chemin": "ch", "impasse": "imp",
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
    "karnataka": "ka", "maharashtra": "mh", "gujarat": "gj", "rajasthan": "rj",
    "telangana": "tg", "kerala": "kl", "punjab": "pb", "haryana": "hr", "bihar": "br",
    "odisha": "od", "orissa": "od", "assam": "as", "goa": "ga", "jharkhand": "jh",
    "uttarakhand": "uk", "chhattisgarh": "cg", "delhi": "dl",
}
ADDR_PHRASES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc", "tamil nadu": "tn", "andhra pradesh": "ap",
    "uttar pradesh": "up", "madhya pradesh": "mp", "west bengal": "wb",
    "himachal pradesh": "hp", "jammu and kashmir": "jk",
}
_PHRASE_RE = re.compile(r"\b(" + "|".join(ADDR_PHRASES) + r")\b")
_collapse = lambda t: re.sub(r"([a-z])\1+", r"\1", t)
STOP = {_collapse(w) for w in NAME_STOP}

MAPS = {"name": {}, "addr": {}}
NOISE = {}


def _base(s):
    return (s if s.isascii() else unidecode(s)).lower()


def name_tokens(s):
    if not s:
        return ""
    m = ALIAS.match(s)
    if m and s[m.end():].strip():
        s = s[m.end():]
    s = _base(s)
    s = re.sub(r"\bwww\.|\.(com|net|org|in|co|fr|us|biz|info)\b|@", " ", s)
    s = re.sub(r"\d{6,}", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[.'`]", "", s)
    return " ".join(_collapse(t) for t in re.sub(r"[^a-z0-9]+", " ", s).split())


def addr_tokens(s):
    if not s or s in ("None", "N/A"):
        return ""
    s = _base(s)
    s = re.sub(r"\bn/a\b|\bndeg", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b0+(\d)", r"\1", s)
    s = _PHRASE_RE.sub(lambda m: ADDR_PHRASES[m.group(1)], s)
    return " ".join(ADDR_ABBR.get(t, t) for t in s.split() if t not in ("none", "na"))


def finish(tokens, kind, country):
    stop = STOP if kind == "name" else set()
    noise = NOISE.get(country, {}).get(kind, set())
    toks = [MAPS[kind].get(t, t) for t in tokens.split()]
    return " ".join([t for t in toks if t not in stop and t not in noise]
                    or [t for t in toks if t not in stop] or toks)


def finish_row(row):
    nm, ad, country = row
    return finish(nm, "name", country), finish(ad, "addr", country)


def learn_map(tq, ts):
    toks = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = pl.DataFrame({"tq": tq, "ts": ts}).with_columns(toks("tq"), toks("ts"))
    count = lambda c: d.select(t=pl.col(c)).explode("t").group_by("t").len(c)
    s1_side = count("tq").join(count("ts"), on="t", how="left").fill_null(0).filter(pl.col("ts") > 0.2 * pl.col("tq"))
    d = (d
         .with_columns(uq=pl.col("tq").list.set_difference("ts"), us=pl.col("ts").list.set_difference("tq"))
         .filter(pl.col("uq").list.len().is_between(1, 3) & pl.col("us").list.len().is_between(1, 3)))
    n_t = d.select("uq").explode("uq").group_by("uq").len("n_t")
    best = (d.select("uq", "us").explode("uq").explode("us").group_by("uq", "us").len("n")
            .sort("n", descending=True).unique("uq", keep="first").join(n_t, on="uq")
            .filter((pl.col("n") >= MAP_MIN) & (pl.col("n") / pl.col("n_t") >= MAP_SHARE)
                    & ~pl.col("uq").str.contains(r"\d") & ~pl.col("uq").is_in(s1_side["t"].implode())
                    & ~((pl.col("uq").str.len_chars() >= 12) & pl.col("uq").str.contains(pl.col("us"), literal=True))
                    & (pl.col("us").str.len_chars() > 1) & ~pl.col("us").is_in(["the", "and", "of", "dba"])))
    return dict(best.select("uq", "us").iter_rows())


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def _tokenize_chunk(args):
    """Worker: tokenize names and addresses of one chunk."""
    names, addrs = args
    return ([name_tokens(x) for x in names], [addr_tokens(x) for x in addrs])


def prep(split):
    tok_dir = f"{WORK}/{split}_tok"
    os.makedirs(tok_dir, exist_ok=True)

    # ------------------------------------------------------------------ PASS 1
    print(f"=== {split}: tokenize ===", flush=True)
    with Pool(THREADS) as pool:
        for src in (1, 2, 3):
            df = read_tsv(f"{DATA}/{split}/{split}_source{src}.tsv")
            n = len(df)
            print(f"  src{src}: {n} rows", flush=True)
            for lo in range(0, n, CHUNK):
                cf = f"{tok_dir}/src{src}_chunk_{lo:09d}.parquet"
                if os.path.exists(cf):
                    print(f"  [ckpt] {cf}", flush=True)
                    continue
                hi = min(lo + CHUNK, n)
                sub = df[lo:hi]
                nm, ad = pool.apply(_tokenize_chunk, ((sub["business_name"].to_list(),
                                                       sub["business_address"].to_list()),))
                out = sub.with_columns(src=pl.lit(src, pl.Int8),
                                       nm=pl.Series(nm), ad=pl.Series(ad))
                out.write_parquet(cf)
                print(f"    {hi}/{n}", flush=True)
                del sub, out, nm, ad
                gc.collect()
            del df
            gc.collect()

    tok_files = sorted(glob.glob(f"{tok_dir}/*.parquet"))
    assert tok_files, "no tokenized chunks produced"

    # ------------------------------------------------------------------ PASS 2
    if split == "train":
        print(f"=== {split}: learn maps ===", flush=True)
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
        s1_needed = set(gt["s1"].unique().to_list())
        q_needed = set(gt["q"].unique().to_list())
        s1_rows, q_rows = [], []
        for f in tok_files:
            sub = pl.read_parquet(f, columns=["entity_id", "nm", "ad"])
            s = sub.filter(pl.col("entity_id").is_in(s1_needed)).rename(
                {"entity_id": "s1", "nm": "nm_s1", "ad": "ad_s1"})
            q = sub.filter(pl.col("entity_id").is_in(q_needed)).rename(
                {"entity_id": "q", "nm": "nm_q", "ad": "ad_q"})
            if len(s): s1_rows.append(s)
            if len(q): q_rows.append(q)
            del sub, s, q
            gc.collect()
        s1_all, q_all = pl.concat(s1_rows), pl.concat(q_rows)
        pairs = gt.join(s1_all, on="s1").join(q_all, on="q")
        MAPS.update(name=learn_map(pairs["nm_q"], pairs["nm_s1"]),
                    addr=learn_map(pairs["ad_q"], pairs["ad_s1"]))
        json.dump(MAPS, open(f"{WORK}/maps.json", "w"), indent=0, sort_keys=True)
        del s1_all, q_all, pairs, s1_rows, q_rows
        gc.collect()
    else:
        maps_path = f"{WORK}/maps.json"
        if os.path.exists(maps_path):
            MAPS.update(json.load(open(maps_path)))
        else:
            print(f"Notice: {maps_path} not found. Running test normalization without precomputed training maps.")

    # ------------------------------------------------------------------ PASS 3
    print(f"=== {split}: filler detection ===", flush=True)
    countries = pl.concat([pl.read_parquet(f, columns=["country"]) for f in tok_files])["country"].unique()
    n = (pl.concat([pl.read_parquet(f, columns=["country", "src"]) for f in tok_files])
         .group_by("country").agg(n1=(pl.col("src") == 1).sum(), n23=(pl.col("src") != 1).sum()))

    def compute_filler(col):
        map_key = {"nm": "name", "ad": "addr"}[col]
        parts = []
        for f in tok_files:
            sub = pl.read_parquet(f, columns=["country", "src", col])
            t = (sub.select("country", "src",
                            t=pl.col(col).str.split(" ").list.eval(pl.element().replace(MAPS[map_key])).list.unique())
                 .explode("t")
                 .filter((pl.col("t") != "") & ~pl.col("t").str.contains(r"^\d+$"))
                 .group_by("country", "t")
                 .agg(s1=(pl.col("src") == 1).sum(), s23=(pl.col("src") != 1).sum()))
            parts.append(t)
            del sub, t
            gc.collect()
        agg = pl.concat(parts).group_by("country", "t").agg(pl.col("s1").sum(), pl.col("s23").sum())
        return (agg.join(n, on="country")
                .filter((pl.col("s23") / pl.col("n23") >= NOISE_LIFT[col] * (pl.col("s1") + 1) / pl.col("n1"))
                        & (pl.col("s23") >= NOISE_MIN * pl.col("n23"))))

    names_agg = compute_filler("nm")
    addrs_agg = compute_filler("ad")
    names = {c: set(g["t"]) for (c,), g in names_agg.group_by("country")}
    addrs = {c: set(g["t"]) for (c,), g in addrs_agg.group_by("country")}
    NOISE.clear()
    NOISE.update({c: {"name": names.get(c, set()) - {""}, "addr": addrs.get(c, set())} for c in countries})
    json.dump({c: {k: sorted(v) for k, v in d.items()} for c, d in NOISE.items()},
              open(f"{WORK}/{split}_noise.json", "w"), indent=0)

    # ------------------------------------------------------------------ PASS 4
    print(f"=== {split}: finish ===", flush=True)
    final_dir = f"{WORK}/{split}_final"
    os.makedirs(final_dir, exist_ok=True)
    with Pool(THREADS) as pool:
        for f in tok_files:
            cf = f"{final_dir}/{os.path.basename(f)}"
            if os.path.exists(cf):
                print(f"  [ckpt] {cf}", flush=True)
                continue
            sub = pl.read_parquet(f)
            out = pool.map(finish_row, zip(sub["nm"], sub["ad"], sub["country"]), chunksize=20_000)
            sub = sub.with_columns(nm=pl.Series([a for a, _ in out]),
                                   ad=pl.Series([b for _, b in out]))
            sub.write_parquet(cf)
            print(f"  wrote {cf}", flush=True)
            del sub, out
            gc.collect()

    # ------------------------------------------------------------------ PASS 5
    print(f"=== {split}: concat ===", flush=True)
    (pl.scan_parquet(f"{final_dir}/*.parquet")
       .sink_parquet(f"{WORK}/{split}.parquet"))

    print(split, {k: len(v) for k, v in MAPS.items()}, "maps;",
          {c: {k: len(v) for k, v in d.items()} for c, d in NOISE.items()}, "filler tokens")


if __name__ == "__main__":
    assert name_tokens("SOLOVA FÁCT L.L.C.") == "solova fact lc"
    assert finish(name_tokens("SOLOVA FÁCT L.L.C."), "name", "US") == "solova fact"
    assert name_tokens("Xylozeta Co formerly known as Cervantes Select Mountain LLC") == "cervantes select mountain lc"
    assert name_tokens("Viozeta formerly: New Delhi Communications") == "new delhi comunications"
    assert name_tokens("internationalforteanimation.com") == "internationalforteanimation"
    assert finish(name_tokens("Fire Master (India) Pvt (Ltd) - 9832661323"), "name", "India") == "fire master india"
    assert finish("lc", "name", "US") == "lc"
    assert addr_tokens("##1824 Brittany Lane, N/A, Edmond, Oklahoma") == "1824 brittany ln edmond ok"
    assert addr_tokens("00806-807 Trinity Orion, GJ, Surat") == "806 807 trinity orion gj surat"
    assert addr_tokens("12 MG Road, Chennai, Tamil Nadu") == "12 mg rd chennai tn"
    assert addr_tokens("N°3 Rue Kant") == "3 rue kant"
    assert addr_tokens("None") == ""

    os.makedirs(WORK, exist_ok=True)
    gt_path = f"{DATA}/train/train_ground_truth.tsv"
    if not os.path.exists(gt_path):
        print("Self-tests passed: all 12 normalization assertions verified.")
        print(f"Notice: Ground truth dataset not found at '{gt_path}'.")
        print(f"To run normalization on competition data, place dataset files under '{DATA}/train' and '{DATA}/test'.")
        sys.exit(0)

    gt = read_tsv(gt_path)
    gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode(
        "matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename(
        {"source1_entity_id": "s1", "matched_entity_ids": "q"}).write_parquet(f"{WORK}/train_gt.parquet")
    for split in ("train", "test"):
        prep(split)