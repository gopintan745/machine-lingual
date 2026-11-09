"""Why does the length filter drop pairs from one source, and what would each rule change recover?

    python -m scripts.inspect_lengths --source news-commentary --pair ar-en
    python -m scripts.inspect_lengths --source news-commentary --pair ar-en --n 5000 --show 8

Reads token counts with the real NLLB tokenizer (the same counts clean_pairs uses). Reports:
  * exclusive reasons a pair is dropped (too short / too long / length ratio)
  * length and ratio percentiles per side
  * example pairs for each reason
  * what-if table: share kept for other max_tokens and ratio ranges

Note: raising max_tokens above data.tokenizer.max_length keeps pairs that training then TRUNCATES
(source and target cut at different places) and raises memory use. Prefer dropping long pairs.
"""
from __future__ import annotations

import argparse

import numpy as np

from configs.config import Config, load_config

GRID_TOKENS = (128, 160, 192, 256)
GRID_RATIOS = ((0.5, 2.0), (0.4, 2.5), (0.33, 3.0))


def token_lengths(tokenizer, texts) -> np.ndarray:
    return np.array([len(x) for x in tokenizer(list(texts), add_special_tokens=False).input_ids])


def diagnose(ds, a: str, b: str, tokenizer, cfg: Config, show: int = 5, seed: int = 0) -> dict:
    c = cfg.data.cleaning
    texts_a, texts_b = list(ds[a]), list(ds[b])
    la, lb = token_lengths(tokenizer, texts_a), token_lengths(tokenizer, texts_b)
    n = len(la)
    ratio = la / np.maximum(lb, 1)                                   # tokens in a / tokens in b
    lo, hi, (r_lo, r_hi) = c.min_tokens, c.max_tokens, c.length_ratio

    short = np.minimum(la, lb) < lo
    long_ = np.maximum(la, lb) > hi
    off = (ratio < r_lo) | (ratio > r_hi)
    reason = np.where(short, "too short", np.where(long_, "too long", np.where(off, "ratio", "kept")))

    print(f"\n{n} pairs, settings: min_tokens={lo}, max_tokens={hi}, length_ratio=[{r_lo}, {r_hi}]")
    print("\nExclusive reason (checked in this order):")
    counts = {}
    for name in ("too short", "too long", "ratio", "kept"):
        counts[name] = int((reason == name).sum())
        print(f"  {name:10s} {counts[name]:>6d}  {counts[name] / n:6.1%}")
    print(f"\nEach rule on its own (a pair can break several):  short {short.mean():.1%} | "
          f"long {long_.mean():.1%} (>{hi} on {a}: {(la > hi).mean():.1%}, on {b}: {(lb > hi).mean():.1%}) | "
          f"ratio {off.mean():.1%}")

    pcts = (5, 25, 50, 75, 95)
    print(f"\nPercentiles {pcts}")
    print(f"  tokens {a}: {np.percentile(la, pcts).round(0).astype(int).tolist()}")
    print(f"  tokens {b}: {np.percentile(lb, pcts).round(0).astype(int).tolist()}")
    print(f"  ratio {a}/{b}: {np.percentile(ratio, pcts).round(2).tolist()}")

    print("\nWhat-if: share of pairs kept (current setting marked *)")
    header = "  max_tokens  " + "  ".join(f"ratio[{x},{y}]" for x, y in GRID_RATIOS)
    print(header)
    for t in GRID_TOKENS:
        cells = []
        for rl, rh in GRID_RATIOS:
            keep = (np.minimum(la, lb) >= lo) & (np.maximum(la, lb) <= t) & (ratio >= rl) & (ratio <= rh)
            mark = "*" if (t, (rl, rh)) == (hi, (r_lo, r_hi)) else " "
            cells.append(f"{keep.mean():>{len(f'ratio[{rl},{rh}]') - 1}.1%}{mark}")
        print(f"  {t:<10d}  " + "  ".join(cells))
    print(f"  (training truncates at data.tokenizer.max_length = {cfg.data.tokenizer.max_length})")

    rng = np.random.default_rng(seed)
    for name in ("too short", "too long", "ratio"):
        idx = np.flatnonzero(reason == name)
        print(f"\n--- examples: {name} ({len(idx)} pairs)")
        for i in rng.permutation(idx)[:show]:
            print(f"  [{a}:{la[i]} tok | {b}:{lb[i]} tok | ratio {ratio[i]:.2f}]")
            print(f"    {a}: {texts_a[i][:120]}")
            print(f"    {b}: {texts_b[i][:120]}")
    return {"n": n, "counts": counts, "ratio_percentiles": np.percentile(ratio, pcts).tolist()}


def main():
    from transformers import AutoTokenizer

    from src.data.data_prep import _load_pair

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--pair", required=True, help="e.g. ar-en")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    a, b = args.pair.split("-")
    source = next((s for s in cfg.data.sources if s.name == args.source), None)
    if source is None:
        raise SystemExit(f"Unknown source '{args.source}'. Known: {[s.name for s in cfg.data.sources]}")
    ds = _load_pair(source, a, b, cfg.data.seed, args.n)
    if ds is None:
        raise SystemExit("Could not load the source (see the warning above).")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, cache_dir=cfg.model.cache_dir)
    diagnose(ds, a, b, tokenizer, cfg, show=args.show)


if __name__ == "__main__":
    main()