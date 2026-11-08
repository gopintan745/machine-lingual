"""See what the language-ID filter does to ONE source, so thresholds are chosen from evidence.

    python -m scripts.inspect_langid --source armelrandy --pair en-sw --lang sw
    python -m scripts.inspect_langid --source amaniopia  --pair en-sw --lang sw --n 1000

--source is the `name` of an entry in data.sources. Prints: pass rate at several thresholds,
what fastText calls the rows it rejects, and example sentences from each confidence band.
"""
from __future__ import annotations

import argparse
from collections import Counter

from configs.config import load_config
from src.data.data_prep import LangID, _load_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--pair", required=True, help="e.g. en-sw")
    ap.add_argument("--lang", required=True, help="which side to inspect, e.g. sw")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--show", type=int, default=6, help="examples per confidence band")
    args = ap.parse_args()

    cfg = load_config(args.config)
    a, b = args.pair.split("-")
    source = next((s for s in cfg.data.sources if s.name == args.source), None)
    if source is None:
        raise SystemExit(f"Unknown source '{args.source}'. Known: {[s.name for s in cfg.data.sources]}")
    ds = _load_pair(source, a, b, cfg.data.seed, args.n)
    if ds is None:
        raise SystemExit("Could not load the source (see the warning above).")

    lid = LangID(cfg.data.cleaning.langid.model_path)
    want = cfg.data.languages[args.lang].fasttext
    texts = list(ds[args.lang])
    preds = lid.predict(texts)
    n = len(texts)

    print(f"\n{args.source} / {args.lang}: {n} sentences (after dropping empty rows)")
    for t in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7):
        ok = sum(lab == want and p >= t for lab, p in preds) / n
        print(f"  label == {want} and confidence >= {t:.1f}: {ok:.2f}")

    wrong = Counter(lab for lab, _ in preds if lab != want)
    print(f"\nLabelled as something else ({sum(wrong.values())} rows), top labels: {wrong.most_common(8)}")

    bands = [("wrong label", lambda l, p: l != want),
             ("right label, conf < 0.2", lambda l, p: l == want and p < 0.2),
             ("right label, 0.2 <= conf < 0.35", lambda l, p: l == want and 0.2 <= p < 0.35),
             ("right label, 0.35 <= conf < 0.5", lambda l, p: l == want and 0.35 <= p < 0.5),
             ("right label, conf >= 0.5", lambda l, p: l == want and p >= 0.5)]
    for name, test in bands:
        rows = [(t, l, p) for t, (l, p) in zip(texts, preds) if test(l, p)]
        print(f"\n--- {name}: {len(rows)} rows")
        for t, l, p in rows[:args.show]:
            print(f"  [{l} {p:.2f}] {t[:140]}")


if __name__ == "__main__":
    main()