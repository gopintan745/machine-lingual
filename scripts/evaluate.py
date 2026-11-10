"""Evaluate systems on the held-out test set. Run from the repo root:

    python -m scripts.evaluate                       # all systems in eval.systems
    python -m scripts.evaluate --systems baseline    # zero-shot baseline only (do this FIRST)
    python -m scripts.evaluate --max-rows 50         # quick plumbing check, written to results_smoke/
    python -m scripts.evaluate --force               # regenerate even if cached
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse

from configs.config import load_config
from src.eval.evaluate import run_evaluation
from src.eval.testsets import load_test_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--systems", default=None, help="comma-separated system names (default: all)")
    ap.add_argument("--max-rows", type=int, default=None, help="use only the first N test sentences")
    ap.add_argument("--force", action="store_true", help="ignore cached hypotheses")
    ap.add_argument("--push", action="store_true", help="upload the results folder to the Hub repo (needs HF_TOKEN)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    test_sets = load_test_sets(cfg, max_rows=args.max_rows)
    for d, (src, _) in test_sets.items():
        print(f"{d}: {len(src)} test sentences")

    from src.eval.generate import loaded_translator          # imports torch: keep out of module scope
    out_dir = cfg.eval.output_dir + ("_smoke" if args.max_rows else "")
    run_evaluation(cfg, test_sets, lambda system: loaded_translator(system, cfg), out_dir=out_dir,
                   force=args.force, systems=args.systems.split(",") if args.systems else None)
    if args.push:
        if args.max_rows:
            raise SystemExit("--push is not allowed with --max-rows (partial results).")
        from src.training import hub
        hub.check_hub_ready(cfg)
        hub.push_results(cfg, out_dir)


if __name__ == "__main__":
    main()