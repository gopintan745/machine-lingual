"""Run the LLM judge on the rows scripts.evaluate sampled. Run from the repo root:

    python -m scripts.judge --dry-run                 # count calls and cache hits, call nothing
    python -m scripts.judge --limit 5                 # tiny paid test: first 5 rows per direction
    python -m scripts.judge                           # full run (resumes from the cache)
    python -m scripts.judge --mode pairwise           # baseline vs finetuned, A/B order randomised
    python -m scripts.judge --export-human-check      # blind CSV for native speakers (no API calls)
    python -m scripts.judge --push                    # upload the judge files to the Hub repo

Needs eval.llm_judge.model and an API key (env var or Kaggle secret named by eval.llm_judge.api_key_env).
"""
from __future__ import annotations

import argparse

from configs.config import load_config
from src.eval.judge import export_human_check, run_judge
from src.eval.judge_clients import make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None, help="default: eval.output_dir")
    ap.add_argument("--mode", choices=["pointwise", "pairwise", "both"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows per direction")
    ap.add_argument("--dry-run", action="store_true", help="count calls and cache hits; make none")
    ap.add_argument("--export-human-check", action="store_true", help="write the blind native-speaker sheet")
    ap.add_argument("--push", action="store_true", help="upload judge files to the Hub repo (needs HF_TOKEN)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = args.results_dir or cfg.eval.output_dir

    if args.export_human_check:
        csv_path, key_path = export_human_check(cfg, results_dir)
        print(f"Wrote {csv_path}\nWrote {key_path}  (keep this file away from the raters)")
        return

    client = None if args.dry_run else make_client(cfg.eval.llm_judge)
    run_judge(cfg, results_dir, client, mode=args.mode, limit=args.limit, dry_run=args.dry_run)

    if args.push and not args.dry_run:
        if args.limit:
            raise SystemExit("--push is not allowed with --limit (partial results).")
        from src.training import hub
        hub.check_hub_ready(cfg)
        hub.push_results(cfg, results_dir)


if __name__ == "__main__":
    main()