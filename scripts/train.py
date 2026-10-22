"""Entry point. Run from the repo root:

    python -m scripts.train --stage all            # data -> HPO -> final training
    python -m scripts.train --stage all --smoke    # tiny end-to-end check (do this first)
    python -m scripts.train --stage hpo            # resume HPO only (study is stored on disk)

On Kaggle set WORK_DIR=/kaggle/working and HF_USERNAME first.
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from configs.config import load_config
from src.data.data_prep import prepare_data
from src.training.hpo import run_hyperparameter_search
from src.training.train_full import full_training


def apply_smoke(cfg):
    """Shrink everything so a full pass finishes in minutes. Uses separate study/output names."""
    cfg.data.sizes.train_per_direction = 200
    cfg.data.sizes.val_per_direction = 50
    cfg.data.sizes.hpo_subset = 100
    cfg.data.sizes.pool_factor = 2
    cfg.hpo.n_trials = 2
    cfg.hpo.max_steps_per_trial = 20
    cfg.hpo.study_name += "_smoke"
    cfg.training.epochs = 1
    cfg.training.eval_steps = cfg.training.save_steps = 10
    cfg.training.output_dir = cfg.training.output_dir.rstrip("/") + "_smoke"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    ap.add_argument("--stage", choices=["data", "hpo", "train", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the whole pipeline")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the tokenized datasets")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, cache_dir=cfg.model.cache_dir)
    train_ds, val_ds = prepare_data(cfg, tokenizer, use_cache=not args.no_cache)
    print(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)}")
    if args.stage == "data":
        return

    if args.stage in ("hpo", "all"):
        run_hyperparameter_search(cfg, train_ds, val_ds, tokenizer)
    if args.stage in ("train", "all"):
        full_training(cfg, train_ds, val_ds, tokenizer)


if __name__ == "__main__":
    main()