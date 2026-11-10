"""Final training with the best HPO parameters. Resumes from the last checkpoint if one exists
(locally, or restored from the Hub when hub.enabled)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import optuna
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from configs.config import Config
from src.models.model import load_model
from src.training import hub
from src.training.hpo import apply_trial_params
from src.training.trainer import build_trainer


class HubCheckpointCallback(TrainerCallback):
    """Upload the newest checkpoint every `hub.push_checkpoints_every` saves."""

    def __init__(self, cfg: Config, meta: dict):
        self.cfg, self.meta, self.n_saves = cfg, meta, 0

    def on_save(self, args, state, control, **kwargs):
        every = self.cfg.hub.push_checkpoints_every
        if every <= 0 or not state.is_world_process_zero:
            return
        self.n_saves += 1
        if self.n_saves % every == 0:
            hub.push_checkpoint(self.cfg, Path(args.output_dir) / f"checkpoint-{state.global_step}", self.meta)


def full_training(cfg: Config, train_ds, val_ds, tokenizer) -> dict:
    cfg = cfg.copy()
    study = optuna.load_study(study_name=cfg.hpo.study_name, storage=cfg.hpo.storage)
    params = study.best_trial.params
    print(f"Training with best HPO params: {params}")
    meta = {"fingerprint": hub.fingerprint(cfg, with_training=True, params=params), "params": params}
    apply_trial_params(cfg, params)                       # applied BEFORE the model/adapter is built

    ckpt_dir = cfg.training.output_dir
    if cfg.hub.enabled:
        hub.restore_checkpoint(cfg, ckpt_dir, meta)       # no-op if a local checkpoint exists

    model = load_model(cfg)                               # fresh model with the best r / alpha
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, cfg)
    if cfg.hub.enabled:
        trainer.add_callback(HubCheckpointCallback(cfg, meta))

    last = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if last:
        print(f"Resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    best_dir = Path(ckpt_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))                     # PeftModel -> adapter weights only
    tokenizer.save_pretrained(str(best_dir))

    metrics = trainer.evaluate()
    (best_dir / "final_metrics.json").write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
    print(f"Saved adapter + tokenizer to {best_dir}. Final eval_loss={metrics['eval_loss']:.4f}")
    if cfg.hub.enabled:
        hub.push_final(cfg, best_dir, params, metrics)
    return metrics