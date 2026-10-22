from __future__ import annotations

import gc
import json
from pathlib import Path

import optuna
import torch
from optuna.importance import get_param_importances
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import TrainerCallback

from configs.config import Config
from src.eval.visuals import plot_hyperparameter_importance, plot_optimization_results
from src.models.model import load_model
from src.training.trainer import build_trainer


class PruneCallback(TrainerCallback):
    """Report eval loss to Optuna at every evaluation so the pruner has data to act on."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.trial.report(metrics["eval_loss"], state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def apply_trial_params(cfg: Config, params: dict) -> None:
    """Write HPO parameters into a (copied) config."""
    if "learning_rate" in params:
        cfg.training.learning_rate = params["learning_rate"]
    if "warmup_ratio" in params:
        cfg.training.warmup_ratio = params["warmup_ratio"]
    if "r" in params:
        cfg.lora.r = params["r"]
    if "lora_alpha" in params:
        cfg.lora.alpha = params["lora_alpha"]


def objective(trial: optuna.Trial, base_cfg: Config, train_ds, val_ds, tokenizer) -> float:
    cfg = base_cfg.copy()                                   # never mutate the shared config
    sp = cfg.hpo.space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", sp["learning_rate"]["low"],
                                             sp["learning_rate"]["high"], log=sp["learning_rate"].get("log", True)),
        "warmup_ratio": trial.suggest_float("warmup_ratio", sp["warmup_ratio"]["low"], sp["warmup_ratio"]["high"]),
        "r": trial.suggest_categorical("r", sp["r"]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", sp["alpha"]),
    }
    apply_trial_params(cfg, params)

    subset = train_ds.shuffle(seed=cfg.data.seed).select(range(min(cfg.data.sizes.hpo_subset, len(train_ds))))
    steps = cfg.hpo.max_steps_per_trial
    model = trainer = None
    try:
        model = load_model(cfg)                             # fresh base weights + new adapter
        trainer = build_trainer(
            model, subset, val_ds, tokenizer, cfg,
            output_dir=f"{cfg.training.output_dir}/hpo/trial_{trial.number}",
            max_steps=steps, for_hpo=True, eval_steps=max(1, steps // 4),
        )
        trainer.add_callback(PruneCallback(trial))
        trainer.train()
        return float(trainer.evaluate()["eval_loss"])
    finally:                                                # free GPU memory between trials
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


def run_hyperparameter_search(cfg: Config, train_ds, val_ds, tokenizer) -> optuna.Study:
    study = optuna.create_study(
        study_name=cfg.hpo.study_name,
        storage=cfg.hpo.storage,
        direction="minimize",
        load_if_exists=True,                                # resume after a crash / new session
        sampler=TPESampler(seed=cfg.data.seed),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1),
    )
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
               for t in study.trials)
    remaining = cfg.hpo.n_trials - done
    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cfg, train_ds, val_ds, tokenizer),
            n_trials=remaining,
            catch=(torch.cuda.OutOfMemoryError,),           # an OOM trial is marked failed, search continues
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No HPO trial completed. Check the logs above (OOM? bad config?).")

    best = study.best_trial
    print(f"Best trial {best.number}: eval_loss={best.value:.4f}, params={best.params}")
    out_dir = Path(cfg.training.output_dir)
    (out_dir / "best_params.json").write_text(json.dumps(best.params, indent=2))

    plot_optimization_results([t.value for t in completed], save_dir=str(out_dir))
    if len(completed) >= 2:                                 # importance needs >= 2 completed trials
        plot_hyperparameter_importance(get_param_importances(study), save_dir=str(out_dir))
    return study