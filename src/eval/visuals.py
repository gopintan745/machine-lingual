"""Plot helpers. Every function saves a PNG into `save_dir` and closes the figure.
Pass show=True only in notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, name: str, save_dir: str, show: bool) -> str:
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return str(path)


def plot_chrf_scores(chrf_scores: dict, save_dir: str = "results", title: str = "chrF++ by direction",
                     show: bool = False) -> str:
    """chrF++ from sacrebleu is on a 0-100 scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(chrf_scores), list(chrf_scores.values()), color="skyblue")
    ax.set(title=title, xlabel="Direction", ylabel="chrF++", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return _finish(fig, "chrf_scores", save_dir, show)


def plot_loss_curve(steps, train_losses, val_steps, val_losses, save_dir: str = "results",
                    name: str = "loss_curve", show: bool = False) -> str:
    """The Trainer logs by step, so the x-axis is the global step (take values from trainer.state.log_history)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_losses, label="Training loss")
    ax.plot(val_steps, val_losses, label="Validation loss", marker="o")
    ax.set(title="Training and validation loss", xlabel="Step", ylabel="Loss")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.7)
    return _finish(fig, name, save_dir, show)


def plot_optimization_results(trial_values, save_dir: str = "results", show: bool = False) -> str:
    """trial_values: eval losses of COMPLETED trials only (failed/pruned trials have no value)."""
    values = [v for v in trial_values if v is not None]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", color="purple")
    ax.set(title="HPO: eval loss per completed trial", xlabel="Completed trial", ylabel="Eval loss")
    ax.grid(linestyle="--", alpha=0.7)
    return _finish(fig, "hpo_results", save_dir, show)


def plot_hyperparameter_importance(importance: dict, save_dir: str = "results", show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(importance), list(importance.values()), color="orange")
    ax.set(title="Hyperparameter importance", xlabel="Hyperparameter", ylabel="Importance")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return _finish(fig, "hpo_importance", save_dir, show)


def plot_chrf_vs_llm_judgement(chrf_scores: list, judge_scores: list, save_dir: str = "results",
                               show: bool = False) -> str:
    """One point per translated sentence (not per direction): chrF++ (0-100) vs judge score (1-5)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(chrf_scores, judge_scores, color="green", alpha=0.5)
    ax.set(title="chrF++ vs LLM judge", xlabel="Sentence chrF++", ylabel="Judge score")
    ax.grid(linestyle="--", alpha=0.7)
    return _finish(fig, "chrf_vs_judge", save_dir, show)