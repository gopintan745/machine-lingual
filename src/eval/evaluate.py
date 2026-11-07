"""Evaluation harness: systems x directions -> chrF++, paired bootstrap vs baseline, judge inputs.

Generation is cached per (system, direction) on disk, so a Kaggle disconnect only costs the unfinished
directions. The translator is injected (translator_factory), so the harness logic is testable without a GPU.
"""
from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Callable, ContextManager, Dict, List, Optional, Tuple

import numpy as np
import sacrebleu

from configs.config import Config, EvalSystem
from src.eval.metrics import compute_chrf_corpus_score, compute_chrf_sentence_scores, make_chrf

Translator = Callable[[List[str], str, str], Tuple[List[str], dict]]
TestSets = Dict[str, Tuple[List[str], List[str]]]


# ----------------------------------------------------------------------------- caching
def _digest(sources: List[str]) -> str:
    return hashlib.md5("\n".join(sources).encode("utf-8")).hexdigest()


def _cache_file(out: Path, system: str, direction: str) -> Path:
    return out / "hypotheses" / f"{system}__{direction}.json"


def _read_cache(out: Path, system: str, direction: str, sources: List[str]) -> Optional[dict]:
    f = _cache_file(out, system, direction)
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    ok = data.get("source_digest") == _digest(sources) and len(data.get("hypotheses", [])) == len(sources)
    return data if ok else None


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------- statistics
def paired_bootstrap(base: np.ndarray, other: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> dict:
    """Bootstrap the mean per-sentence chrF++ difference (other - base) over the SAME sentences.
    Uses sentence-level scores, so it approximates (does not equal) the corpus-score difference."""
    diffs = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "p_better": float((means > 0).mean()), "significant": bool(lo > 0 or hi < 0)}


def _markdown(cfg: Config, summary: dict, directions: List[str], systems: List[str]) -> str:
    base = cfg.eval.baseline
    lines = ["| direction | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for d in directions:
        row = []
        for s in systems:
            cell = f"{summary['results'][s][d]['chrf']:.1f}"
            comp = summary["comparisons"].get(s, {}).get(d)
            if comp:
                lo, hi = comp["ci95"]
                star = "*" if comp["significant"] else ""
                cell += f" ({comp['mean_diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]){star}"
            row.append(cell)
        lines.append(f"| {d} | " + " | ".join(row) + " |")
    macro = " | ".join(f"{summary['macro_avg_chrf'][s]:.1f}" for s in systems)
    lines.append(f"| macro avg | {macro} |")
    lines.append("")
    lines.append(f"chrF++ on the held-out test set. Parentheses: difference vs `{base}` with 95% bootstrap CI; "
                 f"* = CI excludes 0. Macro average is a rough summary across languages.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main entry
def run_evaluation(
    cfg: Config,
    test_sets: TestSets,
    translator_factory: Callable[[EvalSystem], ContextManager[Translator]],
    out_dir: Optional[str] = None,
    force: bool = False,
    systems: Optional[List[str]] = None,
) -> dict:
    ev = cfg.eval
    out = Path(out_dir or ev.output_dir)
    (out / "hypotheses").mkdir(parents=True, exist_ok=True)
    selected = [s for s in ev.systems if not systems or s.name in systems]
    if not selected:
        raise ValueError("No systems selected for evaluation")
    directions = list(test_sets)
    chrf = make_chrf(ev.chrf.word_order)

    hyps: Dict[str, Dict[str, List[str]]] = {}
    truncated: Dict[str, Dict[str, int]] = {}
    for system in selected:
        hyps[system.name], truncated[system.name] = {}, {}
        todo = []
        for d in directions:
            cached = None if force else _read_cache(out, system.name, d, test_sets[d][0])
            if cached:
                hyps[system.name][d] = cached["hypotheses"]
                truncated[system.name][d] = cached.get("truncated_inputs", 0)
            else:
                todo.append(d)
        if not todo:
            print(f"[{system.name}] all directions cached")
            continue
        with translator_factory(system) as translate:
            for d in todo:
                src, tgt = d.split("-")
                sources = test_sets[d][0]
                t0 = time()
                hyp, stats = translate(sources, src, tgt)
                hyps[system.name][d] = hyp
                truncated[system.name][d] = stats.get("truncated_inputs", 0)
                _cache_file(out, system.name, d).write_text(json.dumps({
                    "system": system.name, "direction": d, "source_digest": _digest(sources),
                    "truncated_inputs": truncated[system.name][d], "hypotheses": hyp}, ensure_ascii=False),
                    encoding="utf-8")
                print(f"[{system.name}] {d}: {len(sources)} sentences in {time() - t0:.0f}s "
                      f"(truncated inputs: {truncated[system.name][d]})")
        translate = None                    # drop the last reference to the model, then free GPU memory
        _free_gpu()

    # ---- scores
    names = [s.name for s in selected]
    results: Dict[str, Dict[str, dict]] = {}
    sent: Dict[str, Dict[str, np.ndarray]] = {}
    signature = ""                          # sacrebleu only knows it after the first scoring call
    for name in names:
        results[name], sent[name] = {}, {}
        for d in directions:
            refs = test_sets[d][1]
            corpus = compute_chrf_corpus_score(hyps[name][d], refs, chrf)
            signature = corpus["signature"]
            scores = np.array(compute_chrf_sentence_scores(hyps[name][d], refs, chrf))
            sent[name][d] = scores
            results[name][d] = {"n": len(refs), "chrf": corpus["score"],
                                "chrf_sentence_mean": float(scores.mean()),
                                "truncated_inputs": truncated[name][d]}

    comparisons: Dict[str, Dict[str, dict]] = {}
    if ev.baseline in names:
        for name in names:
            if name == ev.baseline:
                continue
            comparisons[name] = {d: paired_bootstrap(sent[ev.baseline][d], sent[name][d],
                                                     ev.bootstrap_resamples, ev.seed) for d in directions}
    else:
        print(f"[WARN] baseline '{ev.baseline}' not among evaluated systems; skipping comparisons")

    summary = {
        "meta": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "sacrebleu_version": sacrebleu.__version__, "chrf_signature": signature,
                 "test_set": {"kind": ev.test_set.kind, "hub_id": ev.test_set.hub_id,
                              "split": ev.test_set.split, "path": ev.test_set.path},
                 "generation": {"num_beams": ev.generation.num_beams,
                                "max_new_tokens": ev.generation.max_new_tokens},
                 "bootstrap_resamples": ev.bootstrap_resamples, "baseline": ev.baseline},
        "results": results,
        "comparisons": comparisons,
        "macro_avg_chrf": {n: float(np.mean([results[n][d]["chrf"] for d in directions])) for n in names},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    table = _markdown(cfg, summary, directions, names)
    (out / "summary.md").write_text(table, encoding="utf-8")

    _write_judge_inputs(cfg, test_sets, hyps, out)
    print("\n" + table + f"\n\nSaved to {out}  (signature: {signature})")
    return summary


def _write_judge_inputs(cfg: Config, test_sets: TestSets, hyps: Dict[str, Dict[str, List[str]]], out: Path) -> None:
    """Seeded sample of rows (the SAME rows for every system) for the LLM judge and the human check."""
    k = cfg.eval.llm_judge.samples_per_direction
    if k <= 0:
        return
    rng = np.random.default_rng(cfg.eval.seed)
    with (out / "judge_inputs.jsonl").open("w", encoding="utf-8") as f:
        for d, (sources, refs) in test_sets.items():
            for i in sorted(rng.choice(len(sources), size=min(k, len(sources)), replace=False).tolist()):
                f.write(json.dumps({"direction": d, "idx": i, "source": sources[i], "reference": refs[i],
                                    "outputs": {n: hyps[n][d][i] for n in hyps}}, ensure_ascii=False) + "\n")