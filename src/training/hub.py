"""Hugging Face Hub sync, so a Kaggle session ending does not cost the run.

Repo layout (model repo `hub.repo_id`):
    README.md              model card (written at the end of training)
    best_params.json       best HPO parameters
    best_model/            final LoRA adapter + tokenizer + final_metrics.json
    hpo/hpo.db, meta.json  Optuna study (uploaded after every trial) + config fingerprint
    last-checkpoint/       latest full Trainer checkpoint (every hub.push_checkpoints_every saves)
    results/               evaluation summary (scripts.evaluate --push)

Background syncs (checkpoints, HPO state) never stop training: a failed upload prints a warning.
The final upload is strict: if it fails you get an error, and the files are still on local disk.
Restores only happen when the stored fingerprint matches the current config, so a checkpoint from a
different configuration is never resumed by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from configs.config import Config


# ----------------------------------------------------------------------------- access
def get_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.getenv(name):
            return os.getenv(name)
    try:                                                   # Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=get_token())


def check_hub_ready(cfg: Config) -> None:
    """Fail early (before hours of training) if pushing cannot work."""
    if "$" in cfg.hub.repo_id or cfg.hub.repo_id.startswith("/") or "/" not in cfg.hub.repo_id:
        raise SystemExit(f"hub.repo_id is '{cfg.hub.repo_id}'. Set HF_USERNAME (or edit hub.repo_id) "
                         f"so it looks like 'username/name'.")
    if not get_token():
        raise SystemExit("No Hugging Face token found. Set HF_TOKEN (Kaggle: Add-ons -> Secrets -> HF_TOKEN) "
                         "with a token that has WRITE access.")
    try:
        _api().whoami()
    except Exception as e:
        raise SystemExit(f"Hugging Face token was rejected: {e!r}")


def _ensure_repo(api, cfg: Config) -> None:
    api.create_repo(cfg.hub.repo_id, private=cfg.hub.private, repo_type="model", exist_ok=True)


def _guard(strict: bool, what: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
        return True
    except Exception as e:
        if strict:
            raise
        print(f"[hub] WARNING: {what} failed ({e!r}). Training continues; files stay on local disk.")
        return False


# ----------------------------------------------------------------------------- uploads
def push_file(cfg: Config, local_path, path_in_repo: str, message: str, strict: bool = False) -> bool:
    def go():
        api = _api()
        _ensure_repo(api, cfg)
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=cfg.hub.repo_id,
                        repo_type="model", commit_message=message)
    return _guard(strict, f"upload of {path_in_repo}", go)


def push_folder(cfg: Config, folder, path_in_repo: str, message: str, strict: bool = False) -> bool:
    def go():
        api = _api()
        _ensure_repo(api, cfg)
        api.upload_folder(folder_path=str(folder), path_in_repo=path_in_repo, repo_id=cfg.hub.repo_id,
                          repo_type="model", commit_message=message)
    return _guard(strict, f"upload of {path_in_repo}/", go)


# ----------------------------------------------------------------------------- fingerprints
def fingerprint(cfg: Config, *, with_training: bool = False, params: Optional[dict] = None) -> str:
    """Hash of everything a stored HPO study / checkpoint depends on."""
    parts = {
        "data": asdict(cfg.data),
        "base": cfg.model.base,
        "lora_targets": cfg.lora.target_modules,
        "batch": [cfg.training.per_device_batch_size, cfg.training.grad_accum_steps],
        "hpo": {"space": cfg.hpo.space, "max_steps": cfg.hpo.max_steps_per_trial},  # n_trials may grow
    }
    if with_training:
        parts["epochs"] = cfg.training.epochs
        parts["params"] = params or {}
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- HPO state
def push_hpo_state(cfg: Config) -> bool:
    """Upload the Optuna database (after each trial) so HPO can resume in a new session."""
    db = Path(cfg.hpo.storage_path)
    if not db.exists():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(db, Path(tmp) / "hpo.db")
        (Path(tmp) / "meta.json").write_text(json.dumps({"fingerprint": fingerprint(cfg)}))
        return push_folder(cfg, tmp, "hpo", "Update HPO state")


def _download_file(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=get_token())


def _download_folder(repo_id: str, pattern: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, allow_patterns=[pattern], local_dir=local_dir, token=get_token())


def restore_hpo(cfg: Config) -> bool:
    """Download the HPO study if there is none locally and the stored fingerprint matches."""
    db = Path(cfg.hpo.storage_path)
    if db.exists():
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            meta = json.loads(Path(_download_file(cfg.hub.repo_id, "hpo/meta.json", tmp)).read_text())
            if meta.get("fingerprint") != fingerprint(cfg):
                print("[hub] Stored HPO state was made with a different configuration; starting fresh.")
                return False
            src = _download_file(cfg.hub.repo_id, "hpo/hpo.db", tmp)
            db.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, db)
        print(f"[hub] Restored HPO study from {cfg.hub.repo_id}")
        return True
    except Exception as e:
        print(f"[hub] No HPO state restored ({type(e).__name__}); starting fresh.")
        return False


# ----------------------------------------------------------------------------- checkpoints
def push_checkpoint(cfg: Config, ckpt_dir, meta: dict) -> bool:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return False
    (ckpt_dir / "run_meta.json").write_text(json.dumps(meta))
    return push_folder(cfg, ckpt_dir, "last-checkpoint", f"Checkpoint {ckpt_dir.name}")


def restore_checkpoint(cfg: Config, out_dir: str, meta: dict) -> bool:
    """Bring last-checkpoint/ back as out_dir/checkpoint-<step> if none exists locally and it matches."""
    out = Path(out_dir)
    if any(out.glob("checkpoint-*")):
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _download_folder(cfg.hub.repo_id, "last-checkpoint/*", tmp)
            src = Path(tmp) / "last-checkpoint"
            stored = json.loads((src / "run_meta.json").read_text())
            if stored != meta:
                print("[hub] Stored checkpoint was made with a different configuration or HPO result; ignoring it.")
                return False
            step = json.loads((src / "trainer_state.json").read_text())["global_step"]
            out.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, out / f"checkpoint-{step}")
        print(f"[hub] Restored checkpoint-{step} from {cfg.hub.repo_id}")
        return True
    except Exception as e:
        print(f"[hub] No checkpoint restored ({type(e).__name__}); training starts from scratch.")
        return False


# ----------------------------------------------------------------------------- final upload + card
def build_model_card(cfg: Config, params: dict, metrics: dict) -> str:
    langs = ", ".join(f"{v.name} ({k})" for k, v in cfg.data.languages.items())
    sources = "\n".join(
        f"- `{s.name}` ({s.hub_id or s.data_file})" + (f", capped at {s.max_rows} rows" if s.max_rows else "")
        for s in cfg.data.sources)
    p = {**{"r": cfg.lora.r, "lora_alpha": cfg.lora.alpha}, **params}
    base = cfg.model.base
    adapter = cfg.hub.repo_id
    nllb = {k: v.nllb for k, v in cfg.data.languages.items()}
    return f"""---
base_model: {base}
library_name: peft
pipeline_tag: translation
language: {json.dumps(list(cfg.data.languages))}
tags: [translation, lora, nllb]
---

# {adapter.split('/')[-1]}

LoRA adapter for `{base}`, fine-tuned for sentence-level translation between {langs}.
Directions trained: {', '.join(cfg.data.directions)}.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("{base}", src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("{base}")
model = PeftModel.from_pretrained(model, "{adapter}", subfolder="best_model").merge_and_unload().eval()

inputs = tok("The government announced a new policy today.", return_tensors="pt")
out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
                     num_beams=4, max_new_tokens=256)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

NLLB language codes used: {json.dumps(nllb)}. The target language token MUST be forced as shown.

## Training

- Method: LoRA (r={p['r']}, alpha={p['lora_alpha']}, dropout={cfg.lora.dropout}) on {', '.join(cfg.lora.target_modules)}.
- Best HPO parameters: `{json.dumps(params)}`.
- Data: up to {cfg.data.sizes.train_per_direction} pairs per direction, both directions of each pair, mixed.
- Sources:
{sources}
- Cleaning: token-length and length-ratio filters, markup filters, fastText language ID, de-duplication,
  and removal of any pair containing a test-set sentence.
- Final validation loss: {metrics.get('eval_loss', 'n/a')}

## Evaluation

chrF++ on the held-out test set, per direction, against the zero-shot base model: see `results/` in this
repository once evaluation has been pushed. Validation loss above is a proxy, not translation quality.

## Limitations

- Sentence-level model: split articles into sentences first. Arabic output is about 1.8x longer in tokens
  than English, so keep source sentences short when translating into Arabic.
- Swahili training data is a mix of news (MAFAND-MT, Global Voices) and general text; quality and domain vary.
- No human evaluation unless stated in `results/`.

## Licenses

Training data licenses differ by source and were not all verified. MAFAND-MT is CC BY-NC 4.0
(non-commercial). Check every source before reusing this adapter commercially.
"""


def push_final(cfg: Config, best_dir, params: dict, metrics: dict) -> None:
    """Strict: raises if the upload fails (local files are untouched)."""
    best_dir = Path(best_dir)
    with tempfile.TemporaryDirectory() as tmp:
        card, bp = Path(tmp) / "README.md", Path(tmp) / "best_params.json"
        card.write_text(build_model_card(cfg, params, metrics), encoding="utf-8")
        bp.write_text(json.dumps(params, indent=2))
        push_folder(cfg, best_dir, "best_model", "Final adapter", strict=True)
        push_file(cfg, bp, "best_params.json", "Best HPO parameters", strict=True)
        push_file(cfg, card, "README.md", "Model card", strict=True)
    print(f"[hub] Published https://huggingface.co/{cfg.hub.repo_id}")


def push_results(cfg: Config, results_dir) -> None:
    push_folder(cfg, results_dir, "results", "Evaluation results", strict=True)
    print(f"[hub] Pushed evaluation results to {cfg.hub.repo_id}/results")