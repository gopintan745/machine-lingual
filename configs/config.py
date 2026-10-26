"""Strict, typed config loader.

Rules:
  * YAML keys are exactly the attribute names below (no aliases). Unknown keys raise.
  * No global singleton: call load_config() and pass the Config around. Use cfg.copy()
    before changing values (HPO trials do this).
  * Relative paths resolve under WORK_DIR (set WORK_DIR=/kaggle/working on Kaggle).
  * ${ENV_VAR} placeholders in the YAML are expanded from the environment.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin, get_type_hints

import yaml

try:  # optional: local .env support (Kaggle uses real env vars / Secrets instead)
    import dotenv
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(os.getenv("WORK_DIR", PROJECT_ROOT / "outputs"))


def resolve(path: str) -> str:
    """Relative paths live under WORK_DIR; absolute paths are kept."""
    p = Path(path)
    return str(p if p.is_absolute() else WORK_DIR / p)


# ----------------------------------------------------------------------------- data
@dataclass
class Language:
    nllb: str
    fasttext: str
    name: str
    rtl: bool = False


@dataclass
class Source:
    name: str
    hub_id: str
    pairs: List[str]                                            # e.g. ["en-fr"]; order does not matter
    configs: Dict[str, str] = field(default_factory=dict)       # canonical pair -> Hub config name
    column_codes: Dict[str, str] = field(default_factory=dict)  # short code -> key in "translation"
    revision: Optional[str] = None


@dataclass
class Sizes:
    train_per_direction: int = 20000
    hpo_subset: int = 5000
    val_per_direction: int = 500
    pool_factor: int = 3        # clean at most pool_factor * (train + val) rows per pair


@dataclass
class LangID:
    enabled: bool = True
    model_path: str = "lid.176.bin"
    min_confidence: float = 0.5


@dataclass
class Cleaning:
    dedupe: bool = True
    length_ratio: List[float] = field(default_factory=lambda: [0.5, 2.0])
    min_tokens: int = 3
    max_tokens: int = 128
    langid: LangID = field(default_factory=LangID)


@dataclass
class TokenizerCfg:
    max_length: int = 128


@dataclass
class DataConfig:
    seed: int = 42
    languages: Dict[str, Language] = field(default_factory=dict)
    directions: List[str] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    sizes: Sizes = field(default_factory=Sizes)
    cleaning: Cleaning = field(default_factory=Cleaning)
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)

    def pairs(self) -> List[tuple]:
        """Unordered language pairs implied by `directions`, in first-seen order."""
        seen, out = set(), []
        for d in self.directions:
            a, b = d.split("-")
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def nllb(self, lang: str) -> str:
        return self.languages[lang].nllb


# ----------------------------------------------------------------------------- model / training
@dataclass
class ModelConfig:
    base: str = "facebook/nllb-200-distilled-600M"
    cache_dir: Optional[str] = None     # None -> default Hugging Face cache
    max_length: int = 128


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    epochs: int = 2
    per_device_batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    fp16: bool = True
    gradient_checkpointing: bool = False
    report_to: str = "none"             # "none" | "tensorboard" | "wandb"
    seed: int = 42


@dataclass
class GenerationConfig:
    num_beams: int = 4
    max_new_tokens: int = 128


@dataclass
class HPOConfig:
    n_trials: int = 10
    max_steps_per_trial: int = 400
    study_name: str = "machine_lingual_hpo"
    storage_path: str = "optuna/hpo.db"
    space: Dict[str, Any] = field(default_factory=dict)

    @property
    def storage(self) -> str:
        return f"sqlite:///{self.storage_path}"


@dataclass
class HubConfig:
    repo_id: str = "user/nllb-600M-lora"
    private: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    eval: Dict[str, Any] = field(default_factory=dict)   # typed once the eval harness exists

    def copy(self) -> "Config":
        return copy.deepcopy(self)


# ----------------------------------------------------------------------------- loader
def _convert(tp: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if is_dataclass(tp):
        return _build(tp, value, path)
    if origin is list and args:
        return [_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict and len(args) == 2 and is_dataclass(args[1]):
        return {k: _convert(args[1], v, f"{path}.{k}") for k, v in value.items()}
    return value


def _build(cls: Any, data: Optional[dict], path: str) -> Any:
    data = data or {}
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown key(s) under '{path}': {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {f.name: _convert(hints[f.name], data[f.name], f"{path}.{f.name}")
              for f in fields(cls) if f.name in data}
    return cls(**kwargs)


def load_config(yaml_path: Optional[str | Path] = None) -> Config:
    path = Path(yaml_path) if yaml_path else PROJECT_ROOT / "configs" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8"))) or {}
    cfg = _build(Config, raw, "config")

    # relative paths -> under WORK_DIR; the sqlite directory must exist
    cfg.training.output_dir = resolve(cfg.training.output_dir)
    cfg.hpo.storage_path = resolve(cfg.hpo.storage_path)
    cfg.data.cleaning.langid.model_path = resolve(cfg.data.cleaning.langid.model_path)
    Path(cfg.hpo.storage_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)

    # fail early on typos in direction / language names
    for d in cfg.data.directions:
        for code in d.split("-"):
            if code not in cfg.data.languages:
                raise ValueError(f"Direction '{d}' uses unknown language '{code}'")
    return cfg