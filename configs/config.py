import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import dotenv
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

HF_USERNAME = os.getenv("HF_USERNAME", "")

try:
    from peft import LoraConfig, TaskType
    PEFT_AVAILABLE = True
except (ImportError, Exception):
    PEFT_AVAILABLE = False

    class TaskType:
        SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
        CAUSAL_LM = "CAUSAL_LM"

    class LoraConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class ConfigDict(dict):
    """Dictionary supporting both dictionary item access d['k'] and attribute access d.k."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"ConfigDict has no attribute or key '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"ConfigDict has no attribute or key '{name}'")


def wrap_config_dict(val: Any) -> Any:
    """Recursively wrap nested dictionaries with ConfigDict."""
    if isinstance(val, dict):
        return ConfigDict({k: wrap_config_dict(v) for k, v in val.items()})
    elif isinstance(val, list):
        return [wrap_config_dict(item) for item in val]
    return val


KNOWN_LANGUAGES = {
    "en": {"nllb": "eng_Latn", "fasttext": "en", "name": "English", "rtl": False},
    "fr": {"nllb": "fra_Latn", "fasttext": "fr", "name": "French",  "rtl": False},
    "ar": {"nllb": "arb_Arab", "fasttext": "ar", "name": "Arabic",  "rtl": True},
    "sw": {"nllb": "swh_Latn", "fasttext": "sw", "name": "Swahili", "rtl": False},
}

DEFAULT_SOURCE_METADATA = {
    "opus-100": {"source": "Helsinki-NLP/opus-100", "column_codes": {"en": "en", "fr": "fr", "ar": "ar"}},
    "flores-200": {"source": "facebook/flores", "column_codes": {"en": "en", "fr": "fr", "ar": "ar", "sw": "sw"}},
    "news-commentary": {"source": "Helsinki-NLP/news_commentary", "column_codes": {"en": "en", "fr": "fr", "ar": "ar"}},
    "mafand": {"source": "masakhane/mafand", "column_codes": {"en": "en", "fr": "fr", "sw": "swa"}},
    "mafand-mt": {"source": "masakhane/mafand", "column_codes": {"en": "en", "fr": "fr", "sw": "swa"}},
    "tatoeba": {"source": "Helsinki-NLP/tatoeba", "column_codes": {"en": "en", "fr": "fr", "ar": "ar", "sw": "swh"}},
}


class LanguageInfo(ConfigDict):
    """Wrapper that acts as a dict, object, and string for language configuration."""

    def __init__(self, nllb: str, fasttext: str = "", name: str = "", rtl: bool = False, **kwargs):
        fasttext = fasttext or (nllb.split("_")[0] if "_" in nllb else nllb[:2])
        name = name or nllb
        super().__init__(nllb=nllb, fasttext=fasttext, name=name, rtl=rtl, **kwargs)

    @property
    def nllb(self) -> str:
        return self["nllb"]

    @property
    def fasttext(self) -> str:
        return self["fasttext"]

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def rtl(self) -> bool:
        return self["rtl"]

    def __str__(self) -> str:
        return self["nllb"]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self["nllb"] == other
        return super().__eq__(other)


@dataclass
class DataSizesConfig:
    train_per_direction: int = 20000
    hpo_subset: int = 5000
    val_per_direction: int = 500


@dataclass
class DataCleaningConfig:
    dedupe: bool = True
    length_ratio: List[float] = field(default_factory=lambda: [0.5, 2.0])
    min_tokens: int = 3
    max_tokens: int = 128
    check_langid: bool = True

    @property
    def min_token(self) -> int:
        return self.min_tokens

    @min_token.setter
    def min_token(self, val: int):
        self.min_tokens = val

    @property
    def max_token(self) -> int:
        return self.max_tokens

    @max_token.setter
    def max_token(self, val: int):
        self.max_tokens = val

    @property
    def check_langID(self) -> bool:
        return self.check_langid

    @check_langID.setter
    def check_langID(self, val: bool):
        self.check_langid = val


@dataclass
class DataEvalConfig:
    dataset: str = "flores-200"
    split: str = "devtest"
    llm_judge_samples: int = 100


@dataclass
class DataTokenizerConfig:
    max_length: int = 128


@dataclass
class DataConfig:
    seed: int = 42
    languages: Dict[str, LanguageInfo] = field(default_factory=dict)
    directions: List[str] = field(default_factory=lambda: ["en-fr", "fr-en", "en-ar", "ar-en", "en-sw", "sw-en"])
    sources: List[Dict[str, Any]] = field(default_factory=list)
    sizes: DataSizesConfig = field(default_factory=DataSizesConfig)
    cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    eval: DataEvalConfig = field(default_factory=DataEvalConfig)
    tokenizer: DataTokenizerConfig = field(default_factory=DataTokenizerConfig)

    # Flattened properties for convenience and backward compatibility
    @property
    def train_per_direction(self) -> int:
        return self.sizes.train_per_direction

    @train_per_direction.setter
    def train_per_direction(self, val: int):
        self.sizes.train_per_direction = val

    @property
    def hpo_subset(self) -> int:
        return self.sizes.hpo_subset

    @hpo_subset.setter
    def hpo_subset(self, val: int):
        self.sizes.hpo_subset = val

    @property
    def val_per_direction(self) -> int:
        return self.sizes.val_per_direction

    @val_per_direction.setter
    def val_per_direction(self, val: int):
        self.sizes.val_per_direction = val

    @property
    def dedupe(self) -> bool:
        return self.cleaning.dedupe

    @dedupe.setter
    def dedupe(self, val: bool):
        self.cleaning.dedupe = val

    @property
    def length_ratio(self) -> List[float]:
        return self.cleaning.length_ratio

    @length_ratio.setter
    def length_ratio(self, val: List[float]):
        self.cleaning.length_ratio = val

    @property
    def min_tokens(self) -> int:
        return self.cleaning.min_tokens

    @min_tokens.setter
    def min_tokens(self, val: int):
        self.cleaning.min_tokens = val

    @property
    def min_token(self) -> int:
        return self.cleaning.min_tokens

    @min_token.setter
    def min_token(self, val: int):
        self.cleaning.min_tokens = val

    @property
    def max_tokens(self) -> int:
        return self.cleaning.max_tokens

    @max_tokens.setter
    def max_tokens(self, val: int):
        self.cleaning.max_tokens = val

    @property
    def max_token(self) -> int:
        return self.cleaning.max_tokens

    @max_token.setter
    def max_token(self, val: int):
        self.cleaning.max_tokens = val

    @property
    def check_langid(self) -> bool:
        return self.cleaning.check_langid

    @check_langid.setter
    def check_langid(self, val: bool):
        self.cleaning.check_langid = val

    @property
    def check_langID(self) -> bool:
        return self.cleaning.check_langid

    @check_langID.setter
    def check_langID(self, val: bool):
        self.cleaning.check_langid = val

    @property
    def eval_dataset(self) -> str:
        return self.eval.dataset

    @eval_dataset.setter
    def eval_dataset(self, val: str):
        self.eval.dataset = val

    @property
    def eval_split(self) -> str:
        return self.eval.split

    @eval_split.setter
    def eval_split(self, val: str):
        self.eval.split = val

    @property
    def llm_judge_samples(self) -> int:
        return self.eval.llm_judge_samples

    @llm_judge_samples.setter
    def llm_judge_samples(self, val: int):
        self.eval.llm_judge_samples = val

    @property
    def tokenizer_max_length(self) -> int:
        return self.tokenizer.max_length

    @tokenizer_max_length.setter
    def tokenizer_max_length(self, val: int):
        self.tokenizer.max_length = val


@dataclass 
class ModelConfig:
    base: str = "facebook/nllb-200-distilled-600M"
    tokenizer: str = "facebook/nllb-200-distilled-600M"
    dtype: str = "fp32"
    max_length: int = 128
    cached_dir: str = str(PROJECT_ROOT / "models")

    @property
    def model(self) -> str:
        return self.base

    @model.setter
    def model(self, val: str):
        self.base = val

    @property
    def cache_dir(self) -> str:
        return self.cached_dir

    @cache_dir.setter
    def cache_dir(self, val: str):
        self.cached_dir = val


@dataclass
class LoraTrainingConfig(LoraConfig):
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    task_type: Any = "SEQ_2_SEQ_LM"

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            try:
                super().__post_init__()
            except Exception:
                pass
        if not hasattr(self, "peft_type"):
            self.peft_type = "LORA"

    @property
    def alpha(self) -> int:
        return self.lora_alpha

    @alpha.setter
    def alpha(self, val: int):
        self.lora_alpha = val

    @property
    def dropout(self) -> float:
        return self.lora_dropout

    @dropout.setter
    def dropout(self, val: float):
        self.lora_dropout = val

    @property
    def lora_target_modules(self) -> List[str]:
        return self.target_modules

    @lora_target_modules.setter
    def lora_target_modules(self, val: List[str]):
        self.target_modules = val


@dataclass
class TrainingConfig:
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    num_train_epochs: int = 2
    per_device_batch_size: int = 16
    grad_acum_steps: int = 2
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    warmup_ratio: float = 0.05
    logging_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_strategy: str = "steps"
    fp16: bool = True
    gradient_checkpointing: bool = False
    best_metric: str = "chrf"
    save_total_limit: int = 2
    seed: int = 42

    @property
    def epochs(self) -> int:
        return self.num_train_epochs

    @epochs.setter
    def epochs(self, val: int):
        self.num_train_epochs = val

    @property
    def grad_accum_steps(self) -> int:
        return self.grad_acum_steps

    @grad_accum_steps.setter
    def grad_accum_steps(self, val: int):
        self.grad_acum_steps = val

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.grad_acum_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, val: int):
        self.grad_acum_steps = val

    @property
    def evaluation_strategy(self) -> str:
        return self.eval_strategy

    @evaluation_strategy.setter
    def evaluation_strategy(self, val: str):
        self.eval_strategy = val

    @property
    def per_device_train_batch_size(self) -> int:
        return self.per_device_batch_size

    @per_device_train_batch_size.setter
    def per_device_train_batch_size(self, val: int):
        self.per_device_batch_size = val


@dataclass
class GenerationConfig:
    num_beams: int = 4
    max_new_tokens: int = 128

    @property
    def num_beam(self) -> int:
        return self.num_beams

    @num_beam.setter
    def num_beam(self, val: int):
        self.num_beams = val

    @property
    def max_length(self) -> int:
        return self.max_new_tokens

    @max_length.setter
    def max_length(self, val: int):
        self.max_new_tokens = val


@dataclass
class HPOConfig:
    n_trials: int = 10
    max_steps_per_trial: int = 400
    data_subset: str = "hpo_subset"
    storage: str = f"sqlite:///{PROJECT_ROOT}/optuna/optuna.db"
    sampler: str = "tpe"
    pruner: str = "median"
    space: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)


@dataclass
class EvalConfig:
    seed: int = 42
    systems: List[Dict[str, Any]] = field(default_factory=list)
    test_set: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)
    generation: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)
    metrics: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)
    llm_judge: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)
    human_check: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)
    report: Union[Dict[str, Any], ConfigDict] = field(default_factory=ConfigDict)

    @property
    def baseline(self) -> str:
        for sys_item in self.systems:
            if sys_item.get("name") == "baseline":
                return sys_item.get("model", "")
        return "facebook/nllb-200-distilled-600M"

    @property
    def finetuned(self) -> str:
        for sys_item in self.systems:
            if sys_item.get("name") == "finetuned":
                return sys_item.get("model", "")
        return f"{HF_USERNAME}/nllb-200-distilled-lora-merged"

    @property
    def extra(self) -> str:
        for sys_item in self.systems:
            if sys_item.get("name") not in ("baseline", "finetuned"):
                return sys_item.get("model", "")
        return "facebook/nllb-200-1.3B"

    @property
    def test_dataset(self) -> str:
        return self.test_set.get("dataset", "flores-200")

    @property
    def split(self) -> str:
        return self.test_set.get("split", "devtest")

    @property
    def directions(self) -> List[str]:
        return self.test_set.get("directions", ["en-fr", "en-ar", "en-sw", "fr-en", "ar-en", "sw-en"])

    @property
    def chrf_plus_plus(self) -> Dict[str, Any]:
        return self.metrics.get("chrf_plus_plus", {"enabled": True, "word_order": 2})

    @property
    def comet(self) -> Dict[str, Any]:
        return self.metrics.get("comet", {"enabled": False})

    @property
    def report_per_direction_language(self) -> bool:
        return self.report.get("per_direction_table", True)

    @property
    def correlations(self) -> List[str]:
        return self.report.get("correlations", ["spearman_judge_vs_chrf"])

    @property
    def report_output_dir(self) -> str:
        return self.report.get("output_dir", f"{PROJECT_ROOT}/reports")


@dataclass
class HubConfig:
    repo_id: str = f"{HF_USERNAME}/nllb-200-distilled-lora-merged"
    private: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    lora: LoraTrainingConfig = field(default_factory=LoraTrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def val(self) -> EvalConfig:
        return self.eval

    @val.setter
    def val(self, val: EvalConfig):
        self.eval = val

    @property
    def languages(self) -> Dict[str, LanguageInfo]:
        return self.data.languages

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Union[str, Path]] = None) -> "Config":
        return load_config(yaml_path)

    def reload(self, yaml_path: Optional[Union[str, Path]] = None) -> "Config":
        new_config = load_config(yaml_path)
        for f_name in ["data", "training", "model", "hub", "lora", "generation", "hpo", "eval"]:
            setattr(self, f_name, getattr(new_config, f_name))
        return self


def populate_config(raw: Dict[str, Any]) -> Config:
    # 1. Data
    data_dict = raw.get("data", {})
    raw_langs = data_dict.get("languages", {})
    parsed_langs: Dict[str, LanguageInfo] = {}

    for code, info in (raw_langs.items() if raw_langs else KNOWN_LANGUAGES.items()):
        if isinstance(info, str):
            defaults = KNOWN_LANGUAGES.get(code, {})
            parsed_langs[code] = LanguageInfo(
                nllb=info,
                fasttext=defaults.get("fasttext", code),
                name=defaults.get("name", code),
                rtl=defaults.get("rtl", False)
            )
        elif isinstance(info, dict):
            defaults = KNOWN_LANGUAGES.get(code, {})
            merged = {**defaults, **info}
            parsed_langs[code] = LanguageInfo(**merged)
        else:
            parsed_langs[code] = info

    for code, defaults in KNOWN_LANGUAGES.items():
        if code not in parsed_langs:
            parsed_langs[code] = LanguageInfo(**defaults)

    raw_sources = data_dict.get("sources", [])
    merged_sources = []
    for src_item in raw_sources:
        s_name = src_item.get("name", "")
        defaults = DEFAULT_SOURCE_METADATA.get(s_name, {})
        merged = {**defaults, **src_item}
        merged_sources.append(wrap_config_dict(merged))

    sizes = DataSizesConfig(**data_dict.get("sizes", {}))
    cleaning = DataCleaningConfig(**data_dict.get("cleaning", {}))
    eval_sub = DataEvalConfig(**data_dict.get("eval", {}))
    tokenizer_sub = DataTokenizerConfig(**data_dict.get("tokenizer", {}))

    data_cfg = DataConfig(
        seed=data_dict.get("seed", 42),
        languages=parsed_langs,
        directions=data_dict.get("directions", ["en-fr", "fr-en", "en-ar", "ar-en", "en-sw", "sw-en"]),
        sources=merged_sources,
        sizes=sizes,
        cleaning=cleaning,
        eval=eval_sub,
        tokenizer=tokenizer_sub
    )

    # 2. Model
    model_dict = raw.get("model", {})
    base_model = model_dict.get("base", "facebook/nllb-200-distilled-600M")
    tok = model_dict.get("tokenizer", base_model)
    if tok == "same as base":
        tok = base_model
    model_cfg = ModelConfig(
        base=base_model,
        tokenizer=tok,
        dtype=model_dict.get("dtype", "fp32"),
        max_length=model_dict.get("max_length", 128),
        cached_dir=model_dict.get("cached_dir", model_dict.get("cache_dir", str(PROJECT_ROOT / "models")))
    )

    # 3. LoRA
    lora_dict = raw.get("lora", {})
    lora_cfg = LoraTrainingConfig(
        enabled=lora_dict.get("enabled", True),
        r=lora_dict.get("r", 16),
        lora_alpha=lora_dict.get("alpha", lora_dict.get("lora_alpha", 32)),
        lora_dropout=lora_dict.get("dropout", lora_dict.get("lora_dropout", 0.05)),
        target_modules=lora_dict.get("target_modules", lora_dict.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])),
        task_type=lora_dict.get("task_type", "SEQ_2_SEQ_LM")
    )

    # 4. Training
    train_dict = raw.get("training", {})
    training_cfg = TrainingConfig(
        output_dir=train_dict.get("output_dir", str(PROJECT_ROOT / "checkpoints")),
        num_train_epochs=train_dict.get("epochs", train_dict.get("num_train_epochs", 2)),
        per_device_batch_size=train_dict.get("per_device_batch_size", 16),
        grad_acum_steps=train_dict.get("grad_accum_steps", train_dict.get("grad_acum_steps", 2)),
        learning_rate=float(train_dict.get("learning_rate", 2e-4)),
        weight_decay=float(train_dict.get("weight_decay", 0.01)),
        warmup_steps=int(train_dict.get("warmup_steps", 500)),
        warmup_ratio=float(train_dict.get("warmup_ratio", 0.05)),
        logging_steps=int(train_dict.get("logging_steps", 100)),
        eval_strategy=train_dict.get("eval_strategy", "steps"),
        eval_steps=int(train_dict.get("eval_steps", 500)),
        save_steps=int(train_dict.get("save_steps", 500)),
        save_strategy=train_dict.get("save_strategy", "steps"),
        fp16=bool(train_dict.get("fp16", True)),
        gradient_checkpointing=bool(train_dict.get("gradient_checkpointing", False)),
        best_metric=train_dict.get("best_metric", "chrf"),
        save_total_limit=int(train_dict.get("save_total_limit", 2)),
        seed=int(train_dict.get("seed", 42))
    )

    # 5. Generation
    gen_dict = raw.get("generation", {})
    gen_cfg = GenerationConfig(
        num_beams=gen_dict.get("num_beams", gen_dict.get("num_beam", 4)),
        max_new_tokens=gen_dict.get("max_new_tokens", gen_dict.get("max_length", 128))
    )

    # 6. HPO
    hpo_dict = raw.get("hpo", {})
    hpo_cfg = HPOConfig(
        n_trials=hpo_dict.get("n_trials", 10),
        max_steps_per_trial=hpo_dict.get("max_steps_per_trial", 400),
        data_subset=hpo_dict.get("data_subset", "hpo_subset"),
        storage=hpo_dict.get("storage", f"sqlite:///{PROJECT_ROOT}/optuna/optuna.db"),
        sampler=hpo_dict.get("sampler", "tpe"),
        pruner=hpo_dict.get("pruner", "median"),
        space=wrap_config_dict(hpo_dict.get("space", {}))
    )

    # 7. Eval
    eval_dict = raw.get("eval", {})
    systems = eval_dict.get("systems", [])
    systems_wrapped = []
    for sys_item in systems:
        sys_copy = dict(sys_item)
        if HF_USERNAME and sys_copy.get("model") and "<your-hub-username>" in sys_copy["model"]:
            sys_copy["model"] = sys_copy["model"].replace("<your-hub-username>", HF_USERNAME)
        systems_wrapped.append(wrap_config_dict(sys_copy))

    eval_cfg = EvalConfig(
        seed=eval_dict.get("seed", 42),
        systems=systems_wrapped,
        test_set=wrap_config_dict(eval_dict.get("test_set", {})),
        generation=wrap_config_dict(eval_dict.get("generation", {})),
        metrics=wrap_config_dict(eval_dict.get("metrics", {})),
        llm_judge=wrap_config_dict(eval_dict.get("llm_judge", {})),
        human_check=wrap_config_dict(eval_dict.get("human_check", {})),
        report=wrap_config_dict(eval_dict.get("report", {}))
    )

    # 8. Hub
    hub_dict = raw.get("hub", {})
    hub_cfg = HubConfig(
        repo_id=hub_dict.get("repo_id", f"{HF_USERNAME or 'user'}/nllb-200-distilled-lora-merged"),
        private=hub_dict.get("private", False)
    )

    return Config(
        data=data_cfg,
        training=training_cfg,
        model=model_cfg,
        hub=hub_cfg,
        lora=lora_cfg,
        generation=gen_cfg,
        hpo=hpo_cfg,
        eval=eval_cfg
    )


def load_config(yaml_path: Optional[Union[str, Path]] = None) -> Config:
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent / "config.yaml"
        if not yaml_path.exists():
            yaml_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        return Config()

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    return populate_config(raw_data)


# Default singleton instance loaded from configs/config.yaml
config: Config = load_config()

__all__ = [
    "Config",
    "DataConfig",
    "DataSizesConfig",
    "DataCleaningConfig",
    "DataEvalConfig",
    "DataTokenizerConfig",
    "TrainingConfig",
    "ModelConfig",
    "HubConfig",
    "LoraTrainingConfig",
    "GenerationConfig",
    "HPOConfig",
    "EvalConfig",
    "LanguageInfo",
    "ConfigDict",
    "load_config",
    "config",
    "PROJECT_ROOT",
    "HF_USERNAME"
]