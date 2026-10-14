import os
from typing import Dict, List
import dotenv
from pathlib import Path
from dataclasses import dataclass
from peft import LoraConfig, TaskType

PROJECT_ROOT = Path(__file__).parent.parent
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

HF_USERNAME = os.getenv("HF_USERNAME")

@dataclass
class DataConfig:
    seed: int = 42
    languages: Dict[str, str] = {
        "en": "eng_Latn",
        "fr": "fra_Latn",
        "ar": "arb_Arab",
        "sw": "swh_Latn",
    }
    directions: List[str] = ["en-fr", "en-ar", "en-sw"]
    sources: List[Dict[str, str]] = [
        {"name": "opus-100", "directions": ["en-fr", "en-ar", "en-sw"]},
        {"name": "news-commentary", "directions": ["en-fr", "en-ar"]},
        {"name": "mafand-mt", "directions": ["en-sw", "fr-sw"]},
        {"name": "tatoeba", "directions": ["en-fr", "en-ar", "en-sw"]},
    ]
    train_per_direction: int = 20000
    hpo_subset: int = 5000
    val_per_direction: int = 500
    dedupe: bool = True
    length_ratio: List[float] = [0.5, 2.0]
    min_token: int = 3
    max_token: int = 128
    check_langID: bool = True
    eval_dataset: str = "flores-200"
    eval_split: str = "devtest"
    llm_judge_samples: int = 100
    tokenizer_max_length: int = 128


@dataclass
class TrainingConfig:
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    num_train_epochs: int = 2
    per_device_batch_size: int = 16
    grad_acum_steps: int = 2
    learning_rate: float = 2e-5 #HPO : [1e-3, 5e-5]
    weight_decay: float = 0.01
    warmup_steps: int = 500
    warmup_ratio: float = 0.05 
    logging_steps: int = 100
    #save_strategy: str = "epoch"
    eval_strategy: str = "steps"
    fp16: bool = True  # Use mixed precision on GPU
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to save memory
    best_metric: str = "chrf"
    save_total_limit: int = 2


@dataclass
class GenerationConfig:
    num_beam: int = 4
    max_new_tokens: 128


@dataclass 
class ModelConfig:
    base: str = "facebook/nllb-200-distilled-600M"
    cached_dir: str = str(PROJECT_ROOT / "models")
    dtype: str = "fp32"
    max_length: int = 128


@dataclass
class HubConfig:
    repo_id: str = f"{HF_USERNAME}/machine-lingual-nllb-200_distilled-peft"
    private: bool = False


@dataclass
class LoraTrainingConfig(LoraConfig):
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list =  ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    task_type: TaskType = TaskType.SEQ_2_SEQ_LM

@dataclass
class Config:
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    hub: HubConfig = HubConfig()
    lora: LoraTrainingConfig = LoraTrainingConfig()
    generation: GenerationConfig = GenerationConfig()


config: Config = Config()