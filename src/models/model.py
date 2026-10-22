from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

from configs.config import Config


def load_model(cfg: Config):
    """Load a FRESH base model and attach a new LoRA adapter built from cfg.lora.

    Call this once per HPO trial and once for the final run. Reusing one model object
    carries trained weights between runs and ignores later changes to r / alpha.
    """
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.base, cache_dir=cfg.model.cache_dir)
    if cfg.training.gradient_checkpointing:
        base.config.use_cache = False
        base.enable_input_require_grads()          # needed for checkpointing with frozen base weights

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()             # sanity check: only adapter weights should train
    return model