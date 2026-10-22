from __future__ import annotations

import inspect
import math
from dataclasses import fields
from typing import Optional

import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from configs.config import Config


def _total_steps(cfg: Config, n_train: int, max_steps: int) -> int:
    if max_steps and max_steps > 0:
        return max_steps
    n_gpu = max(1, torch.cuda.device_count())
    per_step = cfg.training.per_device_batch_size * cfg.training.grad_accum_steps * n_gpu
    return math.ceil(n_train / per_step) * cfg.training.epochs


def build_trainer(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    cfg: Config,
    output_dir: Optional[str] = None,
    max_steps: int = -1,
    for_hpo: bool = False,
    eval_steps: Optional[int] = None,
) -> Seq2SeqTrainer:
    """Build a Seq2SeqTrainer. Model selection is always by eval_loss (lower is better).

    for_hpo=True: evaluate on a schedule, do not save checkpoints, do not reload a 'best' model.
    """
    t = cfg.training
    out = output_dir or t.output_dir
    total = _total_steps(cfg, len(train_dataset), max_steps)
    steps = eval_steps or t.eval_steps

    args = dict(
        output_dir=out,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=int(t.warmup_ratio * total),        # computed: works across transformers versions
        logging_steps=t.logging_steps,
        eval_steps=steps,
        save_steps=steps if for_hpo else t.save_steps,
        save_strategy="no" if for_hpo else "steps",
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=not for_hpo,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=t.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=t.gradient_checkpointing,
        report_to=t.report_to,
        seed=t.seed,
        data_seed=cfg.data.seed,
        label_names=["labels"],                           # PeftModel hides the forward signature
        remove_unused_columns=False,                      # datasets already hold only model inputs
    )
    if max_steps and max_steps > 0:
        args["max_steps"] = max_steps
    else:
        args["num_train_epochs"] = t.epochs

    names = {f.name for f in fields(Seq2SeqTrainingArguments)}
    args["eval_strategy" if "eval_strategy" in names else "evaluation_strategy"] = "steps"

    trainer_kwargs = dict(
        model=model,
        args=Seq2SeqTrainingArguments(**args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    # newer transformers: processing_class; older: tokenizer
    key = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters \
        else "tokenizer"
    trainer_kwargs[key] = tokenizer
    return Seq2SeqTrainer(**trainer_kwargs)