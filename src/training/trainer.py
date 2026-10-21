from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from configs.config import Config

def build_trainer(
    model, 
    train_dataset, 
    eval_dataset, 
    tokenizer, 
    config: Config,
    metric_for_best_model: str | None = None
):
    # Use explicitly passed metric, or fall back to config, or default to "eval_loss"
    chosen_metric = metric_for_best_model or getattr(config.training, "best_metric", "eval_loss")
    greater_is_better = False if "loss" in chosen_metric else True

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.grad_acum_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        
        # Explicit metric settings
        load_best_model_at_end=True,
        metric_for_best_model=chosen_metric,      # e.g., "eval_loss"
        greater_is_better=greater_is_better,      # False for loss
        
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )