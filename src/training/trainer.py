from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from configs.config import Config
from eval.metrics import compute_chrf_corpus_score

def build_trainer(model, train_dataset, eval_dataset, tokenizer, config: Config):
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        label_names=["labels"], 
        generation_max_length=config.generation.max_length,
        generation_num_beams=config.generation.num_beams,
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.grad_acum_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        evaluation_strategy=config.training.eval_strategy,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        save_total_limit=config.training.save_total_limit
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer