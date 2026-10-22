"""
This is file is used for full training of the model with the optimal hyparapeater values obtained from hyperparameter optimization (HPO). 
It loads the prepared datasets, initializes the model, and trains it using the best hyperparameters found during HPO. 
The training process is managed by the trainer module, which handles the training loop, evaluation, and saving of the model checkpoints.
The script is designed to be run after the hyperparameter optimization has been completed, and it assumes that the best hyperparameters have been identified and an Optuna Database has been created to store the results of the HPO trials.
This code would load the optuna.db file before full training.
"""
import os
from optuna import load_study
from training.trainer import build_trainer
from configs.config import Config

def full_training(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer, config: Config):
    # Load the best hyperparameters from the Optuna study
    study = load_study(study_name="machine_lingual_hpo_study", storage=config.hpo.storage)
    best_trial = study.best_trial

    # Update the config with the best hyperparameters
    config.training.learning_rate = best_trial.params['learning_rate']
    config.training.warmup_steps = int(
        best_trial.params['warmup_ratio'] 
        * config.training.num_train_epochs 
        * len(tokenized_train_dataset) 
        / config.training.per_device_batch_size
    )
    config.lora.r = best_trial.params['r']
    config.lora.lora_alpha = best_trial.params['lora_alpha']

    # Build trainer with the updated config
    trainer = build_trainer(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer, config)

    # Run training
    print("Starting full training...")
    train_results = trainer.train()

    # Save the best model weights (or LoRA adapter) and tokenizer
    best_model_dir = os.path.join(config.training.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    print(f"Saving best model checkpoint to: {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    return train_results