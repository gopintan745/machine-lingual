from configs.config import config
from training.trainer import build_trainer
from optuna import Trial, create_study, TrialPruned
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Any


def objective(trial: Trial, model: Any, tokenized_train_dataset: Any, tokenized_val_dataset: Any, tokenizer: Any) -> Any:
    try:
        r = trial.suggest_categorical('r', [8, 16, 32])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.1)

        # Update the config with the suggested hyperparameters
        config.training.learning_rate = learning_rate
        config.training.warmup_steps = int(warmup_ratio * config.training.num_train_epochs * len(tokenized_train_dataset) / config.training.per_device_batch_size)
        config.lora.r = r
        config.lora.lora_alpha = lora_alpha

        trainer = build_trainer(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer, config)
        results = trainer.train()

        return results["eval_loss"]  # Returns the evaluation loss to be minimized
    
    except Exception as e:
        trial.report(e, step=0)  # Report a failed trial
        if trial.should_prune():
            raise TrialPruned()  # Prune the trial if it should be pruned
        return float('inf')  # Return a large value for failed trials


def run_hyperparameter_search(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer):

    if config.hpo.sampler == "tpe" and config.hpo.pruner == "median":
        sampler = TPESampler()
        pruner = MedianPruner()

    study = create_study(
        direction='minimize', # Minimize the evaluation loss
        storage=config.hpo.storage,
        sampler=sampler,
        pruner=pruner,
        )  # Create a new study for hyperparameter optimization
    study.optimize(lambda trial: objective(trial, model, tokenized_train_dataset, tokenized_val_dataset, tokenizer), n_trials=config.hpo.n_trials)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")