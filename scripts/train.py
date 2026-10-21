from data.data_prep import load_and_prepare_data
from training.hpo import run_hyperparameter_search
from training.trainer import build_trainer
from configs.config import config
from models.model import load_model


if __name__ == "__main__":

    # Load and prepare the data
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    print("Data loaded and prepared successfully.")

    # Load the model
    print("Loading the model...")
    model = load_model(config)
    print("Model loaded successfully.")

    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")

    for dataset_key, item in data.items():
        tokenized_train_dataset = item["train"]
        tokenized_val_dataset = item["val"]
        tokenizer = item["tokenizer"]

        print(f"Running HPO for dataset: {dataset_key}")
        run_hyperparameter_search(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer)
        print(f"HPO completed for dataset: {dataset_key}")