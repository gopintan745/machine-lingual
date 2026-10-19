from configs.config import config
from src.data import data_utils

def load_and_prepare_data():

    prepared_data = {}

    loaded_data, tokenizer, src, tgt = data_utils.load_data_sources(config)

    for dataset_key, dataset in loaded_data.items():
        cleaned_dataset = data_utils.clean_data(dataset, src, tgt, tokenizer, config)
        train_dataset, val_dataset = data_utils.sample_and_split_data(cleaned_dataset, config)
        tokenized_train_dataset = data_utils.tokenize_data(train_dataset, tokenizer, src, tgt, data_utils.nllb_code(config, src), data_utils.nllb_code(config, tgt), config)
        tokenized_val_dataset = data_utils.tokenize_data(val_dataset, tokenizer, src, tgt, data_utils.nllb_code(config, src), data_utils.nllb_code(config, tgt), config)

        prepared_data[dataset_key] = {
            "train": tokenized_train_dataset,
            "val": tokenized_val_dataset,
            "tokenizer": tokenizer,
            "src": src,
            "tgt": tgt
        }

        return prepared_data


if __name__ == "__main__":
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    print("Data loaded and prepared successfully.")
        