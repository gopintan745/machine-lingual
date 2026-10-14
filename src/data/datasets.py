from configs.config import Config
from datasets import load_dataset
from transformers import AutoTokenizer


def load_data_sources(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model, cache_dir=config.model.cached_dir)
    sources = {}
    for dataset in config.data.sources:
        name = dataset["name"]
        directions = dataset["directions"]
        for direction in directions:
            #src_lang, tgt_lang = direction.split("-")
            data = load_dataset(name, f"{direction}", cache_dir=config.model.cached_dir)
            sources[f"{name}_{direction}"] = data
    return sources, tokenizer


def clean_data(dataset, config: Config):
    # Implement data cleaning logic based on the configuration
    # For example, filter by length, deduplicate, etc.
    cleaned_dataset = dataset.filter(lambda x: len(x['translation'][list(x['translation'].keys())[0]].split()) >= config.data.min_token and len(x['translation'][list(x['translation'].keys())[0]].split()) <= config.data.max_token)
    if config.data.dedupe:
        cleaned_dataset = cleaned_dataset.unique()
    return cleaned_dataset


def sample_and_split_data(dataset, config: Config):
    # Sample and split the dataset into training and validation sets based on the configuration
    seed = config.data.seed
    train_dataset = dataset.shuffle(seed=seed).select(range(config.data.train_per_direction))
    val_dataset = dataset.shuffle(seed=seed).select(range(config.data.val_per_direction))
    return train_dataset, val_dataset


def tokenize_data(dataset, tokenizer, config: Config):
    # Tokenize the dataset using the provided tokenizer
    def tokenize(examples):
        return tokenizer(examples['translation'], padding="max_length", truncation=True, max_length=config.data.tokenizer_max_length)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset
