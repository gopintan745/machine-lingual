from configs.config import Config, config
from datasets import load_dataset
from transformers import AutoTokenizer
import fasttext

def lang_ok(texts, expected, min_conf):
    lid = fasttext.load_model("lid.176.bin")
    texts = [t.replace("\n", " ") for t in texts]      # predict rejects newlines
    labels, probs = lid.predict(texts, k=1)
    return [l[0] == f"__label__{expected}" and p[0] >= min_conf
            for l, p in zip(labels, probs)]

def source_key(cfg: Config, source, lang):
    # Used to retrieve the language key eg. "en" from the source datasets
    return source.get("column_codes", {}).get(lang, lang)

def nllb_code(cfg: Config, lang):
    return cfg.languages[lang]["nllb"]

def load_data_sources(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model, cache_dir=config.model.cached_dir)
    datasets = {}

    for dataset in config.data.sources:
        # Use Hub repo path (e.g. "Helsinki-NLP/opus-100") if provided, else fallback to name
        hub_path = dataset.get("source", dataset["name"])
        name = dataset["name"]
        directions = dataset["directions"]

        for direction in directions:
            src_lang, tgt_lang = direction.split("-")
            data = None

            # Try src-tgt then tgt-src (e.g. ar-en instead of en-ar)
            for pair in (f"{src_lang}-{tgt_lang}", f"{tgt_lang}-{src_lang}"):
                try:
                    data = load_dataset(hub_path, pair, cache_dir=config.model.cached_dir, split="train", trust_remote_code=True)
                    break
                except Exception:
                    continue

            if data is not None:
                datasets[f"{name}_{direction}"] = {
                    "dataset": data,
                    "src": src_lang,
                    "tgt": tgt_lang,
                }
            else:
                print(f"Warning: Could not load {hub_path} for direction {direction}")

    return datasets, tokenizer

def clean_data(ds, src, tgt, tokenizer, cfg: Config):
    low, high = cfg.data.min_tokens, cfg.data.max_tokens      # lo must be >= 1
    r_low, r_high = cfg.data.length_ratio

    def keep(batch):
        s = [p[src] for p in batch["translation"]]
        t = [p[tgt] for p in batch["translation"]]
        ls = [len(x) for x in tokenizer(s, add_special_tokens=False).input_ids]
        lt = [len(x) for x in tokenizer(t, add_special_tokens=False).input_ids]
        return [low <= a <= high and low <= b <= high and r_low <= a / b <= r_high
                for a, b in zip(ls, lt)]

    ds = ds.filter(keep, batched=True)

    if cfg.data.dedupe:
        seen = set()
        def first_time(ex):
            key = (ex["translation"][src], ex["translation"][tgt])
            if key in seen:
                return False
            seen.add(key)
            return True
        ds = ds.filter(first_time)          # keep single process
    return ds


def sample_and_split_data(ds, cfg: Config):
    ds = ds.shuffle(seed=cfg.data.seed)
    n_val = min(cfg.data.val_per_direction, len(ds) // 10)
    n_train = min(cfg.data.train_per_direction, len(ds) - n_val)
    val = ds.select(range(n_val))
    train = ds.select(range(n_val, n_val + n_train))   # disjoint from val
    return train, val


def tokenize_data(ds, tokenizer, src, tgt, src_code, tgt_code, cfg: Config):
    tokenizer.src_lang, tokenizer.tgt_lang = src_code, tgt_code

    def tokenize(batch):
        sources = [p[src] for p in batch["translation"]]
        targets = [p[tgt] for p in batch["translation"]]
        return tokenizer(sources, text_target=targets,
                         max_length=cfg.data.tokenizer_max_length,
                         truncation=True)

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def load_and_prepare_data():

    prepared_data = {}

    loaded_data, tokenizer = load_data_sources(config)

    for dataset_key, item in loaded_data.items():
        dataset = item["dataset"]
        src = item["src"]
        tgt = item["tgt"]
        cleaned_dataset = clean_data(dataset, src, tgt, tokenizer, config)
        train_dataset, val_dataset = sample_and_split_data(cleaned_dataset, config)
        tokenized_train_dataset = tokenize_data(train_dataset, tokenizer, src, tgt, nllb_code(config, src), nllb_code(config, tgt), config)
        tokenized_val_dataset = tokenize_data(val_dataset, tokenizer, src, tgt, nllb_code(config, src), nllb_code(config, tgt), config)

        prepared_data[dataset_key] = {
            "train": tokenized_train_dataset,
            "val": tokenized_val_dataset,
            "tokenizer": tokenizer,
            "src": src,
            "tgt": tgt
        }

    return prepared_data

