from configs.config import Config
from datasets import load_dataset
from transformers import AutoTokenizer

import fasttext
lid = fasttext.load_model("lid.176.bin")

def lang_ok(texts, expected, min_conf):
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
        name = dataset["name"]
        directions = dataset["directions"]
        for direction in directions:
            src_lang, tgt_lang = direction.split("-")
            data = load_dataset(name, f"{src_lang}-{tgt_lang}", cache_dir=config.model.cached_dir)
            datasets[f"{name}_{direction}"] = data
    return datasets, tokenizer, src_lang, tgt_lang


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