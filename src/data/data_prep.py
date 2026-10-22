"""Data pipeline.

Flow, per unordered language pair (e.g. en-fr):
  load every source that lists the pair -> pool -> normalise to columns {a, b}
  -> clean (token length, length ratio, language ID) -> dedupe across sources
  -> split train/val (disjoint) -> tokenize in every direction listed in cfg.data.directions.
Everything is then concatenated into ONE training set and ONE validation set.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from configs.config import Config, WORK_DIR

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"  # verify if download fails


# ----------------------------------------------------------------------------- language ID
class LangID:
    """fastText language ID, loaded once."""

    def __init__(self, path: str):
        import fasttext
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading fastText LID model to {p} ...")
            urllib.request.urlretrieve(LID_URL, p)
        self.model = fasttext.load_model(str(p))

    def check(self, texts: list[str], label: str, min_conf: float) -> list[bool]:
        texts = [" ".join(t.split()) for t in texts]          # no newlines, collapsed whitespace
        want = f"__label__{label}"
        try:
            labels, probs = self.model.predict(texts, k=1)
            return [l[0] == want and float(p[0]) >= min_conf for l, p in zip(labels, probs)]
        except ValueError:
            # Some fastText builds break on NumPy 2 inside predict(); use the raw binding.
            out = []
            for t in texts:
                res = self.model.f.predict(t + "\n", 1, 0.0, "strict")
                prob, lab = res[0] if res else (0.0, "")
                out.append(lab == want and prob >= min_conf)
            return out


# ----------------------------------------------------------------------------- loading
def _same_pair(pair: str, a: str, b: str) -> bool:
    return tuple(sorted(pair.split("-"))) == tuple(sorted((a, b)))


def _load_pair(source, a: str, b: str, seed: int, pool: int) -> Optional[Dataset]:
    """Load one source's train split for pair (a, b) as a Dataset with columns {a, b}."""
    ka, kb = source.column_codes.get(a, a), source.column_codes.get(b, b)
    canon, rev = f"{a}-{b}", f"{b}-{a}"
    if canon in source.configs:
        names = [source.configs[canon]]
    elif rev in source.configs:
        names = [source.configs[rev]]
    else:                                   # try both orders; datasets differ on which they use
        names = [f"{ka}-{kb}", f"{kb}-{ka}"]
    ds, last_err = None, None
    for name in names:
        try:
            ds = load_dataset(source.hub_id, name, split="train")
            break
        except Exception as e:               # keep the reason; print it if every candidate fails
            last_err = e
    if ds is None:
        print(f"[WARN] {source.name} / {canon}: could not load (tried {names}). Last error: {last_err!r}")
        return None
    if "translation" not in ds.column_names:
        raise ValueError(f"{source.name}: expected a 'translation' column, found {ds.column_names}. "
                         f"Adjust _load_pair for this schema.")
    ds = ds.shuffle(seed=seed).select(range(min(pool, len(ds))))      # pool BEFORE cleaning (speed)
    return ds.map(
        lambda batch: {a: [p[ka] for p in batch["translation"]],
                       b: [p[kb] for p in batch["translation"]]},
        batched=True, remove_columns=ds.column_names)


# ----------------------------------------------------------------------------- cleaning
def clean_pairs(ds: Dataset, a: str, b: str, tokenizer, cfg: Config, lid: Optional[LangID]) -> Dataset:
    c = cfg.data.cleaning
    assert c.min_tokens >= 1, "min_tokens must be >= 1 (division by zero in the ratio filter)"
    r_lo, r_hi = c.length_ratio

    def keep_length(batch):
        la = [len(x) for x in tokenizer(batch[a], add_special_tokens=False).input_ids]
        lb = [len(x) for x in tokenizer(batch[b], add_special_tokens=False).input_ids]
        return [c.min_tokens <= x <= c.max_tokens and c.min_tokens <= y <= c.max_tokens
                and r_lo <= x / y <= r_hi for x, y in zip(la, lb)]

    ds = ds.filter(keep_length, batched=True, desc="length filter")

    if lid is not None and c.langid.enabled:
        fa, fb = cfg.data.languages[a].fasttext, cfg.data.languages[b].fasttext
        ds = ds.filter(
            lambda batch: [x and y for x, y in zip(lid.check(batch[a], fa, c.langid.min_confidence),
                                                   lid.check(batch[b], fb, c.langid.min_confidence))],
            batched=True, desc="language-ID filter")
    return ds


def dedupe(ds: Dataset, a: str, b: str, blocked: Optional[set] = None) -> Dataset:
    """Drop duplicate pairs (across sources) and, optionally, any pair containing a blocked
    sentence (e.g. FLORES devtest text) so evaluation stays clean. Single process on purpose."""
    seen: set = set()
    blocked = blocked or set()

    def first_time(ex):
        key = (ex[a].strip(), ex[b].strip())
        if key in seen or key[0] in blocked or key[1] in blocked:
            return False
        seen.add(key)
        return True

    return ds.filter(first_time, desc="dedupe")


def split_pair(ds: Dataset, cfg: Config):
    """Disjoint train/val slices of one shuffled pair dataset."""
    ds = ds.shuffle(seed=cfg.data.seed)
    n_val = min(cfg.data.sizes.val_per_direction, len(ds) // 10)
    n_train = min(cfg.data.sizes.train_per_direction, len(ds) - n_val)
    return ds.select(range(n_val, n_val + n_train)), ds.select(range(n_val))


# ----------------------------------------------------------------------------- tokenizing
def check_language_tokens(tokenizer, cfg: Config) -> None:
    """A wrong NLLB code silently maps to <unk>; fail loudly instead."""
    for lang, info in cfg.data.languages.items():
        if tokenizer.convert_tokens_to_ids(info.nllb) == tokenizer.unk_token_id:
            raise ValueError(f"Language code '{info.nllb}' ({lang}) is unknown to the tokenizer")


def tokenize_direction(ds: Dataset, tokenizer, src: str, tgt: str, cfg: Config) -> Dataset:
    tokenizer.src_lang = cfg.data.nllb(src)
    tokenizer.tgt_lang = cfg.data.nllb(tgt)

    def fn(batch):
        return tokenizer(batch[src], text_target=batch[tgt],
                         max_length=cfg.data.tokenizer.max_length, truncation=True)

    return ds.map(fn, batched=True, remove_columns=ds.column_names, desc=f"tokenize {src}->{tgt}")


def show_example(tokenizer, ds: Dataset, label: str) -> None:
    """Print one decoded example so language-code placement can be checked by eye."""
    ex = ds[0]
    print(f"[{label}] input : {tokenizer.convert_ids_to_tokens(ex['input_ids'])[:6]} ...")
    print(f"[{label}] labels: {tokenizer.convert_ids_to_tokens(ex['labels'])[:6]} ...")


# ----------------------------------------------------------------------------- orchestration
def build_datasets(cfg: Config, tokenizer, blocked: Optional[set] = None):
    check_language_tokens(tokenizer, cfg)
    sz = cfg.data.sizes
    pool = sz.pool_factor * (sz.train_per_direction + sz.val_per_direction)
    lid = LangID(cfg.data.cleaning.langid.model_path) if cfg.data.cleaning.langid.enabled else None

    train_parts, val_parts, report = [], [], {}
    for a, b in cfg.data.pairs():
        pieces = []
        for source in cfg.data.sources:
            if not any(_same_pair(p, a, b) for p in source.pairs):
                continue
            ds = _load_pair(source, a, b, cfg.data.seed, pool)
            if ds is None:
                continue
            ds = clean_pairs(ds, a, b, tokenizer, cfg, lid)
            print(f"{source.name:16s} {a}-{b}: {len(ds):>7d} pairs after cleaning")
            pieces.append(ds)
        if not pieces:
            raise RuntimeError(f"No data loaded for pair {a}-{b}. Check sources/pairs in the config.")

        merged = concatenate_datasets(pieces)
        if cfg.data.cleaning.dedupe or blocked:
            merged = dedupe(merged, a, b, blocked)
        tr, va = split_pair(merged, cfg)

        for src, tgt in ((a, b), (b, a)):
            if f"{src}-{tgt}" not in cfg.data.directions:
                continue
            t_tok, v_tok = (tokenize_direction(x, tokenizer, src, tgt, cfg) for x in (tr, va))
            show_example(tokenizer, t_tok, f"{src}->{tgt}")
            train_parts.append(t_tok)
            val_parts.append(v_tok)
            report[f"{src}-{tgt}"] = {"train": len(t_tok), "val": len(v_tok)}

    missing = set(cfg.data.directions) - set(report)
    if missing:
        raise RuntimeError(f"Directions with no data: {sorted(missing)}")
    print("Rows per direction:", json.dumps(report))
    train = concatenate_datasets(train_parts).shuffle(seed=cfg.data.seed)
    val = concatenate_datasets(val_parts).shuffle(seed=cfg.data.seed)
    return train, val


def prepare_data(cfg: Config, tokenizer, use_cache: bool = True, blocked: Optional[set] = None):
    """Build (or load from disk) the tokenized train/val sets. The cache key covers the data config,
    so changing sizes, sources or cleaning rebuilds automatically."""
    key = hashlib.md5(json.dumps(asdict(cfg.data), sort_keys=True).encode()).hexdigest()[:8]
    cache = Path(WORK_DIR) / "data_cache" / key
    if use_cache and (cache / "train").exists() and (cache / "val").exists():
        print(f"Loading cached data from {cache}")
        return load_from_disk(str(cache / "train")), load_from_disk(str(cache / "val"))
    train, val = build_datasets(cfg, tokenizer, blocked)
    cache.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(cache / "train"))
    val.save_to_disk(str(cache / "val"))
    return train, val