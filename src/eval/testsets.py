"""Load the held-out test set (FLORES-style): the same sentences in every language, row-aligned.

Three layouts are supported (set eval.test_set.kind):
  per_language : one Hub config per language (config name = NLLB code), sentence in `text_column`
  wide         : one dataset, one row per sentence, one column per language (`columns` maps code -> column)
  files        : a local directory with one line-aligned text file per language

Row alignment across languages is ASSUMED (same row index = same sentence); only lengths are checked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from configs.config import Config

TestSets = Dict[str, Tuple[List[str], List[str]]]       # direction -> (sources, references)


def _load_texts(cfg: Config, langs: Iterable[str]) -> Dict[str, List[str]]:
    ts = cfg.eval.test_set
    langs = sorted(set(langs))
    texts: Dict[str, List[str]] = {}

    if ts.kind == "files":
        if not ts.path:
            raise ValueError("eval.test_set.path is required for kind: files")
        for lang in langs:
            name = ts.file_pattern.format(nllb=cfg.data.nllb(lang), code=lang, split=ts.split)
            f = Path(ts.path) / name
            if not f.exists():
                raise FileNotFoundError(f"Test file not found: {f}")
            texts[lang] = [line.rstrip("\n") for line in f.open(encoding="utf-8")]

    elif ts.kind == "wide":
        from datasets import load_dataset
        ds = load_dataset(ts.hub_id, ts.config, split=ts.split)
        for lang in langs:
            col = ts.columns.get(lang)
            if col is None or col not in ds.column_names:
                raise ValueError(f"Column for '{lang}' missing. Set eval.test_set.columns; "
                                 f"dataset columns: {ds.column_names}")
            texts[lang] = list(ds[col])

    elif ts.kind == "per_language":
        from datasets import load_dataset
        for lang in langs:
            try:
                ds = load_dataset(ts.hub_id, cfg.data.nllb(lang), split=ts.split)
                texts[lang] = list(ds[ts.text_column])
            except Exception as e:
                raise RuntimeError(
                    f"Could not load {ts.hub_id} / {cfg.data.nllb(lang)} / {ts.split}: {e!r}. "
                    f"For gated datasets accept the terms on the Hub and set HF_TOKEN. "
                    f"Otherwise switch eval.test_set.kind to 'wide' or 'files'.") from e
    else:
        raise ValueError(f"Unknown eval.test_set.kind: {ts.kind!r}")

    lengths = {lang: len(t) for lang, t in texts.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Test set languages are not row-aligned (row counts differ): {lengths}")
    return texts


def load_test_sets(cfg: Config, directions: Optional[List[str]] = None,
                   max_rows: Optional[int] = None) -> TestSets:
    directions = directions or cfg.data.directions
    langs = {c for d in directions for c in d.split("-")}
    texts = _load_texts(cfg, langs)
    if max_rows:
        texts = {k: v[:max_rows] for k, v in texts.items()}
    out: TestSets = {}
    for d in directions:
        s, t = d.split("-")
        out[d] = (texts[s], texts[t])
    return out


def build_blocklist(cfg: Config) -> Set[str]:
    """Every test sentence in every configured language. Pass to prepare_data(blocked=...) so no
    training pair contains test text (exact match after strip)."""
    texts = _load_texts(cfg, cfg.data.languages.keys())
    return {t.strip() for lst in texts.values() for t in lst if t.strip()}