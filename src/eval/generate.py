"""Load a system (base model, optionally + LoRA adapter) and translate with the correct target token.

NLLB needs the TARGET language code forced as the first generated token. A wrong or missing
forced_bos_token_id silently produces text in the wrong language.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Callable, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from configs.config import Config, EvalSystem
from src.eval.gen_utils import capped_rows

Translator = Callable[[List[str], str, str], Tuple[List[str], dict]]


def _make_translate(model, tok, device: str, cfg: Config) -> Translator:
    g = cfg.eval.generation
    max_len = cfg.model.max_length

    def translate(sources: List[str], src: str, tgt: str):
        tok.src_lang = cfg.data.nllb(src)
        forced = tok.convert_tokens_to_ids(cfg.data.nllb(tgt))
        if forced == tok.unk_token_id:
            raise ValueError(f"Target language code '{cfg.data.nllb(tgt)}' is unknown to the tokenizer")

        def run(idx: List[int]):
            enc = tok([sources[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            truncated = (enc["attention_mask"].sum(dim=1) >= max_len).tolist()
            try:
                with torch.no_grad():
                    gen = model.generate(**enc, forced_bos_token_id=forced,
                                         num_beams=g.num_beams, max_new_tokens=g.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                half = len(idx) // 2                      # retry with smaller batches
                return run(idx[:half]) + run(idx[half:])
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            capped = capped_rows(gen.tolist(), tok.eos_token_id, g.max_new_tokens)
            return list(zip(idx, texts, truncated, capped))

        out = [""] * len(sources)
        n_truncated = 0
        n_capped = 0
        nonempty = [i for i, s in enumerate(sources) if s.strip()]
        order = sorted(nonempty, key=lambda i: len(sources[i]), reverse=True)   # similar lengths per batch
        for start in range(0, len(order), g.batch_size):
            for i, text, was_truncated, was_capped in run(order[start:start + g.batch_size]):
                out[i] = text
                n_truncated += int(was_truncated)
                n_capped += int(was_capped)
        return out, {"truncated_inputs": n_truncated, "capped_outputs": n_capped}

    return translate


@contextmanager
def loaded_translator(system: EvalSystem, cfg: Config):
    """with loaded_translator(system, cfg) as translate: hyps, stats = translate(sources, 'en', 'fr')"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(system.base, cache_dir=cfg.model.cache_dir)   # LoRA leaves it unchanged
    model = AutoModelForSeq2SeqLM.from_pretrained(system.base, cache_dir=cfg.model.cache_dir)
    if system.adapter:
        model = PeftModel.from_pretrained(model, system.adapter).merge_and_unload()
    if device == "cuda":
        model = model.half()
    model = model.to(device).eval()
    try:
        yield _make_translate(model, tok, device, cfg)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()