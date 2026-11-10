"""Torch-free helpers for the generation code (kept separate so they can be unit-tested anywhere)."""
from __future__ import annotations

from typing import List, Sequence


def capped_rows(gen_rows: Sequence[Sequence[int]], eos_id: int, max_new_tokens: int) -> List[bool]:
    """Which generated sequences were cut off by max_new_tokens?

    NLLB's decoder output starts with the decoder-start token (= </s>) and may end with a forced </s>
    at the length limit, so only an </s> strictly inside the sequence (positions 1..-2) proves the model
    stopped by itself. A row is 'capped' if the batch reached the limit and no such early </s> exists.
    Approximate: a sentence that ends naturally on the very last allowed token is also flagged.
    """
    out = []
    for row in gen_rows:
        row = list(row)
        reached_limit = len(row) >= max_new_tokens + 1          # +1 for the decoder-start token
        stopped_early = eos_id in row[1:-1]
        out.append(reached_limit and not stopped_early)
    return out