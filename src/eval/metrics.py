"""chrF++ helpers (sacrebleu). chrF++ is CHRF with word_order=2.

Scores are on a 0-100 scale. refs is a FLAT list: one reference string per hypothesis.
Never pool directions into one corpus score; scores are not comparable across languages.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sacrebleu.metrics import CHRF


def make_chrf(word_order: int = 2) -> CHRF:
    return CHRF(word_order=word_order)


def _check(hyps: Sequence[str], refs: Sequence[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses but {len(refs)} references")


def compute_chrf_corpus_score(hyps: Sequence[str], refs: Sequence[str], chrf: Optional[CHRF] = None) -> dict:
    """Corpus-level chrF++ for ONE direction."""
    _check(hyps, refs)
    chrf = chrf or make_chrf()
    result = chrf.corpus_score(list(hyps), [list(refs)])      # wrap: a list of reference sets
    return {"score": result.score, "signature": str(chrf.get_signature())}


def compute_chrf_sentence_scores(hyps: Sequence[str], refs: Sequence[str], chrf: Optional[CHRF] = None) -> list:
    """One chrF++ per sentence (for bootstrap tests and judge-vs-chrF++ correlation)."""
    _check(hyps, refs)
    chrf = chrf or make_chrf()
    return [chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def compute_chrf_sentence_score(hyp: str, ref: str, chrf: Optional[CHRF] = None) -> dict:
    chrf = chrf or make_chrf()
    return {"score": chrf.sentence_score(hyp, [ref]).score, "signature": str(chrf.get_signature())}