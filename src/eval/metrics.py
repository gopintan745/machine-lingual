from typing import Sequence
from sacrebleu.metrics import CHRF


def make_chrf(word_order: int = 2) -> CHRF:
    # word_order=2 gives chrF++; other settings stay at sacrebleu defaults
    return CHRF(word_order=word_order)


def compute_chrf_corpus_score(hyps: Sequence[str], refs: Sequence[str], chrf: CHRF | None = None):
    """Corpus-level chrF++ for ONE direction. refs is a flat list, one reference per hypothesis."""
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses but {len(refs)} references")
    chrf = chrf or make_chrf()
    result = chrf.corpus_score(list(hyps), [list(refs)])   # wrap: list of reference sets
    return {"score": result.score, "signature": str(chrf.get_signature())}


def compute_chrf_sentence_scores(hyps: Sequence[str], refs: Sequence[str], chrf: CHRF | None = None):
    """Per-sentence scores, for the judge-vs-chrF++ correlation."""
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses but {len(refs)} references")
    chrf = chrf or make_chrf()
    return [chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]