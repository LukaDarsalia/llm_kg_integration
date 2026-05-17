"""Standard short-answer QA metrics (HippoRAG / SQuAD style).

These follow the canonical SQuAD-EM/F1 normalization (lowercase, strip articles
and punctuation, collapse whitespace) and pick the max across multiple
acceptable gold answers.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, collapse whitespace.

    Punctuation is deleted (so "cat's" → "cats"); articles are replaced with a
    space (so "the cat" → " cat" → "cat" after the final whitespace collapse).
    """
    s = s.lower()
    s = _PUNCT.sub("", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def exact_match(prediction: str, golds: list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold, else 0.0."""
    pred = normalize_text(prediction)
    return 1.0 if any(pred == normalize_text(g) for g in golds) else 0.0


def _f1_one(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        # MuSiQue / SQuAD convention: F1=1 when both are empty (agree on no-answer),
        # F1=0 if exactly one is empty.
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, golds: list[str]) -> float:
    """Maximum token-overlap F1 across `golds`. MuSiQue's official `answer_f1`."""
    if not golds:
        return 0.0
    return max(_f1_one(prediction, g) for g in golds)


# Alias for callers that want to be explicit about which paper's metric they use.
# MuSiQue's `metrics/answer.py:compute_f1` is functionally identical to `f1_score`.
musique_f1 = f1_score


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of `relevant` items that appear in the first `k` of `retrieved`.

    By convention, returns 1.0 if `relevant` is empty (vacuously true).
    """
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)
