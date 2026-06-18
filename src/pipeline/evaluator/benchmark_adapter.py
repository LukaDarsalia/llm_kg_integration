"""Adapter onto the vendored GraphRAG-Bench evaluation metrics (git submodule).

We call the benchmark's async metric functions directly (not its CLI) so scoring is
identical to the public leaderboard: gpt-4o-mini judge (temp 0, seed 42) + local
BAAI/bge-large-en-v1.5 embeddings, with the same per-question_type metric selection.

Robustness note: the vendored metrics can raise on a single sample (e.g. the judge LLM
returns JSON in a shape the metric's pydantic model rejects — `calculate_factuality` only
catches JSONDecodeError/TypeError, not ValidationError). We wrap every metric so one bad
sample degrades to NaN (excluded from the mean, counted in `failures`) instead of
aborting the whole run — matching the benchmark's "warn and continue" philosophy without
forking its code.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# third_party/GraphRAG-Benchmark (repo root / third_party / ...).
_BENCH_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "GraphRAG-Benchmark"
SEED = 42

# Per-question_type metric selection — copied verbatim from
# Evaluation/generation_eval.py so we stay leaderboard-faithful.
GENERATION_METRIC_CONFIG: Dict[str, List[str]] = {
    "Fact Retrieval": ["rouge_score", "answer_correctness"],
    "Complex Reasoning": ["rouge_score", "answer_correctness"],
    "Contextual Summarize": ["answer_correctness", "coverage_score"],
    "Creative Generation": ["answer_correctness", "coverage_score", "faithfulness"],
}


def _ensure_bench_on_path() -> None:
    if not _BENCH_ROOT.exists():
        raise FileNotFoundError(
            f"GraphRAG-Benchmark submodule not found at {_BENCH_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    p = str(_BENCH_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


# --- judge construction ----------------------------------------------------------------
def build_judge_llm(llm_cfg: Dict[str, Any]):
    """LangChain ChatOpenAI judge mirroring the benchmark's settings."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from src.pipeline.shared.providers import resolve_api_key

    p = llm_cfg.get("params") or {}
    return ChatOpenAI(
        model=llm_cfg["model"],
        base_url=llm_cfg.get("base_url"),
        api_key=SecretStr(resolve_api_key(llm_cfg)),
        temperature=p.get("temperature", 0.0),
        max_retries=3,
        timeout=30,
        model_kwargs={
            "top_p": p.get("top_p", 1),
            "seed": p.get("seed", SEED),
            "presence_penalty": p.get("presence_penalty", 0),
            "frequency_penalty": p.get("frequency_penalty", 0),
        },
        extra_body=llm_cfg.get("extra_body"),
    )


def build_judge_embeddings(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Local BGE embeddings, as the benchmark uses for answer_correctness similarity."""
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(model_name=model_name)


# --- scoring helpers -------------------------------------------------------------------
def _mean(values: List[float]) -> float:
    """Mean ignoring NaN; NaN if there are no valid values (and no RuntimeWarning)."""
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _count_nan(values: List[float]) -> int:
    return int(sum(1 for v in values if v != v))


async def _safe_metric(coro, label: str) -> float:
    """Await a metric coroutine; degrade any failure to NaN (warn + continue)."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - intentional: any judge failure -> NaN
        print(f"  ⚠️  judge metric {label} failed ({type(exc).__name__}: {str(exc)[:90]}); -> NaN")
        return float("nan")


# --- generation ------------------------------------------------------------------------
async def _score_generation_sample(rec, metrics, llm, embeddings, sem) -> Dict[str, float]:
    _ensure_bench_on_path()
    from Evaluation.metrics import (  # noqa: E402  (path set above)
        compute_answer_correctness,
        compute_coverage_score,
        compute_faithfulness_score,
        compute_rouge_score,
    )

    qid = rec.get("id")
    async with sem:
        out: Dict[str, float] = {}
        if "rouge_score" in metrics:
            out["rouge_score"] = await _safe_metric(
                compute_rouge_score(rec["generated_answer"], rec["ground_truth"]),
                f"rouge_score[{qid}]",
            )
        if "answer_correctness" in metrics:
            out["answer_correctness"] = await _safe_metric(
                compute_answer_correctness(
                    rec["question"], rec["generated_answer"], rec["ground_truth"], llm, embeddings
                ),
                f"answer_correctness[{qid}]",
            )
        if "coverage_score" in metrics:
            out["coverage_score"] = await _safe_metric(
                compute_coverage_score(
                    rec["question"], rec["ground_truth"], rec["generated_answer"], llm
                ),
                f"coverage_score[{qid}]",
            )
        if "faithfulness" in metrics:
            out["faithfulness"] = await _safe_metric(
                compute_faithfulness_score(
                    rec["question"], rec["generated_answer"], rec["context"], llm
                ),
                f"faithfulness[{qid}]",
            )
        return out


async def score_generation(records, llm, embeddings, max_concurrency: int = 8) -> Dict[str, Any]:
    """Answer-quality scoring, grouped/aggregated per question_type (NaN failures excluded)."""
    sem = asyncio.Semaphore(max_concurrency)
    scored, tasks = [], []
    for rec in records:
        qt = rec.get("question_type", "Uncategorized")
        metrics = GENERATION_METRIC_CONFIG.get(qt)
        if not metrics:
            continue  # skip undefined question types, exactly like upstream
        scored.append((rec, qt))
        tasks.append(_score_generation_sample(rec, metrics, llm, embeddings, sem))
    results = await asyncio.gather(*tasks)

    by_type: Dict[str, Dict[str, List[float]]] = {}
    samples = []
    for (rec, qt), res in zip(scored, results, strict=True):
        by_type.setdefault(qt, {})
        for metric, value in res.items():
            by_type[qt].setdefault(metric, []).append(value)
        samples.append({"id": rec["id"], "question_type": qt, **res})

    by_question_type = {qt: {m: _mean(vs) for m, vs in md.items()} for qt, md in by_type.items()}
    pooled: Dict[str, List[float]] = {}
    for md in by_type.values():
        for m, vs in md.items():
            pooled.setdefault(m, []).extend(vs)
    overall = {m: _mean(vs) for m, vs in pooled.items()}
    failures = {m: _count_nan(vs) for m, vs in pooled.items()}
    return {
        "by_question_type": by_question_type,
        "overall": overall,
        "failures": failures,
        "samples": samples,
    }


# --- retrieval -------------------------------------------------------------------------
async def _score_retrieval_sample(rec, llm, sem) -> Dict[str, float]:
    _ensure_bench_on_path()
    from Evaluation.metrics import (  # noqa: E402
        compute_context_relevance,
        compute_evidence_recall,
    )

    qid = rec.get("id")
    async with sem:
        relevancy = await _safe_metric(
            compute_context_relevance(rec["question"], rec["context"], llm),
            f"context_relevancy[{qid}]",
        )
        recall = await _safe_metric(
            compute_evidence_recall(rec["question"], rec["context"], rec["evidence"], llm),
            f"evidence_recall[{qid}]",
        )
        return {"context_relevancy": relevancy, "evidence_recall": recall}


async def score_retrieval(records, llm, max_concurrency: int = 8) -> Dict[str, Any]:
    """Retrieval scoring: context_relevancy + evidence_recall (NaN failures excluded)."""
    sem = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(*[_score_retrieval_sample(r, llm, sem) for r in records])
    samples = [{"id": r["id"], **res} for r, res in zip(records, results, strict=True)]
    overall = {
        "context_relevancy": _mean([s["context_relevancy"] for s in samples]),
        "evidence_recall": _mean([s["evidence_recall"] for s in samples]),
    }
    failures = {
        "context_relevancy": _count_nan([s["context_relevancy"] for s in samples]),
        "evidence_recall": _count_nan([s["evidence_recall"] for s in samples]),
    }
    return {"overall": overall, "failures": failures, "samples": samples}
