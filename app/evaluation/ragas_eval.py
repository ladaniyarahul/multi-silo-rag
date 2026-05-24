"""
RAGAS Evaluator — scores RAG pipeline responses for quality.

Three metrics are computed per query:

    Faithfulness     : every claim in the answer is grounded in retrieved context.
                       Target > 0.85
    Answer Relevancy : the answer directly addresses the question asked.
                       Target > 0.80
    Context Recall   : the fraction of ground-truth information present in context.
                       Target > 0.75

The module supports two modes:
    1. Single-query scoring  — score_response()
    2. Batch dataset scoring — score_dataset()

Results are written to evaluation/ragas_results.jsonl for the Streamlit
metrics tab to read.

Dependencies: ragas, langchain, datasets
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ── RAGAS imports (graceful degradation if not installed) ─────────────────────

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
    )
    from datasets import Dataset
    _RAGAS_AVAILABLE = True
    logger.info("RAGAS library loaded successfully.")
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS not installed — evaluation will return null scores. "
        "Run: pip install ragas datasets"
    )

from evaluation.latency_tracker  import track
from evaluation.eval_dataset      import load_dataset, DATASET_PATH

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_PATH = Path(os.getenv("RAGAS_RESULTS_PATH", "evaluation/ragas_results.jsonl"))

METRIC_TARGETS = {
    "faithfulness":     0.85,
    "answer_relevancy": 0.80,
    "context_recall":   0.75,
}

_METRICS = [faithfulness, answer_relevancy, context_recall] if _RAGAS_AVAILABLE else []


# ── Result helpers ────────────────────────────────────────────────────────────

def _make_result(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    scores: dict[str, float],
    elapsed: float,
) -> dict[str, Any]:
    """Assemble a structured result record."""
    passed = {
        metric: scores.get(metric, 0.0) >= target
        for metric, target in METRIC_TARGETS.items()
    }
    return {
        "ts":           datetime.utcnow().isoformat() + "Z",
        "question":     question,
        "answer":       answer,
        "ground_truth": ground_truth,
        "scores":       scores,
        "targets":      METRIC_TARGETS,
        "passed":       passed,
        "all_passed":   all(passed.values()),
        "latency_s":    round(elapsed, 4),
    }


def _append_result(result: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")


def _null_scores() -> dict[str, float]:
    return {m: 0.0 for m in METRIC_TARGETS}


# ── Core scoring ──────────────────────────────────────────────────────────────

def _run_ragas(
    questions:    list[str],
    answers:      list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str],
) -> list[dict[str, float]]:
    """
    Run RAGAS evaluate() and return per-row score dicts.
    Returns null scores on any failure.
    """
    if not _RAGAS_AVAILABLE:
        logger.warning("RAGAS unavailable — returning null scores.")
        return [_null_scores() for _ in questions]

    try:
        dataset = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts_list,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=_METRICS)
        df = result.to_pandas()

        scores_list = []
        for _, row in df.iterrows():
            scores_list.append({
                "faithfulness":     float(row.get("faithfulness",     0.0)),
                "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                "context_recall":   float(row.get("context_recall",   0.0)),
            })
        return scores_list

    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}")
        return [_null_scores() for _ in questions]


# ── Public API ────────────────────────────────────────────────────────────────

@track("ragas_single")
def score_response(
    question:     str,
    answer:       str,
    contexts:     list[str],
    ground_truth: str,
    persist:      bool = True,
) -> dict[str, Any]:
    """
    Score a single RAG response.

    Args:
        question:     The user query.
        answer:       The LLM-generated answer.
        contexts:     List of retrieved context strings (parent chunks).
        ground_truth: Expected answer from the eval dataset.
        persist:      Append result to ragas_results.jsonl.

    Returns:
        Result dict with scores, targets, and pass/fail flags.
    """
    t0 = time.perf_counter()
    scores_list = _run_ragas([question], [answer], [contexts], [ground_truth])
    elapsed = time.perf_counter() - t0

    scores = scores_list[0] if scores_list else _null_scores()
    result = _make_result(question, answer, contexts, ground_truth, scores, elapsed)

    if persist:
        _append_result(result)

    _log_result(result)
    return result


@track("ragas_batch")
def score_dataset(
    answers:   list[str],
    contexts_list: list[list[str]],
    dataset_path: Path = DATASET_PATH,
    persist:   bool = True,
) -> list[dict[str, Any]]:
    """
    Score a full batch of answers against the ground-truth dataset.

    Args:
        answers:       One LLM answer per dataset entry (same order).
        contexts_list: One context list per dataset entry.
        dataset_path:  Path to the eval_dataset.jsonl file.
        persist:       Append each result to ragas_results.jsonl.

    Returns:
        List of result dicts, one per entry.
    """
    entries = load_dataset(dataset_path)

    if len(answers) != len(entries):
        raise ValueError(
            f"answers length ({len(answers)}) != dataset length ({len(entries)})"
        )
    if len(contexts_list) != len(entries):
        raise ValueError(
            f"contexts_list length ({len(contexts_list)}) != dataset length ({len(entries)})"
        )

    questions     = [e["question"]     for e in entries]
    ground_truths = [e["ground_truth"] for e in entries]

    t0 = time.perf_counter()
    scores_list = _run_ragas(questions, answers, contexts_list, ground_truths)
    elapsed_total = time.perf_counter() - t0

    results = []
    for i, (entry, scores) in enumerate(zip(entries, scores_list)):
        result = _make_result(
            question=entry["question"],
            answer=answers[i],
            contexts=contexts_list[i],
            ground_truth=entry["ground_truth"],
            scores=scores,
            elapsed=elapsed_total / len(entries),
        )
        if persist:
            _append_result(result)
        results.append(result)

    _log_batch_summary(results)
    return results


# ── Aggregation helpers (used by Streamlit metrics tab) ──────────────────────

def load_results(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Load all persisted result records."""
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute mean scores and pass-rates across a list of results.

    Returns:
        {
            "n":        int,
            "means":    {metric: float},
            "pass_rate":{metric: float},   # fraction meeting target
            "all_pass_rate": float,        # fraction where ALL metrics pass
        }
    """
    if not results:
        return {"n": 0, "means": {}, "pass_rate": {}, "all_pass_rate": 0.0}

    metrics = list(METRIC_TARGETS.keys())
    sums    = {m: 0.0 for m in metrics}
    passes  = {m: 0   for m in metrics}
    all_ok  = 0

    for r in results:
        scores = r.get("scores", {})
        passed = r.get("passed", {})
        for m in metrics:
            sums[m]   += scores.get(m, 0.0)
            passes[m] += int(passed.get(m, False))
        if r.get("all_passed", False):
            all_ok += 1

    n = len(results)
    return {
        "n":     n,
        "means": {m: round(sums[m] / n, 4) for m in metrics},
        "pass_rate": {m: round(passes[m] / n, 4) for m in metrics},
        "all_pass_rate": round(all_ok / n, 4),
    }


def latest_scores(n: int = 10, path: Path = RESULTS_PATH) -> dict[str, Any]:
    """Aggregate the most recent *n* results — used by the Streamlit live view."""
    results = load_results(path)
    return aggregate_scores(results[-n:])


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_result(result: dict[str, Any]) -> None:
    scores  = result["scores"]
    passed  = result["passed"]
    icons   = {True: "✓", False: "✗"}
    summary = "  ".join(
        f"{icons[passed[m]]} {m}: {scores[m]:.3f}"
        for m in METRIC_TARGETS
    )
    level = "info" if result["all_passed"] else "warning"
    getattr(logger, level)(f"RAGAS [{summary}]  ({result['latency_s']:.2f}s)")


def _log_batch_summary(results: list[dict[str, Any]]) -> None:
    agg = aggregate_scores(results)
    means = agg["means"]
    logger.info(
        f"RAGAS batch ({agg['n']} items) — "
        f"faithfulness: {means.get('faithfulness', 0):.3f}  "
        f"relevancy: {means.get('answer_relevancy', 0):.3f}  "
        f"recall: {means.get('context_recall', 0):.3f}  "
        f"all-pass: {agg['all_pass_rate']:.1%}"
    )