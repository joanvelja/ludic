#!/usr/bin/env python3
"""Posthoc diagnostics for RM training logs.

Parses log files for evaluation metrics and reports trends, progressions, and spikes.
Prefers structured RM_DIAG JSON lines if present; falls back to standard eval lines.

Usage:
    uv run python examples/reward_model/diagnose_rm_logs.py --logs-dir logs
    uv run python examples/reward_model/diagnose_rm_logs.py --logs-dir logs --pattern "slurm_*_rm_v4.log"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


BASELINE_RE = re.compile(
    r"Baseline \| Eval Loss: ([\d.]+) \| Eval Acc: ([\d.]+)% \| Eval Margin: ([-\d.]+) \| "
    r"Eval ECE: ([\d.]+) \| Eval AUROC: ([\d.]+)"
)
STEP_EVAL_RE = re.compile(
    r"Step (\d+) Eval \| Loss: ([\d.]+) \| Acc: ([\d.]+)% \| Margin: ([-\d.]+) \| "
    r"ECE: ([\d.]+) \| AUROC: ([\d.]+)"
)


@dataclass
class EvalPoint:
    step: int
    split: str
    metrics: Dict[str, float]


@dataclass
class LogSummary:
    path: Path
    run_name: Optional[str]
    points: List[EvalPoint]
    metrics_available: List[str]
    best_accuracy: Optional[Tuple[int, float]]
    final_accuracy: Optional[Tuple[int, float]]
    trend_summary: Dict[str, Dict[str, float]]
    spikes: Dict[str, List[Dict[str, float]]]
    confidence_inflation: Dict[str, float]


def _flatten_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}

    def visit(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "calibration_bins":
                    continue
                next_prefix = f"{prefix}{key}_"
                visit(next_prefix, value)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            key = prefix[:-1] if prefix.endswith("_") else prefix
            if key:
                flat[key] = float(obj)

    visit("", payload)
    return flat


def _parse_diag_line(line: str) -> Optional[EvalPoint]:
    if "RM_DIAG " not in line:
        return None
    payload = line.split("RM_DIAG ", 1)[1].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if "step" not in data:
        return None
    step = int(data["step"])
    split = str(data.get("split", "eval"))
    metrics = _flatten_metrics(data)
    return EvalPoint(step=step, split=split, metrics=metrics)


def _parse_fallback_line(line: str) -> Optional[EvalPoint]:
    baseline = BASELINE_RE.search(line)
    if baseline:
        loss, acc, margin, ece, auroc = baseline.groups()
        return EvalPoint(
            step=0,
            split="baseline",
            metrics={
                "loss": float(loss),
                "accuracy": float(acc) / 100.0,
                "margin": float(margin),
                "ece": float(ece),
                "auroc": float(auroc),
            },
        )

    match = STEP_EVAL_RE.search(line)
    if match:
        step, loss, acc, margin, ece, auroc = match.groups()
        return EvalPoint(
            step=int(step),
            split="eval",
            metrics={
                "loss": float(loss),
                "accuracy": float(acc) / 100.0,
                "margin": float(margin),
                "ece": float(ece),
                "auroc": float(auroc),
            },
        )
    return None


def parse_log(path: Path) -> Tuple[Optional[str], List[EvalPoint]]:
    run_name: Optional[str] = None
    diag_points: List[EvalPoint] = []
    fallback_points: List[EvalPoint] = []

    with path.open() as f:
        for line in f:
            if run_name is None and line.startswith("Running:"):
                run_name = line.split("Running:", 1)[1].strip()

            diag_point = _parse_diag_line(line)
            if diag_point is not None:
                diag_points.append(diag_point)
                continue

            fallback_point = _parse_fallback_line(line)
            if fallback_point is not None:
                fallback_points.append(fallback_point)

    points = diag_points if diag_points else fallback_points
    points = sorted(points, key=lambda p: p.step)
    return run_name, points


def _compute_slope(steps: List[int], values: List[float]) -> float:
    if len(steps) < 2:
        return 0.0
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / denom)


def _detect_spikes(
    steps: List[int],
    values: List[float],
    *,
    k: float,
) -> List[Dict[str, float]]:
    if len(values) < 4:
        return []
    diffs = np.diff(values)
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    if mad == 0.0:
        return []
    z = (diffs - median) / (1.4826 * mad)
    spikes: List[Dict[str, float]] = []
    for idx, z_val in enumerate(z):
        if abs(z_val) >= k:
            spikes.append({
                "step": float(steps[idx + 1]),
                "delta": float(diffs[idx]),
                "z": float(z_val),
            })
    return spikes


def _summarize_series(
    points: List[EvalPoint],
    *,
    min_points: int,
    spike_k: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    metrics_keys: List[str] = sorted({key for p in points for key in p.metrics.keys()})
    trend_summary: Dict[str, Dict[str, float]] = {}
    spikes: Dict[str, List[Dict[str, float]]] = {}

    for key in metrics_keys:
        series = [(p.step, p.metrics[key]) for p in points if key in p.metrics and not math.isnan(p.metrics[key])]
        if len(series) < min_points:
            continue
        steps, values = zip(*series)
        steps_list = list(steps)
        values_list = list(values)
        slope = _compute_slope(steps_list, values_list)
        trend_summary[key] = {
            "start": float(values_list[0]),
            "end": float(values_list[-1]),
            "min": float(min(values_list)),
            "max": float(max(values_list)),
            "slope_per_step": slope,
            "slope_per_100_steps": slope * 100.0,
        }
        spikes[key] = _detect_spikes(steps_list, values_list, k=spike_k)
    return trend_summary, spikes


def _confidence_inflation(points: List[EvalPoint]) -> Dict[str, float]:
    acc_series = [(p.step, p.metrics.get("accuracy")) for p in points]
    margin_series = [(p.step, p.metrics.get("margin")) for p in points]
    acc_series = [(s, v) for s, v in acc_series if v is not None and not math.isnan(v)]
    margin_series = [(s, v) for s, v in margin_series if v is not None and not math.isnan(v)]
    if len(acc_series) < 2 or len(margin_series) < 2:
        return {"events": 0.0, "ratio": 0.0}

    acc_by_step = {s: v for s, v in acc_series}
    margin_by_step = {s: v for s, v in margin_series}
    steps = sorted(set(acc_by_step.keys()) & set(margin_by_step.keys()))
    if len(steps) < 2:
        return {"events": 0.0, "ratio": 0.0}

    events = 0
    total = 0
    for idx in range(1, len(steps)):
        prev_step = steps[idx - 1]
        step = steps[idx]
        acc_delta = acc_by_step[step] - acc_by_step[prev_step]
        margin_delta = margin_by_step[step] - margin_by_step[prev_step]
        if acc_delta < 0 and margin_delta > 0:
            events += 1
        total += 1
    ratio = float(events / total) if total > 0 else 0.0
    return {"events": float(events), "ratio": ratio}


def analyze_log(
    path: Path,
    *,
    min_points: int,
    spike_k: float,
) -> Optional[LogSummary]:
    run_name, points = parse_log(path)
    if not points:
        return None

    best_accuracy = None
    final_accuracy = None
    for p in points:
        acc = p.metrics.get("accuracy")
        if acc is None or math.isnan(acc):
            continue
        if best_accuracy is None or acc > best_accuracy[1]:
            best_accuracy = (p.step, acc)
    last_point = max(points, key=lambda p: p.step)
    if "accuracy" in last_point.metrics:
        final_accuracy = (last_point.step, last_point.metrics["accuracy"])

    trend_summary, spikes = _summarize_series(points, min_points=min_points, spike_k=spike_k)
    confidence_inflation = _confidence_inflation(points)
    metrics_available = sorted({k for p in points for k in p.metrics.keys()})

    return LogSummary(
        path=path,
        run_name=run_name,
        points=points,
        metrics_available=metrics_available,
        best_accuracy=best_accuracy,
        final_accuracy=final_accuracy,
        trend_summary=trend_summary,
        spikes=spikes,
        confidence_inflation=confidence_inflation,
    )


def _print_summary(summary: LogSummary, *, max_spikes: int) -> None:
    name = summary.run_name or summary.path.name
    print(f"\n{name}")
    print(f"  log: {summary.path}")
    print(f"  eval points: {len(summary.points)}")
    if summary.best_accuracy:
        step, acc = summary.best_accuracy
        print(f"  best accuracy: {acc:.2%} @ step {step}")
    if summary.final_accuracy:
        step, acc = summary.final_accuracy
        print(f"  final accuracy: {acc:.2%} @ step {step}")

    if "accuracy" in summary.trend_summary:
        trend = summary.trend_summary["accuracy"]
        print(
            "  accuracy trend: "
            f"start={trend['start']:.2%} end={trend['end']:.2%} "
            f"slope/100={trend['slope_per_100_steps']:.3f}"
        )

    if "loss" in summary.trend_summary:
        trend = summary.trend_summary["loss"]
        print(
            "  loss trend: "
            f"start={trend['start']:.4f} end={trend['end']:.4f} "
            f"slope/100={trend['slope_per_100_steps']:.4f}"
        )

    if summary.confidence_inflation["events"] > 0:
        print(
            "  confidence inflation events "
            f"(acc down, margin up): {summary.confidence_inflation['events']:.0f} "
            f"({summary.confidence_inflation['ratio']:.2%})"
        )

    for metric in ("accuracy", "loss", "margin"):
        if metric not in summary.spikes:
            continue
        spikes = summary.spikes[metric][:max_spikes]
        if not spikes:
            continue
        formatted = ", ".join(
            f"step {int(s['step'])}: delta={s['delta']:.4f} z={s['z']:.2f}"
            for s in spikes
        )
        print(f"  spikes ({metric}): {formatted}")


def _summary_to_json(summary: LogSummary) -> Dict[str, Any]:
    return {
        "log": str(summary.path),
        "run_name": summary.run_name,
        "points": len(summary.points),
        "metrics_available": summary.metrics_available,
        "best_accuracy": summary.best_accuracy,
        "final_accuracy": summary.final_accuracy,
        "trend_summary": summary.trend_summary,
        "spikes": summary.spikes,
        "confidence_inflation": summary.confidence_inflation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose RM training logs")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--pattern", type=str, default="slurm_*_rm_*.log")
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--spike-k", type=float, default=4.0)
    parser.add_argument("--max-spikes", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    log_paths = sorted(args.logs_dir.glob(args.pattern))
    if not log_paths:
        print(f"No logs matched {args.pattern} in {args.logs_dir}")
        return

    summaries: List[LogSummary] = []
    for path in log_paths:
        summary = analyze_log(path, min_points=args.min_points, spike_k=args.spike_k)
        if summary is None:
            continue
        summaries.append(summary)
        _print_summary(summary, max_spikes=args.max_spikes)

    if args.json_out:
        payload = [_summary_to_json(summary) for summary in summaries]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
