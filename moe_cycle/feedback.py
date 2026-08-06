"""Measured NAS feedback summaries injected into generation and training prompts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _candidate_index(path: Path) -> int:
    try:
        return int(path.name.removeprefix("B"))
    except ValueError:
        return 10**9


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    return default if value is None else value


def _load_candidate_source_row(candidate_dir: Path) -> Any:
    """Read the generation seed saved by ChatBot without changing Tune.py."""
    dataframe_path = candidate_dir / "dataframe.df"
    if not dataframe_path.is_file():
        return None
    try:
        import pandas as pd

        value = pd.read_pickle(dataframe_path)
        if getattr(value, "ndim", None) == 2:
            return value.iloc[0] if len(value) else None
        return value
    except (OSError, ValueError, TypeError, ImportError):
        return None


def _build_gate_feedback_summary(
    epoch_path: Path,
    used_prompts: list,
    cycle_results: dict,
    session: Any,
) -> dict:
    """Compose the per-cycle gate outcome report injected into the training prompt.

    Reports the gate code that generated the cycle, candidate validity, measured
    accuracy, and improvement relative to each candidate's source architecture.
    Failed evaluations are reported separately and are never treated as zero
    accuracy.
    """
    gate_code = session.gate_source
    if not gate_code:
        gate_file = epoch_path / "gate" / "gate.py"
        gate_code = gate_file.read_text(encoding="utf-8") if gate_file.is_file() else None
    if gate_code and len(gate_code) > 2000:
        gate_code = gate_code[:2000] + "\n# ... (truncated)"
    gate_identity = gate_code or f"[{session.gate_factory_name or 'installed'} gate]"

    synth = epoch_path / "synth_nn"
    candidate_dirs = (
        sorted(synth.glob("B*"), key=_candidate_index) if synth.is_dir() else []
    )
    n_generated = len(candidate_dirs) or len(used_prompts)

    accuracies: list[float] = []
    deltas: list[float] = []
    candidate_outcomes: list[dict[str, Any]] = []
    failure_reasons: dict[str, int] = {}
    source_rows = {
        f"B{index}": row
        for index, (_, row) in enumerate(used_prompts)
    }
    datasets: set[str] = set()
    tasks: set[str] = set()
    for candidate_dir in candidate_dirs:
        source_row = source_rows.get(candidate_dir.name)
        if source_row is None:
            source_row = _load_candidate_source_row(candidate_dir)
        baseline_accuracy = _row_value(source_row, "accuracy")
        try:
            baseline_accuracy = (
                float(baseline_accuracy) if baseline_accuracy is not None else None
            )
        except (TypeError, ValueError):
            baseline_accuracy = None
        dataset = str(_row_value(source_row, "dataset", "?"))
        task = str(_row_value(source_row, "task", "?"))
        source_nn = str(_row_value(source_row, "nn", "?"))
        datasets.add(dataset)
        tasks.add(task)

        summary_file = candidate_dir / "eval_summary.json"
        if not summary_file.is_file():
            error_file = candidate_dir / "error.txt"
            reason = "not evaluated"
            if error_file.is_file():
                try:
                    reason = error_file.read_text(encoding="utf-8").splitlines()[0].strip()
                except (OSError, IndexError):
                    reason = "evaluation failed"
            reason = reason[:200] or "evaluation failed"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            candidate_outcomes.append({
                "candidate": candidate_dir.name,
                "source_nn": source_nn,
                "baseline_accuracy": baseline_accuracy,
                "accuracy": None,
                "accuracy_delta": None,
                "status": "failed",
                "failure": reason,
            })
            continue
        try:
            entries = json.loads(summary_file.read_text(encoding="utf-8"))
            measured_accuracy = None
            for entry in entries:
                value = entry.get("accuracy", entry.get("acc"))
                if value is not None:
                    measured_accuracy = float(value)
            if measured_accuracy is None:
                continue
            accuracies.append(measured_accuracy)
            delta = (
                measured_accuracy - baseline_accuracy
                if baseline_accuracy is not None
                else None
            )
            if delta is not None:
                deltas.append(delta)
            candidate_outcomes.append({
                "candidate": candidate_dir.name,
                "source_nn": source_nn,
                "baseline_accuracy": baseline_accuracy,
                "accuracy": measured_accuracy,
                "accuracy_delta": delta,
                "status": "measured",
                "failure": None,
            })
        except (ValueError, OSError, TypeError):
            continue

    evaluation = (cycle_results or {}).get("evaluation", {})
    n_trained = int(evaluation.get("models_trained", 0))

    datasets.discard("?")
    tasks.discard("?")
    sorted_datasets = sorted(datasets)
    sorted_tasks = sorted(tasks)

    if accuracies:
        mean = sum(accuracies) / len(accuracies)
        variance = sum((value - mean) ** 2 for value in accuracies) / len(accuracies)
        std = math.sqrt(variance)
        best = max(accuracies)
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        improved = sum(delta > 0 for delta in deltas)
        measured = (
            f"Average accuracy of the {len(accuracies)} successfully trained "
            f"candidate networks: {mean:.4f} (std: {std:.4f}); best accuracy: "
            f"{best:.4f}."
        )
        if mean_delta is not None:
            measured += (
                f" {improved}/{len(deltas)} measured candidates improved on their "
                f"source model; mean accuracy delta: {mean_delta:+.4f}."
            )
    else:
        mean = None
        std = None
        best = None
        mean_delta = None
        improved = 0
        measured = (
            "None of the candidate networks trained successfully, "
            "so no measured accuracy is available for this cycle."
        )

    measured_outcomes = sorted(
        (
            outcome for outcome in candidate_outcomes
            if outcome["status"] == "measured"
        ),
        key=lambda outcome: outcome["accuracy"],
        reverse=True,
    )
    outcome_lines = []
    for outcome in measured_outcomes[:5]:
        delta_text = (
            f", delta={outcome['accuracy_delta']:+.4f}"
            if outcome["accuracy_delta"] is not None
            else ""
        )
        outcome_lines.append(
            f"- {outcome['candidate']} from {outcome['source_nn']}: "
            f"accuracy={outcome['accuracy']:.4f}{delta_text}"
        )
    failure_lines = [
        f"- {count}x {reason}"
        for reason, count in sorted(
            failure_reasons.items(), key=lambda item: (-item[1], item[0])
        )[:5]
    ]

    summary = (
        "During the previous generation cycle you used the following MoE router gate code:\n"
        f"<gate>\n{gate_identity}\n</gate>\n"
        f"You generated {n_generated} candidate neural networks; {n_trained} of them were "
        f"trained and evaluated ({', '.join(sorted_tasks) or 'unknown'} task on "
        f"{', '.join(sorted_datasets) or 'unknown'} datasets).\n"
        f"{measured}\n"
        + ("Best measured candidates:\n" + "\n".join(outcome_lines) + "\n" if outcome_lines else "")
        + ("Most frequent failures:\n" + "\n".join(failure_lines) if failure_lines else "")
    )
    return {
        "summary": summary,
        "n_generated": n_generated,
        "n_trained": n_trained,
        "n_measured": len(accuracies),
        "mean_accuracy": mean,
        "std_accuracy": std,
        "best_accuracy": best,
        "mean_accuracy_delta": mean_delta,
        "improved_candidates": improved,
        "datasets": sorted_datasets,
        "tasks": sorted_tasks,
        "candidate_outcomes": candidate_outcomes,
    }
