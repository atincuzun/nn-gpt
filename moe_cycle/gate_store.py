"""On-disk gate experiment store.

One directory per gate candidate under ``nngpt_gate_dir`` (``out/nngpt/gates``):

    gate_000/
      gate.py                 proposed source (reference for the next proposal)
      summary.json            per-gate aggregate + selection score
      epoch_00/
        metrics.json          per inner-epoch measurements
        gate_weights.pt       trained gate weights (optional)
      epoch_01/ ...

Scoring stores everything and ranks on the mean of per-epoch means.  A single
lucky CV model in one epoch must not outrank a consistently better gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

# LEMUR-style vocabulary keeps records recognisable and exportable.
SCORE_OBJECTIVE = "mean_of_epoch_means_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def structural_hash(source: str) -> str:
    """AST hash for deduplication, normalised against cosmetic rewrites.

    Type annotations and docstrings are stripped before hashing: the observed
    failure mode was a bare linear gate echoing the baseline contract with
    annotations added, which changed the raw AST while the architecture was
    identical.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            node.body = [stmt for stmt in node.body if not (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )] or node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            for argument in list(node.args.args) + list(node.args.kwonlyargs) + list(node.args.posonlyargs):
                argument.annotation = None
        if isinstance(node, ast.AnnAssign):
            node.annotation = None
    return fingerprint(ast.dump(tree, include_attributes=False))


def next_gate_id(root: Path) -> int:
    """First gate id not already recorded in the store (0 for a fresh store).

    Lets a persistent store shared across runs continue its numbering instead
    of overwriting earlier ``gate_XXX`` directories.
    """
    root = Path(root)
    if not root.is_dir():
        return 0
    used = {
        int(match.group(1))
        for directory in root.glob("gate_*")
        if (match := re.fullmatch(r"gate_(\d+)", directory.name))
    }
    return max(used) + 1 if used else 0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gate_dir(root: Path, gate_id: int) -> Path:
    return Path(root) / f"gate_{gate_id:03d}"


def epoch_dir(root: Path, gate_id: int, epoch: int) -> Path:
    return gate_dir(root, gate_id) / f"epoch_{epoch:02d}"


def summarize_epoch_metrics(metrics: list[dict]) -> dict:
    """Reduce per-epoch measurements to the stored gate score.

    ``accuracy`` is the mean of per-epoch means over epochs that produced at
    least one measurement.  Bests and counts are kept for later re-analysis.
    """
    means = [m["cv_accuracy_mean"] for m in metrics if m.get("cv_accuracy_mean") is not None]
    bests = [m["cv_accuracy_best"] for m in metrics if m.get("cv_accuracy_best") is not None]
    n_measured = sum(int(m.get("n_measured", 0)) for m in metrics)
    n_attempted = sum(int(m.get("n_attempted", 0)) for m in metrics)
    mean_of_means = sum(means) / len(means) if means else None
    mean_of_bests = sum(bests) / len(bests) if bests else None
    variance = (
        sum((value - mean_of_means) ** 2 for value in means) / len(means)
        if means else None
    )
    return {
        "objective": SCORE_OBJECTIVE,
        "accuracy": mean_of_means,
        "mean_of_epoch_means": mean_of_means,
        "mean_of_epoch_bests": mean_of_bests,
        "best_epoch_mean": max(means) if means else None,
        "std_epoch_mean": math.sqrt(variance) if variance is not None else None,
        "n_inner_epochs_scored": len(means),
        "n_measured": n_measured,
        "n_attempted": n_attempted,
        "success_rate": (n_measured / n_attempted) if n_attempted else None,
        "eligible": bool(means) and n_measured > 0,
    }


def summarize_gate(summary: dict) -> str:
    """Compact feedback line for the next gate proposal prompt."""
    score = summary.get("score") or {}
    accuracy = score.get("accuracy")
    accuracy_text = f"{accuracy:.4f}" if isinstance(accuracy, float) else "unavailable"
    return (
        f"- gate {summary.get('gate_id'):03d}: mean CV accuracy {accuracy_text} "
        f"over {score.get('n_inner_epochs_scored', 0)} epochs, "
        f"{score.get('n_measured', 0)}/{score.get('n_attempted', 0)} candidates measured"
    )


def write_gate_summary(root: Path, gate_id: int, record: dict) -> Path:
    path = gate_dir(root, gate_id) / "summary.json"
    write_json(path, record)
    return path


def load_gate_summaries(root: Path) -> list[dict]:
    """All gate summaries for a run, oldest first, ignoring incomplete gates."""
    root = Path(root)
    if not root.is_dir():
        return []
    summaries = []
    for directory in sorted(root.glob("gate_*")):
        path = directory / "summary.json"
        if not path.is_file():
            continue
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return summaries


def best_gate(root: Path) -> dict | None:
    """Highest-scoring eligible gate seen so far, or None during bootstrap."""
    eligible = [
        summary for summary in load_gate_summaries(root)
        if (summary.get("score") or {}).get("eligible")
        and isinstance((summary.get("score") or {}).get("accuracy"), float)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda summary: summary["score"]["accuracy"])
