"""On-disk gate experiment store.

One self-contained directory per gate candidate under ``--gate-store``:

    gate_000/
      gate.py                     canonical proposed source
      record.json                 canonical gate record (LEMUR-shaped stats)
      weights/                    latest trained gate checkpoint
      epochs/
        A0/
          synth_nn/B0..B9/        generated CV candidates (NNGPT layout)

``record.json`` is the only gate-level JSON. It carries the structural hash,
provenance, aggregate score, LEMUR-shaped ``stats`` rows, per-epoch feedback,
training diagnostics, and proposal/install receipts. ``nn.code`` embeds the
gate source so the record can be converted to LEMUR ``nn``/``stat`` rows
without re-reading ``gate.py``.

Scoring ranks on the mean of per-epoch means; a single lucky CV model in one
epoch must not outrank a consistently better gate. A single ``weights``
directory per gate holds the newest trained checkpoint; older epochs are
recomputed from the record or the training chain rather than kept on disk.

Legacy stores with ``summary.json`` and ``epochs/A*/gate_post_train`` remain
readable; new writes use only the layout above.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# LEMUR-style vocabulary keeps records recognisable and exportable.
SCORE_OBJECTIVE = "mean_of_epoch_means_v1"
GATE_RECORD_SCHEMA_VERSION = 2
GATE_TASK = "moe-gate"
GATE_METRIC = "cv_accuracy"
GATE_TRANSFORM = "moe-gate"
GATE_NN_PREFIX = "moe-gate-"
WEIGHTS_DIR = "weights"
SYNTH_DIR = "synth_nn"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def lemur_id(value: Any) -> str:
    """LEMUR's content id used by ``ab.nn.util.Util.uuid4``."""
    return hashlib.md5(re.sub(r"\s", "", str(value)).encode()).hexdigest()


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


def _gate_number(path: Path) -> int | None:
    match = re.fullmatch(r"gate_(\d+)", path.name)
    return int(match.group(1)) if match else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gate_dir(root: Path, gate_id: int) -> Path:
    return Path(root) / f"gate_{gate_id:03d}"


def epoch_dir(root: Path, gate_id: int, epoch: int) -> Path:
    return gate_dir(root, gate_id) / "epochs" / f"A{epoch}"


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


def _is_checkpoint(path: Path) -> bool:
    return (path / "metadata.json").is_file() and (path / "gate_weights.pt").is_file()


def _latest_trained_checkpoint(gate_path: Path) -> str | None:
    """Newest trained checkpoint: new ``weights/`` first, legacy epochs after."""
    weights = gate_path / WEIGHTS_DIR
    if _is_checkpoint(weights):
        return WEIGHTS_DIR
    checkpoints = [
        path for path in (gate_path / "epochs").glob("A*/gate_post_train")
        if _is_checkpoint(path)
    ]
    if not checkpoints:
        return None

    def epoch_number(path: Path) -> int:
        name = path.parent.name
        return int(name[1:]) if name.startswith("A") and name[1:].isdigit() else -1

    latest = max(checkpoints, key=epoch_number)
    return str(latest.relative_to(gate_path))


def _nn_name(digest: str | None) -> str | None:
    return f"{GATE_NN_PREFIX}{digest[:16]}" if digest else None


def _gate_stats(
    record: dict, gate_id: Any, digest: str | None, nn_name: str | None,
) -> list[dict]:
    """Per-epoch rows shaped like a LEMUR ``new_lemur/train`` JSON row.

    The scalar keys are what nn-dataset's ``DB_Write.save_stat`` writes to the
    ``prm`` table; ``train_stat`` becomes the 1:1 ``train_stat`` row. Keeping
    the row flat (rather than nesting a ``prm`` dict) makes the record directly
    insertable into a LEMUR-shaped database later.
    """
    score = record.get("score") or {}
    dataset = record.get("dataset") or "unknown"
    epochs = record.get("epochs") or []
    epoch_max = len(epochs)
    stats: list[dict] = []
    for epoch in epochs:
        mean = epoch.get("cv_accuracy_mean")
        if not isinstance(mean, (int, float)) or isinstance(mean, bool):
            continue
        epoch_index = int(epoch.get("epoch", len(stats)))
        n_attempted = epoch.get("n_attempted")
        n_measured = epoch.get("n_measured")
        success_rate = (
            float(n_measured) / float(n_attempted)
            if n_attempted else None
        )
        stats.append({
            "accuracy": float(mean),
            "epoch": epoch_index,
            "duration": int(epoch.get("duration") or 0),
            "task": GATE_TASK,
            "dataset": dataset,
            "metric": GATE_METRIC,
            "transform": GATE_TRANSFORM,
            "model": nn_name,
            "gate_id": gate_id,
            "structural_hash": digest,
            "seeded": record.get("seeded"),
            "reference_gate_id": record.get("reference_gate_id"),
            "author_gate_id": record.get("author_gate_id"),
            "carry_source_gate_id": record.get("carry_source_gate_id"),
            "llm_version": record.get("llm_version"),
            "score_objective": score.get("objective"),
            "score_accuracy": score.get("accuracy"),
            "cv_task": record.get("task"),
            "cv_metric": record.get("metric"),
            "cv_accuracy_mean": float(mean),
            "cv_accuracy_best": epoch.get("cv_accuracy_best"),
            "n_measured": n_measured,
            "n_attempted": n_attempted,
            "n_trained": epoch.get("n_trained"),
            "n_generated": epoch.get("n_generated"),
            "success_rate": success_rate,
            "train_stat": {
                "train_loss": epoch.get("train_loss"),
                "test_loss": epoch.get("validation_loss"),
                "train_accuracy": float(mean),
                "epoch_max": epoch_max,
            },
        })
    return stats


def build_gate_record(
    record: dict,
    *,
    gate_path: Path,
    run_root: Path | None = None,
    model: str | None = None,
) -> dict:
    """Canonical, LEMUR-shaped view of one gate candidate.

    ``nn.code`` embeds the gate source and ``nn.id`` uses LEMUR's content id
    (md5 of whitespace-stripped code), so the record can be inserted into a
    LEMUR-shaped ``nn``/``stat`` store without additional lookups. The
    nn-dataset ``Net`` contract is deliberately not applied: a gate is a
    router module, not a CV model, and must live in a separate gate namespace.
    """
    gate_path = Path(gate_path)
    digest = record.get("structural_hash")
    score = record.get("score") or {}
    failure = record.get("failure")
    if failure:
        status = "failed"
    elif score.get("eligible"):
        status = "success"
    elif record.get("running"):
        status = "running"
    else:
        status = "ineligible"

    source = None
    gate_source = gate_path / "gate.py"
    if gate_source.is_file():
        try:
            source = gate_source.read_text(encoding="utf-8")
        except OSError:
            source = None
    nn_name = _nn_name(digest)
    created_at = (
        record.get("created_at")
        or (record.get("provenance") or {}).get("created_at")
        or datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    canonical = {key: value for key, value in record.items() if key != "gate_code"}
    canonical.update({
        "schema_version": GATE_RECORD_SCHEMA_VERSION,
        "status": status,
        "nn": {
            "name": nn_name,
            "code": source,
            "id": lemur_id(source) if source is not None else None,
            "code_file": "gate.py" if source is not None else None,
            "structural_hash": digest,
        },
        "provenance": {
            "gate_id": record.get("gate_id"),
            "seeded": record.get("seeded"),
            "reference_gate_id": record.get("reference_gate_id"),
            "author_mode": record.get("author_mode"),
            "author_gate_id": record.get("author_gate_id"),
            "carry_source_gate_id": record.get("carry_source_gate_id"),
            "rematch_for": record.get("rematch_for"),
            "llm_version": record.get("llm_version"),
            "run_root": str(run_root) if run_root is not None else None,
            "model": model,
            "created_at": created_at,
        },
        "score": score,
        "stats": _gate_stats(record, record.get("gate_id"), digest, nn_name),
        "artifacts": {
            "source": "gate.py" if source is not None else None,
            "weights": WEIGHTS_DIR if _is_checkpoint(gate_path / WEIGHTS_DIR) else None,
            "synth_nn": SYNTH_DIR if (gate_path / "epochs").is_dir() else None,
            "latest_checkpoint": _latest_trained_checkpoint(gate_path),
        },
        "dedup": {
            "structural_hash": digest,
            "status": (
                "bypass"
                if record.get("seeded") or record.get("rematch_for") is not None
                else ("failed" if failure else "accepted")
            ),
            "reason": (
                "seed"
                if record.get("seeded")
                else "rematch"
                if record.get("rematch_for") is not None
                else ("candidate failure" if failure else None)
            ),
        },
    })
    return canonical


def write_gate_record(
    root: Path,
    gate_id: int,
    record: dict,
    *,
    run_root: Path | None = None,
    model: str | None = None,
) -> Path:
    """Write the single canonical ``record.json`` (no legacy sidecars)."""
    path = gate_dir(root, gate_id) / "record.json"
    record.setdefault("gate_id", gate_id)
    write_json(
        path,
        build_gate_record(
            record, gate_path=gate_dir(root, gate_id),
            run_root=run_root, model=model,
        ),
    )
    return path


def write_gate_summary(root: Path, gate_id: int, record: dict) -> Path:
    """Legacy aggregate view kept for old callers and fixtures only."""
    path = gate_dir(root, gate_id) / "summary.json"
    write_json(path, record)
    return path


def load_gate_summaries(root: Path) -> list[dict]:
    """All gate records for a run, oldest first, ignoring incomplete gates.

    ``record.json`` is preferred; ``summary.json`` remains the fallback for
    stores written before the canonical record existed. ``gate_code`` is
    hydrated from ``gate.py`` (canonical), ``nn.code``, or the legacy summary
    so pair/selection callers keep the same interface.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    directories = sorted(
        (path for path in root.glob("gate_*") if _gate_number(path) is not None),
        key=lambda path: _gate_number(path) or 0,
    )
    summaries = []
    for directory in directories:
        data = None
        for path in (directory / "record.json", directory / "summary.json"):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            break
        if data is None:
            continue
        if not data.get("gate_code"):
            gate_source = directory / "gate.py"
            if gate_source.is_file():
                try:
                    data["gate_code"] = gate_source.read_text(encoding="utf-8").rstrip("\n")
                except OSError:
                    pass
        if not data.get("gate_code"):
            code = (data.get("nn") or {}).get("code")
            if code:
                data["gate_code"] = code
        if not data.get("gate_code"):
            legacy = directory / "summary.json"
            if legacy.is_file() and legacy != directory / "record.json":
                try:
                    code = json.loads(legacy.read_text(encoding="utf-8")).get("gate_code")
                    if code:
                        data["gate_code"] = code
                except (OSError, ValueError):
                    pass
        summaries.append(data)
    return summaries


def backfill_gate_records(
    root: Path,
    *,
    run_root: Path | None = None,
    model: str | None = None,
) -> list[Path]:
    """Create ``record.json`` for legacy summary-only gates.

    Existing ``record.json`` files are never overwritten. This is the
    migration path for a store written before the canonical layout.
    """
    root = Path(root)
    written: list[Path] = []
    for directory in sorted(root.glob("gate_*")):
        if (directory / "record.json").is_file():
            continue
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            record = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gate_id = record.get("gate_id")
        if not isinstance(gate_id, int):
            match = re.fullmatch(r"gate_(\d+)", directory.name)
            if match is None:
                continue
            gate_id = int(match.group(1))
        record["gate_id"] = gate_id
        written.append(write_gate_record(
            root, gate_id, record, run_root=run_root, model=model,
        ))
    return written


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
