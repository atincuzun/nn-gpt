#!/usr/bin/env python3
"""Visualize completed A-cycle neural architecture search results.

The script produces three figures:

1. Per-cycle mean accuracy with a 95% Student-t confidence interval and a
   trailing rolling mean.
2. Per-cycle maximum accuracy with the cumulative high-watermark trajectory.
3. Accuracy versus total parameter count for every detected trained model,
   colored by its A-cycle, with the size/accuracy Pareto frontier overlaid.

Parameter counting imports and instantiates the archived, locally generated
``new_lemur/nn/*.py`` model files on CPU.  It does not run a forward pass or
train a model.  Only run this script on generated model code that you trust.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = (
    SCRIPT_DIR / "out" / "moe_gate_ds_v2lite_fourbits_last_total"
)
CYCLE_FILE_RE = re.compile(r"cycle_results_A(\d+)\.json$")
A_DIR_RE = re.compile(r"A(\d+)$")
B_DIR_RE = re.compile(r"B(\d+)$")
MODEL_ID_PREFIX = "moe-gate-cycle-"
EPSILON = 1e-12

# Defaults cover the LEMUR interface and the generated models in this run.
# Recorded evaluation values always override these fallbacks.
DEFAULT_MODEL_PARAMETERS: dict[str, Any] = {
    "batch": 32,
    "batch_size": 32,
    "dropout": 0.2,
    "epoch": 1,
    "epoch_max": 1,
    "lr": 0.01,
    "momentum": 0.9,
    "norm_eps": 1e-6,
    "norm_std": 0.02,
    "stochastic_depth_prob": 0.1,
    "weight_decay": 0.0,
}

# Two-sided 95% Student-t critical values (quantile 0.975).
T_CRITICAL_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    31: 2.040,
    32: 2.037,
    33: 2.035,
    34: 2.032,
    35: 2.030,
    36: 2.028,
    37: 2.026,
    38: 2.024,
    39: 2.023,
    40: 2.021,
}


class Reporter:
    """Collect warnings for both terminal output and the summary JSON."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"warning: {message}", file=sys.stderr)


@dataclass
class CycleMetrics:
    cycle: int
    source_path: Path
    generated: int
    reported_trained: int
    reported_mean: float
    reported_max: float
    detected_trained: int = 0
    sample_accuracies: list[float] = field(default_factory=list)
    sample_source: str = "none"
    sample_mean: float | None = None
    sample_stdev: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    rolling_mean: float | None = None


@dataclass
class EvalArtifact:
    model_id: str
    cycle: int
    candidate: str
    model_dir: Path
    generated_code_path: Path
    accuracy: float | None
    parameters: dict[str, Any]
    dataset: str
    task: str


@dataclass
class TrainObservation:
    model_id: str
    accuracy: float
    stat_path: Path
    row: dict[str, Any]


@dataclass
class TrainedModel:
    model_id: str
    cycle: int
    candidate: str
    accuracy: float
    stat_path: Path
    code_path: Path | None
    dataset: str
    task: str
    parameters: dict[str, Any]
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    parameter_error: str | None = None
    parameter_count_cached: bool = False
    is_pareto: bool = False


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create accuracy trajectory, high-watermark, and parameter/accuracy "
            "Pareto plots from completed cycle_results_A*.json files."
        )
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=str(DEFAULT_EXPERIMENT),
        help=(
            "Experiment directory, or its nngpt subdirectory. "
            f"Default: {DEFAULT_EXPERIMENT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <experiment>/nngpt/cycle_visualizations)",
    )
    parser.add_argument(
        "--rolling-window",
        type=_positive_int,
        default=3,
        help="Trailing cycle window for the rolling mean (default: 3)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        choices=(0.95,),
        help="Confidence level; currently 0.95 is supported (default: 0.95)",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=180,
        help="Output image resolution (default: 180)",
    )
    parser.add_argument(
        "--recount",
        action="store_true",
        help="Ignore cached parameter counts and reconstruct every model",
    )
    return parser.parse_args(argv)


def resolve_nngpt_dir(experiment: Path) -> Path:
    path = experiment.expanduser().resolve()
    candidates = (path, path / "nngpt")
    for candidate in candidates:
        if candidate.is_dir() and any(
            CYCLE_FILE_RE.fullmatch(child.name) for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(
        f"No cycle_results_A*.json files found in {path} or {path / 'nngpt'}"
    )


def read_json(path: Path, reporter: Reporter) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reporter.warn(f"could not read {path}: {type(exc).__name__}: {exc}")
        return None


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def accuracy_fraction(value: Any) -> float | None:
    """Return an accuracy in [0, 1], accepting either fractions or percentages."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    if 1.0 < result <= 100.0:
        result /= 100.0
    if result < 0.0 or result > 1.0:
        return None
    return result


def canonical_model_key(model_id: str) -> str:
    value = str(model_id).strip()
    if value.startswith(MODEL_ID_PREFIX):
        return value[len(MODEL_ID_PREFIX) :]
    return value


def load_cycle_summaries(nngpt_dir: Path, reporter: Reporter) -> list[CycleMetrics]:
    summaries: list[CycleMetrics] = []
    cycle_paths: list[tuple[int, Path]] = []
    for child in nngpt_dir.iterdir():
        match = CYCLE_FILE_RE.fullmatch(child.name)
        if child.is_file() and match:
            cycle_paths.append((int(match.group(1)), child))

    for filename_cycle, path in sorted(cycle_paths):
        payload = read_json(path, reporter)
        if not isinstance(payload, Mapping):
            continue
        cycle = as_int(payload.get("cycle"), filename_cycle)
        if cycle != filename_cycle:
            reporter.warn(
                f"{path.name} says cycle={cycle}; using filename cycle A{filename_cycle}"
            )
            cycle = filename_cycle
        generation = payload.get("generation")
        evaluation = payload.get("evaluation")
        generation = generation if isinstance(generation, Mapping) else {}
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        mean_accuracy = accuracy_fraction(evaluation.get("avg_accuracy"))
        max_accuracy = accuracy_fraction(evaluation.get("best_accuracy"))
        if mean_accuracy is None or max_accuracy is None:
            reporter.warn(
                f"skipping {path.name}: evaluation.avg_accuracy or "
                "evaluation.best_accuracy is missing"
            )
            continue
        summaries.append(
            CycleMetrics(
                cycle=cycle,
                source_path=path,
                generated=as_int(generation.get("total_generated")),
                reported_trained=as_int(evaluation.get("models_trained")),
                reported_mean=mean_accuracy,
                reported_max=max_accuracy,
            )
        )

    if not summaries:
        raise ValueError(f"No usable A-cycle summaries found in {nngpt_dir}")
    return summaries


def load_eval_artifacts(
    nngpt_dir: Path, reporter: Reporter
) -> tuple[dict[str, EvalArtifact], dict[str, EvalArtifact]]:
    exact: dict[str, EvalArtifact] = {}
    canonical: dict[str, EvalArtifact] = {}
    pattern = "gate_candidates/*/epochs/A*/synth_nn/B*/eval_info.json"

    for path in sorted(nngpt_dir.glob(pattern)):
        a_match = A_DIR_RE.fullmatch(path.parents[2].name)
        b_match = B_DIR_RE.fullmatch(path.parent.name)
        if not a_match or not b_match:
            continue
        payload = read_json(path, reporter)
        if not isinstance(payload, Mapping):
            continue
        eval_results = payload.get("eval_results")
        eval_args = payload.get("eval_args")
        eval_results = eval_results if isinstance(eval_results, Mapping) else {}
        eval_args = eval_args if isinstance(eval_args, Mapping) else {}
        model_id = str(eval_results.get("checksum") or "").strip()
        if not model_id:
            reporter.warn(f"{path} has no eval_results.checksum; skipping")
            continue
        raw_parameters = eval_args.get("prm")
        parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
        artifact = EvalArtifact(
            model_id=model_id,
            cycle=int(a_match.group(1)),
            candidate=f"B{int(b_match.group(1))}",
            model_dir=path.parent,
            generated_code_path=path.parent / "new_nn.py",
            accuracy=accuracy_fraction(eval_results.get("accuracy")),
            parameters=parameters,
            dataset=str(eval_args.get("dataset") or "cifar-10"),
            task=str(eval_args.get("task") or "img-classification"),
        )
        previous = exact.get(model_id)
        if previous is not None and (
            previous.cycle != artifact.cycle
            or previous.candidate != artifact.candidate
        ):
            reporter.warn(
                f"model ID {model_id} maps to both A{previous.cycle}/"
                f"{previous.candidate} and A{artifact.cycle}/{artifact.candidate}; "
                "using the first mapping"
            )
            continue
        exact[model_id] = artifact
        key = canonical_model_key(model_id)
        canonical_previous = canonical.get(key)
        if canonical_previous is None:
            canonical[key] = artifact
        elif canonical_previous.model_id != model_id:
            canonical.pop(key, None)

    return exact, canonical


def iter_result_rows(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if "model" in payload or "accuracy" in payload:
            yield dict(payload)
            return
        for key in ("results", "records", "models"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for row in nested:
                    if isinstance(row, Mapping):
                        yield dict(row)
                return
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping):
                yield dict(row)


def infer_model_id_from_stat_path(path: Path) -> str | None:
    directory_name = path.parent.name
    marker = f"_{MODEL_ID_PREFIX}"
    if marker in directory_name:
        return directory_name.split(marker, 1)[1].join((MODEL_ID_PREFIX, ""))
    match = re.search(r"(moe-gate-cycle-[0-9a-fA-F]+)", directory_name)
    return match.group(1) if match else None


def load_train_observations(
    nngpt_dir: Path, reporter: Reporter
) -> dict[str, TrainObservation]:
    train_dir = nngpt_dir / "new_lemur" / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Trained-model statistics directory not found: {train_dir}")

    observations: dict[str, TrainObservation] = {}
    for path in sorted(train_dir.glob("*/*.json")):
        payload = read_json(path, reporter)
        if payload is None:
            continue
        for row in iter_result_rows(payload):
            model_id = str(row.get("model") or "").strip()
            if not model_id:
                model_id = infer_model_id_from_stat_path(path) or ""
            accuracy = accuracy_fraction(row.get("accuracy"))
            if accuracy is None:
                accuracy = accuracy_fraction(row.get("metric_acc"))
            train_stat = row.get("train_stat")
            if accuracy is None and isinstance(train_stat, Mapping):
                accuracy = accuracy_fraction(train_stat.get("best_accuracy"))
            if not model_id or accuracy is None:
                continue
            observation = TrainObservation(
                model_id=model_id,
                accuracy=accuracy,
                stat_path=path,
                row=row,
            )
            previous = observations.get(model_id)
            # If a model has multiple epoch records, its best reported accuracy
            # is the value represented in the cycle's best/average summaries.
            if previous is None or observation.accuracy > previous.accuracy:
                observations[model_id] = observation
    return observations


def find_artifact(
    model_id: str,
    exact: Mapping[str, EvalArtifact],
    canonical: Mapping[str, EvalArtifact],
) -> EvalArtifact | None:
    return exact.get(model_id) or canonical.get(canonical_model_key(model_id))


def merge_model_parameters(
    observation: TrainObservation, artifact: EvalArtifact
) -> dict[str, Any]:
    parameters = dict(DEFAULT_MODEL_PARAMETERS)

    def merge(source: Any) -> None:
        if not isinstance(source, Mapping):
            return
        for key, value in source.items():
            if key in {"model", "task", "dataset", "metric", "train_stat"}:
                continue
            parameters[str(key)] = value
        nested = source.get("train_stat")
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                parameters.setdefault(str(key), value)

    merge(observation.row)
    merge(artifact.parameters)
    return parameters


def build_trained_models(
    nngpt_dir: Path,
    completed_cycles: set[int],
    observations: Mapping[str, TrainObservation],
    exact_artifacts: Mapping[str, EvalArtifact],
    canonical_artifacts: Mapping[str, EvalArtifact],
    reporter: Reporter,
) -> tuple[list[TrainedModel], int]:
    models: list[TrainedModel] = []
    incomplete_cycle_models = 0
    archive_dir = nngpt_dir / "new_lemur" / "nn"

    for observation in observations.values():
        artifact = find_artifact(
            observation.model_id, exact_artifacts, canonical_artifacts
        )
        if artifact is None:
            reporter.warn(
                f"trained model {observation.model_id} has no A/B eval_info mapping; "
                "it cannot be placed on the Pareto plot"
            )
            continue
        if artifact.cycle not in completed_cycles:
            incomplete_cycle_models += 1
            continue
        archived_code = archive_dir / f"{observation.model_id}.py"
        if archived_code.is_file():
            code_path: Path | None = archived_code
        elif artifact.generated_code_path.is_file():
            code_path = artifact.generated_code_path
            reporter.warn(
                f"archived code missing for {observation.model_id}; using "
                f"{artifact.generated_code_path}"
            )
        else:
            code_path = None
        if (
            artifact.accuracy is not None
            and abs(artifact.accuracy - observation.accuracy) > 1e-8
        ):
            reporter.warn(
                f"accuracy mismatch for {observation.model_id}: train record="
                f"{observation.accuracy:.6f}, eval_info={artifact.accuracy:.6f}; "
                "using the train record"
            )
        models.append(
            TrainedModel(
                model_id=observation.model_id,
                cycle=artifact.cycle,
                candidate=artifact.candidate,
                accuracy=observation.accuracy,
                stat_path=observation.stat_path,
                code_path=code_path,
                dataset=str(observation.row.get("dataset") or artifact.dataset),
                task=str(observation.row.get("task") or artifact.task),
                parameters=merge_model_parameters(observation, artifact),
            )
        )

    models.sort(key=lambda model: (model.cycle, int(model.candidate[1:])))
    return models, incomplete_cycle_models


def load_feedback_accuracies(
    nngpt_dir: Path, cycle: int, reporter: Reporter
) -> list[float]:
    values: list[float] = []
    pattern = f"gate_candidates/*/epochs/A{cycle}/nas_feedback.json"
    for path in sorted(nngpt_dir.glob(pattern)):
        payload = read_json(path, reporter)
        if not isinstance(payload, Mapping):
            continue
        outcomes = payload.get("candidate_outcomes")
        if not isinstance(outcomes, list):
            continue
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                continue
            accuracy = accuracy_fraction(outcome.get("accuracy"))
            if accuracy is not None:
                values.append(accuracy)
    return values


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom in T_CRITICAL_975:
        return T_CRITICAL_975[degrees_of_freedom]
    if degrees_of_freedom <= 0:
        return 0.0
    # The experiment generates at most 40 models per cycle.  This asymptotic
    # fallback supports other runs with larger generation batches.
    if degrees_of_freedom < 60:
        return 2.000
    if degrees_of_freedom < 120:
        return 1.980
    return 1.960


def confidence_interval_95(
    center: float, samples: Sequence[float]
) -> tuple[float, float, float | None, float | None]:
    if not samples:
        return center, center, None, None
    sample_mean = statistics.fmean(samples)
    if len(samples) == 1:
        return center, center, sample_mean, 0.0
    sample_stdev = statistics.stdev(samples)
    standard_error = sample_stdev / math.sqrt(len(samples))
    margin = t_critical_975(len(samples) - 1) * standard_error
    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
        sample_mean,
        sample_stdev,
    )


def enrich_cycle_metrics(
    cycles: list[CycleMetrics],
    models: Sequence[TrainedModel],
    nngpt_dir: Path,
    rolling_window: int,
    reporter: Reporter,
) -> None:
    models_by_cycle: dict[int, list[TrainedModel]] = {}
    for model in models:
        models_by_cycle.setdefault(model.cycle, []).append(model)

    for cycle in cycles:
        detected = models_by_cycle.get(cycle.cycle, [])
        detected_accuracies = [model.accuracy for model in detected]
        feedback_accuracies = load_feedback_accuracies(
            nngpt_dir, cycle.cycle, reporter
        )
        cycle.detected_trained = len(detected)

        if len(detected_accuracies) == cycle.reported_trained:
            samples = detected_accuracies
            source = "new_lemur/train"
        elif len(feedback_accuracies) == cycle.reported_trained:
            samples = feedback_accuracies
            source = "nas_feedback.json"
        elif detected_accuracies:
            samples = detected_accuracies
            source = "new_lemur/train (count mismatch)"
        else:
            samples = feedback_accuracies
            source = "nas_feedback.json (count mismatch)"

        cycle.sample_accuracies = samples
        cycle.sample_source = source
        lower, upper, sample_mean, sample_stdev = confidence_interval_95(
            cycle.reported_mean, samples
        )
        cycle.ci_lower = lower
        cycle.ci_upper = upper
        cycle.sample_mean = sample_mean
        cycle.sample_stdev = sample_stdev

        if cycle.detected_trained != cycle.reported_trained:
            reporter.warn(
                f"A{cycle.cycle}: cycle summary reports {cycle.reported_trained} "
                f"trained models, but {cycle.detected_trained} completed train "
                "records were mapped"
            )
        if samples:
            measured_mean = statistics.fmean(samples)
            measured_max = max(samples)
            if abs(measured_mean - cycle.reported_mean) > 1e-8:
                reporter.warn(
                    f"A{cycle.cycle}: reported mean {cycle.reported_mean:.8f} "
                    f"differs from {source} mean {measured_mean:.8f}"
                )
            if abs(measured_max - cycle.reported_max) > 1e-8:
                reporter.warn(
                    f"A{cycle.cycle}: reported max {cycle.reported_max:.8f} "
                    f"differs from {source} max {measured_max:.8f}"
                )

    means = [cycle.reported_mean for cycle in cycles]
    for index, cycle in enumerate(cycles):
        start = max(0, index - rolling_window + 1)
        cycle.rolling_mean = statistics.fmean(means[start : index + 1])


def dataset_shapes(dataset: str) -> tuple[tuple[int, int, int, int], tuple[int, ...]]:
    name = dataset.lower().replace("_", "-").strip()
    if "cifar-100" in name or "cifar100" in name:
        return (1, 3, 32, 32), (100,)
    if "cifar-10" in name or "cifar10" in name:
        return (1, 3, 32, 32), (10,)
    if "fashion-mnist" in name or "fashionmnist" in name or "mnist" in name:
        return (1, 1, 28, 28), (10,)
    if "imagenette" in name:
        return (1, 3, 224, 224), (10,)
    if "imagenet" in name:
        return (1, 3, 224, 224), (1000,)
    return (1, 3, 224, 224), (10,)


def load_parameter_cache(path: Path, reporter: Reporter) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path, reporter)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return {}
    models = payload.get("models")
    return dict(models) if isinstance(models, Mapping) else {}


def write_parameter_cache(path: Path, cache: Mapping[str, Any]) -> None:
    payload = {"schema_version": 1, "models": dict(sorted(cache.items()))}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parameter_fingerprint(model: TrainedModel) -> str:
    if model.code_path is None:
        return ""
    in_shape, out_shape = dataset_shapes(model.dataset)
    digest = hashlib.sha256()
    digest.update(model.code_path.read_bytes())
    digest.update(
        json.dumps(
            {
                "in_shape": in_shape,
                "out_shape": out_shape,
                "parameters": model.parameters,
            },
            sort_keys=True,
            default=repr,
        ).encode("utf-8")
    )
    return digest.hexdigest()


@contextmanager
def torchvision_counting_compatibility(torch: Any) -> Iterator[str]:
    """Provide parameter-equivalent torchvision ops if torchvision is absent.

    Conv2dNormActivation contains only the convolution and optional normalization
    parameters represented below. Permute and StochasticDepth are parameterless,
    so identity stochastic depth is sufficient when no forward pass is performed.
    """
    try:
        from torchvision.ops.misc import Conv2dNormActivation as _native_conv  # noqa: F401
        from torchvision.ops.misc import Permute as _native_permute  # noqa: F401
        from torchvision.ops.stochastic_depth import StochasticDepth as _native_sd  # noqa: F401
    except Exception:
        existing_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "torchvision" or name.startswith("torchvision.")
        }
        for name in tuple(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)

        nn = torch.nn

        class Permute(nn.Module):
            def __init__(self, dims: Sequence[int]) -> None:
                super().__init__()
                self.dims = tuple(dims)

            def forward(self, tensor: Any) -> Any:
                return torch.permute(tensor, self.dims)

        class StochasticDepth(nn.Module):
            def __init__(self, probability: float, mode: str) -> None:
                super().__init__()
                self.probability = probability
                self.mode = mode

            def forward(self, tensor: Any) -> Any:
                return tensor

        class Conv2dNormActivation(nn.Sequential):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int = 3,
                stride: int = 1,
                padding: int | None = None,
                groups: int = 1,
                norm_layer: Any = nn.BatchNorm2d,
                activation_layer: Any = nn.ReLU,
                dilation: int = 1,
                inplace: bool = True,
                bias: bool | None = None,
                **_: Any,
            ) -> None:
                if padding is None:
                    padding = (kernel_size - 1) // 2 * dilation
                if bias is None:
                    bias = norm_layer is None
                layers: list[Any] = [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        groups,
                        bias,
                    )
                ]
                if norm_layer is not None:
                    layers.append(norm_layer(out_channels))
                if activation_layer is not None:
                    try:
                        layers.append(activation_layer(inplace=inplace))
                    except TypeError:
                        layers.append(activation_layer())
                super().__init__(*layers)

        module_names = (
            "torchvision",
            "torchvision.ops",
            "torchvision.ops.misc",
            "torchvision.ops.stochastic_depth",
        )
        compatibility_modules = {
            name: types.ModuleType(name) for name in module_names
        }
        for module in compatibility_modules.values():
            module.__path__ = []  # type: ignore[attr-defined]
        compatibility_modules["torchvision"].ops = compatibility_modules[
            "torchvision.ops"
        ]
        compatibility_modules["torchvision.ops"].misc = compatibility_modules[
            "torchvision.ops.misc"
        ]
        compatibility_modules[
            "torchvision.ops"
        ].stochastic_depth = compatibility_modules[
            "torchvision.ops.stochastic_depth"
        ]
        compatibility_modules[
            "torchvision.ops.misc"
        ].Conv2dNormActivation = Conv2dNormActivation
        compatibility_modules["torchvision.ops.misc"].Permute = Permute
        compatibility_modules[
            "torchvision.ops.stochastic_depth"
        ].StochasticDepth = StochasticDepth
        sys.modules.update(compatibility_modules)
        try:
            yield "compatibility shim"
        finally:
            for name in tuple(sys.modules):
                if name == "torchvision" or name.startswith("torchvision."):
                    sys.modules.pop(name, None)
            sys.modules.update(existing_modules)
    else:
        yield "native torchvision"


def import_and_count_parameters(model: TrainedModel, torch: Any) -> tuple[int, int]:
    if model.code_path is None:
        raise FileNotFoundError("no archived or generated model code was found")
    module_name = (
        "_moe_cycle_parameter_count_"
        + hashlib.sha1(
            f"{model.model_id}:{model.code_path}".encode("utf-8")
        ).hexdigest()
    )
    spec = importlib.util.spec_from_file_location(module_name, model.code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {model.code_path}")
    module = importlib.util.module_from_spec(spec)
    model_instance = None
    inserted_path = False
    parent_string = str(model.code_path.parent)
    if parent_string not in sys.path:
        sys.path.insert(0, parent_string)
        inserted_path = True
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        net_class = getattr(module, "Net")
        in_shape, out_shape = dataset_shapes(model.dataset)
        with torch.no_grad():
            model_instance = net_class(
                in_shape,
                out_shape,
                model.parameters,
                torch.device("cpu"),
            )
        total = sum(parameter.numel() for parameter in model_instance.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model_instance.parameters()
            if parameter.requires_grad
        )
        return int(total), int(trainable)
    finally:
        if model_instance is not None:
            del model_instance
        sys.modules.pop(module_name, None)
        if inserted_path:
            try:
                sys.path.remove(parent_string)
            except ValueError:
                pass
        gc.collect()


def count_all_model_parameters(
    models: Sequence[TrainedModel],
    cache_path: Path,
    recount: bool,
    reporter: Reporter,
) -> tuple[int, str]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to reconstruct generated models and count parameters"
        ) from exc

    cache = {} if recount else load_parameter_cache(cache_path, reporter)
    updated_cache = dict(cache)
    counted = 0
    with torchvision_counting_compatibility(torch) as torchvision_mode:
        print(f"Parameter counting: {torchvision_mode}")
        for index, model in enumerate(models, start=1):
            try:
                fingerprint = parameter_fingerprint(model)
            except OSError as exc:
                model.parameter_error = f"{type(exc).__name__}: {exc}"
                reporter.warn(
                    f"could not fingerprint {model.model_id}: {model.parameter_error}"
                )
                continue

            cached = cache.get(model.model_id)
            if (
                not recount
                and isinstance(cached, Mapping)
                and cached.get("fingerprint") == fingerprint
                and as_int(cached.get("total_parameters")) > 0
            ):
                model.total_parameters = as_int(cached.get("total_parameters"))
                model.trainable_parameters = as_int(
                    cached.get("trainable_parameters")
                )
                model.parameter_count_cached = True
                counted += 1
                continue

            try:
                total, trainable = import_and_count_parameters(model, torch)
                if total <= 0:
                    raise ValueError("model contains no registered parameters")
                model.total_parameters = total
                model.trainable_parameters = trainable
                model.parameter_count_cached = False
                model.parameter_error = None
                counted += 1
                updated_cache[model.model_id] = {
                    "fingerprint": fingerprint,
                    "total_parameters": total,
                    "trainable_parameters": trainable,
                    "code_path": str(model.code_path),
                }
            except Exception as exc:  # generated model imports are heterogeneous
                model.parameter_error = f"{type(exc).__name__}: {exc}"
                reporter.warn(
                    f"parameter count failed for A{model.cycle}/{model.candidate} "
                    f"({model.model_id}): {model.parameter_error}"
                )

            if index % 20 == 0 or index == len(models):
                print(
                    f"  reconstructed {index}/{len(models)} models "
                    f"({counted} parameter counts available)"
                )

    write_parameter_cache(cache_path, updated_cache)
    return counted, torchvision_mode


def mark_pareto_frontier(models: Sequence[TrainedModel]) -> list[TrainedModel]:
    for model in models:
        model.is_pareto = False
    eligible = [
        model
        for model in models
        if model.total_parameters is not None and model.total_parameters > 0
    ]
    groups: dict[int, list[TrainedModel]] = {}
    for model in eligible:
        groups.setdefault(int(model.total_parameters), []).append(model)

    frontier: list[TrainedModel] = []
    best_accuracy_at_smaller_size = -math.inf
    for parameter_count in sorted(groups):
        group = groups[parameter_count]
        group_best_accuracy = max(model.accuracy for model in group)
        if group_best_accuracy > best_accuracy_at_smaller_size + EPSILON:
            tied = [
                model
                for model in group
                if abs(model.accuracy - group_best_accuracy) <= EPSILON
            ]
            for model in tied:
                model.is_pareto = True
            representative = min(
                tied, key=lambda model: (model.cycle, int(model.candidate[1:]))
            )
            frontier.append(representative)
            best_accuracy_at_smaller_size = group_best_accuracy
    return frontier


def require_plotting() -> tuple[Any, Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import ticker
        from matplotlib.colors import Normalize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required. Install the project requirements or run "
            "`python3 -m pip install matplotlib`."
        ) from exc
    return matplotlib, plt, ticker, Normalize


def apply_plot_style(plt: Any) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


def accuracy_limits(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = max(0.0, math.floor((min(values) - 0.05) * 20.0) / 20.0)
    high = min(1.0, math.ceil((max(values) + 0.05) * 20.0) / 20.0)
    if high - low < 0.15:
        midpoint = (high + low) / 2.0
        low = max(0.0, midpoint - 0.075)
        high = min(1.0, midpoint + 0.075)
    return low, high


def plot_mean_trajectory(
    cycles: Sequence[CycleMetrics],
    output_path: Path,
    rolling_window: int,
    dpi: int,
    experiment_name: str,
    plt: Any,
    ticker: Any,
) -> None:
    x = [cycle.cycle for cycle in cycles]
    means = [cycle.reported_mean for cycle in cycles]
    lower = [cycle.ci_lower or cycle.reported_mean for cycle in cycles]
    upper = [cycle.ci_upper or cycle.reported_mean for cycle in cycles]
    rolling = [cycle.rolling_mean or cycle.reported_mean for cycle in cycles]

    fig, axis = plt.subplots(figsize=(13.5, 7.2))
    axis.fill_between(
        x,
        lower,
        upper,
        color="#f59e0b",
        alpha=0.20,
        label="95% Student-t CI",
        zorder=1,
    )
    axis.errorbar(
        x,
        means,
        yerr=(
            [mean - low for mean, low in zip(means, lower)],
            [high - mean for mean, high in zip(means, upper)],
        ),
        fmt="o-",
        color="#d97706",
        ecolor="#f59e0b",
        capsize=4,
        linewidth=1.8,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Reported cycle mean",
        zorder=3,
    )
    axis.plot(
        x,
        rolling,
        color="#1d4ed8",
        linewidth=3.0,
        marker="s",
        markersize=5,
        label=f"Trailing rolling mean (window={rolling_window})",
        zorder=4,
    )

    tick_labels = []
    for cycle in cycles:
        trained = cycle.detected_trained or cycle.reported_trained
        tick_labels.append(f"A{cycle.cycle}\n{trained}/{cycle.generated}")
    axis.set_xticks(x, tick_labels)
    axis.set_xlabel("Generation cycle (detected trained / generated)")
    axis.set_ylabel("Accuracy")
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axis.set_ylim(*accuracy_limits(lower + upper + rolling))
    axis.set_title(
        f"{experiment_name}\nMean accuracy trajectory and uncertainty",
        fontsize=15,
        weight="bold",
    )
    axis.grid(True, which="major", linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.995,
        0.015,
        "CI uses the trained-model accuracies within each A-cycle",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_high_watermark(
    cycles: Sequence[CycleMetrics],
    output_path: Path,
    dpi: int,
    experiment_name: str,
    plt: Any,
    ticker: Any,
) -> None:
    x = [cycle.cycle for cycle in cycles]
    maxima = [cycle.reported_max for cycle in cycles]
    watermark: list[float] = []
    record_indices: list[int] = []
    running = -math.inf
    for index, value in enumerate(maxima):
        if value > running + EPSILON:
            running = value
            record_indices.append(index)
        watermark.append(running)

    best_index = max(range(len(cycles)), key=lambda index: maxima[index])
    best_cycle = cycles[best_index]
    fig, axis = plt.subplots(figsize=(13.5, 7.2))
    axis.plot(
        x,
        maxima,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.7,
        marker="o",
        markersize=6,
        markerfacecolor="#f97316",
        markeredgecolor="white",
        label="Best accuracy in each cycle",
        zorder=2,
    )
    axis.plot(
        x,
        watermark,
        color="#047857",
        linewidth=3.2,
        drawstyle="steps-post",
        label="Cumulative high watermark",
        zorder=3,
    )
    axis.scatter(
        [x[index] for index in record_indices],
        [maxima[index] for index in record_indices],
        marker="D",
        s=75,
        color="#059669",
        edgecolor="white",
        linewidth=0.9,
        label="New record",
        zorder=4,
    )
    axis.scatter(
        [best_cycle.cycle],
        [best_cycle.reported_max],
        marker="*",
        s=310,
        color="#facc15",
        edgecolor="#991b1b",
        linewidth=1.2,
        label="Global best cycle",
        zorder=5,
    )
    axis.annotate(
        f"Highest: A{best_cycle.cycle}\n{best_cycle.reported_max:.2%}",
        xy=(best_cycle.cycle, best_cycle.reported_max),
        xytext=(12, 20),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#f97316"},
        arrowprops={"arrowstyle": "->", "color": "#9a3412"},
    )
    axis.set_xticks(x, [f"A{cycle}" for cycle in x])
    axis.set_xlabel("Generation cycle")
    axis.set_ylabel("Best reported accuracy")
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axis.set_ylim(*accuracy_limits(maxima + watermark))
    axis.set_title(
        f"{experiment_name}\nMaximum accuracy and cumulative high watermark",
        fontsize=15,
        weight="bold",
    )
    axis.grid(True, which="major", linestyle="--", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def human_parameter_count(value: float, _position: Any = None) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def plot_parameter_pareto(
    models: Sequence[TrainedModel],
    frontier: Sequence[TrainedModel],
    output_path: Path,
    dpi: int,
    experiment_name: str,
    plt: Any,
    ticker: Any,
    Normalize: Any,
) -> None:
    eligible = [
        model
        for model in models
        if model.total_parameters is not None and model.total_parameters > 0
    ]
    if not eligible:
        raise ValueError("No model parameter counts are available for the Pareto plot")

    x = [int(model.total_parameters or 0) for model in eligible]
    y = [model.accuracy for model in eligible]
    colors = [model.cycle for model in eligible]
    cycle_values = sorted({model.cycle for model in eligible})
    normalization = Normalize(
        vmin=min(cycle_values) - 0.5,
        vmax=max(cycle_values) + 0.5,
    )
    color_map = plt.get_cmap("turbo")

    fig, axis = plt.subplots(figsize=(14.2, 8.2))
    scatter = axis.scatter(
        x,
        y,
        c=colors,
        cmap=color_map,
        norm=normalization,
        s=58,
        alpha=0.80,
        edgecolors="white",
        linewidths=0.55,
        label="Successfully trained model",
        zorder=2,
    )
    frontier_x = [int(model.total_parameters or 0) for model in frontier]
    frontier_y = [model.accuracy for model in frontier]
    axis.plot(
        frontier_x,
        frontier_y,
        color="#111827",
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Pareto frontier (smaller + more accurate)",
        zorder=4,
    )

    for index, model in enumerate(frontier):
        offset_y = 9 if index % 2 == 0 else -15
        axis.annotate(
            f"A{model.cycle}/{model.candidate}",
            xy=(int(model.total_parameters or 0), model.accuracy),
            xytext=(4, offset_y),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
            zorder=5,
        )

    best_model = max(eligible, key=lambda model: model.accuracy)
    axis.scatter(
        [int(best_model.total_parameters or 0)],
        [best_model.accuracy],
        marker="*",
        s=260,
        facecolor="#fde047",
        edgecolor="#991b1b",
        linewidth=1.2,
        label=(
            f"Highest accuracy: A{best_model.cycle}/{best_model.candidate} "
            f"({best_model.accuracy:.2%})"
        ),
        zorder=6,
    )

    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(human_parameter_count))
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axis.set_xlabel("Total model parameters (log scale; fewer = more efficient)")
    axis.set_ylabel("Reported accuracy")
    axis.set_ylim(*accuracy_limits(y))
    axis.set_title(
        f"{experiment_name}\nModel efficiency and accuracy Pareto frontier",
        fontsize=15,
        weight="bold",
    )
    axis.grid(True, which="major", linestyle="--", alpha=0.35)
    axis.grid(True, which="minor", axis="x", linestyle=":", alpha=0.18)
    axis.legend(loc="lower right", frameon=True)
    axis.text(
        0.012,
        0.985,
        (
            f"{len(eligible)} trained models with exact parameter counts\n"
            "Pareto-optimal: no other model is both smaller and at least as accurate"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "alpha": 0.9},
    )
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.015)
    colorbar.set_label("Generation cycle")
    colorbar.set_ticks(cycle_values)
    colorbar.set_ticklabels([f"A{cycle}" for cycle in cycle_values])
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_cycle_csv(path: Path, cycles: Sequence[CycleMetrics]) -> None:
    fieldnames = (
        "cycle",
        "cycle_label",
        "generated",
        "reported_trained",
        "detected_trained",
        "reported_success_rate",
        "reported_mean_accuracy",
        "reported_max_accuracy",
        "sample_count",
        "sample_source",
        "sample_mean_accuracy",
        "sample_stdev_accuracy",
        "ci_95_lower",
        "ci_95_upper",
        "rolling_mean_accuracy",
        "source_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cycle in cycles:
            writer.writerow(
                {
                    "cycle": cycle.cycle,
                    "cycle_label": f"A{cycle.cycle}",
                    "generated": cycle.generated,
                    "reported_trained": cycle.reported_trained,
                    "detected_trained": cycle.detected_trained,
                    "reported_success_rate": (
                        cycle.reported_trained / cycle.generated
                        if cycle.generated
                        else ""
                    ),
                    "reported_mean_accuracy": cycle.reported_mean,
                    "reported_max_accuracy": cycle.reported_max,
                    "sample_count": len(cycle.sample_accuracies),
                    "sample_source": cycle.sample_source,
                    "sample_mean_accuracy": cycle.sample_mean,
                    "sample_stdev_accuracy": cycle.sample_stdev,
                    "ci_95_lower": cycle.ci_lower,
                    "ci_95_upper": cycle.ci_upper,
                    "rolling_mean_accuracy": cycle.rolling_mean,
                    "source_path": str(cycle.source_path),
                }
            )


def write_models_csv(path: Path, models: Sequence[TrainedModel]) -> None:
    fieldnames = (
        "cycle",
        "cycle_label",
        "candidate",
        "model_id",
        "accuracy",
        "total_parameters",
        "trainable_parameters",
        "is_pareto",
        "parameter_count_cached",
        "parameter_error",
        "dataset",
        "task",
        "code_path",
        "stat_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            writer.writerow(
                {
                    "cycle": model.cycle,
                    "cycle_label": f"A{model.cycle}",
                    "candidate": model.candidate,
                    "model_id": model.model_id,
                    "accuracy": model.accuracy,
                    "total_parameters": model.total_parameters or "",
                    "trainable_parameters": model.trainable_parameters or "",
                    "is_pareto": model.is_pareto,
                    "parameter_count_cached": model.parameter_count_cached,
                    "parameter_error": model.parameter_error or "",
                    "dataset": model.dataset,
                    "task": model.task,
                    "code_path": str(model.code_path or ""),
                    "stat_path": str(model.stat_path),
                }
            )


def experiment_display_name(nngpt_dir: Path) -> str:
    return nngpt_dir.parent.name if nngpt_dir.name == "nngpt" else nngpt_dir.name


def run(args: argparse.Namespace) -> int:
    reporter = Reporter()
    nngpt_dir = resolve_nngpt_dir(Path(args.experiment))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else nngpt_dir / "cycle_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import plotting dependencies before expensive model reconstruction so a
    # missing environment dependency fails immediately and clearly.
    _matplotlib, plt, ticker, Normalize = require_plotting()
    apply_plot_style(plt)

    cycles = load_cycle_summaries(nngpt_dir, reporter)
    completed_cycles = {cycle.cycle for cycle in cycles}
    exact_artifacts, canonical_artifacts = load_eval_artifacts(
        nngpt_dir, reporter
    )
    observations = load_train_observations(nngpt_dir, reporter)
    models, incomplete_cycle_models = build_trained_models(
        nngpt_dir,
        completed_cycles,
        observations,
        exact_artifacts,
        canonical_artifacts,
        reporter,
    )
    enrich_cycle_metrics(
        cycles,
        models,
        nngpt_dir,
        args.rolling_window,
        reporter,
    )

    cache_path = output_dir / "parameter_count_cache.json"
    counted_models, torchvision_mode = count_all_model_parameters(
        models, cache_path, args.recount, reporter
    )
    frontier = mark_pareto_frontier(models)

    mean_plot = output_dir / "01_mean_accuracy_trajectory.png"
    watermark_plot = output_dir / "02_accuracy_high_watermark.png"
    pareto_plot = output_dir / "03_accuracy_parameter_pareto.png"
    cycle_csv = output_dir / "cycle_metrics.csv"
    model_csv = output_dir / "trained_models.csv"
    summary_json = output_dir / "visualization_summary.json"
    display_name = experiment_display_name(nngpt_dir)

    plot_mean_trajectory(
        cycles,
        mean_plot,
        args.rolling_window,
        args.dpi,
        display_name,
        plt,
        ticker,
    )
    plot_high_watermark(
        cycles, watermark_plot, args.dpi, display_name, plt, ticker
    )
    plot_parameter_pareto(
        models,
        frontier,
        pareto_plot,
        args.dpi,
        display_name,
        plt,
        ticker,
        Normalize,
    )
    write_cycle_csv(cycle_csv, cycles)
    write_models_csv(model_csv, models)

    best_cycle = max(cycles, key=lambda cycle: cycle.reported_max)
    best_model = max(models, key=lambda model: model.accuracy) if models else None
    generated_total = sum(cycle.generated for cycle in cycles)
    reported_trained_total = sum(cycle.reported_trained for cycle in cycles)
    detected_trained_total = sum(cycle.detected_trained for cycle in cycles)
    summary = {
        "experiment": str(nngpt_dir),
        "completed_cycles": [cycle.cycle for cycle in cycles],
        "cycle_count": len(cycles),
        "generated_total": generated_total,
        "reported_trained_total": reported_trained_total,
        "detected_trained_total": detected_trained_total,
        "incomplete_cycle_models_excluded": incomplete_cycle_models,
        "parameterized_models": counted_models,
        "parameter_count_mode": torchvision_mode,
        "rolling_window": args.rolling_window,
        "confidence_level": args.confidence,
        "best_cycle": {
            "cycle": best_cycle.cycle,
            "label": f"A{best_cycle.cycle}",
            "accuracy": best_cycle.reported_max,
        },
        "best_model": (
            {
                "model_id": best_model.model_id,
                "cycle": best_model.cycle,
                "candidate": best_model.candidate,
                "accuracy": best_model.accuracy,
                "total_parameters": best_model.total_parameters,
            }
            if best_model
            else None
        ),
        "pareto_frontier": [
            {
                "model_id": model.model_id,
                "cycle": model.cycle,
                "candidate": model.candidate,
                "accuracy": model.accuracy,
                "total_parameters": model.total_parameters,
            }
            for model in frontier
        ],
        "warnings": reporter.warnings,
        "outputs": {
            "mean_accuracy_plot": str(mean_plot),
            "high_watermark_plot": str(watermark_plot),
            "pareto_plot": str(pareto_plot),
            "cycle_metrics_csv": str(cycle_csv),
            "trained_models_csv": str(model_csv),
            "parameter_count_cache": str(cache_path),
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print(f"Completed A-cycles: {len(cycles)}")
    print(
        f"Detected trained models: {detected_trained_total}/"
        f"{generated_total} generated"
    )
    print(
        f"Highest reported accuracy: A{best_cycle.cycle} "
        f"({best_cycle.reported_max:.2%})"
    )
    print(f"Models on Pareto frontier: {len(frontier)}")
    if incomplete_cycle_models:
        print(
            f"Excluded {incomplete_cycle_models} model(s) from cycles without a "
            "completed cycle_results_A*.json snapshot"
        )
    print(f"Outputs written to: {output_dir}")
    for path in (mean_plot, watermark_plot, pareto_plot, cycle_csv, model_csv, summary_json):
        print(f"  {path.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
