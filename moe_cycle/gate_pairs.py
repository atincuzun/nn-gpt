"""Phase B seam: build (lower -> higher) gate training pairs.

Phase A improves gate proposals in context, with no weight updates, because the
gate archive starts empty.  Once enough eligible gates exist, Phase B can train
the LLM on these pairs so it authors better gates directly.  The mechanism is
deliberately left undecided and disabled by default; this module only prepares
the data.

A pair is only comparable when the two gates were evaluated under the same CV
setting, so pairs are formed within ``(task, dataset, metric)`` groups.
"""

from __future__ import annotations

from .gate_store import load_gate_summaries

PAIR_REQUIRED_MATCH = ("task", "dataset", "metric")


def eligible_gates(root, *, required_match: tuple[str, ...] = PAIR_REQUIRED_MATCH) -> list[dict]:
    """Gates with a usable score, grouped-friendly for pairing."""
    gates = []
    for summary in load_gate_summaries(root):
        score = summary.get("score") or {}
        if not score.get("eligible"):
            continue
        if not isinstance(score.get("accuracy"), float):
            continue
        if not summary.get("gate_code"):
            continue
        if any(summary.get(key) is None for key in required_match):
            continue
        gates.append(summary)
    return gates


def build_gate_pairs(
    root,
    *,
    min_accuracy_gap: float = 0.0,
    max_pairs: int | None = None,
    required_match: tuple[str, ...] = PAIR_REQUIRED_MATCH,
) -> list[dict]:
    """Every (lower, higher) pair within a comparable CV setting.

    Sorted by the largest improvement first, so a limited training budget uses
    the clearest examples.  ``min_accuracy_gap`` skips pairs whose difference is
    within noise.
    """
    if min_accuracy_gap < 0:
        raise ValueError("min_accuracy_gap must not be negative")

    groups: dict[tuple, list[dict]] = {}
    for gate in eligible_gates(root, required_match=required_match):
        key = tuple(gate.get(field) for field in required_match)
        groups.setdefault(key, []).append(gate)

    pairs: list[dict] = []
    for key, gates in groups.items():
        gates = sorted(gates, key=lambda g: g["score"]["accuracy"])
        for lower_index, lower in enumerate(gates):
            for higher in gates[lower_index + 1:]:
                gap = higher["score"]["accuracy"] - lower["score"]["accuracy"]
                if gap <= min_accuracy_gap:
                    continue
                pairs.append({
                    "pair_id": f"{lower['gate_id']:03d}->{higher['gate_id']:03d}",
                    "lower_gate_id": lower["gate_id"],
                    "higher_gate_id": higher["gate_id"],
                    "lower_source": lower["gate_code"],
                    "higher_source": higher["gate_code"],
                    "lower_accuracy": lower["score"]["accuracy"],
                    "higher_accuracy": higher["score"]["accuracy"],
                    "accuracy_gap": gap,
                    **{field: value for field, value in zip(required_match, key)},
                })

    pairs.sort(key=lambda pair: -pair["accuracy_gap"])
    return pairs[:max_pairs] if max_pairs is not None else pairs


def describe_pairs(pairs: list[dict]) -> str:
    if not pairs:
        return "No comparable, eligible gate pairs yet; Phase B cannot start."
    best = pairs[0]
    return (
        f"{len(pairs)} comparable gate pair(s); best improvement "
        f"{best['lower_gate_id']:03d}->{best['higher_gate_id']:03d} "
        f"({best['lower_accuracy']:.4f} -> {best['higher_accuracy']:.4f})"
    )
