"""Gate-only experiment storage; never writes to LEMUR's CV-model tables."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def source_hash(source: str) -> str:
    """Ignore formatting/comments when deduplicating architecture targets."""
    return fingerprint(ast.dump(ast.parse(source), include_attributes=False))


def summarize_scores(outcomes: list[dict], *, min_success_rate: float,
                     min_measured: int) -> dict:
    """Mean of finite measured accuracies, with a separate eligibility floor.

    Missing/failed measurements are NOT zero accuracies. Unknown/infrastructure
    failures make a trial incomplete rather than poisoning proposer supervision.
    """
    if not 0 <= min_success_rate <= 1 or min_measured < 1:
        raise ValueError("Invalid gate score eligibility thresholds")
    measured = []
    for outcome in outcomes:
        if outcome.get("status") == "measured":
            value = outcome.get("accuracy")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Measured accuracy must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Measured accuracy must be finite and in [0, 1]")
            measured.append(float(value))
    incomplete = any(o.get("status") not in {"measured", "invalid"} for o in outcomes)
    rate = len(measured) / len(outcomes) if outcomes else 0.0
    mean = sum(measured) / len(measured) if measured else None
    variance = sum((v - mean) ** 2 for v in measured) / len(measured) if measured else None
    return {
        "n_attempted": len(outcomes), "n_measured": len(measured),
        "success_rate": rate, "mean_accuracy": mean,
        "std_accuracy": math.sqrt(variance) if variance is not None else None,
        "incomplete": incomplete,
        "eligible": not incomplete and len(measured) >= min_measured and rate >= min_success_rate,
        "min_success_rate": min_success_rate, "min_measured": min_measured,
        "objective": "mean_measured_accuracy_with_success_floor_v1",
    }


class GateStore:
    """Append trials to a namespaced SQLite table (including unsuccessful trials)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS moe_gate_trials_v1 (
                trial_id TEXT PRIMARY KEY,
                protocol_id TEXT NOT NULL,
                architecture_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS moe_gate_protocol_v1 ON moe_gate_trials_v1(protocol_id)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def add(self, record: dict) -> dict:
        record = dict(record)
        record["architecture_id"] = source_hash(record["source"])
        if record["protocol_id"] != fingerprint(record["protocol"]):
            raise ValueError("Gate protocol fingerprint does not match its payload")
        expected = summarize_scores(
            record["outcomes"], min_success_rate=record["score"]["min_success_rate"],
            min_measured=record["score"]["min_measured"],
        )
        if expected != record["score"]:
            raise ValueError("Gate score does not match measured outcomes")
        with self.connect() as db:
            db.execute("INSERT INTO moe_gate_trials_v1 VALUES (?, ?, ?, ?)", (
                record["trial_id"], record["protocol_id"], record["architecture_id"],
                canonical_json(record),
            ))
        return record

    def records(self, protocol_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT record_json FROM moe_gate_trials_v1 WHERE protocol_id=? ORDER BY trial_id",
                (protocol_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def training_subset(self, protocol_id: str, *, max_examples: int,
                        top_fraction: float, seed: int) -> list[dict]:
        """Sample top-performing *architectures*, averaging repeated valid trials.

        Do not cherry-pick the best random trial of a frequently sampled gate.
        An ineligible repeat disqualifies that architecture from SFT this round.
        """
        if max_examples < 1 or not 0 < top_fraction <= 1:
            raise ValueError("Invalid gate SFT subset settings")
        groups: dict[str, list[dict]] = {}
        for record in self.records(protocol_id):
            groups.setdefault(record["architecture_id"], []).append(record)
        ranked = []
        for trials in groups.values():
            if not all(t["score"]["eligible"] for t in trials):
                continue
            record = dict(trials[-1])
            record["selection_accuracy"] = sum(t["score"]["mean_accuracy"] for t in trials) / len(trials)
            record["selection_trials"] = [t["trial_id"] for t in trials]
            ranked.append(record)
        ranked.sort(key=lambda r: (-r["selection_accuracy"], r["architecture_id"]))
        pool = ranked[:max(1, math.ceil(len(ranked) * top_fraction))]
        rng = random.Random(seed)
        return rng.sample(pool, min(max_examples, len(pool)))
