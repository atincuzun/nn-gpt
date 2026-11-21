"""Summarize the controlled MoE router study into a report.

Reads every ``experiment_result.json`` under ``--root`` (and the gate stores
of the outer-loop arms) and writes:

* ``report.md``  -- per-arm tables, paired contrasts, dose-response, s4 loop
* ``arms.csv``   -- one row per arm and round
* ``slots.csv``  -- one row per generated CV model (arm, round, seed prompt)

Contrasts are paired: two arms are compared on the same seed prompt in the
same round, which in this design also means the same sampling seed. Three
per-prompt outcomes are compared:

* ``yield``  accuracy, with a failed or unusable generation counted as 0
* ``valid``  1 if the generated model trained and was measured, else 0
* ``acc``    accuracy where both arms produced a measured model

Uncertainty: 95% bootstrap interval resampling seed prompts (all rounds of a
prompt move together, because the same prompt recurs every round), and a
two-sided sign-flip permutation p-value over prompts.

    python moe_experiments/analyze.py --root out/moex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

BOOTSTRAP = 4000
PERMUTATIONS = 4000


def load_arms(root: Path) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for path in sorted(root.glob("*/*/experiment_result.json")):
        stage = path.parent.parent.name
        if stage.startswith("smoke"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        arms[f"{stage}/{path.parent.name}"] = data
    return arms


def slot_table(arms: dict[str, dict]) -> list[dict]:
    rows = []
    for key, data in arms.items():
        config = data.get("config", {})
        for record in data.get("rounds", []):
            r = record.get("round")
            for slot in record.get("measurement", {}).get("slots", []):
                measured = slot.get("status") == "measured" and slot.get("accuracy") is not None
                rows.append({
                    "arm": key,
                    "round": r,
                    "gen_seed": (config.get("gen_seed") or 0) + (r or 0),
                    "seed_set": config.get("gen_seed"),
                    "source_nn": slot.get("source_nn"),
                    "slot": slot.get("slot"),
                    "status": slot.get("status"),
                    "accuracy": slot.get("accuracy") if measured else None,
                    "baseline_accuracy": slot.get("baseline_accuracy"),
                    "failure": (slot.get("failure") or "")[:120],
                    "code_sha1": slot.get("code_sha1"),
                })
    return rows


def _values(rows: Iterable[dict], arm: str, rounds: set[int] | None) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        if row["arm"] != arm or (rounds is not None and row["round"] not in rounds):
            continue
        out[(row["gen_seed"], row["source_nn"])] = row
    return out


def _paired(rows, arm_a, arm_b, rounds_a, rounds_b=None, metric="yield"):
    """Per-prompt differences A - B on matching (sampling seed, prompt)."""
    a = _values(rows, arm_a, rounds_a)
    b = _values(rows, arm_b, rounds_b if rounds_b is not None else rounds_a)
    clusters: dict[Any, list[float]] = defaultdict(list)
    for key in sorted(set(a) & set(b), key=str):
        ra, rb = a[key], b[key]
        if metric == "yield":
            da = ra["accuracy"] if ra["accuracy"] is not None else 0.0
            db = rb["accuracy"] if rb["accuracy"] is not None else 0.0
        elif metric == "valid":
            da = 1.0 if ra["accuracy"] is not None else 0.0
            db = 1.0 if rb["accuracy"] is not None else 0.0
        else:
            if ra["accuracy"] is None or rb["accuracy"] is None:
                continue
            da, db = ra["accuracy"], rb["accuracy"]
        clusters[key[1]].append(da - db)
    return clusters


def _stats(clusters: dict[Any, list[float]], seed: int = 7) -> dict[str, Any]:
    groups = [values for values in clusters.values() if values]
    n = sum(len(values) for values in groups)
    if n == 0:
        return {"n": 0}
    mean = sum(sum(values) for values in groups) / n
    rng = random.Random(seed)
    boot = []
    for _ in range(BOOTSTRAP):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        size = sum(len(values) for values in sample)
        boot.append(sum(sum(values) for values in sample) / size)
    boot.sort()
    low = boot[int(0.025 * BOOTSTRAP)]
    high = boot[int(0.975 * BOOTSTRAP) - 1]
    sums = [sum(values) for values in groups]
    observed = abs(sum(sums))
    extreme = 0
    for _ in range(PERMUTATIONS):
        flipped = sum(value if rng.random() < 0.5 else -value for value in sums)
        if abs(flipped) >= observed - 1e-12:
            extreme += 1
    return {
        "n": n,
        "prompts": len(groups),
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "p": (extreme + 1) / (PERMUTATIONS + 1),
    }


def _fmt(value, digits=4, signed=False):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def contrast_line(rows, label, arm_a, arm_b, rounds_a, rounds_b=None) -> str | None:
    cells = []
    any_data = False
    for metric in ("yield", "valid", "acc"):
        stats = _stats(_paired(rows, arm_a, arm_b, rounds_a, rounds_b, metric))
        if stats.get("n"):
            any_data = True
            p_text = "<0.001" if stats["p"] < 0.001 else f"={stats['p']:.3f}"
            cells.append(
                f"{_fmt(stats['mean'], signed=True)} [{_fmt(stats['ci_low'], signed=True)}, "
                f"{_fmt(stats['ci_high'], signed=True)}] p{p_text} (n={stats['n']})"
            )
        else:
            cells.append("—")
    if not any_data:
        return None
    return f"| {label} | " + " | ".join(cells) + " |"


def arm_rows(arms: dict[str, dict]) -> list[dict]:
    rows = []
    for key, data in arms.items():
        config = data.get("config", {})
        install = data.get("install", {})
        for record in data.get("rounds", []):
            summary = record.get("measurement", {}).get("summary", {})
            diag = record.get("diagnostics", {})
            training = record.get("training", {}) or {}
            rows.append({
                "arm": key,
                "complete": data.get("complete"),
                "gate": install.get("gate_label") or config.get("gate_spec"),
                "train": config.get("train"),
                "noise": config.get("router_noise"),
                "gate_lr": config.get("gate_learning_rate"),
                "layers": ",".join(str(x) for x in install.get("installed_layers", [])),
                "round": record.get("round"),
                "steps_total": record.get("optimizer_steps_total"),
                "n_slots": summary.get("n_slots"),
                "valid_rate": summary.get("valid_rate"),
                "mean_accuracy": summary.get("mean_accuracy"),
                "yield_accuracy": summary.get("yield_accuracy"),
                "improved_rate": summary.get("improved_rate"),
                "flip_all": diag.get("flip_any_all_layers"),
                "flip_replaced": diag.get("flip_any_replaced_layers"),
                "kl": diag.get("kl_native_to_current"),
                "top1_agreement": diag.get("top1_agreement"),
                "probe_nll_delta": diag.get("probe_nll_delta"),
                "train_loss_first10": training.get("loss_first10"),
                "train_loss_last10": training.get("loss_last10"),
                "val_loss": training.get("validation_loss"),
                "measure_min": (summary.get("seconds") or 0) / 60,
                "train_min": (training.get("seconds") or 0) / 60,
            })
    return rows


def _rounds(arms, key) -> set[int]:
    return {record.get("round") for record in arms.get(key, {}).get("rounds", [])}


def s4_tables(root: Path) -> list[str]:
    lines: list[str] = []
    for store in sorted((root / "s4").glob("*/gate_store")):
        arm = store.parent.name
        records = []
        for path in sorted(store.glob("*/record.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        if not records:
            continue
        records.sort(key=lambda r: r.get("gate_id", 0))
        lines += [f"### s4/{arm}", "",
                  "| gate | kind | proposer | status | mean CV acc | measured | best so far | note |",
                  "|---|---|---|---|---|---|---|---|"]
        best = None
        for record in records:
            score = record.get("score") or {}
            acc = score.get("accuracy")
            if isinstance(acc, float) and (best is None or acc > best):
                best = acc
            note = (record.get("failure") or "")[:80].replace("|", "/")
            lines.append(
                f"| {record.get('gate_id')} | {record.get('kind')} {record.get('seed_name') or ''} | "
                f"{str(record.get('llm_version'))[-8:]} | "
                f"{'failed' if record.get('failure') else ('running' if record.get('running') else 'done')} | "
                f"{_fmt(acc)} | {score.get('n_measured', 0)}/{score.get('n_attempted', 0)} | "
                f"{_fmt(best)} | {note} |"
            )
        failed = sum(1 for record in records if record.get("failure"))
        llm = [r for r in records if r.get("kind") == "llm"]
        lines += ["", f"LLM proposals: {len(llm)}, failed candidates: {failed}/{len(records)}", ""]
    return lines


def build_report(root: Path) -> str:
    arms = load_arms(root)
    slots = slot_table(arms)
    table = arm_rows(arms)
    lines = [
        "# Controlled MoE router study",
        "",
        "Generated by `moe_experiments/analyze.py`. Every arm generates CV models from the same "
        "pinned seed prompts with the same per-round sampling seeds, so contrasts are paired per "
        "prompt and round. Cells: mean difference A−B, 95% bootstrap CI over prompts, sign-flip "
        "permutation p, n = paired observations. **yield** counts a failed generation as accuracy 0; "
        "**valid** is the trained-and-measured rate; **acc** compares only prompts both arms "
        "turned into a measured model.",
        "",
        "## Arms by round",
        "",
        "| arm | gate | train | round | steps | valid | mean acc | yield | improved | flips (replaced/all) | KL | top1 | ΔNLL | loss first→last | val loss | min (train/measure) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['arm']}{'' if row['complete'] else ' *(running)*'} | {row['gate']}"
            f"{' +noise ' + str(row['noise']) if row['noise'] else ''} | {row['train']}"
            f"{' lr ' + str(row['gate_lr']) if row['train'] in ('gate', 'lora+gate') else ''} | "
            f"{row['round']} | {row['steps_total']} | {_fmt(row['valid_rate'], 2)} | "
            f"{_fmt(row['mean_accuracy'])} | {_fmt(row['yield_accuracy'])} | {_fmt(row['improved_rate'], 2)} | "
            f"{_fmt(row['flip_replaced'], 3)}/{_fmt(row['flip_all'], 3)} | {_fmt(row['kl'], 4)} | "
            f"{_fmt(row['top1_agreement'], 3)} | {_fmt(row['probe_nll_delta'], 4, signed=True)} | "
            f"{_fmt(row['train_loss_first10'], 3)}→{_fmt(row['train_loss_last10'], 3)} | "
            f"{_fmt(row['val_loss'], 3)} | {row['train_min']:.0f}/{row['measure_min']:.0f} |"
        )

    header = ["| contrast (A − B) | yield | valid | acc |", "|---|---|---|---|"]

    def section(title: str, specs: list[tuple]) -> None:
        body = []
        for label, a, b, ra, rb in specs:
            if a not in arms or b not in arms:
                continue
            line = contrast_line(slots, label, a, b, ra, rb)
            if line:
                body.append(line)
        if body:
            lines.extend(["", f"## {title}", "", *header, *body])

    trained_rounds = {1, 2, 3}
    section("s0 — measurement validity (round 0, same sampling seed)", [
        ("exact repeat − frozen", "s0/frozen_repeat", "s0/frozen", {0}, {0}),
        *[(f"{name} − frozen", f"s0/{name}", "s0/frozen", {0}, {0})
          for name in ("noise_0.05", "noise_0.2", "noise_1.0", "constant")],
    ])
    s1_gates = sorted(key.split("/")[1] for key in arms if key.startswith("s1/cat"))
    section("s1 — gate-only training: architecture vs trained native router (rounds 1–3)", [
        ("native(trained) − frozen", "s1/native", "s0/frozen", trained_rounds, trained_rounds),
        *[(f"{name} − native", f"s1/{name}", "s1/native", trained_rounds, None)
          for name in s1_gates if not name.endswith("lr1e-3")],
        ("native lr1e-3 − frozen", "s1/native_lr1e-3", "s0/frozen", trained_rounds, trained_rounds),
        ("cat01 lr1e-3 − native lr1e-3", "s1/cat01_lr1e-3", "s1/native_lr1e-3", trained_rounds, None),
    ])
    section("s1 — final round only (round 3)", [
        *[(f"{name} − native", f"s1/{name}", "s1/native", {3}, None)
          for name in s1_gates if not name.endswith("lr1e-3")],
    ])
    section("s2 — the LoRA path (rounds 1–3)", [
        ("lora − frozen", "s2/lora", "s0/frozen", trained_rounds, trained_rounds),
        ("lora+native gate − lora", "s2/lora_native", "s2/lora", trained_rounds, None),
        ("lora+best gate − lora+native gate", "s2/lora_best", "s2/lora_native", trained_rounds, None),
        ("lora+second gate − lora+native gate", "s2/lora_second", "s2/lora_native", trained_rounds, None),
        ("lora+best gate − lora", "s2/lora_best", "s2/lora", trained_rounds, None),
        ("lora − native gate only", "s2/lora", "s1/native", trained_rounds, None),
    ])
    section("s3 — replicates on fresh seeds (rounds 1–3)", [
        ("native(trained) − frozen", "s3/native", "s3/frozen", trained_rounds, trained_rounds),
        ("best gate − native", "s3/best", "s3/native", trained_rounds, None),
        ("lora − frozen", "s3/lora", "s3/frozen", trained_rounds, trained_rounds),
        ("lora+native gate − lora", "s3/lora_native", "s3/lora", trained_rounds, None),
        ("lora+best gate − lora+native gate", "s3/lora_best", "s3/lora_native", trained_rounds, None),
    ])

    # Code identity of the exact repeat: is generation deterministic?
    if "s0/frozen_repeat" in arms and "s0/frozen" in arms:
        a = _values(slots, "s0/frozen_repeat", {0})
        b = _values(slots, "s0/frozen", {0})
        shared = [key for key in set(a) & set(b) if a[key]["code_sha1"] and b[key]["code_sha1"]]
        same = sum(1 for key in shared if a[key]["code_sha1"] == b[key]["code_sha1"])
        both = [key for key in shared if a[key]["accuracy"] is not None and b[key]["accuracy"] is not None
                and a[key]["code_sha1"] == b[key]["code_sha1"]]
        diffs = [a[key]["accuracy"] - b[key]["accuracy"] for key in both]
        sd = (math.sqrt(sum(d * d for d in diffs) / len(diffs) / 2) if diffs else None)
        lines += ["", "## Measurement noise", "",
                  f"- Exact repeat: {same}/{len(shared)} prompts produced byte-identical model code.",
                  f"- CIFAR-training noise on identical code: per-model SD ≈ {_fmt(sd)} "
                  f"(from {len(diffs)} identical-code pairs)."]
    frozen = [row for row in table if row["arm"] == "s0/frozen"]
    if len(frozen) > 1:
        yields = [row["yield_accuracy"] for row in frozen if row["yield_accuracy"] is not None]
        if yields:
            lines.append(
                f"- Frozen model across sampling seeds (rounds 0–3): yield "
                f"{min(yields):.4f}–{max(yields):.4f} (spread {max(yields) - min(yields):.4f})."
            )

    # Dose-response: routing change vs outcome change against the frozen model.
    dose = []
    for row in table:
        if row["arm"] in ("s0/frozen", "s0/frozen_repeat") or row["round"] is None:
            continue
        ref = "s0/frozen" if row["arm"].startswith(("s0", "s1", "s2")) else "s3/frozen"
        if ref not in arms or row["round"] not in _rounds(arms, ref):
            continue
        stats = _stats(_paired(slots, row["arm"], ref, {row["round"]}, {row["round"]}, "yield"))
        if stats.get("n"):
            dose.append((row, stats))
    if dose:
        lines += ["", "## Routing change vs outcome change (each arm-round vs the frozen model, same seed)", "",
                  "| arm | round | flips (replaced) | flips (all) | KL | Δyield vs frozen [95% CI] |",
                  "|---|---|---|---|---|---|"]
        for row, stats in sorted(dose, key=lambda item: (item[0]["flip_all"] or 0)):
            lines.append(
                f"| {row['arm']} | {row['round']} | {_fmt(row['flip_replaced'], 3)} | "
                f"{_fmt(row['flip_all'], 3)} | {_fmt(row['kl'], 4)} | "
                f"{_fmt(stats['mean'], signed=True)} [{_fmt(stats['ci_low'], signed=True)}, "
                f"{_fmt(stats['ci_high'], signed=True)}] |"
            )

    s4 = s4_tables(root)
    if s4:
        lines += ["", "## s4 — outer self-improvement loop (LLM-proposed gates, scored after training)", "", *s4]

    failures = defaultdict(int)
    for row in slots:
        if row["accuracy"] is None and row["failure"]:
            failures[row["failure"][:90]] += 1
    if failures:
        lines += ["", "## Most frequent generation/evaluation failures (all arms)", ""]
        for reason, count in sorted(failures.items(), key=lambda item: -item[1])[:12]:
            lines.append(f"- {count}× {reason}")
    return "\n".join(lines) + "\n", table, slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("out/moex"))
    args = parser.parse_args()
    report, table, slots = build_report(args.root)
    (args.root / "report.md").write_text(report, encoding="utf-8")
    for name, rows in (("arms.csv", table), ("slots.csv", slots)):
        if rows:
            with open(args.root / name, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"[ANALYZE] {len(table)} arm-rounds, {len(slots)} slots -> {args.root / 'report.md'}")


if __name__ == "__main__":
    main()
