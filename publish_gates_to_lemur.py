"""Publish measured gate artifacts into the LEMUR database.

Gates are routers, not CV models, so this does not call LEMUR's CV trainer or
``check_nn``. It uses LEMUR's own write layer: the gate source becomes
``nn.code`` and its measured CV score becomes a ``stat`` row under the
``moe-gate`` task. The source keeps the LEMUR model-code contract
(``class Net``, ``supported_hyperparameters``, ``train_setup``/``learn``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).resolve().parent / "gate_library")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="LEMUR root containing db/ab.nn.db")
    parser.add_argument("--task", default="moe-gate")
    parser.add_argument("--metric", default="acc")
    parser.add_argument("--transform", default="norm_32_flip")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(args.root)
    sys.path.insert(0, str(args.root))
    from ab.nn.util.db.Write import save_nn
    from ab.nn.util.Util import uuid4

    from moe_cycle.gate_catalog import GATE_CATALOG

    catalog = {name: source for name, source in GATE_CATALOG}
    published = 0
    for record_path in sorted(args.library.glob("gate-*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        score = record.get("score") or {}
        accuracy = score.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            continue
        mechanism = record.get("seed_name") or record.get("name") or record_path.parent.name
        source = catalog.get(mechanism)
        if source is None:
            source = (record_path.parent / "gate.py").read_text(encoding="utf-8")
        digest = uuid4(source)
        name = f"gate-{mechanism}-{digest}"
        protocol = record.get("protocol") or {}
        fixed = protocol.get("fixed_eval_hyperparameters") or {}
        transform = fixed.get("transform") or args.transform
        stats = record.get("stats") or []
        duration = int(stats[0].get("duration", 0)) if stats else 0
        dataset = record.get("dataset") or "cifar-10"
        prm = {
            "uid": uuid4(f"{name}:{duration}"),
            "accuracy": float(accuracy),
            "duration": duration,
            "transform": transform,
            "batch": fixed.get("batch", 64),
            "lr": (stats[0].get("train_stat") or {}).get("lr") if stats else None,
            "gate_name": name,
            "gate_mechanism": mechanism,
            "gate_code_hash": digest,
            "gate_score_objective": score.get("objective"),
            "gate_train_steps": protocol.get("gate_train_steps"),
            "gate_learning_rate": protocol.get("gate_learning_rate"),
            "llm_version": record.get("llm_version"),
            "gate_protocol": json.dumps(protocol, sort_keys=True, default=str),
            "gate_stats": json.dumps(stats, sort_keys=True, default=str),
        }
        prm = {key: value for key, value in prm.items() if value is not None}
        if args.dry_run:
            print(f"DRY {name} | task={args.task} dataset={dataset} metric={args.metric} "
                  f"epoch=1 accuracy={accuracy} transform={transform}")
            published += 1
            continue
        saved = save_nn(
            source,
            args.task,
            dataset,
            args.metric,
            1,
            prm,
            force_name=name,
        )
        print(f"published {saved} | task={args.task} dataset={dataset} "
              f"metric={args.metric} accuracy={accuracy}")
        published += 1
    print(f"{'would publish' if args.dry_run else 'published'} {published} measured gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
