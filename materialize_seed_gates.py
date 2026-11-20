"""Write the 50 built-in seed gates as named, hashed Python artifacts.

Each entry becomes ``<out>/gate-<seed-name>-<md5>/gate.py`` plus metadata.
No index is written, so the first training run assigns runtime gate ids and
reuses these folders; after evaluation they also hold ``record.json``,
``stats.json`` and ``weights/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from moe_cycle.gate_seeds import SEED_GATES
from moe_cycle.gate_store import gate_library_name, lemur_id, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "gate_library",
        help="Library root (default: nn-gpt/gate_library)",
    )
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for index, (name, source) in enumerate(SEED_GATES):
        target = root / gate_library_name(source, seed_name=name, kind="seed")
        target.mkdir(parents=True, exist_ok=True)
        (target / "gate.py").write_text(source.rstrip() + "\n", encoding="utf-8")
        write_json(
            target / "metadata.json",
            {
                "seed_index": index,
                "seed_name": name,
                "name": target.name,
                "class_name": "Net",
                "code_hash": lemur_id(source),
            },
        )
    print(f"materialized {len(SEED_GATES)} seed gates -> {root}")


if __name__ == "__main__":
    main()
