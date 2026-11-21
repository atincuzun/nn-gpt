"""Sequential, resumable queue for the controlled MoE router study.

Every arm runs as its own process (``run_moe_experiment.py`` or, for the
outer self-improvement loop, ``run_moe_gate_cycle.py``), one after another on
the single GPU. Arms whose result is complete are skipped, so the queue can
be restarted after an interruption and continues where it stopped.

    python moe_experiments/dispatch.py --plan          # print the plan
    setsid nohup .venv/bin/python -u moe_experiments/dispatch.py \
        > logs/moex/dispatch.log 2>&1 < /dev/null &

Stages (each answers one question; see out/moex/report.md as results arrive):

s0  measurement validity. Frozen native model at four sampling seeds (noise
    floor), an exact repeat (generation determinism + CIFAR-training noise),
    router perturbations of increasing size and a constant router (does the
    CV metric respond to routing at all, and how strongly?).
s1  gate-only training: six catalog gate architectures against the native
    router trained identically (same data, steps, lr, seeds), plus a 10x lr
    for native and one gate. Answers: does a router *architecture* help
    beyond training the router?
s2  the LoRA path: LoRA SFT every round (generate -> evaluate -> train, the
    NNGPT loop) with the native router frozen, with the native router
    trainable, and with the best s1 gate architecture. Answers: does an
    architectural change add anything on top of ordinary fine-tuning?
s3  replicates of the key arms on fresh seeds (sampling, data order, CIFAR
    training), so no conclusion rests on one seed and the s1 "best" gate is
    re-measured instead of trusted (winner's curse).
s4  the outer self-improvement loop (LLM-proposed gates) as a 2x2 of
    keep-best (carry-forward) x proposer SFT, scored after training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "moex"
LOGS = ROOT / "logs" / "moex"
SEED_FILE = OUT / "seed_prompts.json"
STATUS_FILE = OUT / "dispatch_status.json"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
MODEL = os.environ.get(
    "MOEX_MODEL",
    str(Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B"
        "/snapshots/b01e354a56ff3d155b36e50557817761e9db50c0"),
)
ADAPTER = os.environ.get("MOEX_ADAPTER", "out/lfm25_nn_gate_adapter")
#: Routers replaced in every gate arm: the last eight MoE layers.
LAYERS = [str(layer) for layer in range(16, 24)]
FIXED_EVAL = json.dumps({"batch": 64, "transform": "norm_32_flip"})
MIN_FREE_GPU_MIB = int(os.environ.get("MOEX_MIN_FREE_GPU_MIB", "15000"))

COMMON = [
    "--model", MODEL,
    "--adapter", ADAPTER,
    "--gate-expert-quant", "int8",
    "--device-map", "cuda:0",
    "--dtype", "bfloat16",
    "--dataset", "cifar-10",
    "--conf-keys", "improve_classification_only",
    "--temperature", "0.2", "--top-k", "80", "--top-p", "0.9",
    "--nn-train-epochs", "1",
    "--fixed-eval-hyperparameters", FIXED_EVAL,
    "--test-nn", "24",
    "--generation-batch", "12",
    "--pinned-seed-prompts",
    "--generation-max-new-tokens", "3072",
    "--max-length", "4096",
    "--max-prompts", "512",
    "--batch-size", "1",
    "--validation-steps", "16",
]
SEEDS_A = ["--seed", "42", "--gen-seed", "1000", "--gate-eval-seed", "1234"]
SEEDS_B = ["--seed", "43", "--gen-seed", "2000", "--gate-eval-seed", "2234"]
TRAIN = ["--epochs", "3", "--gate-train-steps", "100", "--grad-accum", "4",
         "--gate-learning-rate", "1e-4", "--lora-lr", "1e-4"]
CATALOG_S1 = {
    "cat01": "catalog:01-additive-mlp-gelu",
    "cat05": "catalog:05-multiplicative-modulation",
    "cat06": "catalog:06-low-rank-gelu",
    "cat08": "catalog:08-layernorm-mlp",
    "cat12": "catalog:12-swiglu",
    "cat14": "catalog:14-sigmoid-gated-branch",
}


@dataclass
class Arm:
    stage: str
    name: str
    args: list[str] | Callable[[], list[str]]
    kind: str = "experiment"  # or "cycle"
    notes: str = ""
    resolved: list[str] = field(default_factory=list)

    @property
    def output(self) -> Path:
        return OUT / self.stage / self.name

    @property
    def log(self) -> Path:
        return LOGS / self.stage / f"{self.name}.log"

    def complete(self) -> bool:
        if self.kind == "cycle":
            return (self.output / "nngpt" / "one_cycle_results.json").is_file()
        result = self.output / "experiment_result.json"
        if not result.is_file():
            return False
        try:
            return bool(json.loads(result.read_text(encoding="utf-8")).get("complete"))
        except (OSError, ValueError):
            return False

    def command(self) -> list[str]:
        args = self.args() if callable(self.args) else list(self.args)
        self.resolved = args
        prefix = f"xsi-{self.stage}-{self.name}".replace(".", "p").replace("_", "-")
        if self.kind == "cycle":
            return [PYTHON, "-u", "run_moe_gate_cycle.py", "--output", str(self.output),
                    "--nn-name-prefix", prefix, *COMMON, *args]
        return [PYTHON, "-u", "run_moe_experiment.py", "--arm", self.name,
                "--stage", self.stage, "--output", str(self.output),
                "--nn-name-prefix", prefix, *COMMON, *args]


def _arm_rounds(stage: str, name: str) -> list[dict]:
    path = OUT / stage / name / "experiment_result.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("rounds", [])
    except (OSError, ValueError):
        return []


def best_stage1_gate() -> str:
    """Catalog gate with the largest paired yield gain over trained native.

    Paired per seed prompt and round (rounds >= 1), failures counted as 0.
    Falls back to cat01 while stage 1 is incomplete.
    """
    def slot_yield(rounds: list[dict]) -> dict[tuple[int, str], float]:
        values: dict[tuple[int, str], float] = {}
        for record in rounds:
            if record.get("round", 0) < 1:
                continue
            for slot in record.get("measurement", {}).get("slots", []):
                key = (record["round"], str(slot.get("source_nn")))
                values[key] = float(slot["accuracy"]) if slot.get("status") == "measured" and slot.get("accuracy") is not None else 0.0
        return values

    native = slot_yield(_arm_rounds("s1", "native"))
    best_name, best_gain = "cat01", float("-inf")
    for name in CATALOG_S1:
        gate = slot_yield(_arm_rounds("s1", name))
        shared = set(native) & set(gate)
        if len(shared) < 10:
            continue
        gain = sum(gate[key] - native[key] for key in shared) / len(shared)
        if gain > best_gain:
            best_name, best_gain = name, gain
    print(f"[DISPATCH] best stage-1 gate: {best_name} (paired yield gain {best_gain:+.4f})", flush=True)
    return best_name


def _best_spec() -> str:
    return CATALOG_S1[best_stage1_gate()]


def _second_spec() -> str:
    best = best_stage1_gate()
    return CATALOG_S1["cat12" if best == "cat01" else "cat01"]


def plan() -> list[Arm]:
    L = ["--layers", *LAYERS]
    arms = [
        # s0: measurement validity
        Arm("s0", "frozen", [*SEEDS_A, "--gate-spec", "none", "--train", "none",
                             "--measure-initial", "--epochs", "3"],
            notes="native model, no training, measured at 4 sampling seeds"),
        Arm("s0", "frozen_repeat", [*SEEDS_A, "--gate-spec", "none", "--train", "none",
                                    "--measure-initial", "--epochs", "0"],
            notes="exact repeat of frozen R0"),
        *[
            Arm("s0", f"noise_{scale}", [*SEEDS_A, *L, "--gate-spec", "native",
                                         "--router-noise", scale, "--train", "none",
                                         "--measure-initial", "--epochs", "0"],
                notes=f"native router + noise {scale} x weight std (dose-response)")
            for scale in ("0.05", "0.2", "1.0")
        ],
        Arm("s0", "constant", [*SEEDS_A, *L, "--gate-spec", "constant", "--train", "none",
                               "--measure-initial", "--epochs", "0"],
            notes="routing by expert bias only (broken router)"),
        # s1: gate-only training, architecture vs trained native router
        Arm("s1", "native", [*SEEDS_A, *L, *TRAIN, "--gate-spec", "native", "--train", "gate"],
            notes="control: native router architecture, trained"),
        *[
            Arm("s1", name, [*SEEDS_A, *L, *TRAIN, "--gate-spec", spec, "--train", "gate"])
            for name, spec in CATALOG_S1.items()
        ],
        Arm("s1", "native_lr1e-3", [*SEEDS_A, *L, *TRAIN, "--gate-spec", "native",
                                    "--train", "gate", "--gate-learning-rate", "1e-3"]),
        Arm("s1", "cat01_lr1e-3", [*SEEDS_A, *L, *TRAIN, "--gate-spec", CATALOG_S1["cat01"],
                                   "--train", "gate", "--gate-learning-rate", "1e-3"]),
        # s2: the LoRA path
        Arm("s2", "lora", [*SEEDS_A, *TRAIN, "--gate-spec", "none", "--train", "lora"],
            notes="ordinary LoRA SFT loop, native router frozen"),
        Arm("s2", "lora_native", [*SEEDS_A, *L, *TRAIN, "--gate-spec", "native",
                                  "--train", "lora+gate"]),
        Arm("s2", "lora_best", lambda: [*SEEDS_A, *L, *TRAIN, "--gate-spec", _best_spec(),
                                        "--train", "lora+gate"]),
        Arm("s2", "lora_second", lambda: [*SEEDS_A, *L, *TRAIN, "--gate-spec", _second_spec(),
                                          "--train", "lora+gate"]),
        # s3: replicates on fresh seeds
        Arm("s3", "frozen", [*SEEDS_B, "--gate-spec", "none", "--train", "none",
                             "--measure-initial", "--epochs", "3"]),
        Arm("s3", "native", [*SEEDS_B, *L, *TRAIN, "--gate-spec", "native", "--train", "gate"]),
        Arm("s3", "best", lambda: [*SEEDS_B, *L, *TRAIN, "--gate-spec", _best_spec(),
                                   "--train", "gate"]),
        Arm("s3", "lora", [*SEEDS_B, *TRAIN, "--gate-spec", "none", "--train", "lora"]),
        Arm("s3", "lora_native", [*SEEDS_B, *L, *TRAIN, "--gate-spec", "native",
                                  "--train", "lora+gate"]),
        Arm("s3", "lora_best", lambda: [*SEEDS_B, *L, *TRAIN, "--gate-spec", _best_spec(),
                                        "--train", "lora+gate"]),
    ]
    # Run order: the cheapest decisive answers first (measurement validity,
    # then "does training the router / LoRA change anything at all"), the
    # architecture sweep next, replicates and the long outer loop last.
    priority = [
        "s0/frozen", "s0/frozen_repeat", "s0/constant", "s0/noise_1.0",
        "s0/noise_0.2", "s0/noise_0.05", "s1/native", "s2/lora",
    ]
    by_label = {f"{arm.stage}/{arm.name}": arm for arm in arms}
    arms = [by_label[label] for label in priority] + [
        arm for arm in arms if f"{arm.stage}/{arm.name}" not in priority
    ]
    outer = [
        "--seed", "42", "--gate-eval-seed", "1234", *L,
        "--gate-outer-search", "--measure-after-training", "--epochs", "1",
        "--gate-candidates", "10", "--gate-catalog-candidates", "2",
        "--gate-reuse-policy", "evaluate", "--gate-prompt-key", "reference_best",
        "--gate-train-steps", "400", "--gate-learning-rate", "1e-4",
        "--gate-max-new-tokens", "2048", "--gate-generation-attempts", "8",
    ]
    for name, extra in (
        ("plain", []),
        ("keepbest", ["--gate-carry-forward", "best"]),
        ("sft", ["--gate-outer-sft", "--gate-sft-every", "3", "--gate-min-pairs", "1"]),
        ("keepbest_sft", ["--gate-carry-forward", "best", "--gate-outer-sft",
                          "--gate-sft-every", "3", "--gate-min-pairs", "1"]),
    ):
        arm = Arm("s4", name, None, kind="cycle")
        arm.args = [*outer, "--gate-store", str(arm.output / "gate_store"), *extra]
        arms.append(arm)
    return arms


def _gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()
        return int(out[0])
    except Exception:
        return None


def _wait_for_gpu() -> None:
    waited = 0
    while True:
        free = _gpu_free_mib()
        if free is None or free >= MIN_FREE_GPU_MIB:
            return
        if waited % 1800 == 0:
            print(f"[DISPATCH] waiting for GPU: {free} MiB free < {MIN_FREE_GPU_MIB}", flush=True)
        time.sleep(60)
        waited += 60


def _status(update: dict) -> None:
    status = {}
    if STATUS_FILE.is_file():
        try:
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
    status.update(update)
    status["updated"] = datetime.now().isoformat(timespec="seconds")
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _analyze() -> None:
    subprocess.run(
        [PYTHON, str(ROOT / "moe_experiments" / "analyze.py"), "--root", str(OUT)],
        cwd=ROOT, check=False,
    )


def _overlay() -> str:
    """transformers 5.x overlay the MoE runs need (the venv pins 4.46).

    LFM2-MoE exists only in transformers 5.x; the earlier gate runs put the
    same overlay first on PYTHONPATH. A persistent copy is preferred over
    /tmp, which does not survive a reboot.
    """
    for candidate in (
        os.environ.get("MOEX_OVERLAY", ""),
        str(Path.home() / "dev_nngpt" / "overlays" / "transformers-5.17"),
        "/tmp/kilo/nn-gpt-resolved-overlay",
    ):
        if candidate and (Path(candidate) / "transformers").is_dir():
            return candidate
    raise RuntimeError("no transformers 5.x overlay found; set MOEX_OVERLAY")


def experiment_env() -> dict[str, str]:
    """The environment of the earlier gate-cycle dispatch, plus seed pinning."""
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "NNGPT_MOE_EVAL_CUDA": "0",
        # Evaluated CV models are not written back to LEMUR: every arm then
        # trains on the same corpus, so arms differ only in the factor under
        # test (and the shared database is not filled with experiment rows).
        "NNGPT_EVAL_SAVE_TO_DB": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "NNGPT_STRIP_THINK_OUTPUT": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": f"{_overlay()}:{ROOT}",
        "NNGPT_SEED_FILE": str(SEED_FILE),
        "NNGPT_SEED_FILE_EXTERNAL": "1",
        "NNGPT_PIN_SEEDS": "1",
    })
    return env


def run(arms: list[Arm], *, retries: int) -> None:
    env = experiment_env()
    if not SEED_FILE.is_file():
        subprocess.run([PYTHON, str(ROOT / "moe_experiments" / "seed_prompts.py"),
                        "--out", str(SEED_FILE)], cwd=ROOT, env=env, check=True)
    done, failed = [], []
    for arm in arms:
        label = f"{arm.stage}/{arm.name}"
        if arm.complete():
            print(f"[DISPATCH] skip {label} (complete)", flush=True)
            done.append(label)
            continue
        for attempt in range(retries + 1):
            _wait_for_gpu()
            command = arm.command()
            arm.output.mkdir(parents=True, exist_ok=True)
            arm.log.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f"[DISPATCH] {datetime.now():%Y-%m-%d %H:%M:%S} start {label} "
                  f"(attempt {attempt + 1}) -> {arm.log}", flush=True)
            _status({"current": label, "current_started": datetime.now().isoformat(timespec="seconds"),
                     "current_log": str(arm.log)})
            (arm.output / "command.json").write_text(json.dumps(command, indent=1), encoding="utf-8")
            with open(arm.log, "a", encoding="utf-8") as log:
                log.write(f"\n##### {datetime.now():%Y-%m-%d %H:%M:%S} attempt {attempt + 1}\n")
                log.flush()
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT).returncode
            hours = (time.monotonic() - started) / 3600
            if code == 0 and arm.complete():
                print(f"[DISPATCH] done {label} in {hours:.2f} h", flush=True)
                done.append(label)
                break
            print(f"[DISPATCH] {label} exited {code} after {hours:.2f} h", flush=True)
            time.sleep(120)
        else:
            failed.append(label)
        _status({"done": done, "failed": failed, "current": None})
        _analyze()
    print(f"[DISPATCH] queue finished: {len(done)} done, {len(failed)} failed {failed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    parser.add_argument("--stages", default="s0,s1,s2,s3,s4")
    parser.add_argument("--only", nargs="*", help="run only these stage/name arms")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    stages = set(args.stages.split(","))
    arms = [arm for arm in plan() if arm.stage in stages]
    if args.only:
        arms = [arm for arm in arms if f"{arm.stage}/{arm.name}" in set(args.only)]
    if args.plan:
        for arm in arms:
            state = "complete" if arm.complete() else "pending"
            shown = "<resolved at run time>" if callable(arm.args) else " ".join(arm.args)
            print(f"{arm.stage}/{arm.name:16s} [{state}] {arm.notes}\n    {shown}")
        return
    os.chdir(ROOT)
    run(arms, retries=args.retries)


if __name__ == "__main__":
    main()
