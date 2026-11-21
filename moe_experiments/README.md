# Controlled MoE router study

Does an LLM get better at writing computer-vision models when it changes its
own router architecture while it learns? This folder holds the queue that
tests that question empirically, and the analysis that reads the answer out.

## Why a separate harness

In the gate cycle (`run_moe_gate_cycle.py`) every gate starts as an exact copy
of the native router (step-zero logits match to 0.0), and each inner epoch
measured CV accuracy *before* training the gate. With `--epochs 1` the score
was therefore taken while every gate still computed exactly what the native
router computes: in `gate_library/arm_b`, four different gates produced
byte-identical CV model code and their 0.647–0.682 spread was CIFAR-training
noise. One CV model per gate also gave no statistical power.

`run_moe_experiment.py` (implementation: `moe_cycle/experiment.py`) measures
one fully specified arm per process:

- **router** (`--gate-spec`): `none`, `native` (trainable exact copy — the
  control), `catalog:<name>`, `constant` (broken), `file:<path>`;
  `--router-noise` perturbs the copied projection.
- **what learns each round** (`--train`): `none`, `gate`, `lora` (the NNGPT
  LoRA SFT loop), `lora+gate`.
- **protocol**: every round is *train → measure*. A measurement generates
  `--test-nn` (24) CV models from the same pinned seed prompts
  (`out/moex/seed_prompts.json`, via upstream `NNGPT_SEED_FILE`) with the same
  per-round sampling seed in every arm, then trains each for one epoch on
  CIFAR-10. Outcomes are stored per prompt, so arms are compared pairwise.
- **diagnostics** (`moe_cycle/routing_probe.py`): after every phase, the
  fraction of probe tokens whose top-k expert set changed versus the native
  model (per MoE layer), KL(native‖current) of the next-token distribution and
  the probe NLL. A null result can then be told apart from "the router never
  changed".

The gate cycle itself gained `--measure-after-training`,
`--generation-batch` and `--pinned-seed-prompts` (used by stage s4), and the
tag stopping criterion now stops each batch row at its own `</nn>` (it used to
stop the whole batch on row 0 and never fired for blocks longer than its
256-token window).

## Running

```bash
cd ~/dev_nngpt/nn-gpt
.venv/bin/python moe_experiments/dispatch.py --plan
setsid nohup .venv/bin/python -u moe_experiments/dispatch.py \
    > logs/moex/dispatch.log 2>&1 < /dev/null &
```

The queue is sequential and resumable (complete arms are skipped). It sets
the same environment as the earlier gate-cycle dispatch, including the
transformers 5.x overlay (`~/dev_nngpt/overlays/transformers-5.17`; override
with `MOEX_OVERLAY`). Progress: `out/moex/dispatch_status.json`, one log per
arm under `logs/moex/<stage>/`, and `out/moex/report.md`, which
`analyze.py` refreshes after every arm.

## Stages

| stage | question |
|---|---|
| s0 | Is the measurement valid? Noise floor (frozen model, 4 sampling seeds), exact repeat, router perturbations of increasing size, constant router. |
| s1 | Gate-only training: do six router architectures beat the native router trained identically? (+10× lr) |
| s2 | The LoRA path: does a router architecture add anything on top of LoRA SFT each round? |
| s3 | Replicates of the key arms on fresh seeds; the s1 winner is re-measured, not trusted. |
| s4 | The outer self-improvement loop (LLM-proposed gates): keep-best × proposer-SFT, scored after training. |

## Reading the report

Contrast cells are `mean(A−B) [95% CI] p (n)`, paired per seed prompt and
round. **yield** counts a failed generation as accuracy 0, **valid** is the
trained-and-measured rate, **acc** compares prompts both arms turned into a
measured model. Intervals resample prompts (a prompt recurs every round);
p-values are sign-flip permutations over prompts.
