# Gate-Architecture Outer SFT Loop

This document describes the opt-in loop that learns to **generate better MoE gate
architectures**, as opposed to the pre-existing inner loop that trains the *weights*
of one already-generated gate.

## Two loops

| Loop | Question it answers | Where it lives |
| --- | --- | --- |
| Inner (existing) | "Can these gate weights generate better CV networks?" | `_run_epochs` + `moe_gate_only.training` + `Tune.py` CV evaluation |
| Outer (this feature) | "Which gate *architecture* adapts best?" | `moe_cycle.outer_loop` |

The shared CV generation/evaluation path (`ab.gpt.util.Tune`) and the paired
CV-network data preparation (`moe_gate_only.nngpt_data`, `NNGenPrompt`) are **not**
modified by the outer loop.

## Enabling it

```bash
python run_moe_gate_cycle.py \
  --gate-outer-sft \
  --fixed-eval-hyperparameters '{"batch":64,"lr":0.01,"momentum":0.9,"transform":"norm_32"}' \
  --gate-candidates 8 \
  --gate-benchmark ./out/gate_benchmark.json \
  --gate-db ./out/gate_archive.sqlite3
```

The loop is off by default; without `--gate-outer-sft` the cycle behaves exactly as
before.

## Round structure

1. **Prepare once** (`gate_benchmark.prepare_benchmark`)
   Freeze the training rows and the comparison prompts into a manifest with a content
   hash. Reusing a manifest across runs is only allowed when the settings hash matches.
2. **Select SFT examples** (`gate_store.GateStore.training_subset`)
   Read measured gate trials, average repeated trials per architecture, keep only
   architectures above the success-rate/measured-count floor, and sample the top
   fraction.
3. **Train the proposer** (`gate_proposer.train_proposer`)
   Supervised fine-tuning of a persistent LoRA adapter on `requirements -> high-scoring
   gate source` pairs. Bootstraps with the native proposer until enough eligible
   examples exist.
4. **Propose** (`gate_source._generate_gate`)
   Generate a gate with the adapter attached and an isolated gate-only system prompt.
5. **Run the inner loop** (`cycle._run_epochs`)
   Install the gate, train its weights, and evaluate generated CV networks.
6. **Score the trained checkpoint** (`gate_benchmark.evaluate_candidate`)
   Generate from the frozen prompts with the post-training gate, evaluate through the
   shared evaluator, and reduce the outcomes.
7. **Archive and repeat** (`outer_loop.GateOuterLoop.record`)
   Append the trial (including failures) to the SQLite archive and continue.

## Scoring contract

`summarize_scores` averages only finite, in-range **measured** accuracies. Failures are
never treated as zero accuracy:

- `invalid` — the gate produced a CV network that failed verification/quality checks.
- `unavailable` — infrastructure/unknown failure; makes the trial `incomplete`.

A trial is `eligible` for SFT only when it is not incomplete, has at least
`--gate-min-measured` measurements, and meets `--gate-min-success-rate`. The reported
objective string is `mean_measured_accuracy_with_success_floor_v1`.

## Isolation guarantees

- The proposer adapter is attached only while training/proposing and is **unloaded
  without merging**; base weights and trainability flags are restored before any
  candidate installation, CV generation, or evaluation.
- Candidate gate weights are never carried into the proposer and are restored by the
  session context after each candidate.
- In outer mode every candidate sees **empty gate feedback** in its generation prompts,
  so a gate is not scored in a prompt that mentions itself.
- The proposer's own SFT does not run during the inner comparison.

## Environment

Use the project virtual environment, e.g. `nn-gpt/.venv/bin/python`.

## Known limitations

- The frozen benchmark is **adaptive search data**, not an unbiased final test set.
- CV evaluation randomness is upstream; seeds are matched where possible but full CV
  determinism is not guaranteed by the shared evaluator.
- Full GPU end-to-end validation requires the real MoE model; the logic is covered by
  CPU unit tests plus a CPU LoRA proposer smoke test.
