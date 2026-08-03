# NNGPT Agent Guide

## Project Objective

This project is a neural architecture search (NAS) loop driven by an LLM:

1. Select an evaluated neural network as a reference architecture.
2. Ask the LLM to generate improved LEMUR-compatible neural-network code, hyperparameters, and transforms.
3. Validate and execute the generated code, train the resulting network, and measure it on a real task and dataset.
4. Turn measured outcomes into formatted prompt/response examples.
5. Adapt the code-generating LLM, then repeat generation and evaluation.

Accuracy is the usual image-classification feedback signal, but conceptually the objective is the configured metric for the task. Preserve `task`, `dataset`, `metric`, training budget, and other evaluation conditions when comparing candidates. Do not describe a candidate as improved based only on syntax, language-model loss, or plausible-looking code.

The measured metric is generally indirect training feedback rather than a differentiable loss through the generated network. `NNGenPrompt` joins weaker and stronger evaluated architectures; the weaker architecture and score become the user context, and the stronger architecture becomes the assistant target. Failed or invalid generations affect generation success rate but must not be treated as zero-accuracy trained models unless an experiment explicitly defines that reward.

## System Boundaries

Keep these two model layers distinct:

- The code-generating LLM produces candidate PyTorch source.
- The generated CV neural network is separately instantiated, trained, and scored by NNEval.

This branch additionally studies the MoE gate inside the code-generating LLM. That gate is not an MoE layer in the generated CV network. Gate training optimizes causal language-model loss on formatted NNGPT examples; CV evaluation in each pipeline epoch provides downstream evidence about the actual NAS objective.

## Core Data Flow

- `ab/gpt/util/Tune.py::nn_gen` builds generation prompts from LEMUR rows, calls `ChatBot`, and writes candidate artifacts.
- `ab/gpt/conf/prompt/test/NN_gen.json` defines generation-time prompts.
- `ab/gpt/NNEval.py` collects candidate requests, manages workers, records success or failure, copies successful results into LEMUR-visible data, and writes cycle summaries.
- `ab/gpt/util/Eval.py::Eval.evaluate` performs substantive source validation, duplicate checks, and the `ab.nn.api.check_nn` training/evaluation call.
- `ab/gpt/util/prompt/NNGenPrompt.py` queries evaluated data and formats chat examples.
- `ab/gpt/conf/prompt/train/NN_gen.json` defines the current weaker-to-stronger improvement examples.
- `ab/gpt/util/LoRA.py` implements the upstream SFT/LoRA adaptation path.
- `ab/gpt/iterative_finetune.py` is the explicit multi-cycle selection and data-augmentation path.
- `ab/gpt/util/CycleResults.py` aggregates accuracy-shaped scores and generation/evaluation success information.

## Directory Naming Convention

- **A{N} directories** (`A0`, `A1`, `A2`, ...) are dynamically generated epoch directories. `epoch_dir(epoch)` in `ab/gpt/util/Const.py:76-80` resolves to `{nngpt_dir}/llm/epoch/A{epoch}`. The epoch index increments each time a new generation cycle runs; `nn_gen()` creates the directory on demand.
- **B{N} directories** (`B0`, `B1`, ...) are dynamically generated individual candidate model directories created inside the current epoch's `synth_nn/` subdirectory by `nn_gen()` in `ab/gpt/util/Tune.py:248,387,492`. Each call to `nn_gen()` creates a new set of `B{idx}` directories for the candidates generated in that epoch.
- `nngpt_dir` defaults to `out/nngpt` but can be overridden via the `NNGPT_DIR_OVERRIDE` environment variable.
- `NNEval.py` reads these directories but does not create them; `nn_gen()` in `Tune.py` is the sole creator of A and B directories.

Candidate directories normally contain `new_nn.py`, `hp.txt`, `tr.py`, `full_output.txt`, and `dataframe.df`. Successful evaluation adds `1.json`, `eval_info.json`, and `eval_summary.json`; failed evaluation adds `error.txt`. Treat `1.json` as the full nn-dataset trial record, not a disposable summary.

Current evaluation limitations must remain visible when interpreting experiments:

- Although the conceptual objective is the configured metric, NNEval and `CycleResults` currently extract and report `accuracy`/`acc` fields. Verify the backend result schema before using another metric.
- NNEval loads generated `hp.txt`, then updates it with the reference row's `dataframe.df` parameters; overlapping reference values therefore override generated values. Explicit `prm_json` overrides come later, and the run's `nn_train_epochs` always controls `epoch`.
- The normal NNEval path does not import generated `tr.py`, and `copy_to_lemur` does not copy it. Do not claim that generated transform source affected a score unless the execution path has been changed and verified to load it.

## Generated NN Contract

The normal response format is exactly:

```text
<hp>{valid JSON hyperparameters}</hp>
<tr>{complete transform code}</tr>
<nn>{complete PyTorch model code}</nn>
```

Generated model code must be complete and parseable. The LEMUR contract requires:

- A standalone `supported_hyperparameters` function returning the supported parameter names.
- A `Net` implementation with the methods expected by the prompt and nn-dataset, including `__init__`, `forward`, `train_setup`, and `learn`.
- An `<hp>` object containing obligatory `batch` and `transform` keys plus every key declared by `supported_hyperparameters`, without unrelated placeholder keys.
- Every declared hyperparameter to be used by the implementation.
- Code that can be imported and trained under the requested task, dataset, transform, metric, and budget.

Do not weaken evaluation to make generated code pass. `ab/gpt/util/Util.py::verify_nn_code` is not the substantive validator; preserve the checks in `Eval.evaluate` and the nn-dataset execution path. Keep full tracebacks and per-candidate failure artifacts when improving reliability.

## MoE Gate Experiment

`run_moe_gate_cycle.py` is the current full gate experiment for this branch:

1. Load the Hugging Face MoE LLM and discover native router sites.
2. Ask the LLM to generate `LLMGeneratedGate` source.
3. Validate and install the generated scorer, initialize its base projection from the native router, and verify that model forward execution still works.
4. Freeze all parameters except replacement gates.
5. For each pipeline epoch, generate and evaluate CV candidates in its `A{epoch}` directory.
6. Build `NNGenPrompt` data after that epoch's evaluation so successful results can enter the feedback dataset.
7. Train only replacement gates on causal language-model loss.
8. Continue to the next pipeline epoch using the updated gate parameters.
9. Record per-epoch CV metrics and routing diagnostics plus gate-training and post-training validation loss.

The `moe_gate_only/` package is the transactional gate replacement, training, checkpoint, and metrics API. `moe_surgery/` is a separate lower-level replacement API with different lifecycle semantics. Do not mix them without explicitly handling discovery, initialization, verification, freezing, restoration, and model-specific routing contracts.

A generated gate must:

- Import only `torch`/`torch.nn` in the validated cycle path.
- Define `LLMGeneratedGate(nn.Module)` with constructor `(model_dim, num_experts)`.
- Define `base = nn.Linear(model_dim, num_experts, bias=False)`.
- Accept `(..., model_dim)` and return finite raw logits `(..., num_experts)`.
- Leave softmax or sigmoid, top-k, auxiliary loss, and expert dispatch to the native MoE wrapper.
- Initialize any residual path so installation can reproduce native step-zero routing.

The current gate validator checks imports, class/base shape, output shape, finite logits, and successful model execution. It does not numerically compare native and replacement logits, so zero-residual initialization is a contract rather than a proven equivalence. Add an explicit equivalence check when A0 must reproduce native routing exactly.

Use `MoEGateSession` as a context manager unless persistence is deliberate. Verify that only replacement-gate parameters are trainable before optimization. Gate checkpoints (`metadata.json`, `gate_weights.pt`, optional `gate.py` and `optimizer.pt`) are not PEFT/LoRA adapters and are not interchangeable with them.

`A0`, `A1`, and subsequent `A{N}` names always denote NNGPT pipeline epoch directories, not named MoE stages or a before/after ablation. Each `nn_gen` call queries LEMUR again, and successful evaluations from one pipeline epoch can change the database before the next epoch's reference rows are sampled. Use fixed persisted prompts and identical decoding seeds if an experiment must isolate only the effect of gate-weight updates.

Related entry points have narrower meanings:

- `train_moe_gates.py`: standalone built-in or supplied gate training.
- `finetune_deepseek_lora.py`: LoRA baseline, not gate training.
- `verify_lora.py`: held-out language-model-loss comparison, not CV accuracy verification.
- `compare_lora_generation.py`: output-format and generation comparison, not end-to-end NAS evaluation.

Many `ab/gpt/Tune*.py`, agent, RL, and older MoE files are research variants or snapshots. Before using one, verify its call signatures against `ab/gpt/util/Tune.py`, confirm its configuration fields are consumed, and trace its output and database behavior. Do not assume every documented legacy command currently runs.

## Database Safety

LEMUR prompt construction and NNEval share mutable nn-dataset state under `db/`; `ab/gpt/NNEval.py` defaults to `SAVE_TO_DB = True`. A unique output directory does not isolate concurrent jobs from this database.

- Inspect `squeue --me` and coordinate before starting a database-writing experiment.
- Do not run concurrent writers unless the database backend and experiment have been explicitly designed and tested for it.
- Never delete or replace the database, `.lock`, journal, WAL, or shared result data to recover a stalled job without explicit user authorization and a verified backup.
- Do not follow legacy README instructions that remove `db/` without explicit authorization.
- Preserve comparability fields and complete trial records when inserting or copying results.
- Clear or refresh API caches through the existing code paths; do not work around stale reads by deleting shared state.

Generated Python and Hugging Face `trust_remote_code=True` are executable code, not data. The import checks in `run_moe_gate_cycle.py` are not a security sandbox, and standalone gate-source paths may execute less restricted code. Run generated or remote code only in an isolated Slurm job without unnecessary credentials. Never print access tokens or the complete environment.

## Result Interpretation

A process exit code of zero or a final "completed" message is not enough to claim NAS success. Inspect at least:

- Scheduler state and exit code.
- Per-candidate `eval_info.json`, `eval_summary.json`, `1.json`, or `error.txt`.
- Number of generated candidates, successful evaluations, and actually trained models.
- Best and average reported score under comparable evaluation settings, accounting for the current accuracy-shaped result schema.
- Per-pipeline-epoch copies of cycle results for the gate experiment.
- Gate trainability, checkpoint metadata, LM train/validation loss, and routing diagnostics when relevant.

Do not substitute LM loss for the downstream NAS metric. Report both when both are available. If no CV model trained successfully, report the cycle as an evaluation failure even if gate or LoRA training completed.

## Testing Policy

Allowed directly in the editor allocation:

- Syntax compilation of changed Python files.
- `bash -n` for changed shell or Slurm scripts.
- Formatting, static analysis, and small CPU-only tests.
- Targeted tiny-config tests such as `test/test_moe_gate_only.py` and `test/test_moe_surgery_smoke.py`, provided the installed Transformers version supports their model classes and no checkpoint is downloaded.

Use a separate Slurm job for any `from_pretrained` model load, GPU operation, generation, NNEval run, gate/LoRA training, dataset processing, large download, or end-to-end test.

Do not run broad `pytest` by default. Several `test/test*.py` files are executable research or integration scripts that query the database, load models, train, or perform work at import time. Select only tests whose side effects and resource needs have been inspected.

Dependency profiles differ across upstream NNGPT, gate experiments, and newer router tests. There is no single lockfile known to cover all workflows. Check Python, PyTorch, Transformers, PEFT/TRL, CUDA, and flash-attention compatibility before creating a job environment; do not blindly install a wheel built for another Python version.

## Cluster Execution Rules

This repository runs on Slurm cluster `julia2`. Keep code and small logs in `/home`, disposable job files in node-local `/tmp`, and large or persistent artifacts in `/data`.

### Storage

Use this persistent project root:

```bash
DATA_ROOT=/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment
```

Jobs that use Hugging Face must export:

```bash
export HF_HOME="$DATA_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

- Keep source code, scripts, configuration, and small Slurm logs under `/home`.
- Store datasets, model weights, adapters, checkpoints, Hugging Face caches, and persistent results under `$DATA_ROOT`.
- Pass an explicit unique output under `$DATA_ROOT/outputs`, normally including `${SLURM_JOB_ID}`. Do not accept entry-point defaults that write persistent experiments to repository `out/` or `.cache/`.
- Create a fresh virtual environment for each job under `${TMPDIR:-/tmp}`. Install with `pip --no-cache-dir`; do not persist virtual environments or package caches in `/home` or `/data`.
- Request enough `gres/tmp` for the environment, installed packages, staged files, and intermediates. Verify local space before staging large data.
- Move an existing large artifact out of `/home` only after confirming that no running job uses it.

Before a large download or run, inspect storage:

```bash
du -sh /home/s497179 /data/42-julia-hpc-ai-cv-students/s497179
df -h /home /data
```

`df` shows filesystem-wide capacity, not the user's quota. Never describe storage as unlimited. Exact quotas require confirmation from cluster administrators.

### Editor Allocation

The user manually creates a compute allocation before connecting VS Code/Kilo. Its node, partition, GPUs, memory, job ID, and environment can change between sessions.

- Never assume the current host is `jn004`, a login node, or equipped with a particular GPU.
- Do not infer available resources from `SLURM_JOB_ID` alone. The editor process may not retain `SLURM_JOB_PARTITION`, `CUDA_VISIBLE_DEVICES`, or `TMPDIR`.
- Run only the lightweight checks listed above directly in the editor allocation.
- Submit a separate Slurm job for resource-heavy or executable research workflows.
- Never print or persist the complete environment; editor processes can contain credentials.

### Slurm Identity

The Linux user is `s497179`. The Slurm account and default QOS are `computervision`; these are separate from the username and Unix groups.

Every project batch script must include:

```bash
#SBATCH --account=computervision
#SBATCH --qos=computervision
```

Scheduler permissions can change. Query `sacctmgr`, `scontrol`, and `sinfo` when current policy matters. Blank limit fields mean "not shown at this level," not "unlimited."

### Partition Selection

- Never use the `test` partition.
- Use `gpu_computervision` by default for GPU testing, inference, and training within its current time limit.
- Use `gpu_computervision_long` only when required runtime exceeds the normal GPU partition limit.
- Use `standard` only for a documented hardware, compatibility, or scheduling reason not met by the computer-vision partitions.
- Use `cpu_standard`, `small_cpu`, or `large_cpu` for CPU-only workloads according to resource needs.
- Do not use `cpu_long` unless a current scheduler query confirms access.
- Do not request GPUs from a CPU partition unless current policy explicitly permits it.

Do not hardcode `#SBATCH --nodelist`. Let Slurm select a node unless a documented hardware or debugging requirement demands a submission-time constraint.

Use an untyped GPU request when every GPU in the selected partition is suitable. Request a GPU type only when it affects correctness, capacity, or reproducibility. Always request explicit nodes, tasks, CPUs, memory, wall time, GPUs when needed, and sufficient `gres/tmp`. Keep wall time within the partition's current maximum.

Tracked scripts under `slurm/` may predate these rules. Audit account, QOS, partition, node constraints, source-directory resolution, output location, environment creation, cache variables, and `gres/tmp` before every submission. Do not submit a stale template unchanged merely because it exists in the repository.

### Job Submission

- Resolve the source directory from explicit `PROJECT_DIR`, then `${SLURM_SUBMIT_DIR}`, and only then `pwd`.
- Inspect `squeue --me` before submission to avoid duplicate jobs, database contention, and output collisions.
- Use `sbatch`; never substitute `&`, `nohup`, or another unmanaged background process for Slurm.
- Use a unique persistent output path for every job. Never allow two jobs to write the same checkpoint or result directory concurrently.
- Long or preemptible work must save restartable checkpoints under `$DATA_ROOT/outputs`; node-local files are disposable.
- After submission, report the job ID, partition, log path, persistent output path, and scheduler status.
- A pending job is not a failure. Inspect its scheduler reason before changing resources or resubmitting, and never create duplicates merely because a job is waiting.

### Verification

Submission is not verification. Claim functionality is verified only after:

1. `sacct` reports `COMPLETED` with exit code `0:0`.
2. The relevant log contains no hidden failure or incomplete stage.
3. Expected files exist under the persistent output path and have been inspected.
4. Generated and evaluated model counts are nonzero when required.
5. Metrics and artifacts are structurally valid and comparable for the experiment.

On failure, inspect the Slurm log, state, exit code, elapsed time, and resource usage such as `MaxRSS` before changing code or resubmitting. Inspect GPU utilization when available and relevant. Fix the identified cause rather than blindly increasing resources.
