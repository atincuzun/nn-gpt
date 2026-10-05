# Changelog

All notable changes to NNGPT are documented in this file.

The project follows [Semantic Versioning](https://semver.org/) and releases are
tagged on GitHub and published on PyPI as `nn-gpt`.

## [Unreleased]

### Added
- JOSS submission support: continuous integration, contribution guidelines, and
  this changelog.

## [1.0.7] - 2026-07-10

This release adds the first tracks for models that must actually run on a phone, for training the generator from preferences rather than raw supervision, and for deciding when training can stop early. The common thread is avoiding wasted work: evaluate only when it can change a decision, stop training once final accuracy is predictable, and keep evaluation processes warm.

### Added
- **Edge/mobile generation track** (`ab/gpt/edge/`), because accuracy alone does not say whether a model is deployable. `EdgeGen_k4.py` reuses the curriculum tuner but sets `sft_max_length=16384` and `packing_strategy='wrapped'`: trl's 1024-token default silently truncated the ~10–14k-token k=4 prompts, and trl≥0.24 rejects a custom collator in BFD packing mode, so both are overridden locally instead of in the shared trainer. `_install_reference_filter(max_params, min_acc)` narrows the reference prefix to models the edge target can actually imitate.
- **Edge scoring** (`EdgeBench.py`, `EdgeScore.py`): `count_params`, TFLite INT8 export/convertibility, and CPU latency (litert interpreter, with the PyTorch fallback recorded as `latency_source` so mixed runs stay comparable). Runs are ranked by efficiency = accuracy/latency under a <6M-parameter budget, so the best epoch is picked by the metric that matters on device, not by raw accuracy.
- **Accuracy predictor** (`ab/gpt/AccPredictor.py`, `ab/gpt/util/acc_client.py`): a QLoRA Qwen3-8B predicts `best_accuracy`/`best_epoch` from the first epochs' accuracies plus the model code. Training a candidate to convergence is the most expensive part of the loop, so predicting the peak lets the pipeline stop early; `infer_arch_family` and per-family splits exist because accuracy trajectories differ by architecture family, and the `acc_client` vLLM path keeps the predictor off the training GPU.
- **PPO as a second RL algorithm** (`ab/gpt/TunePPO.py`): PPO uses the real CIFAR-10 test accuracy of the trained candidate as the reward, since architecture quality is only visible after training; an explicit `ValueHead` and separate rollout/update phases let it be compared against the existing GRPO path rather than replacing it.
- **Self-contained KTO pipeline** (`ab/gpt/kto_pipeline/`, `ab/gpt/util/KTO.py`): KTO learns from unpaired desirable/undesirable completions, which is exactly what a generate→evaluate loop produces, so the pipeline bootstraps its own preference data instead of requiring paired preferences. Each cycle generates, evaluates, novelty-prefilters, buckets and trains with an adaptive `_effective_threshold` and dynamic `_class_weights` (failures outnumber successes), and a MinHash-Jaccard `SimilarityIndex` penalises near-duplicates to stop the generator collapsing onto one solution.
- **Curriculum pipeline** (`ab/gpt/curriculum/`, `ab/gpt/util/Tune_Curriculum.py`): references are ordered L1 (similar) → L2 → L3 (distant) so the LLM first learns evaluator-compatible patterns it can later vary; each step merges its best adapter into the cumulative backbone. Steps whose dataset has no viable similarity band (e.g. MNIST, CelebA-Gender, CIFAR-100) are skipped instead of failing, and `--resume` continues after a crash.
- **Persistent NNEval worker pool** (`nneval_worker_pool.py`, `nneval_worker_bootstrap.py`): evaluations previously reloaded models and rebuilt the CUDA context every time; workers now stay alive per GPU, and `build_nneval_worker_plan` sizes the pool by free GPU memory so evaluation can share GPUs with training without OOM, restarting only the affected worker on a fatal CUDA error.
- **Mobile deployment** (`mobile_iterative_finetune.py`): a standalone `--mobile_deployment` extension, so the shared iterative pipeline is untouched. INT8 TFLite export and on-device `adb` benchmarking are gated by non-decreasing score enforcement, because post-quantisation accuracy and real latency differ from host numbers, so a mobile step must never make the accepted result worse.
- **Shared infrastructure**: `ArchDiscovery.py` AST family/graph signatures (structural identity across cosmetic rewrites, for novelty and predictor family splits); `training_runtime.py` GPU-role plan and checkpoint resume; `DatasetSplit.py` stratified `trainvaltest`; `GenerationDType.py` (`DTypeSafeLinearWrapper`) to stop quantised-head dtype mismatches corrupting generation; `eval_checkpoint.py`, `torchvision_prewarm.py`, `lemur_enrichment.py`, `EditUtil.py`, `rl_pipeline/*`, layerwise-LR helpers, `db_query.py`, `plot_rl_reward.py`; and new tests under `test/`.

### Changed
- `requirements.txt`: `nn-dataset>=2.2.11`; added `packaging`, `ninja` and `httpx` (kernel builds and the predictor client); removed the `torch`/`diffusers` pins so installs follow `nn-dataset`'s compatible versions. The in-tree version string was bumped 1.0.5 → 1.0.6 during this cycle, so 1.0.7 is the first release after 1.0.5.
- `Chatbot.py`: system-prompt support, ` thinking`-stripping (`NNGPT_STRIP_THINK_OUTPUT`) so reasoning output cannot leak into extracted code, dtype-aligned generation heads, `add_special_tokens=False`, and `raise RuntimeError` instead of silently returning `None` so failures surface.
- `Tune.py` / `TuneNNGen.py` / `TuneBackbone.py`: system prompts, joint-NN seeds (`JoinConf`), backbone `nn_prefixes`/dataset filtering, a classification path, prompt batching, and a trl-version-introspected `SFTConfig`; plus `classification_mode`, `context_length`, `only_best_accuracy`, `load_in_4bit` and `mobile_*` flags and auto-merge via `MergeLLM.rebuild_from_lineage`.
- `Reward.py`: a multi-horizon reward (score candidates across training horizons, so a short run cannot mis-rank a late-improving model), persistent GPU eval workers, CPU smoke pre-validation and OOM backoff so bad candidates fail before consuming GPU time, and dual-backbone evaluation.
- `Const.py`: `AB_GPT_NNGPT_DIR`/`NNGPT_DIR_OVERRIDE` output redirection, so concurrent experiments write to separate output trees.

### Fixed
- `Chatbot._direct_generate_batch`: corrected `padded_input_length` slicing and the vocab clamp, which had corrupted batched generation whenever padding was present.
- `Eval.py`: per-process isolated trainer tmp modules, preventing races between concurrent evaluation workers, and checkpoint-aware retraining.
- `CycleResults._extract_accuracy_from_eval_info`: tolerant parsing of legacy/dict `eval_results`, fixing metric collection across trl and 4-bit variants.

### Removed
- `ab/gpt/util/Merge.py` (superseded by `MergeLLM.py`), `TuneNNGen_1_3B_code_ONNX.py`, the local NN Dataset install reference (the pip package is used instead), and generated FractalNet architecture artifacts.

### Documentation
- README: the AccPredictor pipeline, `TuneNNGen_delta` (arXiv:2605.04903), and the LangGraph multi-agent workflow with the optional `--use_predictor` accuracy-prediction node.

## [1.0.5] - 2026-03-26

### Added
- **LangGraph multi-agent orchestration** (`ab/gpt/agents/`): the epoch loop is decomposed into thin `generator`/`evaluator`/`finetuner` nodes that wrap `Tune`, with a `manager_node` that owns only routing and the epoch stop condition, so training logic stays in `Tune.py` and the workflow can change without touching it. Evaluation is wired directly after generation (no manager round-trip), because evaluation always follows generation — a failed generation therefore cannot strand the loop; `evaluator`, `finetuner` and the optional `predictor` return to the manager. `MemorySaver` checkpointing lets a crash mid-epoch resume from the last completed node instead of epoch 0, which matters for multi-hour single-GPU runs. Enabled via `tune(..., use_agents, use_predictor)`.
- **Optional predictor node** (`predictor.py`): calls `AccPredictor.predict_best_accuracy` on state collected by `evaluate_step` after two epochs. It is opt-in (`--use_predictor`) and returns `partial_success` when the predictor or its inputs are absent, so the default pipeline never pays for prediction it does not use.
- **ONNX generation pipeline** (`OnnxExport.py`, `OnnxWrapper.py`, `Tune_Onnx.py`): generation is the latency and VRAM hotspot, so a dual-model loop trains with PyTorch FP16/QLoRA but generates with an Optimum `ORTModelForCausalLM` (opset 17, CUDA EP) via `--onnx_run`. The base model is exported unquantized because exported from a quantized checkpoint it produced corrupted output; the ONNX graph is freed before fine-tuning and reloaded afterwards so both models never coexist on one GPU.
- **LoRA adapter merge system** (`Merge.py`, `MergeLLM.py`, `Mergedecision.py`): merging every epoch adapter would accumulate regressions, so a buffer collects `BUFFER_SIZE=5` candidates, only the most accurate is considered, and a per-epoch KEEP/REVERT decision keyed on `cycle_results.json` accuracy (`IMPROVEMENT_EPS=1e-4`, strict improvement) keeps only adapters that actually help. Decisions are persisted to `accepted_adapters.json` and `merge_decision.json` — and written even on REVERT — so the chain is reproducible; `rebuild_from_lineage` rebuilds from the recorded set. Wired to `enable_merge`.
- **Delta code generation** (`DeltaUtil.py`): full-file generation is token-expensive and drifts, so the model emits a unified diff over a baseline (`compute_delta`/`apply_delta`), applied with the system `patch` and a manual unified-diff parser fallback so it works where `patch` is missing. `repair_code`/`validate_delta` gate malformed edits, and `_MAX_DELTA_RETRIES=2` retries feed the error back. `compute_novelty_jaccard` (MinHash, τ=0.90) rejects near-duplicates so edits stay genuinely new rather than memorized.
- **Backbone/SFT tuning** (`SFTUtil.py`, `prompt/SFTGenPrompt.py`): generation is constrained to a fixed skeleton (TorchVision wrapper, `FractalBlock`/`FractalUnit`) where only `<block>/<init>/<forward>` XML sections are filled, which keeps output compilable and within a known-good infrastructure.
- **Iterative fine-tuning pipeline** (`iterative_finetune.py`, `iterative_pipeline/`): `novelty_checker` and `structural_reranker` prioritize high-potential candidates before expensive evaluation; `training_data_manager` accumulates successful models as chat data; `gpu_memory_manager` guards against OOM; `analyze_cycles`/`aggregate_cycles`/`evaluate_cycle_models`/`plot_cycles`/`pipeline_validation` report across cycles.
- **Iterative prompt improvement** (`iterative_prompt_improvement/`): closes the loop on the prompt itself — generate → evaluate → improve — per arXiv:2603.12091, learning from prior improvement attempts instead of repeating them.
- **Transform code-generation pipeline** (`prompt/TransformGenPrompt.py`, `Tune.trans_gen`, `trans_mode`).
- **Metrics & cycle reporting** (`Metrics.py`, `CycleResults.py`, `DatasetCompareBuilder.py`, `Reward.compute_cv_reward_simple`, `nn_sftcodegen_rag.py`): BLEU and a machine-readable `cycle_results.json` give each cycle a durable record that also drives the merge decision.
- **Batched prompt inference** (`Chatbot.chat_batch`, `_direct_generate_batch`, `prompt_batch`), opt-in with safe fallback because batching is faster but must not change single-prompt behavior.

### Changed
- **`Tune.py` rewritten as the single source of truth** (`nn_gen`, `trans_gen`, `_evaluate_epoch`, `_finetune_epoch` plus thin `generate_step`/`evaluate_step`/`finetune_step` used by the agents), adding evolutionary base-model chaining so each epoch builds on the best model so far.
- **`LLM.py`**: optional Unsloth fast path (`use_unsloth`/`load_in_4bit` via `FastModel`), bfloat16 default, DeepSpeed ZeRO-3 guard (incompatible with Unsloth), `pad_token` mapped to `eos_token`.
- **`Chatbot.py`**: ONNX-aware construction, prompt-prefix stripping and normalized pipeline output.
- **Dependencies**: torch 2.9.1, `nn-dataset>=2.2.8`, `nn-rag`/`nn-dup>=2.2.4`, `langgraph>=1.0.4`, `unsloth`, `optimum[onnxruntime-gpu]`, `onnxruntime-gpu`, `datasketch`, `diffusers==0.33.0`; `Const.py` gains `nngpt_dir`, `nngpt_upload`, `base_llm` and variadic `epoch_dir(*args)`.

### Fixed
- Removed the fake-accuracy fallback in `Merge.py` — it reported numbers that hid failures and polluted merge decisions.
- `Chatbot` prompt-batch generation corruption (bad padding/slicing produced garbage output).
- Code-extraction issue; ONNX export now deletes a stale quantized HF cache (an old quantized checkpoint made the export emit gibberish); JSON property quoting in `Util.py`; CUDA kernel bug.

### Removed
- Deleted `LLM_ONNX.py`, `Chatbot_Onnx.py`, `Util_Onnx.py`, `Const_Onnx.py` (superseded by the unified `Tune_Onnx` path), `Testing.py`, `generate_variants.py`, `base_nn/`; dropped the `rag` extra and the `deepspeed` pin.

### Documentation
- README gains a "LangGraph Multi-Agent Workflow" section (agent flow, crash recovery, `--use_agents`/`--use_predictor`, agent file table).

## [1.0.4] - 2025-10-27

### Added
- **Joint multi-NN prompt data** — `NNGenPrompt.get_raw_dataset` passes a single `ab.nn.api.JoinConf` (`num_joint_nns`, `same_columns`, `diff_columns`, `enhance_nn`) to `lemur.data`, replacing the old "fetch addon rows once, then re-filter with pandas for every prompt" path. One database-side join is far cheaper than per-row filtering, and it makes "keep these columns equal, vary those" declarative in `NN_gen.json` (`num_joint_nns`, `nn_prefixes`, `_2`-suffixed addon fields) instead of depending on implicit same-prefix matching.
- **GA mutation search** (`ab/gpt/brute/ga/mutation/`), kept separate from the architecture GA: `MutNet_evolvable.py` (`SEARCH_SPACE`), `genetic_algorithm.py` (tournament selection, crossover, elitism, pickled checkpoints) and `run_mut_evolution.py` (CIFAR-10 fitness, MD5 dedup, `ga-champ-mut.py`). Scoring layer choices directly against measured accuracy gives an LLM-free baseline and a search that can resume from a checkpoint.
- **Structured logging** (`util/simple_logger.py`): prompts/completions/rewards go to `rl_output/generation_log.json`, making RL runs diagnosable and success rates countable without re-evaluating.
- **Transformer path and sampling controls** — `Util.extract_transform` / `Const.transformer_file='tr.py'` make `ChatBot.chat` return the `<tr>` block, and `temperature`/`top_k`/`top_p` become `ChatBot`/`tune` arguments with CLI flags, so one generation can emit the transform it assumes and the diversity/quality tradeoff is tunable per model and stage instead of hard-coded.

### Changed
- `NNEval` → `Eval` (`ab/gpt/util/Eval.py`), reflecting its role as the shared evaluation entry point.
- `TuneRL.py` rebuilt as an explicit SFT→GRPO curriculum on `deepseek-ai/deepseek-coder-6.7b-instruct`: `mask_layer_definitions`/`format_for_sft` build a "fill in the masked layers" SFT set, then GRPO optimizes `compute_reward` (real `api.check_nn` accuracy), saving only positive-reward models. The stronger code model plus the two-stage curriculum bias generation toward code that actually trains, replacing the looser 1.3B path.
- `TuneAccPrediction.py` rebuilt around `Qwen/Qwen3-8B` QLoRA (`MAX_LENGTH=1024`, `gradient_checkpointing_enable(use_reentrant=False)`, `paged_adamw_8bit`, `create_peft_config(["q_proj","k_proj"])`, RMSE/MAE/≤5%/≤10%). The larger backbone with tightened memory settings fits the run without OOM, and the tolerance metrics match the early-stopping decision the predictor serves.
- `TuneNNGen.py`: `--start_from_layer/--end_at_layer` → `--start_layer/--end_layer`, and `MAX_PROMPTS` 36K→4K until a faster SQL-backed prompt query exists, since a full LEMUR pull made iteration too slow; `AlterNNFN.py` adds `nn.MaxPool2d(kernel_size=3, stride=2)` and extends enumerated combos to 2–5 elements so the fractal search actually covers downsampling.
- Packaging/prompts: `pyproject.toml` gains AI/Code-Generator classifiers, keywords and a Bug Tracker URL; `requirements.txt` bumps `nn-dataset>=2.1.1` (needed for `JoinConf`); `NNRLPrompt.py` builds `text` via `apply_chat_template`.

### Fixed
- `ab/gpt/NNEval.py`: loads LLM-recommended hyperparameters from `hp_file` (falling back to CLI defaults) so each model trains with the params its generator proposed; fixed the `args.only_epoch` crash and the duplicated `-nae` short flags (`-ae/-oe/-te`).
- `Tune.py`: `release_memory()` after LoRA training and after evaluation to curb OOM; `extract_hyperparam` also handles the `</ hp >` variant.

### Removed
- `NNGenPrompt` global `models_with_the_same_prefix` and per-row addon pandas filtering (superseded by `JoinConf`); the old SFT path and 4-bit config from `TuneRL.py`; GA scripts moved to `ab/gpt/brute/ga/architecture/`.

### Documentation
- README Docker section relocated under "Docker" and updated to `abrainone/ai-linux:llm`.

## [1.0.3] - 2025-10-03

### Added
- `ab/gpt/TuneNNGen.py`, a single generic fine-tuning driver, because every model variant previously carried its own near-identical copy of the LoRA/`TrainingArguments` setup, so fixing or changing a default meant editing all of them. `main()` now builds `TrainingArguments` + `LoraConfig` and calls `Tune.tune`, exposing layer range, LoRA `r`/`alpha`/`dropout`/`target_modules`, LR, batch/grad-accum, optimizer, `max_prompts`, `max_new_tokens`, `save_llm_output`, `use_deepspeed`, `nn_name_prefix` and `nn_train_epochs` as CLI flags, so a new experiment is a command line rather than a new file and any run stays reproducible from its arguments.
- Thin entrypoints (`TuneNNGen_1_3B_code.py`, `TuneNNGen_7B.py`, `TuneNNGen_7B_code.py`, `TuneNNGen_8B.py`, `TuneNNGen_1_3B_code_nngpt.py`, `TuneNNGen_7B_code_olympic.py`) that only pin model-specific defaults and delegate to `TuneNNGen.main`, keeping per-model differences explicit while sharing one code path.
- `ab/gpt/util/Code.py` (`improve_code` = `normalize_top_indent` + `dedup_imports` + `strip_comments`), because LLM output routinely came back with a stray first-line indent, duplicated torch imports or prose comments that broke `exec`; routing `Util.extract_code`/`extract_hyperparam` through it makes every extraction site tolerant of the same defects instead of each caller re-patching them.
- `ab/gpt/util/RAG_AlterNN.py`, a block-based RAG pipeline, because retrieving whole reference files flooded the prompt with unrelated code. It detects the block type (`_detect_block_type`/`_get_prompt_for_block`), prepends the retrieved block verbatim so the saved file is self-contained, strips imports for packages not installed (`_strip_unavailable_imports`), caches blocks via `BlockExtractor`, retries up to `MAX_TRIES=3` feeding the traceback back to the model so it can self-correct, and enforces a 2M `PARAM_BUDGET` to avoid OOM at training time.
- Captioning pipeline `ab/gpt/util/AlterCaptionNNtune.py` with `NNAlterCaption_8B.py`, adding `_extract_best_code`/`_syntax_ok`/`_ensure_net_methods_ast` plus an LLM repair loop so malformed captions are corrected before they reach evaluation.
- FractalNet generation (`ab/gpt/brute/fract/AlterNNFN.py`, `NNAlterFractalNet.py`, rewritten `Fractal_template.py`): enumerates element permutations into up to 1200 template variants so the architecture space is swept systematically without depending on an LLM.
- New AST mutation package under `ab/gpt/brute/ast/mutator/`.
- `AlterNN.alter(..., n=)` with `format_prompt_with_supporting_models`, because prompts previously saw a single reference network; fetching n supporting models per prompt (and sampling away the original) gives the LLM several accurate models for the same task, handling fewer-than-n gracefully.

### Changed
- `util/Tune.py`: `tune()` handles `nn_train_epochs`/`max_prompts`/`save_llm_output`/`max_new_tokens`/`nn_name_prefix`, and in-loop evaluation is delegated to `NNEval.main` instead of the duplicate evaluator that lived here, so evaluation logic exists in one place and can be run standalone.
- `NNEval.py` refactored into a callable `main()` plus `__main__` CLI, adding `--only_epoch`, `--custom_synth_dir` and `--epoch_limit_minutes` so long evaluations can be bounded and resumed.
- `Chatbot.chat` `max_words` → `max_new_tokens` and `Prompt.get_dataset` `n_training_prompts` → `max_prompts`, naming the actual unit/limit.
- `requirements.txt`: `nn-dataset>=2.0.6` for the newer dataset API, `rag_NNAlter_7b.py` renamed `RAG_NNAlter_7b.py` for naming consistency with `RAG_AlterNN`; unused `mpi4py` dropped.

### Fixed
- `NNEval.evaluate` deduplicates its checksum lookup with a single early return, `extract_hyperparam` normalizes `< hp >`, and `extract_str` no longer leaks trailing text.
- `Fractal_template.py`: corrected `Net` method indentation/AST, dynamic `in_features`, `_norm4d` 4D/5D handling, device sync, AMP, checkpointing and non-finite-loss batch skip so generated variants survive training.
- `AlterCaptionNN`: distinct PAD token/attention mask to silence generation warnings.

### Removed
- `ab/gpt/TuneRAG.py`, `ab/gpt/util/rag_AlterNN.py`, `ab/gpt/NNAlterCaptionAll.py` and `ab/gpt/brute/fract/Generate_variants.py`, superseded by the consolidated driver and block-based RAG pipeline.

### Documentation
- README: `pip install nn-gpt` instructions, CUDA 12.6 / 24 GB note, new logo, updated citation.

## [1.0.2] - 2025-08-20

### Added
- RAG-based NN synthesis (`ab/gpt/rag_NNAlter_7b.py` → `ab/gpt/util/rag_AlterNN.py:alter`): rather than asking the LLM for an entire architecture at once, each prompt is given a verified building block retrieved through `ab.rag.retriever.Retriever` (`BLOCKS_100`) and only the wrapping code is generated, so synthesis is grounded in real, known-good layers instead of invented APIs, and the block is prepended verbatim so every model written to `synth_dir` is self-contained.
- Retry/validation gate in `rag_AlterNN.py`: a candidate must compile and execute, expose `Net` and `supported_hyperparameters`, and stay under `PARAM_BUDGET=2_000_000` in `numel()`, so broken or OOM-prone nets are rejected before training instead of crashing a long run; on failure the traceback is fed back and the retry switches to sampling (temp 0.2 → 0.7), so the model can escape a repeated mistake rather than reproduce it.
- Caption-guided synthesis (`ab/gpt/util/AlterCaptionNN.py:alter`; new `NNAlterCaptionAll.py` with `--mode {structural,numeric,both}`), which builds prompts from the best-accuracy row per `nn` so variants inherit the design of models that actually trained well, exporting them as `<nn>_ALTER_<N>.py` for direct reuse.
- New entry points `NNAlter_30B_code_qw.py` (`Qwen3-Coder-30B-A3B-Instruct-FP8`) and `NNAlter_7B_code_olymp.py` (`OlympicCoder-7B`), to compare a large general coder against a code-specialised small one.
- Optional dependency groups `flash`, `stat`, `rag` in `pyproject.toml`, so the base install stays light and heavy or situational packages are pulled in only when used.

### Changed
- Safer LLM/config loading (`LLM.py`): the local `config.json` is read and sanitised (a `null` `quantization_config` is dropped and the config loaded through a temp dir with `AutoConfig`), and `quantization_config` is forwarded only when set; stale/null quantization fields and passing `None` into HF internals were crashing model load, while `pad_token_id=0` is preserved so existing tokenisation behaviour does not change.
- Comment/docstring stripping (`Util.py`): the new tokenize-based `strip_comments` is applied in `extract_code`/`extract_hyperparam`, and `extract_str` now returns `None` when the start token is absent, because generated text often wraps code in prose that broke `exec`; tokenizing removes comments without mangling string literals, whereas the missing-token case previously returned the whole output instead of nothing.
- Generation length raised to `64*1024` tokens in `AlterNN.alter`, `NNLayer.alter` and `AlterCaptionNN.alter`, since full architectures were being truncated mid-file and the new limit is large enough for a complete model (and its caption) to finish.
- Hyperparameter-use check (`NNEval.py`): declared `supported_hyperparameters` are AST-parsed and a model is rejected when a declared key is never referenced, which stops the pipeline from tuning knobs the network ignores and producing meaningless hyperparameter results.
- GA and packaging: `run_evolution.py` names outputs by checksum (`ga-<checksum>`, champion `ga-champ.py`) so runs are reproducible and deduplicable, and `pyproject.toml` merges `requirements-no-isolation.txt` into `dependencies` while `requirements.txt` bumps `nn-dataset>=2.0.3`, keeping one source of dependencies that matches the new RAG API.

### Fixed
- GA channel indexing (`AlexNet_evolvable.py`): `in_channels = in_shape[0]` → `in_shape[1]` in both `Net` and `generate_model_code_string`, because the leading element is not the channel count, so generated AlexNets previously built a wrongly sized first convolution and failed evaluation.

### Removed
- `ab/gpt/util/retrieval.py` (FAISS `CodeRetrieval`), because retrieval now lives in the shared `ab.rag.retriever`, removing a duplicated implementation to maintain.

### Documentation
- README install steps corrected to `req-no-isolation.txt`, matching the requirements file actually shipped, so the documented non-isolated flash-attention install works.

## [1.0.1] - 2025-08-12

### Changed
- Packaging moved from `setup.py` to a declarative `pyproject.toml` (PEP 621, `setuptools.build_meta`). Version, runtime dependencies and the optional `no-isolation` extra are now read from `version`, `requirements.txt` and `requirements-no-isolation.txt`, so each fact is stated once and the package metadata can no longer drift from the files that actually drive installs.
- `MANIFEST.in` deleted: it declared the same `ab/*.py|json|md`, `requirements.txt`, `README.md`, `LICENSE` and `version` entries that `pyproject.toml` now expresses via `include-package-data`, `package-data` and `data-files`, so the sdist/wheel contents are defined in one place instead of two that could disagree.
- `requirements.txt` refreshed: `nn-dataset>=1.2.6` → `>=2.0.0`, because the code now relies on the newer dataset API; `torch==2.7.1` pinned explicitly instead of relying on a transitive version, so CUDA/torchvision builds are reproducible; the `flash_attn==2.7.4.post1` pin dropped and the list reordered, since a package that needs `pip install --no-build-isolation` does not belong in the default install set.
- New `requirements-no-isolation.txt` pinning `flash_attn>=2.8.2`, so the special `--no-build-isolation` install path is opt-in and versioned separately from the ordinary dependency set.
- `NNAlter_7B_coda.py` → `NNAlter_7B_code.py`, `TuneNNGen_1_3B_coda.py` → `TuneNNGen_1_3B_code.py`, `TuneNNGen_7B_coda.py` → `TuneNNGen_7B_code.py`: `coda` was a misnomer for the `code` model entry points, and the rename makes each script's target model unambiguous. The file contents are unchanged.
- Version bumped `1.0.0` → `1.0.1`.

### Documentation
- README install block adds the `--no-build-isolation` step and clarifies the roles of `NNAlter*`, `NNEval` and `TuneNNGen*`, so the new packaging path is usable and the renamed entry points are discoverable.

## [1.0.0] - 2025-08-11

The first tagged release of the refactored generator: the ad-hoc `finetune_*` scripts of 0.1.0 were replaced by one reusable fine-tuning loop, so supporting a new model or task no longer means copying and editing a script.

### Added
- `ab/gpt/util/LLM.py` (`LLM`): unifies base-model/tokenizer loading with 4-bit `BitsAndBytesConfig`, local caching and optional `gguf_file`/DeepSpeed, because every pipeline had duplicated loader code that drifted apart; `get_max_length()` lets callers size prompts from the model's real context.
- `ab/gpt/util/Tune.py` (`tune`/`nn_gen`): the multi-epoch generate→finetune loop, so generation, evaluation and training run as one pipeline; it builds prompts from LEMUR best models, parses `<nn>`/`<hp>` via `ChatBot`, evaluates with `NNEval`, regenerates code after exceptions (one bad sample no longer aborts an epoch) and copies accepted models into the dataset.
- `ab/gpt/util/AlterNN.py` (`alter`) with `NNLayer.py`: generate altered NN code from seed LEMUR models via `apply_chat_template`, so mutation/improvement is decoupled from fine-tuning and can be run on its own.
- `ab/gpt/util/NNEval.py` (`NNEval`): wraps `ab.nn.api.check_nn`, validates the required functions and skips duplicates by `uuid4` checksum, so code is evaluated and stored only when valid and new — avoiding wasted evaluation and duplicate LEMUR rows.
- `ab/gpt/util/LoRA.py` (`LoRA`): QLoRA/DoRA trainer (r=32, alpha=64, `use_dora=True`, automatic linear target discovery), so all entry points share one training implementation instead of per-script setups.
- `MergeLLM.py` and `Util.py` (`extract_code`, `extract_hyperparam`, `copy_to_lemur`, `verify_nn_code`, `nn_accepted`): shared parse/accept/merge helpers, so code extraction and dataset import behave identically everywhere.
- Prompt abstraction `util/prompt/` (`Prompt`, `NNGenPrompt`, `NNPrompt`, `NNRLPrompt`) replacing the old preprocessors; `Prompt.get_dataset` centralizes tokenization, length filtering and shuffling, so a new task only needs `get_raw_dataset`.
- Entry points `TuneNNGen_1_3B_coda.py`/`TuneNNGen_7B.py`/`TuneNNGen_7B_coda.py`/`TuneNNGen_8B.py` and `NNAlter_7B.py`/`NNAlter_7B_coda.py`/`NNAlter_8B.py`, all thin wrappers over `Tune.tune`, so model choice is a config/flag rather than a new script.
- `TuneAccPrediction.py` (`LLMFineTuner`): predicts `best_accuracy`/`best_epoch` from epoch-1/2 features with group-aware `stratified_split`, so accuracy prediction can steer early stopping.
- `TuneRL.py`: SFT + GRPO where `reward_fn`/`compute_reward` score generated code through `api.check_nn`, tying the optimizer to measured accuracy instead of next-token loss alone.
- `TuneRAG.py`: retrieval-augmented fine-tuning injecting GitHub/dataset snippets via `CodeRetrieval` + FAISS, so the model can condition on real code it was never trained on.
- `ab/gpt/brute/`: GA search (`GeneticAlgorithm`, `AlexNet_evolvable.py`, CIFAR-10 `run_evolution.py`), the fractal generator (`fract/`) and `trans/GenerateTransform.py`, giving non-LLM baselines for the generated architectures.
- `ab/gpt/conf/Conf.py`: RAG config constants, so retrieval paths and models are not hard-coded in scripts.

### Changed
- `ChatBot.chat()` now returns `(nn_code, hyperparams, full_output)` and extraction moved into `Util`, because callers needed hyperparameters and the raw reply, not just code; `self.model.eval()` disables dropout so inference is deterministic.
- `Const.py` centralizes the output layout and loads `base_llm` from `conf/llm/*.json`, so paths and the active model live in one place instead of being repeated per script.
- `requirements.txt`: `nn-dataset>=1.2.6`, added `gguf` and `trl` (GGUF loading and the RL pipeline), and loosened pins so installs are not blocked by stale upper bounds.
- Renames (`finetune_hyperparameters.py`→`TuneHyperparameters.py`, `finetune_rag.py`→`TuneRAG.py`, `LoRATrainer.py`→`LoRA.py`, `preprocessors/`→`prompt/`, `PreprocessorBase.py`→`Prompt.py`), so names describe the pipeline rather than the old script layout.
- `ab/gpt/__init__.py` sets `WANDB_MODE=disabled` globally, so no run unexpectedly blocks on or requires W&B credentials.

### Fixed
- Duplicate evaluations are skipped by checksum in `NNEval.evaluate`; `nn_gen` captures tracebacks and can ask the LLM to repair failing code; 7B training filters by `context_length`; and `--nn_name_prefix` names generated models predictably — each fix prevents a wasted evaluation, an aborted epoch or an oversized prompt.

### Removed
- Legacy `finetune_nn_gen.py`, `finetune_acc_prediction.py`, `initial_generation_chg.py` and helpers `ModelLoader.py`, `CVModelEvaluator.py`, `merge_ft_model.py`, old preprocessors and `conf/config.py`, now superseded by `Tune.py`/`LLM.py`/`NNEval.py`/`prompt/`, leaving a single code path.

### Documentation
- README rework: LEMUR setup/update, Docker install/run, the `NNAlter*`/`TuneNNGen*` workflow and the NNGPT/HPGPT citations, so the pipeline can be reproduced without reading the source.

## [0.1.0] - 2025-04-01

First packaged release of `nn-gpt`, turning the initial `# code-llm` stub into an installable LLM-driven neural-network generator; before this the work lived as ad-hoc scripts run from a checkout, with no version, no installable name, and no shared output layout.

### Added
- **Packaging** (`setup.py`, `version`, `requirements.txt`): the project could only be run by hand from its source tree, so `setup.py` now installs the `nn-gpt` name with the version read from the `version` file, Python `>=3.10`, and `extras_require={'stat': ['nn-stat']}` keeping the accuracy-statistics dependency optional; `requirements.txt` pins `nn-dataset`, `transformers~=4.48`, `peft`, `bitsandbytes`, `deepspeed`, `sentence-transformers` and `faiss-cpu` so a fresh environment reproduces the same LLM-plus-CV stack instead of silently drifting.
- **Standalone generation** (`ab/gpt/initial_generation_chg.py`): generating candidate networks needed a cheap, single-shot entry point before the expensive fine-tune loop, so the script loads `DeepSeek-R1-Distill-Qwen-7B`, reads `conf/test_nn_chg_prompts_generation.json`, samples best-accuracy rows from `ab.nn.api` and fills prompt templates via `input_list`/`addon_list`, writing each candidate's `code.py` plus its provenance `dataframe.df` under `out/Models/epochs/A{epoch}/synth_cv_models/B{i}` so the evaluator can later recover the exact task/prm it was generated for.
- **Iterative pipeline** (`ab/gpt/finetune_nn_gen.py`): raw generation produced many nets that would not run, so this driver closes the loop per epoch — generate → evaluate → repair → fine-tune — and, when evaluation raises, feeds the traceback back to `ChatBot` for up to two repair attempts while logging `error_{tries}.txt`, rather than discarding the sample; accepted results are copied into `out/Dataset/nn` and `stat`, and `-s/--skip` and `-p/--peft` let long runs resume instead of restarting from scratch.
- **Model handling** (`util/ModelLoader.py`, `util/merge_ft_model.py`, `util/Const.py`): repeatedly pulling the base model from the Hub was slow and fragile, so `ModelLoader` caches the tokenizer under `out/Tokenizers` and the raw model under `out/Models`, is wired for a 4-bit `BitsAndBytesConfig`, `get_max_length()` exposes the context size the preprocessors need, and it switches between the DeepSpeed path and `device_map="auto"` because supplying a `device_map` under DeepSpeed conflicts with its own memory mapping; `merge_ft_model.py` folds a trained LoRA adapter back into the base weights.
- **Training/eval utilities** (`util/LoRATrainer.py`, `util/Chatbot.py`, `util/CVModelEvaluator.py`): full fine-tuning is out of reach on one GPU, so `LoRATrainer` builds a DoRA adapter (`r=32`, `lora_alpha=64`, `use_dora=True`), auto-detects linear target modules, prepares k-bit training, and guards against stacking a second adapter (`already_peft`); `Chatbot` wraps a text-generation pipeline with `extra_instructions` that force a PyTorch `Net` class and strip explanation text; `CVModelEvaluator` reads a candidate's `code.py` and calls `ab.nn.api.check_nn`, replacing the old local `codeEvaluator` so evaluation and database write-back share one path.
- **Dataset preprocessors** (`util/preprocessors/PreprocessorBase.py` and variants): each fine-tuning mode needed LEMUR rows turned into tokenized instruction/response pairs that fit GPU memory, so `PreprocessorBase` centralizes `get_raw_dataset()` plus tokenize/filter/shuffle/truncate to `max_len`, and `CodePromptPreprocessor` (SFT), `CodeImprovePromptPreprocessor` (improvement) and `CodeChgPrmPromptPreprocessorSFT` (hyperparameter change) only supply the prompt construction, so the training loop and all variants share identical tokenization and filtering.
- **Fine-tuning variants** (`finetune_hyperparameters.py`, `finetune_acc_prediction.py`, `finetune_rag.py`): the generator's usefulness depends on predicting how a network will behave before training it, so these add a 4-bit QLoRA trainer that maps a model version to a base checkpoint and predicts hyperparameters from a model's code and target accuracy, a `ModelFinetuner` that trains per architecture on its top-accuracy LEMUR models, and a retrieval-augmented QLoRA + DeepSpeed ZeRO-3 run on `DeepSeek-R1-Distill-Qwen-1.5B` that injects real code snippets into the training examples.
- **Retrieval & data setup** (`util/retrieval.py`, `util/data_loader.py`, `util/setup_data.py`, `util/lemur_dataset_preparation.py`): snippet-augmented training and hyperparameter prediction need an external code corpus, so `setup_data.py` clones `pytorchcv`, `pytorch_vision` and `fastai` and seeds dataset descriptions, `data_loader.py` chunks them (`detect_libraries`, `chunk_code`, `load_full_corpus`), `CodeRetrieval` embeds the chunks with `sentence-transformers` and searches a `faiss.IndexFlatL2` index, and `DatasetPreparation` converts raw LEMUR records into the question/answer JSON the fine-tuners consume.

### Changed
- `README.md` was expanded from a stub into a full guide (overview, venv/`pip install`, nn-dataset upgrade, Docker, usage), because none of the new install and workflow steps were discoverable without reading the code.
- Scripts and modules were standardized on the `gpt`/`finetune_*` naming and all outputs consolidated under `out/` (`Const.epoch_dir`), since the earlier scripts wrote to scattered, inconsistent locations that the evaluation and dataset-copy steps could not agree on.

### Fixed
- Stale `dataframe.df` files are now deleted when a prompt yields no addon information, so the evaluator no longer pairs generated code with the provenance of a previous candidate and evaluates it against the wrong task or parameters.
- Evaluation now forces `prm['epoch']` and shrinks large transforms (`512`/`256`/`299` → `128`), because the sampled LEMUR settings were long and memory-heavy enough to OOM the single-GPU box.
- Code extraction was made robust by requiring at least two ``` fences in the raw reply before attempting the regex match and skipping otherwise, so truncated model output no longer yields a broken `code.py` or spins the generation loop.

### Removed
- The old `codeEvaluator` module, superseded by `util/CVModelEvaluator.py`, so evaluation has a single implementation backed by `ab.nn.api.check_nn`.

### Documentation
- Added badges, the citation BibTeX (Qin et al., 2024), and MIT/DeepSeek licensing notes, so the newly packaged project could be cited and reused with clear terms.
