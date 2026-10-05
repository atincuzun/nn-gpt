---
title: 'NNGPT: an open-source framework for LLM-driven neural network generation and AutoML'
tags:
  - Python
  - AutoML
  - neural architecture search
  - large language models
  - code generation
  - PyTorch
authors:
  - name: Roman Kochnev
    affiliation: 1
  - name: Waleed Khalid
    affiliation: 1
  - name: Tolgay Atinc Uzun
    affiliation: 1
  - name: Xi Zhang
    affiliation: 1
  - name: Yashkumar Sanjaybhai Dhameliya
    affiliation: 1
  - name: Furui Qin
    affiliation: 1
  - name: Chandini Vysyaraju
    affiliation: 1
  - name: Raghuvir Duvvuri
    affiliation: 1
  - name: Avi Goyal
    affiliation: 1
  - name: Dmitry Ignatov
    affiliation: 1
  - name: Radu Timofte
    affiliation: 1
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    index: 1
date: 3 October 2026
bibliography: paper.bib
---

# Summary

`NNGPT` is an open-source Python framework for generating, evaluating, and
optionally adapting neural-network programs proposed by a code-capable large
language model (LLM), in support of automated machine learning (AutoML). AutoML
aims to remove the manual trial-and-error of choosing a network architecture, its
data preprocessing, and its training hyperparameters. Classical AutoML tools
search these choices as a black box and need many training runs to find a good
configuration. `NNGPT` instead asks a model to *write* a candidate training
program in Python, checks and repairs it against the interface required by the
LEMUR neural-network dataset [@goodarzi2025lemur], evaluates it under a
standardized protocol, and can feed the measured result back to improve the
generator.

The package combines several cooperating capabilities around a common LEMUR-based
evaluation and logging substrate: prompt-driven generation and editing of network
source code, hyperparameter recommendation, prediction of final accuracy and of
the epoch at which it peaks, retrieval-augmented reuse of existing PyTorch
blocks, and a reinforcement-learning loop. From a task configuration, `NNGPT`
retrieves related networks from LEMUR, generates and repairs a candidate,
evaluates it through the LEMUR evaluator, archives the result in the LEMUR
database, and can update the underlying LLM with low-rank adapters, forming a
generate--check--execute--log--update loop that keeps expanding the corpus of
executable networks.

`NNGPT` is intended for machine-learning researchers and practitioners who want
to automate network design and configuration, and for researchers who study
LLM-driven code generation and automated search. The source is MIT-licensed.
The LLM pipelines target Linux with an NVIDIA GPU and CUDA,
together with native dependencies such as PyTorch and bitsandbytes, and some
components (for example AST validation and data preparation) run on CPU.
Installation is a single `pip` command, and the project documents a public Docker
image.

# Statement of need

Researchers studying LLM-generated neural-network programs need reusable
infrastructure for generating candidates, checking interface conformity,
evaluating their behaviour, and reusing measured outcomes in subsequent
generation and adaptation. AutoML and neural-architecture-search tools provide
parts of this, and recent LLM-based methods can propose architectures or
hyperparameters, but these pieces are usually tied to a specific method or
harness rather than a reusable, configurable pipeline.

`NNGPT` addresses this gap by making the candidate an executable program that is
checked against a standard interface and then evaluated, and by feeding results
back to the generator. It combines, in one reusable open-source package, three
capabilities that are otherwise spread across separate research prototypes and
scripts: (i) an initial generation call that produces a candidate training
program, followed by repair and bounded retries where needed before standardized
evaluation; (ii) a predictor that estimates final accuracy and the epoch at which
it peaks from early training signals; and (iii) a feedback mechanism that turns
execution results into new training data and reward signal for the generator.
Because each accepted candidate is validated once and its code and measurements
are stored, an identical program and configuration can be looked up instead of
re-evaluated. This removes duplicate evaluations, but searching still costs the
evaluation of new candidates, failed attempts, generation, and fine-tuning. It is
designed for inspectable AutoML rather than a black-box service, and it doubles as
a research instrument for studying how LLMs reason about code and architecture.
The broader aim is a shared substrate on which LLM-driven and search-based AutoML
can be compared.

# State of the field

Hyperparameter optimization tools such as Optuna [@akiba2019optuna], Hyperopt
[@bergstra2011hyperopt], and Google Vizier [@golovin2017vizier] perform strong
black-box search but do not synthesize full training programs. NAS systems such
as ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras [@jin2019autokeras]
extend search to architectures but remain compute-intensive and tied to narrow
search spaces, while tabular benchmarks such as NAS-Bench-101
[@ying2019nasbench101] avoid training during search by serving precomputed
evaluations for a fixed cell-based space.

LLM-based methods are closer in spirit. EvoPrompting [@chen2023evoprompting]
uses an LLM as an adaptive mutation and crossover operator and already trains,
evaluates, and reuses candidates for prompt tuning; LLAMBO [@liu2024llambo]
couples LLM suggestions with Bayesian optimization; and LLMatic
[@nasir2024llmatic] and GPT-NAS [@yu2023gptnas] generate or complete candidate
architectures. Closing a feedback loop is therefore not by itself novel. What is less developed is a shared layer coupling generation to a
common executable-program interface and evaluation substrate. `NNGPT` provides
such a layer: a configurable package that generates, repairs, evaluates, and
reuses programs through the LEMUR interface across tasks and backbones, with an
accuracy and peak-epoch predictor and execution-informed generator adaptation.
We built a separate layer rather than extending these projects because the
closest LLM-NAS prototypes implement a fixed search procedure for their own
harness and would require reimplementing generation, repair, evaluation, storage,
and adaptation to reach this scope; LEMUR [@goodarzi2025lemur] is a dataset and
evaluation substrate, not a generator, so `NNGPT` sits on top of it. A detailed
comparison is in the companion methods paper
[@kochnev2026nngpt].

# Software design

`NNGPT` is written in Python (installable as `nn-gpt`, Python >= 3.10).

The core workflow retrieves seed networks from LEMUR, builds a prompt, generates
a candidate program, repairs it, evaluates it through the LEMUR evaluator, and
stores successful results. Within this loop, generator fine-tuning runs each
cycle but can be skipped; hyperparameter recommendation, accuracy prediction, and
reinforcement learning are additional components with their own entry points. Across iterations, the available candidate corpus
and retrieved examples can change; adapter weights are updated when generator
fine-tuning is enabled. "Learning from execution" here means that accepted
programs and their measured outcomes become fine-tuning data and, on the RL path,
reward signal.

The package is organized around the following components.

* **Models, prompts, and adapters.** A chat/LLM abstraction loads Hugging Face
  models [@wolf2020transformers] and applies LoRA [@hu2021lora] and QLoRA
  [@dettmers2023qlora] adapters via `peft`, with optional 4-bit quantization.
  Backbones and prompt templates
  are selected by declarative JSON and assembled at runtime with task, dataset,
  and seed-model context, so a new model or prompt is adopted by configuration
  rather than code; a backend with genuinely new capabilities may still require
  implementation work.
* **Generation and repair.** From a task prompt and seed models retrieved through
  the LEMUR API, the LLM emits a `Net` class and training code. Extractors parse
  the output, and an AST-based post-processor repairs imports, input-shape
  unpacking, class and method names, and unused hyperparameters before evaluation;
  an AST-only checker validates the contract without executing code. In delta mode
  the model emits unified diffs with up to two validation retries. Repair enforces
  structural conformity but does not establish that a program trains correctly.
* **Hyperparameter recommendation.** A separate fine-tuning pipeline trains code
  LLMs to emit hyperparameter values from task, dataset, and a target accuracy,
  and can produce a full set in one call from a fine-tuned adapter; it is a
  standalone component.
* **Execution substrate.** Generated models must implement the LEMUR `Net`
  contract [@paszke2019pytorch]. Evaluation runs through the LEMUR evaluator;
  successful runs are archived in its SQLite database through the LEMUR API and
  queried again for later generation. This reuses a standardized multi-task
  evaluation and an existing model corpus, at the cost of requiring generated
  programs to conform to the interface. Retrieval-augmented generation builds on
  the companion `nn-rag` and `nn-dup` packages.
* **Accuracy and peak-epoch prediction.** A `Qwen3-8B` model is fine-tuned with
  4-bit QLoRA to predict `best_accuracy`/`best_epoch` from early-epoch accuracies
  and zero-cost proxies (for example synflow, NWOT, and gradient norm); source
  code is optional. The trained predictor is released as a Hugging Face
  checkpoint with a public inference function that estimates final accuracy and
  peak epoch from early training signals.
* **Reinforcement learning.** A standalone RL entry point updates the generator
  from an execution reward rewarding build success, correct output shape, a
  completed training step, and a loss drop, with optional efficiency, critic, and
  KL terms.
* **Programmatic baselines.** Programmatic generators (genetic search, fractal
  networks, AST mutation, and mixture-of-experts assembly) let LLM-driven results
  be compared against non-LLM search under the same evaluation.

Two design choices matter. First, the same pipeline can run as a classic loop or
through an optional agent-orchestration mode (a LangGraph workflow over the
generation, evaluation, and fine-tuning steps); sharing one module between the two
keeps them consistent while the extension boundary stays clear as new pipelines
are added, and LLM fine-tuning can resume from a trainer checkpoint rather than
restarting. Second, generated programs are never
trusted: they are structurally checked and repaired, then executed under one
standardized protocol. Structural checks and repair enforce interface conformity
before runtime evaluation, and evaluation through a common LEMUR interface
supports comparison under a specified task configuration. The framework runs on
Linux with an NVIDIA GPU and CUDA for the LLM pipelines, alongside PyTorch and a
compatible LEMUR installation for evaluation; its analysis and validation tools
also run on CPU, and a CPU-only dependency set is provided for lightweight
checks. The `test/` directory contains checks for generation filters and pruning.

# Research impact statement

`NNGPT` is the generative engine of an active research program: its developers
and collaborators have used it to produce peer-reviewed and preprint work, and
its outputs are citable artifacts. Three examples: the delta-generation pipeline
underlies the delta-based NAS study [@adhikari2026delta] (valid-generation rate
50.6% to 75.3%); the NN-RAG component is the basis of the retrieval-augmented
system [@khalid2025nnrag] (1,289 blocks extracted, 941 validated, about 72% of
novel LEMUR architectures); and the closed-loop generator drives the
architecture-synthesis study [@khalid2026creativity] (455 novel architectures
over 22 cycles). Related work covers hyperparameter tuning [@kochnev2025optuna],
few-shot generation [@duvvuri2025fewshot], and transformation design
[@shrestha2026transforms]. Generated models are contributed back to LEMUR [@goodarzi2025lemur;
@uzun2026lemur2], and the fine-tuned models and checkpoints are publicly released
through the project's Hugging Face namespace. The repository
(https://github.com/ABrain-One/nn-gpt) has been developed openly since
December 2024, with about fifty contributors (contributor graph) and
merged pull requests beyond the original authors. It is published on PyPI (1.0.6, with 1.0.7 in the repository) and documented for
the public `abrainone/ai-linux` Docker image.

# AI usage disclosure

`NNGPT` is itself a tool for LLM-based code generation, and generative AI is
central to what the software does; the models it runs include DeepSeek-Coder,
Qwen, and Mistral checkpoints configured under `ab/gpt/conf/llm`. Separately,
DeepSeek v4.1 Flash was used during development of the software, its
documentation, and the drafting of this paper (code suggestions, refactoring,
documentation, and copy-editing), and ChatGPT was used to review drafts. Human
authors reviewed, edited, and validated all AI-assisted output, tested the
software, and made the core design decisions. No AI tools were used in
interactions with editors or reviewers.

# Acknowledgements

We thank the contributors to the `NNGPT` repository and the developers of the
LEMUR dataset and its ecosystem, on which this work builds. This work was
partially supported by the Alexander von Humboldt Foundation.

# References
