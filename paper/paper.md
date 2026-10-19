---
title: 'NNGPT: generating, evaluating, and improving neural network programs with large language models'
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
date: 18 October 2026
bibliography: paper.bib
---

# Summary

Selecting a neural network architecture, its data preprocessing, and its training settings is still a manual and repetitive task. `NNGPT` reduces part of this work by asking a code-capable language model to write complete training programs, checking each program against a fixed program interface, running it, and recording what happened. Programs that pass the check are archived, become examples for later generations, and are used to adapt the generator through low-rank adapters.

These capabilities are provided as components with their own entry points. They cover program generation and editing, structural repair, hyperparameter recommendation, prediction of final accuracy and of the epoch at which it peaks, reinforcement learning from execution results, preference optimization, an agent mode that routes work between them, and generators that do not use a language model and exist for comparison. Accepted programs are written back to the evaluation service and reused by later runs.

`NNGPT` is written for researchers who want to automate the design of neural networks and who want to inspect how a language model makes design decisions. It is distributed under the MIT licence. The language model components require Linux, an NVIDIA graphics processor, and CUDA, while the static checks, the repair step, and the data preparation run on the processor alone. The package is published on PyPI as `nn-gpt`, and the project documents a public Docker image.

# Statement of need

Language models are increasingly used to propose neural network architectures, and the demonstrations in the literature are usually research prototypes. Generation, checking, execution, and reuse are written as separate scripts bound to one model, one dataset, and one experiment. Moving to a different model or dataset requires rewriting the surrounding code, and the results of earlier runs are rarely available to the next run. A study that compares several generation strategies repeats this work for every strategy.

`NNGPT` provides this machinery as one package in which the steps are separate components, each usable on its own, and the pipeline is assembled from them. Model selection, prompt templates, and generation settings are read from declarative configuration files, so a new language model or prompt is adopted by configuration instead of by changing code. The design also keeps the work observable. Generated programs, the prompts that produced them, and the measurements they obtained are written to disk, so a run that stops can be inspected to determine which step produced which artifact.

The framework needs three operations from an evaluation backend. It retrieves stored records, it submits a candidate program for training, and it archives an accepted program together with its measurements. The companion `nn-dataset` distribution [@nndataset2026] provides these operations on top of the LEMUR dataset [@goodarzi2025lemur], where the accepted programs of this work are archived.

# State of the field

Black-box search tools such as Optuna [@akiba2019optuna], Hyperopt [@bergstra2011hyperopt], and Google Vizier [@golovin2017vizier] search hyperparameters well but cannot propose a training program, and they need many evaluations before they find a good setting. Architecture search systems such as ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras [@jin2019autokeras] extend the search to architectures within a fixed space and remain expensive. Tabular benchmarks such as NAS-Bench-101 [@ying2019nasbench101] remove the need for training by serving stored evaluations, which restricts them to the space that was enumerated.

A second line of work places a language model inside the search. EvoPrompting [@chen2023evoprompting] uses one as a mutation and crossover operator, LLAMBO [@liu2024llambo] combines its suggestions with Bayesian optimization, and LLMatic [@nasir2024llmatic] and GPT-NAS [@yu2023gptnas] generate or complete candidate architectures. This work establishes that a closed loop between generation and evaluation is useful, and that a language model can write code that runs.

`NNGPT` builds on this line and differs in what it packages. It provides configurable components with separate entry points, a documented program interface that generated programs must satisfy, a repair step that fixes common structural errors before evaluation, an editing mode that changes a stored program through a validated diff, a predictor that estimates the outcome of a run before it finishes, and adaptation of the generator from recorded results. The nearest prototypes implement one fixed search procedure and do not separate these capabilities, so they cannot be reused when the model, the dataset, or the step under study changes.

# Software design

The unit of work is a training program that defines a network, its training loop, and its hyperparameter set. Programs are written against a fixed interface, which is the interface used by the LEMUR dataset. A program names a network class, declares its hyperparameters, and provides the training procedure. A static checker parses a program as Python source and verifies that the required class, methods, and module level declarations are present. It does not run the program, which separates questions about program shape from questions about infrastructure.

Generation assembles a prompt from a task description, a dataset, and stored programs retrieved from the evaluation service, then asks the configured model for a program. Two output modes are supported. The first returns a complete program. The second, delta mode, returns a unified diff against a stored program, which is applied, validated, and measured for similarity to the program it started from so that near duplicates are rejected. A diff that fails validation is requested again, at most twice.

Programs that nearly follow the interface are repaired rather than discarded. A post-processing step reads the program as Python syntax and applies five fixes in a fixed order. It adds a missing functional import, corrects a shape unpacking that names three variables where four are needed, renames the network class and the training method when they differ from the required names, and removes declared hyperparameters the program never reads. These fixes change only the shape of the program and do not establish that it trains.

Evaluation of generated programs is handled by a worker pool that keeps a fixed number of workers alive across runs and allocates graphics processor memory through configurable budgets. Programs whose evaluation already completed in the same directory are not trained again, and programs with a recorded verification failure are skipped. Results are written back to the service, and each cycle records its outcome for the next one.

Adaptation of the generator is optional and uses LoRA [@hu2021lora] and QLoRA [@dettmers2023qlora] adapters through `peft`, with four bit quantization when the base model is loaded in that mode [@wolf2020transformers]. DeepSpeed and Unsloth are available for training. Because training programs are long, examples are split into windows before training, and a run resumes from the last trainer checkpoint rather than starting over. The agent mode replaces the fixed loop with a workflow in which a manager node decides between generation, evaluation, and fine tuning, and an optional predictor node can intervene. The stage functions are shared with the fixed loop, and the workflow resumes from the last completed node after an interruption.

Prediction of training outcome is a separate entry point. A Qwen3-8B model is fine tuned with four bit QLoRA to predict the best accuracy reached and the epoch at which it is reached, from the accuracies observed in the first epochs together with quantities computed without training, including synflow, NWOT, gradient norm, the logarithm of the parameter count, and network depth. The fine tuned predictor is published as a checkpoint with a function that applies it to a running training. A second entry point fine tunes a model to emit hyperparameter values for a task, a dataset, and a target accuracy.

Reinforcement learning is available as its own entry point. A candidate program is executed and the measured outcome is converted into a scalar reward that combines successful construction, a correct output shape, a completed training step, and a reduction in loss. Optional terms cover validation gain, inference latency, parameter count, computation cost, a critic estimate, and a divergence penalty against the reference model. A policy gradient method updates the adapters of the generator, and reward computation is isolated in workers so that a failed candidate does not end the run. Preference optimization is offered for two objectives, a paired preferred and rejected completion and a single binary label, and both trainers keep the same checkpoint behaviour.

Generators that do not use a language model are included for comparison. They comprise a genetic algorithm over an evolvable network, fractal construction templates, mutation of a stored program at the syntax level, heterogeneous mixture of experts assembly, pruning, learning rate schedules, layer wise learning rates, and generation of data transformation programs. Three further pipelines address narrower questions. One generates networks under a parameter budget, scores them by accuracy divided by inference latency, and rejects those that cannot be converted to a quantized mobile format. One improves its prompt between iterations and keeps a memory of earlier attempts. One generates captioning models. The generator itself can also be exported to ONNX for serving.

# Research impact statement

Studies that used these components include the description of the generation pipeline [@kochnev2026nngpt], a study of architecture change through diffs [@adhikari2026delta], and a study of novel architecture synthesis [@khalid2026creativity]. Further studies used them for hyperparameter tuning [@kochnev2025optuna], few shot generation [@duvvuri2025fewshot], transformation design [@shrestha2026transforms], and extraction of network blocks from external repositories [@khalid2025nnrag]. Accepted programs are archived in the LEMUR dataset [@goodarzi2025lemur] and in its extension [@uzun2026lemur2].

Development is open and has been public since December 2024. The repository has 68 recorded contributors, 8 tagged releases, a changelog, a contribution guide, and a code of conduct. Continuous integration byte compiles the package and runs the test suite on Python 3.10 and 3.11, and the tests cover generation filters, pruning, response parsing, and the reinforcement learning path. The package is published on PyPI as `nn-gpt`, and the project documents a public Docker image and cluster launch scripts.

# AI usage disclosure

Generative AI tools were used in the development of this software and in the preparation of this manuscript. DeepSeek V4.1 Flash was used for code suggestions, refactoring, documentation, and drafting, and ChatGPT was used to review drafts. The human authors reviewed, edited, and validated all assisted output, tested the software, and made the design decisions. No AI tools were used in interactions with editors or reviewers.

# Acknowledgements

We thank the contributors of the `NNGPT` repository and the developers of the LEMUR dataset and its ecosystem, on which this work builds. This work was partially supported by the Alexander von Humboldt Foundation.

# References