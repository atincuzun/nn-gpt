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

`NNGPT` is a Python package that turns a code-capable large language model into a
pipeline for designing neural networks. A language model is asked to write a
complete training program. The package checks the program against a fixed
interface, repairs the parts that do not conform, runs the program, and uses the
measured outcome to improve the generator. The same loop also proposes
hyperparameters, predicts how a run will end, and edits a program that already
exists instead of writing one from scratch.

The pipeline is built from components with their own entry points. They cover
program generation and editing, structural repair, hyperparameter recommendation,
prediction of final accuracy and of the epoch at which it peaks, reinforcement
learning from execution results, preference optimization, an agent mode that
routes work between them, and generators that do not use a language model and
exist for comparison.

`NNGPT` is written for researchers who want to automate the design of neural
networks and who want to inspect how a language model makes design decisions. It
is distributed under the MIT licence. The language model components require Linux,
an NVIDIA graphics processor, and CUDA, while the static checks, the repair step,
and the data preparation run on the processor alone. The package is published on
PyPI as `nn-gpt`, and the project documents a public Docker image.

# Statement of need

A language model that writes neural network code is easy to demonstrate and hard
to use at scale. In research prototypes, the parts of the process are usually
separate scripts bound to one model, one dataset, and one experiment. Asking for
a program, checking it, running it, and deciding what to ask for next are
re-implemented for every study. A study that compares several generation
strategies repeats this work for every strategy.

`NNGPT` provides the process as one package in which the steps are separate
components, each usable on its own, and the loop is assembled from them. Model
selection, prompt templates, and generation settings are read from declarative
configuration files, so a new language model or prompt is adopted by
configuration instead of by changing code. The work is also observable.
Generated programs, the prompts that produced them, and the outcome of each step
are written to disk, so a run that stops can be inspected to see which step
produced which artifact.

The package needs three things from an evaluation backend. It needs stored
programs to use as examples, it needs a way to submit a candidate program for
training, and it needs a way to hand an accepted program back. The companion
`nn-dataset` distribution [@nndataset2026] provides these on top of the LEMUR
dataset [@goodarzi2025lemur].
# State of the field

Black-box search tools such as Optuna [@akiba2019optuna], Hyperopt
[@bergstra2011hyperopt], and Google Vizier [@golovin2017vizier] search
hyperparameters well but cannot write a training program, and they need many
evaluations before they find a good setting. Architecture search systems such as
ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras [@jin2019autokeras]
extend the search to architectures within a fixed space and remain expensive.

A second line of work places a language model inside the search. EvoPrompting
[@chen2023evoprompting] uses one as a mutation and crossover operator, LLAMBO
[@liu2024llambo] combines its suggestions with Bayesian optimization, and LLMatic
[@nasir2024llmatic] and GPT-NAS [@yu2023gptnas] generate or complete candidate
architectures. This work establishes that a closed loop between generation and
evaluation is useful, and that a language model can write code that runs.
Benchmarks such as NAS-Bench-101 [@ying2019nasbench101] reach the same end by
serving evaluations of an enumerated space, which fixes the space in advance.

`NNGPT` builds on this line and differs in what it packages. It provides the
steps of the loop as configurable components with separate entry points, so that a
reader can change the model, the dataset, or the step under study without
rewriting the surrounding code. The nearest prototypes implement one fixed search
procedure and do not separate these capabilities.

# Software design

The unit of work is a training program that defines a network, its training loop,
and its hyperparameter set. Programs are written against a fixed interface in
which a program names a network class, declares its hyperparameters, and
provides the training procedure. A static checker parses a program as Python
source and verifies that the required class, methods, and module level
declarations are present. It does not run the program, which separates questions
about program shape from questions about infrastructure.

Generation assembles a prompt from a task description, a dataset, and examples
retrieved from the evaluation backend, then asks the configured model for a
program. Two output modes are supported. The first returns a complete program.
The second, delta mode, returns a unified diff against an existing program,
which is applied, validated, and measured for similarity to the program it
started from so that near duplicates are rejected. A diff that fails validation
is requested again, at most twice. A separate entry point generates from
prepared retrieval data and rejects a candidate that resembles the material the
model was tuned on.

Programs that nearly follow the interface are repaired rather than discarded. A
post-processing step reads the program as Python syntax and applies five fixes
in a fixed order. It adds a missing functional import, corrects a shape
unpacking that names three variables where four are needed, renames the network
class and the training method when they differ from the required names, and
removes declared hyperparameters the program never reads. These fixes change
only the shape of the program and do not establish that it trains.

Execution of generated programs is handled by a worker pool that keeps a fixed
number of workers alive across runs and allocates graphics processor memory
through configurable budgets. Programs that were already executed in the same
directory are not run again, and programs with a recorded verification failure
are skipped. The outcome of a cycle is passed to the next cycle. The parameter
count and the operation count of a generated program can also be obtained
without training it, so the cost of a candidate is known before it is run.

Adaptation of the generator is optional and uses LoRA [@hu2021lora] and QLoRA
[@dettmers2023qlora] adapters through `peft`, with four bit quantization when the
base model is loaded in that mode [@wolf2020transformers]. DeepSpeed and Unsloth
are available for training. Because training programs are long, examples are
split into windows before training, and a run resumes from the last trainer
checkpoint rather than starting over. An adapter can be applied to a chosen
range of layers, trained on the output head alone, or trained as a soft prompt
that leaves the base weights unchanged, and several adapters can be merged into
one model afterwards. Generation can be limited to a fixed number of output
tokens, and a reasoning block opened by the model can be closed before its
answer is read.

The agent mode replaces the fixed loop with a workflow in which a manager node
decides between generation, evaluation, and fine tuning, and an optional
predictor node can intervene. The stage functions are shared with the fixed
loop, and the workflow resumes from the last completed node after an
interruption.
Prediction of training outcome is a separate entry point. A Qwen3-8B model is
fine tuned with four bit QLoRA to predict the best accuracy reached and the epoch
at which it is reached, from the accuracies observed in the first epochs
together with quantities computed without training, including synflow, NWOT,
gradient norm, the logarithm of the parameter count, and network depth. The fine
tuned predictor is published as a checkpoint with a function that applies it to
a running training. A second entry point fine tunes a model to emit
hyperparameter values for a task, a dataset, and a target accuracy.

Reinforcement learning is available as its own entry point. A candidate program
is executed and the outcome is converted into a scalar reward that combines
successful construction, a correct output shape, a completed training step, and
a reduction in loss. Optional terms cover validation gain, inference latency,
parameter count, computation cost, a critic estimate, and a divergence penalty
against the reference model. A policy gradient method updates the adapters of
the generator, and reward computation is isolated in workers so that a failed
candidate does not end the run. Preference optimization is offered for two
objectives, a paired preferred and rejected completion and a single binary
label, and both trainers keep the same checkpoint behaviour.

Generators that do not use a language model are included for comparison. They
comprise a genetic algorithm over an evolvable network, fractal construction
templates, mutation of an existing program at the syntax level, heterogeneous
mixture of experts assembly, pruning, learning rate schedules, layer wise
learning rates, and generation of data transformation programs. Three further
pipelines address narrower questions. One generates networks under a parameter
budget and scores them by accuracy divided by inference latency. One improves
its prompt between iterations and keeps a memory of earlier attempts. One
generates captioning models. The generator itself can also be exported to ONNX
for serving.

# Research impact statement

Studies that used these components include the description of the generation
pipeline [@kochnev2026nngpt], a study of architecture change through diffs
[@adhikari2026delta], and a study of novel architecture synthesis
[@khalid2026creativity]. Further studies used them for hyperparameter tuning
[@kochnev2025optuna], few shot generation [@duvvuri2025fewshot], transformation
design [@shrestha2026transforms], and extraction of network blocks from external
repositories [@khalid2025nnrag]. The programs the pipeline produced were
contributed to the LEMUR dataset [@goodarzi2025lemur] and to its extension
[@uzun2026lemur2].

Development is open and has been public since December 2024. The repository has
68 recorded contributors, 8 tagged releases, a changelog, a contribution guide,
and a code of conduct. Continuous integration byte compiles the package and runs
the test suite on Python 3.10 and 3.11, and the tests cover generation filters,
pruning, response parsing, and the reinforcement learning path. The package is
published on PyPI as `nn-gpt`, and the project documents a public Docker image
and cluster launch scripts.

# AI usage disclosure

Generative AI tools were used in the development of this software and in the
preparation of this manuscript. DeepSeek V4.1 Flash was used for code
suggestions, refactoring, documentation, and drafting, and ChatGPT was used to
review drafts. The human authors reviewed, edited, and validated all assisted
output, tested the software, and made the design decisions. No AI tools were
used in interactions with editors or reviewers.

# Acknowledgements

We thank the contributors to the `NNGPT` repository and the developers of the
LEMUR dataset and its ecosystem, on which this work builds. This work was
partially supported by the Alexander von Humboldt Foundation.

# References