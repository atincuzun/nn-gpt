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
date: 26 October 2026
bibliography: paper.bib
---

# Summary

[`NNGPT`](https://github.com/ABrain-One/nn-gpt) is a standalone open source Python
package for generating, evaluating, and adapting neural network programs with
language models. A language model writes source code that defines a network and
how it learns from data. The package brings together program generation and
editing, structural analysis and repair, execution based evaluation, and learning
from measured outcomes. These outcomes can become examples or feedback for
adapting the generator, allowing researchers to investigate how execution
experience changes the programs it produces.

The output includes inspectable Python source for network architectures and
training methods, together with training settings and evaluation results.
Generation, evaluation, and adaptation have separate entry points and can be
studied individually or combined into repeated cycles. Additional components
recommend training settings and estimate the outcome of a partially completed
training run. The software is available on PyPI as
[`nn-gpt`](https://pypi.org/project/nn-gpt/) under the MIT licence.

# Statement of need

Research on automated neural network design increasingly treats source code as
the object being searched. Comparing generators requires more than changing a
prompt. Candidate programs must be associated with tasks and datasets, checked
for compatibility with an evaluator, executed under stated training conditions,
and retained alongside their outcomes. Experiments that also adapt the generator
must turn these outcomes into training examples or rewards. Repeating this
integration separately for every generation strategy makes comparisons harder to
maintain and inspect.

`NNGPT` targets researchers in automated machine learning and language model
program synthesis who need reusable components for these experiments. It brings
program generation, editing, evaluation, and generator adaptation into a shared
codebase, with configurable models and prompts. The research object is both the
generated network and the process that produced it.

The package supports method development as well as controlled comparisons.
Researchers can vary the language model, prompt examples, generation mode, or
feedback objective while reusing the remaining workflow. Separate entry points
also support focused studies of program structure, architecture editing, early
performance prediction, and hyperparameter recommendation.

# State of the field

Optuna [@akiba2019optuna], Hyperopt [@bergstra2011hyperopt], and Google Vizier
[@golovin2017vizier] optimize user supplied objectives over parameterized search
spaces. ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras
[@jin2019autokeras] search architectures using specified representations.
NAS-Bench-101 [@ying2019nasbench101] supplies recorded evaluations for an
enumerated architecture space. Together, these tools provide established
approaches to optimization and architecture benchmarking.

Language model approaches differ in their search representations and feedback.
EvoPrompting [@chen2023evoprompting] and LLMatic [@nasir2024llmatic] combine code
generation with evolutionary search. LLAMBO [@liu2024llambo] provides modular
language model assistance for Bayesian optimization. GPT-NAS [@yu2023gptnas]
uses a generative transformer trained on architecture representations within
evolutionary search. Related research also investigates language models as
hyperparameter recommenders [@kochnev2025optuna]. These studies establish several
ways to incorporate language models into optimization and architecture design.

`NNGPT` organizes research around executable programs and evaluation feedback.
It brings program generation, interface repair, execution, prediction, and
generator adaptation into one configurable package. Keeping these stages
separate allows researchers to compare generation strategies while reusing
evaluation and adapter training functions. The package therefore serves both
as a research workflow and as a collection of components for building new
program generation methods.

# Software design

The common artifact is a PyTorch program defining a network, its training
methods, and its supported hyperparameters. A shared program interface specifies
how a network is constructed, configured, and trained. Representing candidates
as Python source allows changes to both architecture and training behaviour,
while the shared interface gives generation and evaluation components a
consistent way to work with them. Generation methods can therefore evolve
independently of the routines used to evaluate their outputs.

The static analysis utility parses Python syntax and reports the presence of
required classes, methods, and declarations without executing the program.
The repair stage combines syntax analysis and text rewriting in five ordered
passes addressing a missing functional import, a particular input shape
unpacking pattern, network class naming, training method naming, and unused
declared hyperparameters. These tools support inspecting program structure and
processing recurring interface patterns before evaluation.

Model configurations and prompt templates separate experiment settings from
shared generation logic. Prompts combine task and dataset information with
reference programs. Retrieval workflows can also use prepared code examples or
extracted blocks. Full program generation supports changes throughout a
candidate, while delta generation requests a unified diff against an existing
program. Diffs provide a compact representation of proposed edits and make
local changes easy to inspect. They undergo application and syntax checks, with
bounded feedback retries and a full code fallback. Approximate similarity
checks support comparing generated code with supplied corpora and earlier
candidates.

Reusable worker processes execute evaluation jobs and collect candidate
outcomes, reducing repeated process setup. Successful local results can be
reused, and each cycle records generated source, sampled reference records,
model responses, and evaluation summaries. These artifacts support inspecting
candidates and following progress between cycles. Separating evaluation from
generation allows the same candidate to be investigated without retraining the
language model.

Generator adaptation builds on Transformers [@wolf2020transformers] and `peft`,
using LoRA [@hu2021lora] with optional four bit QLoRA
[@dettmers2023qlora]. Updating adapters rather than all model weights reduces
the memory required for experiments. Input length filtering and truncation bound
training sequence lengths. Selected workflows accept explicit trainer
checkpoints for resuming optimization. Optional graph orchestration combines
shared generation, evaluation, and fine tuning functions in a state based
workflow.

Execution feedback can also train the generator through reinforcement learning
or preference optimization. Reward workflows assess properties such as model
construction, output shape, and training behaviour, with configurable objective
terms. Direct preference optimization [@rafailov2023dpo] uses preferred and
rejected completions, whereas KTO [@ethayarajh2024kto] learns from binary
desirability labels. These alternatives support experiments with graded rewards,
paired rankings, and binary feedback using the same kind of generated program
artifact.

A separate Qwen3-8B predictor estimates the best accuracy and the epoch at which
it occurs from early training accuracies and architecture features. A
[public adapter checkpoint](https://huggingface.co/ABrain/Accuracy-Prediction)
and an inference function allow researchers to apply the trained predictor to
new training observations. Another component learns hyperparameter
recommendations conditioned on a task, dataset, and target accuracy.
Non-language-model alternatives, including genetic search, fractal templates,
and syntax level mutation, provide comparison workflows.

The package requires Python 3.10 or later. The documented language model
environment uses Linux, NVIDIA GPUs, and CUDA. Static analysis and source repair
run on the CPU. Docker and cluster launch examples are provided in the
repository.

# Research impact statement

`NNGPT` has supported research by its developers and collaborators. The
framework publication describes and evaluates its generation and adaptation
pipelines [@kochnev2026nngpt]. Subsequent studies explicitly use the framework
for iterative architecture synthesis [@khalid2026creativity], modification
through code diffs [@adhikari2026delta], and few shot prompting and duplicate
validation [@duvvuri2025fewshot]. The few shot study identifies its prompting
and validation contributions as NNGPT implementations and reports 1,900 generated
architectures.

Further publications investigate synthesis from extracted module specifications
[@khalid2025nnrag] and performance guided data transformation design
[@shrestha2026transforms]. These applications and related studies illustrate
research directions spanning architecture synthesis, code editing, reusable
block generation, and data processing.

[Public pull requests](https://github.com/ABrain-One/nn-gpt/pull/1) date to
January 2025, and package releases on PyPI span 2025 and 2026. The repository
provides installation and usage instructions, tagged versions, a changelog,
contribution guidelines, and a code of conduct. Automated tests and a GitHub
Actions workflow support ongoing development.

# AI usage disclosure

Generative AI tools were used in software development and manuscript
preparation. DeepSeek V4.1 Flash assisted with code suggestions, refactoring,
documentation, and drafting. ChatGPT assisted with draft review. Kilo Code with
OpenAI GPT-6.1 Sol assisted with manuscript revision, source checks, and citation
validation. The human authors reviewed, edited, and validated AI-assisted
outputs, tested the software, and made the core design decisions. The authors
remain responsible for all claims and submitted materials. No AI tools were
used in interactions with editors or reviewers.

# Acknowledgements

We thank the contributors to the `NNGPT` repository. This work was partially
supported by the Alexander von Humboldt Foundation.

# References
