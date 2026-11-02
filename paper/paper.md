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

`NNGPT` is an open-source Python package for research experiments in which large language models write the source code for neural networks. A language model writes or edits PyTorch source code for a neural network. NNGPT repairs common code errors and repeatedly runs program generation and language-model fine-tuning. It can fine-tune the language model from those experiments with a numerical reward, with pairs of better and worse programs, or with good or bad labels. This lets researchers compare language models, prompts, and fine-tuning methods on the same problems.

Program generation and language-model fine-tuning can be used separately. For comparison, the package includes program generators that do not use a language model. Further tools recommend training settings, rank networks for small devices by accuracy divided by inference time, and predict the highest accuracy of a network and the epoch at which it occurs from the first two training epochs. The software is available on PyPI as `nn-gpt` under the MIT licence.


# Statement of need

Researchers in automated machine learning, neural architecture search, and language-model program synthesis use language models to write source code for neural networks. The generated code can contain errors such as missing imports, wrong class names, or incorrect tensor shapes. If each study repairs these errors separately, a comparison between language models or prompts also includes differences in the repair code. Existing architecture-search tools assume a fixed architecture or parameter space, and general code-generation tools do not provide a common repair and fine-tuning procedure.

NNGPT provides one implementation of these steps. It generates and edits PyTorch source code, repairs common code errors, and fine-tunes the language model over repeated rounds. A study can change the language model, prompt, or fine-tuning method without rewriting the repair step. The package also includes non-language-model generators for comparison, ranks networks for small devices by accuracy divided by inference time, recommends hyperparameters, and predicts the highest accuracy of a network and the epoch at which it occurs.

# State of the field

Related tools differ in what they search and how they use the result. Optuna [@akiba2019optuna], Hyperopt [@bergstra2011hyperopt], and Google Vizier [@golovin2017vizier] search user-defined parameter spaces. A trial evaluates a configuration, an objective returns a score, and a sampling rule chooses the next trial. ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras [@jin2019autokeras] search architectures within their own representations, and validation results guide the search. NAS-Bench-101 [@ying2019nasbench101] returns recorded evaluations for an enumerated space.

Language model methods use a model to propose candidates. EvoPrompting [@chen2023evoprompting] and LLMatic [@nasir2024llmatic] modify or recombine network code and select candidates by validation accuracy. LLAMBO [@liu2024llambo] assists Bayesian optimization, and related work uses language models for hyperparameter recommendation [@kochnev2025optuna]. GPT-NAS [@yu2023gptnas] proposes architecture components within an evolutionary search. These tools search over configurations, architectures, or candidate programs, and each uses its own procedure to evaluate and select them.

NNGPT works with the source code of a neural network, not only with its hyperparameters or layer names. It writes and edits that code, recommends hyperparameters, predicts best accuracy and the epoch at which it occurs, reuses code blocks from other programs, and fine-tunes the language model after runs. All five steps use the same source file and the same recorded result. Each step can use the output of the earlier steps.


# Software design

`NNGPT` has three main parts: it generates a program, runs the program, and
updates the language model from the result. Each part can be used on its own,
and a fixed loop or an optional agent mode can combine them.

**Program structure.** A generated program is a Python file that defines a
PyTorch network and the code that trains it. The program can change the layers,
optimizer, loss, and training loop, not only choose from a fixed list of layers.
This gives the generator control over both the network and its training
procedure.

**Generation and editing.** A program can be written from scratch or changed
with a diff. Full generation allows broad changes to architecture and training
code. A diff describes a local change to an existing program and keeps the
surrounding code visible. The package applies the diff, checks the resulting
Python syntax, asks the model to correct a diff that cannot be applied, and falls
back to a full program if editing still fails. Generation prompts combine the
task, dataset, metric, training settings, and a reference program.

**Configuration.** The language model and prompt template are read from
configuration files. Sampling and training settings are supplied as options. The
same generation, evaluation, and update steps can run in a fixed loop or in an
agent mode that controls the order of the steps.

**Repair and evaluation.** The repair step runs before evaluation and fixes five
common errors: a missing import for functional operations, incorrect unpacking
of the input shape, a network class with the wrong name, a training method with
the wrong name, and declared hyperparameters the program never uses. The
evaluator then scores the repaired program. Passing repair does not mean the
program is good; the measured result decides.

**Stored records.** For each generated program, the package stores the source
code, the reference program when one was used, the model response, and the
evaluation score. This record links a program to the request that produced it
and to its result, and supplies examples and rewards for later steps.

**Accuracy prediction and hyperparameter recommendation.** An optional predictor
estimates the best accuracy and the epoch where it occurs from the first two
epochs and a few network measurements. A separate tool recommends hyperparameters
for a task, dataset, and target accuracy. Both can be used without changing the
generated program.

**Adapter training.** Repeated fine-tuning is expensive, so the package uses the
Transformers library [@wolf2020transformers] and LoRA adapters [@hu2021lora]
instead of updating all weights. Adapters can be saved separately and reused
across runs with the same base model. Four-bit QLoRA [@dettmers2023qlora] reduces
memory use when GPU memory is limited. This makes it practical to compare several
prompts or feedback settings with one pretrained model.

**Feedback methods.** Execution results can train the model in three ways. A
scalar reward gives points for building the network, producing the right output
shape, completing a training step, and reducing the loss, with optional terms for
accuracy, speed, model size, and how far the model drifts from the reference.
Direct preference optimization [@rafailov2023dpo] uses pairs of preferred and
rejected programs. KTO [@ethayarajh2024kto] uses a single good or bad label for
each program. These three methods use the same generated programs and evaluation
results.

# Research impact statement

`NNGPT` has supported research across its generation, editing, and adaptation
workflows. The framework publication describes and evaluates its generation and
adaptation pipelines [@kochnev2026nngpt]. Subsequent studies apply it to
iterative architecture synthesis [@khalid2026creativity], modification through
code diffs [@adhikari2026delta], and few shot prompting and duplicate validation
[@duvvuri2025fewshot]. The few shot study identifies its prompting and validation
contributions as NNGPT implementations and reports 1,900 generated
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
