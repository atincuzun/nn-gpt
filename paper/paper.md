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

`NNGPT` separates source-code generation from the training engine, so generation,
repair, and fine-tuning can be used with an evaluation backend. The generated
source file is the unit passed among generation, repair, and fine-tuning. JSON
configuration files hold model and prompt settings. The prompt can contain task
instructions, a metric, and a reference program. The model returns full PyTorch
source code or a diff. The package applies the diff, checks the Python syntax,
repairs common code errors, and passes the source to the evaluation step. The
returned result can be used to fine-tune the language model. A fixed loop is the
choice for repeatable runs. An optional agent mode is the choice when a study
needs a different order of steps.

**Program representation.** `NNGPT` represents a candidate as PyTorch source
code. A fixed list of layers would simplify checking, but it would limit the
model to operations in the list. Source code lets the model change the network
and its internal operations. This expands the programs a study can compare and
lets experiments address code-level design rather than layer selection alone. A
study that needs code-level changes uses source code. A study that needs only
layer selection can use a fixed-list method.

**Generation and editing.** A program can be written from scratch or changed
with a diff. Full generation is the choice for broad changes. Diff editing is the
choice for local changes because it reduces output length and keeps the baseline
visible. A shorter output leaves more room in the context window, so a study can
isolate the effect of a small edit. The package applies the diff, checks the
Python syntax, and asks the model to correct a patch that does not apply. A
fallback to full generation keeps the editing path reliable. A study that needs a
wide change uses full generation. A study that needs a small change uses a diff.

**Prompt and model configuration.** The language model and prompt template are
selected by JSON files instead of being hard-coded. A JSON model file supplies a
Hugging Face model identifier, context length, quantization, and chat template. A
JSON prompt file supplies task instructions, reference programs, and the expected
output structure. This separation lets a study replace a model or prompt without
editing the generation code, so comparisons use the same generation path. A study
that adds a model or prompt uses a new JSON file. A study with fixed settings
keeps one configuration.

**Repair.** The repair step fixes five recurring errors. These are a missing
import for functional operations, incorrect unpacking of the input shape, a
network class with the wrong name, a required function with the wrong name, and
declared hyperparameters the program never uses. Repair is the choice when one of
these patterns is present. Discarding the program would lose the candidate and
require another generation call. Repair keeps the candidate, so compute goes to
evaluation rather than regeneration. A study that needs other checks can add them
separately.

**Prediction and hyperparameter recommendation.** An optional predictor estimates
the best accuracy and the epoch at which it occurs from early observations. A
separate tool recommends hyperparameters for a task and target accuracy. The
predictor is the choice when early scheduling matters. An early estimate can stop
low-value runs and leave more compute for other candidates. The recommender is
the choice when a starting configuration is useful. A full search is the choice
when the best settings are required. Measured results remain the basis for
comparison.

**Fine-tuning and feedback.** Adapters are the choice when memory or reuse
matters. The package uses the Transformers library [@wolf2020transformers] and
LoRA adapters [@hu2021lora]. Four-bit QLoRA [@dettmers2023qlora] reduces memory
use when GPU memory is limited. Lower memory per run lets several prompt or
feedback conditions share one base model, so comparisons keep the base model
fixed. Adapters can be saved and reused. A scalar reward is the choice for graded
outcomes. Direct preference optimization [@rafailov2023dpo] is the choice for a
preferred program and a rejected program. KTO [@ethayarajh2024kto] is the choice
for one good or bad label. The three methods use the same generated programs and
returned results.

**Comparison tools and edge scoring.** The package includes generators that do
not use a language model. These are genetic search, syntax-tree mutation, fractal
and mixture-of-experts templates, pruning, learning-rate schedules, and
data-transformation generation. They produce alternative programs in the same
source-code format. A shared format lets a study compare language-model and
non-language-model generation without changing the rest of the experiment. They
are the choice when a non-language-model comparison is needed. A separate tool
ranks generated networks for small devices by accuracy divided by inference time.
It is the choice when inference time matters alongside accuracy.

The package targets Python and PyTorch. Generation and fine-tuning use Hugging
Face Transformers and PEFT. Code repair and configuration work on the CPU, while
language-model work uses a GPU when available. Automated tests cover generation
filters, response parsing, repair, and feedback paths, so changes to these parts
are checked before release.

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
