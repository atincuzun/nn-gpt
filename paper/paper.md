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
repair, and fine-tuning can be used with an evaluation backend. JSON configuration
files select the language model and prompt template. The prompt can contain task
instructions, a metric, and a reference program. The model returns full PyTorch
source code or a diff. The package applies the diff, checks the Python syntax,
repairs common code errors, and passes the source to the evaluation step. The
returned result can be used to fine-tune the language model. The same steps can
run in a fixed loop or an optional agent mode. This separation allows the
evaluation backend to be replaced without changing the generation and
fine-tuning code.

**Program representation and generation.** `NNGPT` represents a candidate as
PyTorch source code, which allows the model to change the network and its
internal operations beyond a fixed list of layers. A program can be written from
scratch or changed with a diff. Full generation allows broad changes. A diff
makes a local change and keeps the surrounding code visible. Diffs reduce output
length and preserve context. Patch checks, bounded retries, and a fallback to
full generation keep editing reliable. The package applies the diff, checks the
Python syntax, and asks the model to correct a patch that does not apply before
using the full-generation fallback.

**Prompt and model configuration.** The language model and prompt template are
selected by JSON files. A custom JSON prompt supplies task instructions,
reference programs, and the expected output structure. A custom model JSON
supplies a Hugging Face model identifier, context length, quantization, and chat
template. This lets a study change the model or prompt without changing the
generation code. Sampling settings are supplied as options.

**Repair and prediction.** The repair step fixes five common errors. These are a
missing import for functional operations, incorrect unpacking of the input shape,
a network class with the wrong name, a required function with the wrong name, and
declared hyperparameters the program never uses. Repair handles these recurring
errors before evaluation, which lets the experiment focus on design choices. An
optional predictor provides an early estimate of the best accuracy and the epoch
at which it occurs from early observations. A separate tool recommends
hyperparameters for a task and target accuracy. These tools support scheduling
and initial settings while measured results remain the basis for comparison.

**Language-model fine-tuning and feedback.** Parameter-efficient adapters keep
repeated fine-tuning practical. The package uses the Transformers library
[@wolf2020transformers] and LoRA adapters [@hu2021lora]. Adapters can be saved
separately and reused across runs with the same base model. Four-bit QLoRA
[@dettmers2023qlora] reduces memory use when GPU memory is limited. A scalar
reward assigns one value to each generated program. Direct preference
optimization [@rafailov2023dpo] uses a preferred program and a rejected program.
KTO [@ethayarajh2024kto] uses one good or bad label for each program. The three
methods support different forms of feedback and use the same generated programs
and returned results.

**Dependencies and comparison tools.** The package requires Python and PyTorch.
Generation and fine-tuning use Hugging Face Transformers and PEFT. Code repair
and configuration work on the CPU, while language-model work uses a GPU when
available. The package also includes generators that do not use a language
model. These are genetic search, syntax-tree mutation, fractal and
mixture-of-experts templates, pruning, learning-rate schedules, and
data-transformation generation. These provide alternative programs in the same
source-code format for comparison with language-model generation. A separate
tool ranks generated networks for small devices by accuracy divided by inference
time. An optional agent mode provides an
alternative control path.

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
