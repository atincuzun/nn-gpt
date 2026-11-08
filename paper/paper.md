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

NNGPT is an open-source Python package for experiments in which language models write the code for neural networks and data transformations. Designing such code by hand takes time and expert knowledge, and NNGPT lets researchers study how well language models can do this work instead. A language model writes new PyTorch programs or changes existing ones. NNGPT automatically fixes common errors in the generated code, so that programs with small mistakes can still be trained and evaluated with the LEMUR framework. The training results are then used to fine-tune the language model, and the process repeats. The same steps for writing, fixing, and evaluating programs support several ways of fine-tuning. The language model can learn from successful programs, from a numerical score, from pairs of better and worse programs, or from programs labelled good or bad. Researchers can therefore compare language models, prompts, and fine-tuning methods on the same tasks and datasets, with the same evaluation. NNGPT also includes program generators that do not use a language model, as a baseline for these comparisons. NNGPT can also predict the highest accuracy of a network from its first two training epochs.


# Statement of need

Neural architecture search looks for network designs that perform well on a task. Most methods choose from a fixed set of layers and connections that is defined before the search begins. A language model can instead write the code of a network directly. Because the code is written as text, the network is not limited to a fixed set of operations, and the language model can use what it learned about neural networks during pretraining. The language model can also be fine-tuned on the results of the networks it wrote, so that it improves as the search goes on. Researchers in automated machine learning, neural architecture search, and code generation use this approach to find new networks.

Code written by a language model is not guaranteed to run. It can contain errors such as a missing import, a wrong class name, or layers whose tensor shapes do not match. Each program must therefore be checked, fixed where possible, trained, and evaluated before its result can be used to fine-tune the language model. NNGPT provides these steps in one package. It fixes common errors directly, can test each network on example input before full training, and can return error messages to the language model so that it can correct its output. Researchers can then change the language model, the prompt, or the fine-tuning method while the other steps stay the same, so that differences in results come from the part that was changed.

# State of the field

Related tools differ in what they search and how they use the result. Optuna [@akiba2019optuna], Hyperopt [@bergstra2011hyperopt], and Google Vizier [@golovin2017vizier] search user-defined parameter spaces. A trial evaluates a configuration, an objective returns a score, and a sampling rule chooses the next trial. ENAS [@pham2018enas], DARTS [@liu2018darts], and AutoKeras [@jin2019autokeras] search architectures within their own representations, and validation results guide the search. NAS-Bench-101 [@ying2019nasbench101] returns recorded evaluations for an enumerated space.

Language model methods use a model to propose candidates. EvoPrompting [@chen2023evoprompting] and LLMatic [@nasir2024llmatic] modify or recombine network code and select candidates by validation accuracy. LLAMBO [@liu2024llambo] assists Bayesian optimization, and related work uses language models for hyperparameter recommendation [@kochnev2025optuna]. GPT-NAS [@yu2023gptnas] proposes architecture components within an evolutionary search. These tools search over configurations, architectures, or candidate programs, and each uses its own procedure to evaluate and select them.

NNGPT also uses a language model to write network code at the token level. Unlike the tools above, it covers the whole process from writing code to fine-tuning the language model in one package. It checks and repairs the generated code, trains and evaluates every program in the same way, and fine-tunes the language model on the results. Every search method that uses a language model needs these steps, so NNGPT provides them in a separate package that several search methods share. One of these methods is the package’s own version of the LLMatic search, which can therefore be used with fine-tuning and compared with other methods under the same evaluation.

# Software design

`NNGPT` is designed so that a language model learns from the results of the networks it writes. Each round of writing, checking, training, and evaluation produces data that is used to fine-tune the language model, and the next round starts from the improved model. The language model, the prompt, the search strategy, and the fine-tuning method can each be replaced while the others stay fixed, so researchers can see which part changes the results. Training and evaluation are done by the LEMUR framework [@nndataset2026], so `NNGPT` focuses on writing programs and fine-tuning the language model.

**Models and prompts.** Language models and prompts are set in JSON files that are loaded at the start of a run. A model file names any language model on Hugging Face, loaded through Transformers [@wolf2020transformers] or Unsloth. A prompt file holds the instructions, the template, and the rules that pick database records for it, such as the task, the dataset, and the number of reference programs. A new model or prompt therefore needs a short file rather than new code, so researchers spend their time on experiments instead of changing the pipeline.

**Generation and editing.** The language model writes complete PyTorch programs that follow the LEMUR conventions, so each program goes straight to training and can use any layer that PyTorch offers. Writing from the beginning gives the most freedom for new designs. Editing a working program keeps most of it unchanged, so the result is more likely to run, and a diff is much shorter than a full program. A structured edit names only a few choices, such as width or pooling, and the package writes the code. A template built on pretrained TorchVision networks lets the model write only the parts that join them, so a new design starts from learned features. Blocks that `nn-rag` [@khalid2025nnrag] takes from public repositories add working code written by people and make the generated networks more diverse.

**Checks, repair, and novelty.** Programs with small mistakes, such as a wrong class name, are repaired directly, which costs far less than asking the language model to write them again. Several modes run each new network once on random input to check its output shape, a quick test that keeps GPU time for programs that run. Error messages from edits go back to the language model so it can correct its output. Each program also receives a structure signature that ignores names and formatting, and a text similarity score, so near copies of earlier programs are found before training.

**Search and fine-tuning.** Writing a strong network is hard, but checking one is simple, because it only has to be trained and evaluated. `NNGPT` lets the language model write programs and checks them automatically, so every run produces new examples without manual work. These examples improve the language model in two ways. Search strategies place them in the prompt, for example by moving from similar to distant reference programs, by giving an edit that improved one network as a hint for another, or by rewriting the prompt from results. Fine-tuning learns from them directly, with supervised fine-tuning on successful programs by default, or with GRPO, PPO, DPO [@rafailov2023dpo], and KTO [@ethayarajh2024kto], which also learn from weaker programs. LoRA [@hu2021lora] and QLoRA [@dettmers2023qlora] let several runs share one base model.

**Prediction and recommendation.** Full training is the most expensive step, because a network can need many epochs to reach its best accuracy. A fine-tuned language model reads the code of a network and its accuracy after two epochs, and predicts its best accuracy and the epoch at which it occurs. This gives the researcher the option to stop long runs early when the predicted accuracy is low, and to spend the compute on promising or efficient networks. The predictor is optional and switched on by a setting, so full training stays the default. A second fine-tuned model recommends training settings for a task and a target accuracy, which gives a starting point without a separate search. Both are published on Hugging Face, the predictor as [Accuracy-Prediction](https://huggingface.co/ABrain/Accuracy-Prediction) and the recommender in two sizes, [HPGPT 1.3B](https://huggingface.co/ABrain/HPGPT-DeepSeek-Coder-1.3b-Base-R) and [HPGPT 7B](https://huggingface.co/ABrain/HPGPT-DeepSeek-R1-Distill-Qwen-7B-R).

**Mobile deployment.** Phones and other small devices have limited memory, power, and response time, so a network must also be small and fast to be useful there. The mobile track asks for networks under a parameter limit, exports each network to the eight-bit TFLite format used on phones, and ranks networks by accuracy divided by running time. Because accuracy after conversion and running time on a real phone can differ from desktop numbers, an optional step measures networks on a connected phone and keeps a result only when its score stays equal or improves.

**Baselines.** The package also includes algorithmic generators, such as genetic search, syntax tree mutation, and pruning. Their programs pass through the same checks and evaluation, so a language model is compared with them on equal terms.

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

Generative AI tools assisted with software development and with writing this
paper. DeepSeek V4.1 Flash helped with code, refactoring, and documentation, and
Claude Opus 5.5 helped with drafting, editing, and checking sources. The authors
reviewed and tested all AI-assisted output, made the design decisions, and
remain responsible for the content. No AI tools were used in interactions with
editors or reviewers.

# Acknowledgements

We thank the contributors to the `NNGPT` repository. This work was partially
supported by the Alexander von Humboldt Foundation.

# References
