# Contributing to NNGPT

Thanks for your interest in improving NNGPT. This document explains how to
propose changes, add models or pipelines, and run the tests.

By participating you agree to abide by our [Code of Conduct](CODE-OF-CONDUCT.md).

## Ways to contribute

- Report bugs and request features via GitHub Issues.
- Fix bugs or improve documentation via pull requests.
- Add or improve generation pipelines, prompt configurations, or model backends.
- Contribute generated models and their statistics back to the LEMUR dataset.

## Development setup

```bash
git clone https://github.com/ABrain-One/nn-gpt
cd nn-gpt
python -m pip install --upgrade pip
pip install -e .
```

An NVIDIA GPU with CUDA is required for the LLM pipelines; CPU-only components
(data handling, AST validation, analysis) run without a GPU.

## Running the tests

```bash
pip install pytest
pytest -q test
```

Some end-to-end tests exercise training and therefore require a GPU and network
access; run the subset that matches your environment.

## Pull-request process

1. Fork the repository and create a branch from `main`.
2. Keep changes focused and add or update tests where practical.
3. Make sure `python -m compileall -q ab` and the test suite pass.
4. Describe the motivation and the change in the pull-request description.
5. A maintainer will review and, once approved, merge the change.

## Reporting issues

Include the command you ran, the configuration used, the expected and observed
behaviour, and the full traceback.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
