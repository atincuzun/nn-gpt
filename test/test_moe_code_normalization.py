"""Regression coverage for newline-escaped LLM code payloads."""

from pathlib import Path

from moe_cycle.code_normalization import unescape_single_line_code
from moe_cycle.epochs import _normalize_escaped_generated_code

CODE = """import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x
"""


def test_unescape_single_line_code_recovers_the_module():
    assert unescape_single_line_code(CODE.replace("\n", "\\n")) == CODE


def test_normal_multiline_code_is_untouched():
    assert unescape_single_line_code(CODE) == CODE


def test_epoch_normalization_only_rewrites_compilable_files(tmp_path: Path):
    synth = tmp_path / "synth_nn"
    escapable = synth / "B0" / "new_nn.py"
    broken = synth / "B1" / "new_nn.py"
    escapable.parent.mkdir(parents=True)
    broken.parent.mkdir(parents=True)
    escapable.write_text(CODE.replace("\n", "\\n"), encoding="utf-8")
    broken.write_text("def broken(:\\n    pass", encoding="utf-8")

    assert _normalize_escaped_generated_code(tmp_path) == 1
    assert escapable.read_text(encoding="utf-8") == CODE
    assert broken.read_text(encoding="utf-8") == "def broken(:\\n    pass"
