import json

import numpy as np
import pandas as pd

from ab.gpt.util.eval.Seed import SEED_FILE_NAME
from ab.gpt.util.tune import Tune


class _RecordingBot:
    def __init__(self):
        self.prompts = []
        self.tokenizer = self

    def _build_messages(self, prompt):
        return [{"role": "user", "content": prompt}]

    def apply_chat_template(self, messages, *, add_generation_prompt):
        return list(messages[0]["content"])

    def chat(self, prompt, *, engineer_prompt, max_new_tokens):
        self.prompts.append(prompt)
        return "class Net: pass", None, None, "<nn>class Net: pass</nn>"


def _prompt_config():
    return {"cv": {
        "task": "img-classification",
        "dataset": "cifar-10",
        "nn_prefixes": ["ga-"],
        "input_list": [{"para": "nn_code", "value": "nn_code"}],
        "prompt": ["{nn_code}"],
    }}


def test_backbone_generation_passes_dataset_filters_to_lemur_data(tmp_path, monkeypatch):
    seed_file = tmp_path / SEED_FILE_NAME
    monkeypatch.setenv("NNGPT_PIN_SEEDS", "1")
    monkeypatch.setenv("NNGPT_SEED_FILE", str(seed_file))
    monkeypatch.setenv("NNGPT_BEST_OF_N", "1")
    monkeypatch.delenv("NNGPT_EDIT_MODE", raising=False)
    monkeypatch.setattr(Tune, "release_memory", lambda: None)
    monkeypatch.setattr(Tune.SFTUtil, "assemble_backbone_xml_completion", lambda _: "class Net: pass")

    def fake_data(*, only_best_accuracy, task, nn_prefixes, dataset):
        assert only_best_accuracy
        assert (task, nn_prefixes, dataset) == ("img-classification", ("backbone-",), "cifar-100")
        return pd.DataFrame([{"nn": "backbone-one", "nn_code": "class Net: pass"}])

    monkeypatch.setattr(Tune.lemur, "data", fake_data)
    bot = _RecordingBot()
    Tune.nn_gen(0, tmp_path, bot, ["cv"], 1, _prompt_config(), 1, 2048,
                True, "backbone", None, 1, use_backbone=True,
                sft_dataset="cifar-100", sft_nn_prefixes=("backbone-",))

    assert len(bot.prompts) == 1
    assert json.loads(seed_file.read_text())["nn"] == ["backbone-one"]
    assert (Tune.synth_dir(tmp_path) / "B0" / Tune.new_nn_file).is_file()


def test_moe_generation_uses_legacy_draw_without_pin_or_refill(tmp_path, monkeypatch):
    frame = pd.DataFrame([
        {"nn": "ga-c", "nn_code": "fits but not drawn"},
        {"nn": "ga-b", "nn_code": "trial one"},
        {"nn": "ga-a", "nn_code": "too long " * 10},
        {"nn": "ga-b", "nn_code": "trial two"},
    ])
    seed_file = tmp_path / SEED_FILE_NAME
    pinned = '{"nn":["ga-c"]}'
    seed_file.write_text(pinned)
    monkeypatch.setenv("NNGPT_PIN_SEEDS", "1")
    monkeypatch.setenv("NNGPT_SEED_FILE", str(seed_file))
    monkeypatch.setenv("NNGPT_BEST_OF_N", "3")
    monkeypatch.delenv("NNGPT_EDIT_MODE", raising=False)
    monkeypatch.setattr(Tune.lemur, "data", lambda **kwargs: frame)
    monkeypatch.setattr(Tune, "release_memory", lambda: None)
    bot = _RecordingBot()

    state = np.random.get_state()
    try:
        np.random.seed(19)
        expected = frame.groupby(by="nn").sample(n=1)[:2].iloc[1]
        np.random.seed(19)
        Tune.nn_gen(0, tmp_path, bot, ["cv"], 1, _prompt_config(), 2, 2048,
                    True, "moe-gate-cycle", 20, 1, legacy_prompt_selection=True)
    finally:
        np.random.set_state(state)

    assert bot.prompts == [expected["nn_code"] + "\n"]
    assert seed_file.read_text() == pinned
    models_dir = Tune.synth_dir(tmp_path)
    assert [directory.name for directory in models_dir.iterdir()] == ["B1"]
    assert pd.read_pickle(models_dir / "B1" / "dataframe.df").equals(expected)


from _unittest_bridge import load_tests_from  # noqa: E402

load_tests = load_tests_from(globals())
