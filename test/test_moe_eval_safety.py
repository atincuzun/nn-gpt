import json
import os
from pathlib import Path

import pandas as pd

from moe_cycle.eval_safety import (
    _moe_cv_eval_environment,
    _wrap_evaluate_epoch,
    ensure_moe_eval_safety,
    quarantine_invalid_hyperparameters,
)


def test_quarantine_invalid_hyperparameters_preserves_valid_dicts(tmp_path: Path) -> None:
    models_dir = tmp_path / "synth_nn"
    valid_dir = models_dir / "B0"
    invalid_dir = models_dir / "B1"
    broken_dir = models_dir / "B2"
    valid_dir.mkdir(parents=True)
    invalid_dir.mkdir()
    broken_dir.mkdir()

    valid_hp = valid_dir / "hp.txt"
    invalid_hp = invalid_dir / "hp.txt"
    broken_hp = broken_dir / "hp.txt"
    valid_hp.write_text(json.dumps({"batch": 64}), encoding="utf-8")
    invalid_hp.write_text(json.dumps(["batch", "transform"]), encoding="utf-8")
    broken_hp.write_text("{not-json", encoding="utf-8")

    quarantined = quarantine_invalid_hyperparameters(models_dir)

    assert valid_hp.is_file()
    assert json.loads(valid_hp.read_text(encoding="utf-8")) == {"batch": 64}
    assert not invalid_hp.exists()
    assert not broken_hp.exists()
    assert len(quarantined) == 2
    assert json.loads((invalid_dir / "hp.txt.moe-invalid").read_text()) == [
        "batch",
        "transform",
    ]
    assert (broken_dir / "hp.txt.moe-invalid").read_text() == "{not-json"


def test_moe_wrapper_sanitizes_before_calling_shared_evaluator(tmp_path: Path) -> None:
    models_dir = tmp_path / "synth_nn"
    model_dir = models_dir / "B27"
    model_dir.mkdir(parents=True)
    hp_path = model_dir / "hp.txt"
    hp_path.write_text('["batch", "transform"]', encoding="utf-8")
    observed = {}

    def shared_evaluator(
        epoch,
        out_path,
        nn_name_prefix,
        nn_train_epochs,
        trans_mode,
        classification_mode=False,
        custom_synth_dir=None,
        trainer_model=None,
        aug_mode=False,
        *,
        prm_json=None,
        batch=None,
    ):
        observed["hp_exists"] = hp_path.exists()
        observed["arguments"] = (
            epoch,
            out_path,
            nn_name_prefix,
            nn_train_epochs,
            trans_mode,
            classification_mode,
            custom_synth_dir,
            trainer_model,
            aug_mode,
            prm_json,
            batch,
        )
        return {"epoch": epoch}

    guarded = _wrap_evaluate_epoch(shared_evaluator)
    fixed_prm = {
        "batch": 64,
        "dropout": 0.2,
        "lr": 0.01,
        "momentum": 0.9,
        "transform": "norm_32_flip",
    }
    trainer = object()
    result = guarded(5, tmp_path, "moe-gate-cycle", 1, False, False,
                     models_dir, trainer, False, prm_json=fixed_prm, batch=64)

    assert result == {"epoch": 5}
    assert observed["hp_exists"] is False
    assert observed["arguments"] == (
        5,
        tmp_path,
        "moe-gate-cycle",
        1,
        False,
        False,
        models_dir,
        trainer,
        False,
        fixed_prm,
        64,
    )
    assert (model_dir / "hp.txt.moe-invalid").is_file()


def test_moe_wrapper_is_idempotent() -> None:
    def shared_evaluator(*args, **kwargs):
        return None

    guarded = _wrap_evaluate_epoch(shared_evaluator)
    assert _wrap_evaluate_epoch(guarded) is guarded


def test_moe_wrapper_targets_canonical_tune(monkeypatch):
    from ab.gpt.util.tune import Tune

    original = Tune._evaluate_epoch
    monkeypatch.setattr(Tune, "_evaluate_epoch", original)
    ensure_moe_eval_safety()

    assert Tune._evaluate_epoch._moe_original is original


def test_moe_inline_evaluation_keeps_batch_precedence_and_does_not_retry_oom(tmp_path, monkeypatch):
    from ab.gpt.util.tune import Tune

    nneval = Tune.NNEval
    out_path = tmp_path / "A2"
    models_dir = Tune.synth_dir(out_path)
    for model_id in ("B0", "B1"):
        candidate = models_dir / model_id
        candidate.mkdir(parents=True)
        (candidate / Tune.new_nn_file).write_text("class Net: pass\n")
    (models_dir / "B1" / "hp.txt").write_text('{"batch": 32, "lr": 0.2, "momentum": 0.84}')
    pd.Series({"nn": "ga-seed", "prm": {"batch": 96, "lr": 0.3, "momentum": 0.87}}).to_pickle(
        models_dir / "B1" / "dataframe.df"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NNGPT_EVAL_PRM_MODE", "fixed")
    monkeypatch.setenv("NNGPT_OOM_RETRY", "1")
    monkeypatch.setenv("NNGPT_EVAL_SAVE_TO_DB", "0")
    monkeypatch.delenv("NNGPT_VALIDITY_GATE", raising=False)
    monkeypatch.setattr(nneval, "nngpt_dir", tmp_path)
    monkeypatch.setattr(nneval, "copy_to_lemur", lambda *args: None)
    monkeypatch.setattr(nneval, "release_memory", lambda: None)
    monkeypatch.setattr(Tune, "release_memory", lambda: None)
    pool = nneval.NNEvalWorkerPool
    monkeypatch.setattr(pool, "prewarm_nneval_workers", lambda **kwargs: None)
    monkeypatch.setattr(pool, "shutdown_nneval_workers", lambda: None)
    received = []

    def fake_evaluate(entries, *, use_all_visible_gpus):
        requests = [entry["payload"] for entry in entries]
        received.append(requests)
        assert all(not request["save_to_db"] for request in requests)
        return [
            {"success": False, "error": "out of memory", "is_oom": True},
            {"success": True, "accuracy": 0.75},
        ]

    monkeypatch.setattr(pool, "evaluate_model_entries", fake_evaluate)
    fixed_prm = {"lr": 0.125, "transform": "norm_32_flip", "epoch": 99}
    with _moe_cv_eval_environment():
        Tune._evaluate_epoch(2, out_path, "moe", 1, False,
                             custom_synth_dir=models_dir, prm_json=fixed_prm, batch=64)

    assert len(received) == 1
    fallback, seeded = (request["prm"] for request in received[0])
    assert fallback["batch"] == 64
    assert seeded["batch"] == 96
    assert seeded["momentum"] == 0.87
    for prm in (fallback, seeded):
        assert prm["lr"] == 0.125
        assert prm["transform"] == "norm_32_flip"
        assert prm["epoch"] == 1
    assert os.environ["NNGPT_EVAL_PRM_MODE"] == "fixed"
    assert os.environ["NNGPT_OOM_RETRY"] == "1"
    assert (models_dir / "B0" / "error.txt").is_file()
    assert (models_dir / "B1" / "eval_info.json").is_file()
    assert (tmp_path / "cycle_results_2.json").is_file()


def test_tune_subprocess_passes_json_and_moe_fallback_batch(tmp_path, monkeypatch):
    from ab.gpt.util.tune import Tune

    out_path = tmp_path / "A3"
    models_dir = Tune.synth_dir(out_path)
    models_dir.mkdir(parents=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS", "1")
    monkeypatch.setattr(Tune, "release_memory", lambda: None)
    calls = []

    def fake_run(cmd, *, check, env):
        calls.append(cmd)
        assert cmd[:3] == [Tune.sys.executable, "-m", "ab.gpt.act.eval.Eval"]
        assert check is True
        assert env["CUDA_VISIBLE_DEVICES"] == "7"
        assert env["NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS"] == "1"
        assert env["NNGPT_EVAL_PRM_MODE"] == "seed"
        assert env["NNGPT_OOM_RETRY"] == "0"

    monkeypatch.setattr(Tune.subprocess, "run", fake_run)
    fixed_prm = {"batch": 96, "transform": "a JSON string with spaces"}
    with _moe_cv_eval_environment():
        Tune._evaluate_epoch(3, out_path, "moe", 2, False,
                             custom_synth_dir=models_dir, prm_json=fixed_prm, batch=64)
    (cmd,) = calls
    assert json.loads(cmd[cmd.index("--prm_json") + 1]) == fixed_prm
    assert cmd[cmd.index("--batch_size") + 1] == "64"
    assert cmd[cmd.index("--only_epoch") + 1] == "3"
    assert cmd[cmd.index("--custom_synth_dir") + 1] == str(models_dir)
