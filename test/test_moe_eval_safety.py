import json
from pathlib import Path

from moe_cycle.eval_safety import (
    _wrap_evaluate_epoch,
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
        prm_json=None,
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
            prm_json,
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
    result = guarded(5, tmp_path, "moe-gate-cycle", 1, False,
                     custom_synth_dir=models_dir, prm_json=fixed_prm)

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
        fixed_prm,
    )
    assert (model_dir / "hp.txt.moe-invalid").is_file()


def test_moe_wrapper_is_idempotent() -> None:
    def shared_evaluator(*args, **kwargs):
        return None

    guarded = _wrap_evaluate_epoch(shared_evaluator)
    assert _wrap_evaluate_epoch(guarded) is guarded
