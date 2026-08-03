from moe_gate_only.nngpt_data import _fit_completion_to_context


def test_fit_completion_trims_only_prompt_tokens() -> None:
    tokenized = {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "labels": [-100] * 8 + list(range(8, 12)),
        "response_length": 4,
    }

    fitted = _fit_completion_to_context(tokenized, context_length=8)

    assert fitted["fits_context"] is True
    assert fitted["input_ids"] == [0, 1, 6, 7, 8, 9, 10, 11]
    assert fitted["labels"] == [-100] * 4 + [8, 9, 10, 11]
    assert fitted["truncated_prompt_tokens"] == 4


def test_fit_completion_rejects_response_larger_than_context() -> None:
    tokenized = {
        "input_ids": list(range(8)),
        "attention_mask": [1] * 8,
        "labels": [-100] + list(range(1, 8)),
        "response_length": 7,
    }

    fitted = _fit_completion_to_context(tokenized, context_length=6)

    assert fitted["fits_context"] is False
