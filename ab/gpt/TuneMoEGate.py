"""
ab/gpt/TuneMoEGate.py — Entry point for MoE gate experiment.

Usage:
  Set NNGPT_DIR_OVERRIDE to a writable directory (defaults to out/nngpt).
  The LLM config (--llm_conf) enables gate experiment mode via moe_gate_experiment: true.

Example:
  NNGPT_DIR_OVERRIDE=out/moe_gate_exp python ab/gpt/TuneMoEGate.py --llm_conf deepseek_v2_lite_moe_gate_exp.json
"""
import argparse
import sys

from ab.gpt.util.Const import (
    conf_llm_dir, conf_test_dir, conf_train_dir,
    NN_TRAIN_EPOCHS, nngpt_dir,
)

# ── Defaults (minimal — gate experiment bypasses PEFT/LoRA) ────────────────
LLM_CONF = "deepseek_v2_lite_moe_gate_exp.json"
LLM_TUNE_CONF = "NN_gen.json"
NN_GEN_CONF = "Gate_gen_with_gate.json"
CONF_KEYS = "improve_classification_gate"

SKIP_EPOCHES = -1
TEST_NN = 10
MAX_NEW_TOKENS = 16384
NN_NAME_PREFIX = "moe_gate_exp"
MAX_PROMPTS = 1024
TEMPERATURE = 0.8
TOP_K = 70
TOP_P = 0.9


def main(
    llm_conf=LLM_CONF,
    llm_tune_conf=LLM_TUNE_CONF,
    nn_gen_conf=NN_GEN_CONF,
    conf_keys=CONF_KEYS,
    test_nn=TEST_NN,
    max_prompts=MAX_PROMPTS,
    max_new_tokens=MAX_NEW_TOKENS,
    nn_name_prefix=NN_NAME_PREFIX,
    nn_train_epochs=NN_TRAIN_EPOCHS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    skip_epoches=SKIP_EPOCHES,
):
    # Dummy training args / peft config — gate experiment skip Lora entirely.
    from transformers import TrainingArguments
    from peft import LoraConfig
    training_args = TrainingArguments(
        output_dir=str(nngpt_dir / "outputs"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        report_to="none",
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
    )

    from ab.gpt.util.Tune import tune

    tune(
        test_nn=test_nn,
        nn_train_epochs=nn_train_epochs,
        skip_epoch=skip_epoches,
        llm_path=None,
        llm_tune_conf=llm_tune_conf,
        nn_gen_conf=nn_gen_conf,
        conf_keys=conf_keys,
        llm_conf=llm_conf,
        training_args=training_args,
        peft_config=peft_config,
        max_prompts=max_prompts,
        save_llm_output=True,
        max_new_tokens=max_new_tokens,
        nn_name_prefix=nn_name_prefix,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        moe_gate_experiment=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Gate Experiment")
    parser.add_argument("--llm_conf", type=str, default=LLM_CONF)
    parser.add_argument("--llm_tune_conf", type=str, default=LLM_TUNE_CONF)
    parser.add_argument("--nn_gen_conf", type=str, default=NN_GEN_CONF)
    parser.add_argument("--conf_keys", type=str, default=CONF_KEYS)
    parser.add_argument("--test_nn", type=int, default=TEST_NN)
    parser.add_argument("--nn_train_epochs", type=int, default=NN_TRAIN_EPOCHS)
    parser.add_argument("--max_prompts", type=int, default=MAX_PROMPTS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--nn_name_prefix", type=str, default=NN_NAME_PREFIX)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--skip_epoches", type=int, default=SKIP_EPOCHES)
    args = parser.parse_args()
    main(**vars(args))
