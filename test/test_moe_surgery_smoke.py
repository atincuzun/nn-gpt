from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_surgery import detect_model_info, list_moe_blocks, modify_config, replace_router
from moe_surgery.types import RouterPattern


def zero_gate(x: torch.Tensor, model_dim: int, num_experts: int) -> torch.Tensor:
    return torch.zeros((*x.shape[:-1], num_experts), device=x.device, dtype=x.dtype)


def causal_forward_ok(model, vocab_size: int = 32) -> None:
    with torch.no_grad():
        out = model(torch.randint(0, vocab_size, (1, 3)))
    assert out.logits.shape[:2] == (1, 3)
    assert torch.isfinite(out.logits).all()


def seq2seq_forward_ok(model, vocab_size: int = 32) -> None:
    ids = torch.randint(3, vocab_size, (1, 3))
    with torch.no_grad():
        out = model(input_ids=ids, decoder_input_ids=ids)
    assert out.logits.shape[:2] == (1, 3)
    assert torch.isfinite(out.logits).all()


def test_qwen2_top_k_alias_and_forward():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    cfg = Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=3,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    model = Qwen2MoeForCausalLM(cfg)
    info = detect_model_info(model)
    assert info.num_experts == 4
    assert info.top_k == 3
    assert replace_router(model, zero_gate) == len(list_moe_blocks(model)) == 2
    causal_forward_ok(model)


def test_switch_nested_paths_forward():
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration

    cfg = SwitchTransformersConfig(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        num_experts=4,
        expert_capacity=16,
        encoder_sparse_step=1,
        decoder_sparse_step=1,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = SwitchTransformersForConditionalGeneration(cfg)
    info = detect_model_info(model)
    assert info.pattern == RouterPattern.SWITCH_CAPACITY
    assert len(list_moe_blocks(model)) == 4
    assert replace_router(model, zero_gate) == 4
    seq2seq_forward_ok(model)


def test_nllb_hyphenated_model_type_forward():
    from transformers import NllbMoeConfig, NllbMoeForConditionalGeneration

    cfg = NllbMoeConfig(
        vocab_size=32,
        d_model=16,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        num_experts=4,
        expert_capacity=16,
        encoder_sparse_step=1,
        decoder_sparse_step=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=0,
    )
    model = NllbMoeForConditionalGeneration(cfg)
    info = detect_model_info(model)
    assert info.model_type == "nllb-moe"
    assert info.pattern == RouterPattern.NLLB_CAPACITY
    assert len(list_moe_blocks(model)) == 4
    assert replace_router(model, zero_gate) == 4
    seq2seq_forward_ok(model)


def test_dbrx_nested_config_raw_logits_forward():
    from transformers import DbrxConfig, DbrxForCausalLM

    cfg = DbrxConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        vocab_size=32,
        ffn_config={"ffn_hidden_size": 16, "hidden_size": 16, "moe_num_experts": 4, "moe_top_k": 1},
        attn_config={"attn_pdrop": 0.0, "clip_qkv": 1.0, "kv_n_heads": 1, "rope_theta": 10000.0},
    )
    model = DbrxForCausalLM(cfg)
    info = detect_model_info(model)
    assert info.pattern == RouterPattern.RAW_LOGITS
    assert info.num_experts == 4
    assert info.top_k == 1
    assert replace_router(model, zero_gate) == 1
    causal_forward_ok(model)


def test_bias_and_layer_level_routers_forward():
    from transformers import AfmoeConfig, AfmoeForCausalLM, Gemma4ForCausalLM, Gemma4TextConfig, MiniMaxM2Config, MiniMaxM2ForCausalLM

    cases = []

    cases.append(
        AfmoeForCausalLM(
            AfmoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=16,
                num_hidden_layers=1,
                num_dense_layers=0,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                num_shared_experts=1,
                global_attn_every_n_layers=1,
                sliding_window=8,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
    )
    cases.append(
        MiniMaxM2ForCausalLM(
            MiniMaxM2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=2,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
    )
    cases.append(
        Gemma4ForCausalLM(
            Gemma4TextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                enable_moe_block=True,
                num_experts=4,
                top_k_experts=2,
                moe_intermediate_size=16,
                hidden_size_per_layer_input=0,
                vocab_size_per_layer_input=32,
                tie_word_embeddings=False,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
    )

    for model in cases:
        blocks = list_moe_blocks(model)
        assert blocks
        assert replace_router(model, zero_gate) == len(blocks)
        causal_forward_ok(model)


def test_sparse_dispatch_and_inline_linear_forward():
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM, Lfm2MoeConfig, Lfm2MoeForCausalLM

    granite = GraniteMoeForCausalLM(
        GraniteMoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=4,
            num_experts_per_tok=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    lfm = Lfm2MoeForCausalLM(
        Lfm2MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_dense_layers=0,
            num_experts=4,
            num_experts_per_tok=2,
            layer_types=["full_attention", "full_attention"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )

    for model in (granite, lfm):
        blocks = list_moe_blocks(model)
        assert blocks
        assert replace_router(model, zero_gate) == len(blocks)
        causal_forward_ok(model)


def test_other_specialized_router_contracts_forward():
    from transformers import (
        Cohere2MoeConfig,
        Cohere2MoeForCausalLM,
        Ernie4_5_MoeConfig,
        Ernie4_5_MoeForCausalLM,
        Glm4MoeConfig,
        Glm4MoeForCausalLM,
        Llama4ForCausalLM,
        Llama4TextConfig,
        PhimoeConfig,
        PhimoeForCausalLM,
    )

    cases = [
        Cohere2MoeForCausalLM(
            Cohere2MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                num_shared_experts=0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
        Ernie4_5_MoeForCausalLM(
            Ernie4_5_MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                moe_num_experts=4,
                moe_k=2,
                moe_layer_start_index=0,
                moe_layer_interval=1,
                moe_num_shared_experts=0,
                head_dim=8,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
        Glm4MoeForCausalLM(
            Glm4MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                n_shared_experts=1,
                n_routed_experts=4,
                num_experts_per_tok=2,
                first_k_dense_replace=0,
                n_group=1,
                topk_group=1,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
        Llama4ForCausalLM(
            Llama4TextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                intermediate_size_mlp=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=1,
                moe_layers=[0],
                use_qk_norm=False,
                tie_word_embeddings=False,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
        PhimoeForCausalLM(
            PhimoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                num_local_experts=4,
                num_experts_per_tok=2,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
    ]

    for model in cases:
        blocks = list_moe_blocks(model)
        assert blocks
        assert replace_router(model, zero_gate) == len(blocks)
        causal_forward_ok(model)


def test_modify_config_propagates_to_nested_and_router_attrs():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    model = Qwen2MoeForCausalLM(
        Qwen2MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    )
    modify_config(model, num_experts_per_tok=1)
    assert model.config.num_experts_per_tok == 1
    for _, _, block in list_moe_blocks(model):
        assert block.gate.top_k == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
