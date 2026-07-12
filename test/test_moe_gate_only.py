from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_gate_only import (  # noqa: E402
    GATE_FACTORIES,
    GateInstall,
    assert_hf_native_model,
    count_parameters,
    explain_gate_candidates,
    find_moe_gates,
    freeze_except_gates,
    get_gate_candidate_report,
    get_gate_logits,
    install_gates,
    teacher_student_distillation_loss,
    trainable_parameter_names,
)
from ab.gpt.util.Chatbot import _validate_input_ids  # noqa: E402
from moe_gate_only.universal import (  # noqa: E402
    GateSite,
    _DeepSeekV2Gate,
    _DeepSeekV2TeacherStudentGate,
    _extract_logits_from_gate_output,
    _initialize_from_original_projection,
    _new_gate_for_site,
)


def zero_gate(model_dim: int, num_experts: int) -> torch.nn.Module:
    g = torch.nn.Linear(model_dim, num_experts, bias=False)
    torch.nn.init.zeros_(g.weight)
    return g


def wrong_shape_gate(model_dim: int, num_experts: int) -> torch.nn.Module:
    return torch.nn.Linear(model_dim, num_experts + 1, bias=False)


class _NestedGate(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.inner = torch.nn.Sequential(torch.nn.Linear(model_dim, num_experts, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class _ToyMoeBlock(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate = _NestedGate(model_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        if logits.shape[-1] != self.num_experts:
            raise RuntimeError("bad gate shape")
        return x


class _ToyMoeModel(torch.nn.Module):
    def __init__(self, model_dim: int = 8, num_experts: int = 4, vocab_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_experts=num_experts, vocab_size=vocab_size)
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim)
        self.layers = torch.nn.ModuleList([_ToyMoeBlock(model_dim, num_experts)])
        self.lm_head = torch.nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x))


class _ChoiceBlock(torch.nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.score_proj = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.paths = torch.nn.ModuleList(
            [torch.nn.Linear(model_dim, model_dim, bias=False) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        logits = self.score_proj(flat)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), 1, dim=-1)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.paths):
            token_idx, top_idx = torch.where(indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            out[token_idx] += expert(flat[token_idx]) * weights[token_idx, top_idx].unsqueeze(-1)
        return out.reshape(original_shape)


class _DynamicChoiceModel(torch.nn.Module):
    def __init__(self, model_dim: int = 8, num_experts: int = 4, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim)
        self.blocks = torch.nn.ModuleList([_ChoiceBlock(model_dim, num_experts)])
        self.lm_head = torch.nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x)
        return SimpleNamespace(logits=self.lm_head(x))


# --- tiny config factories ---

def _tiny_lfm():
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM
    cfg = Lfm2MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1,
        num_dense_layers=0, num_experts=4, num_experts_per_tok=2,
        layer_types=["full_attention"] * 2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return Lfm2MoeForCausalLM(cfg)


def _tiny_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return MixtralForCausalLM(cfg)


def _tiny_qwen2():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    cfg = Qwen2MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_cache=False,
    )
    return Qwen2MoeForCausalLM(cfg)


def _tiny_dbrx():
    from transformers import DbrxConfig, DbrxForCausalLM
    cfg = DbrxConfig(
        d_model=16, n_layers=1, n_heads=2, vocab_size=32,
        ffn_config={"ffn_hidden_size": 16, "hidden_size": 16,
                     "moe_num_experts": 4, "moe_top_k": 1},
        attn_config={"attn_pdrop": 0.0, "clip_qkv": 1.0,
                      "kv_n_heads": 1, "rope_theta": 10000.0},
    )
    return DbrxForCausalLM(cfg)


def _tiny_granite():
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
    cfg = GraniteMoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return GraniteMoeForCausalLM(cfg)


def _tiny_cohere2():
    from transformers import Cohere2MoeConfig, Cohere2MoeForCausalLM
    cfg = Cohere2MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_experts=4, num_experts_per_tok=2,
        num_shared_experts=0, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_cache=False,
    )
    return Cohere2MoeForCausalLM(cfg)


def _tiny_afmoe():
    from transformers import AfmoeConfig, AfmoeForCausalLM
    cfg = AfmoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=1, num_dense_layers=0,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
        global_attn_every_n_layers=1, sliding_window=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return AfmoeForCausalLM(cfg)


def _tiny_glm4():
    from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
    cfg = Glm4MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
        n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=2,
        first_k_dense_replace=0, n_group=1, topk_group=1,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return Glm4MoeForCausalLM(cfg)


def _tiny_minimax():
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM
    cfg = MiniMaxM2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return MiniMaxM2ForCausalLM(cfg)


def _tiny_deepseek_v4():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    cfg = DeepseekV4Config(
        vocab_size=32, hidden_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        head_dim=8, q_lora_rank=8, num_experts_per_tok=2,
        n_routed_experts=4, n_shared_experts=1,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["hash_moe", "moe"],
        sliding_window=8, o_groups=1, o_lora_rank=8,
        index_n_heads=1, index_head_dim=4, index_topk=2,
        hc_mult=2, use_cache=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return DeepseekV4ForCausalLM(cfg)


# --- tests ---

def _check_discovery(model_name, model, min_gates):
    sites = find_moe_gates(model)
    assert len(sites) >= min_gates, (
        f"{model_name}: expected >= {min_gates} gates, found {len(sites)}"
    )
    for site in sites:
        assert site.model_dim > 0
        assert site.num_experts > 0
        assert site.pattern in ("linear", "wrapped_linear", "parameter_gate")
    print(f"  {model_name}: {len(sites)} gates discovered, patterns={set(s.pattern for s in sites)}")


def _check_install_and_forward(model_name, model, vocab_size=32):
    installs = install_gates(model, zero_gate)
    assert len(installs) > 0
    freeze_except_gates(model, installs)
    trainable, total = count_parameters(model)
    assert trainable > 0
    assert trainable < total

    with torch.no_grad():
        out = model(torch.randint(3, vocab_size, (1, 4)))
    assert torch.isfinite(out.logits).all()
    print(f"  {model_name}: {len(installs)} gates replaced, "
          f"trainable={trainable}/{total}, forward OK")


def test_sparse_sigmoid_extractor_prefers_logits():
    scores = torch.tensor([[0.0, 0.4, 0.0, 0.6]], dtype=torch.float32)
    logits = torch.randn(1, 4, requires_grad=True)
    extracted = _extract_logits_from_gate_output((scores, logits), num_experts=4)
    assert extracted is logits


def test_input_validation_accepts_added_special_tokens_within_model_vocab():
    class Tokenizer:
        vocab_size = 100000

        def __len__(self):
            return 100002

    model = _ToyMoeModel(vocab_size=102400)
    inputs = {"input_ids": torch.tensor([[100000, 100001]])}
    _validate_input_ids(inputs, model, Tokenizer())
    assert inputs["input_ids"].tolist() == [[100000, 100001]]


def test_deepseek_replacement_preserves_float32_native_routing():
    class NativeDeepSeekGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16))
            self.n_routed_experts = 4
            self.top_k = 2
            self.topk_method = "greedy"
            self.scoring_func = "softmax"
            self.seq_aux = False
            self.norm_topk_prob = False
            self.routed_scaling_factor = 1.0
            self.alpha = 0.0

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.nn.functional.linear(flat.float(), self.weight.float())
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            weights, indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            return indices, weights, None

    class GeneratedGate(torch.nn.Module):
        def __init__(self, model_dim, num_experts):
            super().__init__()
            self.base = torch.nn.Linear(model_dim, num_experts, bias=False)
            self.residual = torch.nn.Linear(model_dim, num_experts, bias=False)
            torch.nn.init.zeros_(self.residual.weight)

        def forward(self, x):
            return self.base(x) + self.residual(x)

    native = NativeDeepSeekGate()
    block = torch.nn.Module()
    block.gate = native
    site = GateSite(0, block, native, "gate", 8, 4, "parameter_gate")
    generated = _new_gate_for_site(GeneratedGate, site)
    assert generated.base.weight.dtype == torch.float32
    _initialize_from_original_projection(generated, site)
    replacement = _DeepSeekV2Gate(generated, top_k=2)
    for name in (
        "topk_method", "scoring_func", "seq_aux", "norm_topk_prob",
        "routed_scaling_factor", "alpha",
    ):
        setattr(replacement, name, getattr(native, name))

    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    native_indices, native_weights, _ = native(hidden_states)
    replacement_indices, replacement_weights, _ = replacement(hidden_states)
    assert torch.equal(native_indices, replacement_indices)
    torch.testing.assert_close(native_weights, replacement_weights, rtol=0, atol=0)


def test_deepseek_teacher_student_shadow_trains_only_random_student():
    class NativeDeepSeekGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))
            self.top_k = 2
            self.topk_method = "greedy"
            self.scoring_func = "softmax"
            self.seq_aux = False
            self.norm_topk_prob = False
            self.routed_scaling_factor = 1.0
            self.alpha = 0.0

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            scores = torch.nn.functional.linear(flat.float(), self.weight.float()).softmax(-1)
            weights, indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            return indices, weights, None

    teacher = NativeDeepSeekGate()
    student = torch.nn.Linear(8, 4, bias=False)
    wrapper = _DeepSeekV2TeacherStudentGate(teacher, student, top_k=2)
    for name in (
        "topk_method", "scoring_func", "seq_aux", "norm_topk_prob",
        "routed_scaling_factor", "alpha",
    ):
        setattr(wrapper, name, getattr(teacher, name))
    block = torch.nn.Module()
    block.gate = wrapper
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(8, 8)
    model.block = block
    site = GateSite(0, block, teacher, "gate", 8, 4, "parameter_gate")
    install = GateInstall(
        site,
        teacher,
        wrapper,
        block,
        "gate",
        teacher,
        mode="teacher_student",
        teacher_gate=teacher,
        student_gate=student,
    )

    hidden_states = torch.randn(2, 3, 8)
    expected = teacher(hidden_states)
    actual = wrapper(hidden_states)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    freeze_except_gates(model, [install])
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert trainable_ids == {id(parameter) for parameter in student.parameters()}
    loss = teacher_student_distillation_loss([install])
    loss.backward()
    assert teacher.weight.grad is None
    assert student.weight.grad is not None
    assert torch.isfinite(student.weight.grad).all()
    assert float(student.weight.grad.norm()) > 0


def test_deepseek_teacher_student_blends_logits_before_topk():
    class NativeDeepSeekGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
            ]))
            self.top_k = 2
            self.topk_method = "greedy"
            self.scoring_func = "softmax"
            self.seq_aux = False
            self.norm_topk_prob = False
            self.routed_scaling_factor = 1.0
            self.alpha = 0.0

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            scores = torch.nn.functional.linear(flat.float(), self.weight.float()).softmax(-1)
            weights, indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            return indices, weights, None

    teacher = NativeDeepSeekGate()
    student = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        student.weight.copy_(torch.tensor([
            [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        ]))
    wrapper = _DeepSeekV2TeacherStudentGate(
        teacher, student, top_k=2, student_weight=0.25
    )
    for name in (
        "topk_method", "scoring_func", "seq_aux", "norm_topk_prob",
        "routed_scaling_factor", "alpha",
    ):
        setattr(wrapper, name, getattr(teacher, name))

    hidden_states = torch.tensor([[[2.0, 1.0], [1.0, -2.0]]])
    flat = hidden_states.reshape(-1, 2)
    teacher_logits = torch.nn.functional.linear(flat, teacher.weight)
    student_logits = student(flat)
    expected_logits = torch.lerp(teacher_logits, student_logits, 0.25)
    expected_weights, expected_indices = torch.topk(
        expected_logits.softmax(-1), 2, dim=-1, sorted=False
    )
    indices, weights, _ = wrapper(hidden_states)
    torch.testing.assert_close(wrapper._last_gate_logits, expected_logits)
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)


def test_nested_wrapped_linear_rolls_back_on_verify_failure():
    model = _ToyMoeModel()
    site = find_moe_gates(model)[0]
    old_inner = site.gate.inner[0]

    try:
        install_gates(model, wrong_shape_gate, verify=True)
        raise AssertionError("install_gates should have failed verification")
    except RuntimeError as exc:
        assert "rolled back" in str(exc)

    assert site.gate.inner[0] is old_inner
    with torch.no_grad():
        out = model(torch.randint(3, 16, (1, 4)))
    assert torch.isfinite(out.logits).all()


def test_verify_forward_does_not_leave_stale_gate_logits():
    model = _ToyMoeModel()
    install_gates(model, zero_gate, verify=True)
    assert get_gate_logits(model) == []

    with torch.no_grad():
        model(torch.randint(3, 16, (1, 4)))
    logits = get_gate_logits(model)
    assert len(logits) == 1
    assert logits[0].shape[-1] == 4


def test_remote_code_guard_rejects_transformers_modules():
    class RemoteLeaf(torch.nn.Module):
        def forward(self, x):
            return x

    RemoteLeaf.__module__ = "transformers_modules.fake.modeling_remote"

    class Host(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.remote = RemoteLeaf()

    model = Host()
    try:
        assert_hf_native_model(model)
        raise AssertionError("remote-code guard should have rejected the model")
    except RuntimeError as exc:
        assert "Remote-code modules found" in str(exc)

    assert_hf_native_model(model, allow_remote_code=True)


def test_dynamic_discovery_does_not_need_gate_names():
    model = _DynamicChoiceModel()
    sample = torch.randint(3, 16, (1, 4))

    sites = find_moe_gates(model, sample_input=sample)
    assert len(sites) == 1
    assert sites[0].path == "blocks.0.score_proj"
    assert sites[0].num_experts == 4
    assert "runtime_output_matches_experts" in sites[0].evidence


def test_dynamic_install_and_candidate_report():
    model = _DynamicChoiceModel()
    sample = torch.randint(3, 16, (1, 4))

    reports = explain_gate_candidates(model, sample)
    assert any(r.accepted and r.path == "blocks.0.score_proj" for r in reports)

    installs = install_gates(model, zero_gate, sample_input=sample)
    assert len(installs) == 1
    assert installs[0].site.path == "blocks.0.score_proj"
    assert get_gate_candidate_report()

    with torch.no_grad():
        out = model(sample)
    assert torch.isfinite(out.logits).all()


def test_lfm_discovery():
    print("LFM discovery:")
    _check_discovery("lfm2_moe", _tiny_lfm(), min_gates=2)


def test_mixtral_discovery():
    print("Mixtral discovery:")
    _check_discovery("mixtral", _tiny_mixtral(), min_gates=2)


def test_qwen2_discovery():
    print("Qwen2-MoE discovery:")
    _check_discovery("qwen2_moe", _tiny_qwen2(), min_gates=2)


def test_dbrx_discovery():
    print("DBRX discovery:")
    _check_discovery("dbrx", _tiny_dbrx(), min_gates=1)


def test_granite_discovery():
    print("Granite discovery:")
    _check_discovery("granitemoe", _tiny_granite(), min_gates=1)


def test_cohere2_discovery():
    print("Cohere2-MoE discovery:")
    _check_discovery("cohere2_moe", _tiny_cohere2(), min_gates=1)


def test_afmoe_discovery():
    print("AFMoE discovery:")
    _check_discovery("afmoe", _tiny_afmoe(), min_gates=1)


def test_glm4_discovery():
    print("GLM4-MoE discovery:")
    _check_discovery("glm4_moe", _tiny_glm4(), min_gates=1)


def test_minimax_discovery():
    print("MiniMax discovery:")
    _check_discovery("minimax_m2", _tiny_minimax(), min_gates=1)


def test_deepseek_v4_discovery():
    print("DeepSeek V4 discovery:")
    _check_discovery("deepseek_v4", _tiny_deepseek_v4(), min_gates=2)


def test_lfm_install_forward():
    print("LFM install + forward:")
    _check_install_and_forward("lfm2_moe", _tiny_lfm())


def test_mixtral_install_forward():
    print("Mixtral install + forward:")
    _check_install_and_forward("mixtral", _tiny_mixtral())


def test_qwen2_install_forward():
    print("Qwen2-MoE install + forward:")
    _check_install_and_forward("qwen2_moe", _tiny_qwen2())


def test_dbrx_install_forward():
    print("DBRX install + forward:")
    _check_install_and_forward("dbrx", _tiny_dbrx())


def test_granite_install_forward():
    print("Granite install + forward:")
    _check_install_and_forward("granitemoe", _tiny_granite())


def test_cohere2_install_forward():
    print("Cohere2-MoE install + forward:")
    _check_install_and_forward("cohere2_moe", _tiny_cohere2())


def test_afmoe_install_forward():
    print("AFMoE install + forward:")
    _check_install_and_forward("afmoe", _tiny_afmoe())


def test_glm4_install_forward():
    print("GLM4-MoE install + forward:")
    _check_install_and_forward("glm4_moe", _tiny_glm4())


def test_minimax_install_forward():
    print("MiniMax install + forward:")
    _check_install_and_forward("minimax_m2", _tiny_minimax())


def test_deepseek_v4_install_forward():
    print("DeepSeek V4 install + forward:")
    model = _tiny_deepseek_v4()
    sample = torch.randint(3, 32, (1, 4))
    installs = install_gates(model, zero_gate, sample_input={"input_ids": sample})
    assert len(installs) == 2
    assert {type(getattr(i.site.block, i.site.gate_attr)).__name__ for i in installs} == {
        "_HashScoreFnGate", "_ScoreFnTopKGate",
    }
    freeze_except_gates(model, installs)
    trainable, total = count_parameters(model)
    assert trainable > 0
    assert trainable < total
    with torch.no_grad():
        out = model(input_ids=sample)
    assert torch.isfinite(out.logits).all()
    print(f"  deepseek_v4: {len(installs)} gates replaced, "
          f"trainable={trainable}/{total}, forward OK")


if __name__ == "__main__":
    tests = sorted(
        (name, obj) for name, obj in globals().items() if name.startswith("test_")
    )
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}\n")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}\n")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
