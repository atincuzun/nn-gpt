# ab/gpt/util/LLM.py
from ab.nn.util.Const import out_dir
from ab.gpt.util.Const import llm_dir, llm_tokenizer_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Util import exists

import os
import json
import tempfile
import shutil
from copy import deepcopy
import torch
import torch.cuda
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)


class LLM:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig = None,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None,
                 use_deepspeed=False,
                 base_path=out_dir,
                 context_length=None,
                 gguf_file=None,
                 training_args=None,
                 use_unsloth=False,
                 load_in_4bit=True,
                 moe_edit_config=None):
        self.model_path = model_path
        self.context_length = context_length
        self._use_unsloth = use_unsloth
        self._moe_edit_config = deepcopy(moe_edit_config) if moe_edit_config is not None else None
        self._moe_edit_runtime = {
            "enabled": False,
            "mode": "disabled",
            "base_model_name": model_path,
        }
        
        # ===== Unsloth Fast Path =====
        if use_unsloth:
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=model_path,
                dtype = None,
                max_seq_length=context_length or 4096,
                load_in_4bit=load_in_4bit,
                token=access_token,
                full_finetuning = False,
            )
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            print(f"[Unsloth] Loaded {model_path}, 4bit={load_in_4bit}")
            return
        
        # ===== Original HuggingFace Path =====
        # --- Tokenizer ---
        tok_fl_nm = llm_tokenizer_dir(base_path, model_path)
        raw_fl_nm = llm_dir(base_path, model_path)
        tokenizer_exists = exists(tok_fl_nm)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_fl_nm if tokenizer_exists else model_path,
            trust_remote_code=True, token=access_token, gguf_file=gguf_file
        )
        self.tokenizer.add_eos_token = True
        if self.tokenizer.pad_token_id is None:
            # Map pad_token to eos_token to avoid accidental masking or unk-id behavior
            # This is safer for LLaMA-like models (e.g., DeepSeek-Coder)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if tokenizer_exists:
            print("Loading Tokenizer from local files:", tok_fl_nm)
        else:
            print("Downloading Tokenizer...")
            self.tokenizer.save_pretrained(tok_fl_nm, access_token=access_token)
            print("Tokenizer saved to: ", tok_fl_nm)

        # --- Determine source dir for local model files (if any) ---
        src_dir = None
        if exists(local_path):
            src_dir = local_path
        elif exists(raw_fl_nm):
            src_dir = raw_fl_nm

        # --- Build a safe config without relying on from_dict/get_config_dict ---
        config = None
        if src_dir is not None and os.path.exists(os.path.join(src_dir, "config.json")):
            # Read local config, sanitize if needed, and load via a temporary folder
            with open(os.path.join(src_dir, "config.json"), "r") as f:
                cfg_dict = json.load(f)
            if cfg_dict.get("quantization_config", "absent") is None:
                # Remove null to prevent HF internals from calling .to_dict() on None
                del cfg_dict["quantization_config"]

            # Write sanitized config into a small temp dir and load from there
            tmp_cfg_dir = tempfile.mkdtemp(prefix="sanitized_cfg_")
            try:
                with open(os.path.join(tmp_cfg_dir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f)
                config = AutoConfig.from_pretrained(
                    tmp_cfg_dir, trust_remote_code=True, token=access_token
                )
            finally:
                # We can keep or clean; keeping is usually fine, but we’ll clean to avoid clutter.
                shutil.rmtree(tmp_cfg_dir, ignore_errors=True)

        if config is None:
            # Remote (or no local config found) → normal path
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, token=access_token
            )

        # --- Model ---
        # Figure out if ZeRO-3 is enabled (deepspeed arg can be dict or path)
        # Check training_args.deepspeed first if available, otherwise use use_deepspeed boolean
        use_zero3 = False
        if training_args is not None:
            deepspeed_cfg = getattr(training_args, "deepspeed", None)
            use_zero3 = bool(deepspeed_cfg)
        elif use_deepspeed:
            # Fallback: if use_deepspeed is True, assume ZeRO-3 might be used
            use_zero3 = True
        
        # Build model kwargs (sanitize for ZeRO-3)
        deepspeed_specific_prm = {} if use_zero3 else {"device_map": "auto"}
        model_kwargs = dict(
            trust_remote_code=True,
            max_memory={i: max_memory for i in range(torch.cuda.device_count())},
            token=access_token,
            torch_dtype=torch.bfloat16,  # QLoRA compute
            gguf_file=gguf_file,
            config=config,
            **deepspeed_specific_prm
        )
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        
        # --- ZeRO-3 guard: strip incompatible args ---
        # NOTE: Other files using device_map (RAG_AlterNN.py, TuneRL.py, MergeLLM.py, etc.)
        # are safe because they don't use DeepSpeed/ZeRO-3. Only this LLM class needs sanitization
        # when training_args.deepspeed is set.
        if use_zero3:
            # Absolutely no device_map / low_cpu_mem_usage on ZeRO-3
            model_kwargs.pop("device_map", None)
            model_kwargs.pop("low_cpu_mem_usage", None)
            # (optional) these can also trip sharding heuristics—keep it simple on ZeRO-3:
            model_kwargs.pop("max_memory", None)
            model_kwargs.pop("offload_folder", None)
        
        # Debug: verify nothing slipped through
        print("[DEBUG from_pretrained kwargs]", {k: ("***" if k == "token" else v) for k, v in model_kwargs.items()})
        
        base_model = local_path if exists(local_path) else raw_fl_nm if exists(raw_fl_nm) else model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        if exists(local_path):
            print("Loading Model from local files:", "'" + local_path + "'")
        elif exists(raw_fl_nm):
            print(f"Loading Raw Model from local files: '{raw_fl_nm}'")
        else:
            self.model.save_pretrained(raw_fl_nm, access_token=access_token)
            print("Model saved to: ", raw_fl_nm)

    def apply_moe_edit_if_enabled(self, model=None, selected_model_id=None):
        model = model or self.model
        from ab.gpt.moe.config import build_edit_config, normalize_moe_edit_config, is_moe_edit_enabled
        from ab.gpt.moe.hf_moe_editor import HFMoEEditor

        runtime_cfg = normalize_moe_edit_config(self._moe_edit_config, base_model_name=self.model_path)
        if not is_moe_edit_enabled(runtime_cfg):
            self._moe_edit_runtime = runtime_cfg
            self.model = model
            return model, runtime_cfg

        if getattr(model, "_moe_edit_result", None) is not None:
            runtime_cfg["selected_model_id"] = selected_model_id or self.model_path
            self._moe_edit_runtime = runtime_cfg
            self.model = model
            return model, runtime_cfg

        editor = HFMoEEditor(build_edit_config(runtime_cfg))
        edited_model, edit_result = editor.apply(model, selected_model_id=selected_model_id or self.model_path)
        runtime_cfg.update(
            {
                "selected_model_id": edit_result.selected_model_id,
                "routing_before": edit_result.routing_before,
                "routing_after": edit_result.routing_after,
                "routing_changes": edit_result.routing_changes,
                "adapter_attached": edit_result.adapter_attached,
                "adapter_summary": edit_result.adapter_summary,
            }
        )
        self._moe_edit_runtime = runtime_cfg
        self.model = edited_model
        return edited_model, runtime_cfg

    def get_moe_edit_runtime(self):
        return deepcopy(self._moe_edit_runtime)

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

    def get_max_length(self) -> int:
        if self.context_length:
            return self.context_length
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        if not max_length:
            max_length = 4096
        return max_length
