# ab/gpt/infer/DPGenerator.py

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
import deepspeed

import ab.nn.api as lemur
from ab.gpt.dist.DistEnv import DistEnv
from ab.gpt.util.Const import synth_dir, new_out_file, new_nn_file, hp_file
from ab.nn.util.Util import create_file
from ab.gpt.util.Util import extract_code, extract_hyperparam
from transformers import AutoTokenizer, AutoModelForCausalLM


# Prompt engineering constants (yours)
extra_instructions = (
    " Use PyTorch for the implementation. Keep the code short. Name the main class of the model \"Net\"."
    " The model code must include default parameters for initialization in the constructor. "
    "Provide only the code. Don't provide any explanation. Remove any text from this reply. "
    "Don't include comments in the code."
)


def _tp_compat_or_raise(cfg, mp_size: int):
    """Light sanity checks so TP doesn’t silently mis-shard."""
    try:
        n_heads = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        if n_heads and (n_heads % mp_size != 0):
            raise RuntimeError(f"Tensor-parallel mp_size={mp_size} must divide num_attention_heads={n_heads}.")
        if hidden and (hidden % mp_size != 0):
            # Not always required by all impls, but catches many odd setups.
            raise RuntimeError(f"Tensor-parallel mp_size={mp_size} should divide hidden_size={hidden}.")
    except Exception as e:
        raise


class DPGenerator:
    @staticmethod
    def _build_prompts(conf_keys, prompt_dict, test_nn) -> List[Tuple[str, Dict[str, Any]]]:
        prompts: List[Tuple[str, Dict[str, Any]]] = []
        for key in conf_keys:
            prompt_tmpl = ''
            for pr in prompt_dict[key]['prompt']:
                prompt_tmpl += pr + '\n'
            data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by='nn').sample(n=1)[:test_nn]
            addon_data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]
                addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
                for it in prompt_dict[key]['addon_list']:
                    para_dict[it['para']] = addon_row[it['value']]
                prompts.append((prompt_tmpl.format(**para_dict), row.to_dict()))
        return prompts

    @staticmethod
    def _worker(local_rank: int,
                base_model_name: str,
                access_token: str | None,
                conf_keys,
                prompt_dict: dict,
                test_nn: int,
                out_path: str | Path,
                max_new_tokens: int,
                save_llm_output: bool,
                nn_name_prefix: str | None,
                seed: int | None = None):

        DistEnv.init(seed=seed)
        world_size = DistEnv.world_size()
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # --- HF model on CPU first ---
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            token=access_token
        )
        model.eval()

        # Basic TP-compat check (avoids weird runtime errors)
        _tp_compat_or_raise(model.config, world_size)

        # --- DeepSpeed TP (public API; no private module_inject/policies) ---
        engine = deepspeed.init_inference(
            model=model,
            mp_size=world_size,
            dtype=torch.float16,
            replace_method="auto",
            replace_with_kernel_inject=False  # avoid Diffusers/policy paths
        )
        model = engine.module  # HF-compatible module

        # --- Build prompts on rank-0; broadcast to all ---
        if DistEnv.is_main():
            prompts = DPGenerator._build_prompts(conf_keys, prompt_dict, test_nn)
        else:
            prompts = None
        if world_size > 1:
            obj = [prompts]
            dist.broadcast_object_list(obj, src=0)
            prompts = obj[0]
        else:
            assert prompts is not None

        models_dir = synth_dir(Path(out_path))
        if DistEnv.is_main():
            models_dir.mkdir(parents=True, exist_ok=True)

        # --- TP rule: ALL ranks call generate() with IDENTICAL inputs ---
        for idx, item in tqdm(enumerate(prompts), total=len(prompts),
                              disable=not DistEnv.is_main(), desc="DeepSpeed TP Inference"):
            prompt_text, origdf_dict = item
            engineered_prompt = prompt_text + extra_instructions

            enc = tokenizer(engineered_prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True
                )

            # Only rank-0 decodes/saves to disk
            if DistEnv.is_main():
                full_out = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)

                # --- print generated text for inspection ---
                print(f"[GEN][TP][idx={idx}]\n{full_out}\n")

                code = extract_code(full_out)
                hp = extract_hyperparam(full_out)

                if not code or not hp:
                    print(f"[WARN] Generation failed for prompt {idx}. Skipping.")
                    continue

                model_dir = models_dir / f'B{idx}'
                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)

                try:
                    hp_json = json.loads(hp.replace("'", '"'))
                    model_dir.mkdir(parents=True, exist_ok=True)
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp_json, f)
                    create_file(model_dir, new_nn_file, code)

                    origdf = pd.Series(origdf_dict)
                    df_file = model_dir / 'dataframe.df'
                    if 'nn' in origdf and 'nn_code' in origdf:
                        create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                        origdf.to_pickle(df_file)
                except Exception as e:
                    print(f"[WARN] Failed to parse/save results for prompt {idx}: {e}")
                    continue
        DistEnv.barrier()
        # Removed DistEnv.barrier() here — it’s broken in your runtime
        #DistEnv.destroy()  # does its own plain dist.barrier() + destroy

    @staticmethod
    def run(base_model_name: str,
            access_token: str | None,
            use_deepspeed: bool,
            context_length: int | None,
            conf_keys,
            prompt_dict: dict,
            test_nn: int,
            out_path: Path,
            max_new_tokens: int,
            save_llm_output: bool,
            nn_name_prefix: str | None,
            seed: int | None = None) -> None:

        # Use your existing spawner (autospawn_if_needed -> spawn_if_needed)
        DistEnv.spawn_if_needed(
            DPGenerator._worker,
            base_model_name,
            access_token,
            conf_keys,
            prompt_dict,
            test_nn,
            str(out_path),
            max_new_tokens,
            save_llm_output,
            nn_name_prefix,
            seed
        )
