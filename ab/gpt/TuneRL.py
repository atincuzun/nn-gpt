#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import PPOTrainer, PPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
from ab.gpt.util.Util import extract_code
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.Const import conf_train_dir, conf_test_dir
from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
from ab.gpt.format_reward import evaluate_code_and_reward
import ab.nn.api as api

import json
import os

from ab.gpt.util.simple_logger import SimpleCodeLogger

from typing import Tuple, Any

# ===== Configuration Options =====
LOAD_EXISTING_MODEL = False  # Whether to load saved model
SAVED_MODEL_PATH = "final_model_masked"  # Saved model path
GENERATED_CODE_DIR = "rl_output/generated_models"  # Directory for storing generated models
B_index = 0
# ==================================
def reward_fn(code: str) -> Tuple[float, Any]:
    """Calculate reward and return API result"""
    base_reward = 0.0
    try:
        try:
            # required_snippets = ["class Net", "def forward", "def train_setup", "def learn"]
            # if all(snippet in code for snippet in required_snippets):
            #     base_reward = max(base_reward, 0.2)
            res = api.check_nn(code, task='img-classification', 
                                dataset='cifar-10', metric='acc', 
                                prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
                                    'transform': 'norm_256_flip', 'epoch': 1}, 
                                save_to_db=False)
            code_logger.log_to_file(f"API evaluation result: {res}")
            # res = (model_name, accuracy, accuracy_to_time, res['score'])
            # acc = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else 0.0
            # final_reward = base_reward*0.5 + acc*0.5
            final_reward = res[3]
            return final_reward, res
        except Exception as e:
            code_logger.log_to_file(f"API evaluation failed: {e}")
            return base_reward, None
    except Exception as e:
        code_logger.log_to_file(f"Reward function crashed: {e}")
        return base_reward, None

def compute_reward(prompts, completions, **kwargs):
    global B_index
    rewards = []
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        from pprint import pprint
        code_logger.log_to_file("="*50)
        # print(f"Processing sample {i+1}/{len(prompts)}")
        # print(f"Prompt:\n{prompt}\n")
        
        final_code = extract_code(completion)
        if not final_code or final_code.strip() == "":
            final_code = completion
        print(final_code)
        api_result = None
        try:
            if final_code.strip():
                # score, api_result = reward_fn(final_code)
                # score = stage1_format_reward(final_code)[0]
                res = evaluate_code_and_reward(
                    final_code,
                    in_shape=(1, 3, 224, 224),  # Assuming input is an RGB image of size 224x224
                    out_shape=(10,),
                    prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
                                    'transform': 'norm_256_flip', 'epoch': 1},
                    device="cpu",
                    val_metric_baseline=0.10,
                )
                code_logger.log_to_file(f"Format reward evaluation result: {res}")
                score = res['reward']
            else:
                score = 0.0
            
            # code_logger.log_generation(prompt, completion, score)
            # code_logger.log_generation(prompt, completion, score, api_result)

            if score > 0:
                score1, api_result = reward_fn(final_code)
                code_logger.log_to_file(f"[INFO]API evaluation result: {api_result}")
                if api_result is not None:
                    out_path = epoch_dir(0)
                    model_dir = synth_dir(out_path) / f"B{B_index}"
                    code_file = model_dir / new_nn_file
                    code_logger.log_to_file(f"[INFO]Saving code to: {code_file}")
                    code_file.parent.mkdir(exist_ok=True, parents=True)  # Move here to avoid empty folder
                    with open(code_file, 'w') as file:
                        file.write(final_code)
                    create_file(model_dir, new_out_file, completion)
                    code_logger.log_to_file("Generated successful code:")
                    code_logger.log_to_file(final_code)
                    B_index += 1
            code_logger.log_generation(prompt, completion, score, api_result)
        except Exception as e:
            code_logger.log_to_file(f"Reward calculation failed: {e}")
            score = 0.0
            code_logger.log_generation(prompt, completion, score, None)

        rewards.append(score)
    return rewards
# """Below is a partial PyTorch neural network with masked layers:

# {masked_code}

# Please generate the complete PyTorch neural network code by replacing the [MASKED LAYER] placeholders.
# Requirements:
# 1. Output the complete Python/PyTorch code wrapped in ```python``` code blocks
# 2. Keep the same class structure (Net class) and method signatures exactly
# 3. Replace each '# [MASKED LAYER]' with appropriate PyTorch layer definitions
# 4. Never change train_setup, learn, forward methods
# 5. supported_hyperparameters function must remain exactly the same outside the Net class, only have ['lr', 'momentum', 'dropout']
# 6. Make sure the Net.__init__ method signature matches: __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device)
# 7. prm only contains 'lr', 'momentum', 'batch', 'transform', 'epoch'
# Now generate the complete working code wrapped in ```python``` blocks:"""

def mask_layer_definitions(code: str) -> tuple[str, str]:
    try:
        lines = code.splitlines()
        masked_lines = list(lines)
        extracted_layers = []

        import re
        for i, line in enumerate(lines):
            if re.search(r'self\.\w+\s*=\s*nn\.', line):
                extracted_layers.append(line.strip())
                masked_lines[i] = "        # [MASKED LAYER]"

        return "\n".join(masked_lines), "\n".join(extracted_layers)
    except Exception:
        return code, ""
                
PROMPT_TEMPLATE = """
Below is a partial PyTorch neural network working on cifar-10 with masked layers:

{masked_code}

Please generate the complete PyTorch neural network code by replacing the [MASKED LAYER] placeholders.
Requirements:
1. Output the complete Python/PyTorch code wrapped in ```python``` code blocks
2. Keep the same class structure (Net class) and method signatures exactly
3. Replace each '# [MASKED LAYER]' with appropriate PyTorch layer definitions
4. Keep train_setup, learn, forward methods exactly the same
5. supported_hyperparameters function must remain exactly the same outside the Net class, only have ['lr', 'momentum', 'dropout']
6. Make sure the Net.__init__ method signature matches: __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device)
7. prm only have ['lr', 'momentum', 'batch', 'transform', 'epoch'], use other prm key will be strictly penalized
8. Try to generate more different network structures, try to use different layers provided by torch.
Now generate the complete working code wrapped in ```python``` blocks:
"""

def load_rl_dataset(tokenizer):
    """
    Load and prepare dataset for RL training
    Returns: Dataset ready for RL training with proper prompts
    """
    data = api.data(only_best_accuracy=True, task='img-classification', dataset='cifar-10')
    print(f"Loaded {len(data)} high-quality NN examples from API")

    prompts = []

    for idx, row in data.iterrows():
        nn_code = row.get('nn_code', '')
        if not nn_code:
            continue

        masked_code, extracted_layers = mask_layer_definitions(nn_code)

        user_prompt = PROMPT_TEMPLATE.format(masked_code=masked_code)

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        prompts.append({
            "prompt": prompt_str,
            "original_code": nn_code,
            "masked_code": masked_code,
            "task": row.get('task', ''),
            "dataset": row.get('dataset', ''),
            "metric": row.get('metric', {}),
        })

    from datasets import Dataset
    rl_dataset = Dataset.from_list(prompts)
    rl_dataset = rl_dataset.shuffle(seed=42)

    print(f"Created RL dataset with {len(rl_dataset)} examples")
    return rl_dataset

def main():
    torch.cuda.empty_cache()  
    
    # base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"  
    base_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    # Load tokenizer first for dataset preparation
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load RL dataset
    rl_dataset = load_rl_dataset(tokenizer).select(range(500))

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Check whether to load saved model
    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading existing model from {SAVED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, SAVED_MODEL_PATH)
        print("Existing model loaded successfully!")
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, 
        # layers_to_transform=list(range(18, 24)),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # === Setup RL Training ===
    print(f"RL dataset size: {len(rl_dataset)}")
    
    grpo_config = GRPOConfig(
        # learning_rate=5e-5,
        learning_rate=1e-4,
        max_completion_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        remove_unused_columns=False,
        logging_steps=10,
        output_dir="./grpo_outputs",
        eval_strategy="no",
        bf16=True,
        # gradient_checkpointing=True,
        num_generations=4,
   
        generation_kwargs={
            "max_new_tokens": 2048,      
            "do_sample": True,
            "top_p": 0.95,              
            "top_k": 50,               
            "temperature": 0.7,
            "repetition_penalty": 1.1,  
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            # "renormalize_logits": True,
        },
        # use_peft=True
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward, 
        args=grpo_config,
        processing_class=None
    )
    trainer.train()
    
    # print("RL training completed!")
    # # Save logs and successful codes to rl_output directory
    # code_logger.save_log()
    # # code_logger.export_success_codes()
    
    # print(f"Saving model to {SAVED_MODEL_PATH}...")
    # model.save_pretrained(SAVED_MODEL_PATH)
    # print("Model saved successfully!")
    
    return model

if __name__ == "__main__":
    code_logger = SimpleCodeLogger("rl_output")
    main()
