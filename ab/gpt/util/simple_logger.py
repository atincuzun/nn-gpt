#!/usr/bin/env python3
import json
import time
import os
from datetime import datetime
from typing import Any

class SimpleCodeLogger:
    
    def __init__(self, output_dir: str = "rl_output"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "generation_log.json")
        self.training_log_file = os.path.join(output_dir, "training_progress.log")
        self.success_codes_file = os.path.join(output_dir, "success_codes.py")
        
        self.total_count = 0
        self.success_count = 0
        self.success_cases = []
        self.start_time = time.time()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.training_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== RL Training start {datetime.now().isoformat()} ===\n")
            f.write(f"Output directory: {os.path.abspath(self.output_dir)}\n")

    def log_to_file(self, message: str):
        with open(self.training_log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    
    def log_generation(self, prompt: str, completion: str, reward: float, api_result: Any = None):
        self.total_count += 1
        
        self.log_to_file(f"Generation {self.total_count}: Reward={reward:.4f}")
        
        if (reward > 0) and (api_result is not None):
            self.success_count += 1
            
            success_case = {
                'id': self.success_count,
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'completion': completion,
                'reward': reward,
                'api_result': api_result,
                'code_length': len(completion)
            }
            
            self.success_cases.append(success_case)
            
            self.log_to_file(f"Success case #{self.success_count}: Reward={reward:.4f}, code length={len(completion)}")
            self.log_to_file(f"   API result: {api_result}")
        else:
            self.log_to_file(f"Fail case: Reward={reward:.4f}")
        
        if self.total_count % 10 == 0:
            current_success_rate = (self.success_count / self.total_count * 100)
            self.log_to_file(f"progress: {self.total_count} generation，{self.success_count} success，success rate {current_success_rate:.1f}%")
     
    def save_log(self):
        log_data = {
            'summary': {
                'total_count': self.total_count,
                'success_count': self.success_count,
                'success_rate': (self.success_count / self.total_count) if self.total_count > 0 else 0,
                'start_time': self.start_time,
                'duration_minutes': (time.time() - self.start_time) / 60
            },
            'success_cases': self.success_cases
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"log save to {self.log_file}")



