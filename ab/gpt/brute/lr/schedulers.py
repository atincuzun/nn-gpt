import os
import shutil
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Target the directory NNEval is scanning
OUTPUT_DIR = "out/nngpt/llm/epoch/A0/synth_nn"

# Use "A0_" prefix to match the parent folder "epoch/A0".
BASE_MODEL_NAME = "A0_"

# -----------------------------------------------------------------------------
# 0. Setup & Cleanup
# -----------------------------------------------------------------------------
print(f"--- Setup Phase ---")

# 1. Ensure the full path exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Created/Verified model directory: {OUTPUT_DIR}")

# 2. Recursive __init__.py creation
# NNEval often needs to import these files as a module (e.g. out.nngpt...).
# We must ensure every folder in the path has an __init__.py.
path_parts = pathlib.Path(OUTPUT_DIR).parts
current_path = ""
for part in path_parts:
    current_path = os.path.join(current_path, part) if current_path else part
    init_file = os.path.join(current_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Package marker for NNEval\n")
        print(f"    + Created missing package marker: {init_file}")

# 3. Clean up OLD generated files (B200*, A0_*) and artifacts like __pycache__
print("--- Cleaning up previous files ---")
pycache_path = os.path.join(OUTPUT_DIR, "__pycache__")
if os.path.exists(pycache_path):
    shutil.rmtree(pycache_path)
    print(f"✅ Removed directory confusing NNEval: {pycache_path}")

# We search for any .py file starting with our prefixes to ensure a clean slate.
# Note: The original code snippet had an issue here as it tried to list files in a new directory,
# but we are only restoring the functional script now.
# This loop assumes OUTPUT_DIR exists and may be temporarily empty after the deletion and clone.

# 4. Auto-delete the database to ensure a fresh scan
db_path = "db/ab.nn.db"
if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"✅ Deleted old database ({db_path}) to force a fresh scan.")
    except Exception as e:
        print(f"⚠️ Could not delete database: {e}")

# -----------------------------------------------------------------------------
# 1. The Model Template (Test block removed)
# -----------------------------------------------------------------------------
model_template = """
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

# Supported hyperparameters for this model
def supported_hyperparameters():
    return {{'lr', 'momentum', 'dropout'}}

# -------------------------------------------------------------------------
# Model Definition: Net (ResNet18 adapted for CIFAR-10)
# -------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 32, 32), out_shape=(10,), prm=None, device='cpu'):
        '''
        Initialize ResNet18 model.
        Parameters match NNEval's expected signature.
        '''
        super(Net, self).__init__()
        
        if prm is None:
            prm = {{'lr': 0.1, 'momentum': 0.9, 'dropout': 0.2}}
        
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # Initialize standard ResNet18
        self.model = models.resnet18(weights=None)
        
        # Modify the final layer to include Dropout and set correct output classes
        num_classes = out_shape[0] if isinstance(out_shape, tuple) else out_shape
        self.model.fc = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        # Training components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)
    
    def train_setup(self, prm):
        '''
        Initialize training components (criterion, optimizer, scheduler).
        Required by NNEval for training evaluation.
        '''
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Initialize Optimizer
        self.optimizer = optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.1), 
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )
        
        # Create local variable reference for scheduler creation
        optimizer = self.optimizer
        
        # --- SCHEDULER LOGIC START ---
        {scheduler_logic}
        # --- SCHEDULER LOGIC END ---
        
        self.scheduler = scheduler
    
    def learn(self, train_data):
        '''
        Training loop. Process one batch or epoch of training data.
        '''
        self.train()
        for (inputs, labels) in train_data:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                print(f"[WARN] Skipping batch due to non-finite loss: {{loss.item()}}")
                continue
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
        
        # Step scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()

# -------------------------------------------------------------------------
# Hyperparameters & Scheduler Configuration
# Strategy: {strategy_name}
# -------------------------------------------------------------------------
def get_model_and_optimizer(in_shape=(1, 3, 32, 32), out_shape=(10,), max_epoch=90, learning_rate=0.1, dropout=0.2, momentum=0.9, device='cpu'):
    '''
    Factory function required by NNEval.
    '''
    # Prepare hyperparameters
    prm = {{
        'lr': learning_rate,
        'momentum': momentum,
        'dropout': dropout,
        'max_epoch': max_epoch
    }}
    
    # Initialize model with the parameters
    model = Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
    
    # Setup training (this will also initialize optimizer and scheduler)
    model.train_setup(prm)
    
    return model, model.optimizer, model.scheduler

# -------------------------------------------------------------------------
# Metadata for LLM Analysis
# -------------------------------------------------------------------------
# description: ResNet18 model using {strategy_name}
# category: Computer Vision
# scheduler_type: {scheduler_type}
"""

# -----------------------------------------------------------------------------
# 2. Scheduler Strategies
# -----------------------------------------------------------------------------
strategies = [
    { "name": "StepLR", "type": "Step-based", "code": "scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)" },
    { "name": "MultiStepLR", "type": "Step-based", "code": "scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)" },
    { "name": "ExponentialLR", "type": "Step-based", "code": "scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)" },
    { "name": "CosineAnnealingLR", "type": "Cosine", "code": "scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm.get('max_epoch', 90))" },
    { "name": "CyclicLR", "type": "Cyclic", "code": "scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1)/10, max_lr=prm.get('lr', 0.1), step_size_up=5, mode='triangular')" },
    { "name": "OneCycleLR", "type": "Cyclic", "code": "scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=prm.get('lr', 0.1), steps_per_epoch=100, epochs=prm.get('max_epoch', 90))" },
    { "name": "CosineAnnealingWarmRestarts", "type": "Cosine", "code": "scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)" }
]

# -----------------------------------------------------------------------------
# 3. Execution: Generate the Files
# -----------------------------------------------------------------------------
print(f"\n--- Generation Phase ---")
generated_files = []
module_names = []

for i, strategy in enumerate(strategies):
    # Naming: A0_001, A0_002, etc.
    module_name = f"{BASE_MODEL_NAME}{i+1:03d}"
    
    # Create a subdirectory for each model (NNEval expects this structure)
    model_dir = os.path.join(OUTPUT_DIR, module_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # File goes inside the model directory as 'new_nn.py'
    filepath = os.path.join(model_dir, "new_nn.py")
    
    try:
        # Fill the template
        file_content = model_template.format(
            strategy_name=strategy['name'],
            scheduler_type=strategy['type'],
            scheduler_logic=strategy['code'].strip()
        )
        
        with open(filepath, "w") as f:
            f.write(file_content)
            
        print(f"Generated: {filepath}")
        generated_files.append(filepath)
        module_names.append(module_name)
    except Exception as e:
        print(f"❌ Error generating {filepath}: {e}")

# -----------------------------------------------------------------------------
# 4. Finalize __init__.py in synth_nn (Optional for module imports)
# -----------------------------------------------------------------------------
# Since models are now in subdirectories, the __init__.py is optional
# NNEval will scan each subdirectory directly
init_path = os.path.join(OUTPUT_DIR, "__init__.py")

with open(init_path, "w") as f:
    f.write("# Package marker for NNEval\n")
    f.write("# Models are organized in subdirectories, each with new_nn.py\n")

print(f"✅ Created {init_path} as package marker.")

# -----------------------------------------------------------------------------
# 5. Final Cleanup and Verification
# -----------------------------------------------------------------------------
# Rerunning cleanup immediately before verification to eliminate the __pycache__ issue.
print("\n--- Final Cleanup and Verification ---")

pycache_path = os.path.join(OUTPUT_DIR, "__pycache__")
if os.path.exists(pycache_path):
    shutil.rmtree(pycache_path)
    print(f"✅ Rerunning cleanup: Removed directory {pycache_path}")

if generated_files and os.path.exists(generated_files[0]):
    print(f"✅ File 1 exists: {generated_files[0]}")
elif generated_files:
    print(f"❌ Error: File 1 missing.")
else:
    print(f"❌ Error: No files were generated.")

print("\n=======================================================")
print("✅ PREPARATION COMPLETE")
print("1. Run this script: python generate_lr_schedulers.py")
print("2. THEN run NNEval: python -m ab.gpt.NNEval")
print("=======================================================")
