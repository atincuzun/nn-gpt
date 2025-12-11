# AI Agent System - Multi-Agent AutoML

Multi-agent system for automated neural network discovery and optimization using LLM-based architecture generation.

## Overview

This system implements a three-agent AutoML workflow:
1. **Generator Agent**: Generates neural architectures using fine-tuned LLM
2. **Predictor Agent**: Predicts final accuracy from early training metrics
3. **Manager Agent**: Coordinates GPU resources across agents

## Current Status

- ✅ **Generator Agent**: Complete and functional
- ⏳ **Predictor Agent**: In development (teammate)
- ⏳ **Manager Agent**: Planned
- ⏳ **LangGraph Integration**: Planned

## Generator Agent Features

### Core Functionality
- Neural architecture generation using nn-gpt (DeepSeek Coder 1.3B)
- 2-epoch quick evaluation for early performance signals
- Automatic hyperparameter extraction

### Intelligent Caching
- Database lookup for duplicate models
- Retrieves cached metrics instead of re-training
- Reduces compute time by ~90% for known architectures

### Robust Extraction
- Multiple parsing strategies with fallbacks
- Handles XML tags, markdown code blocks, and plain text
- Extracts from full LLM output when primary parsing fails

## Architecture
```
Generator Agent
    ↓
nn-gpt (LLM) → Generate Architecture
    ↓
Checksum Check
    ↓
  Exists in DB?
  ├─ Yes → Query cached metrics
  └─ No  → Train 2 epochs → Save metrics
    ↓
Return to Multi-Agent System
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU
- 32GB+ RAM recommended

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database (first run only)
# Database auto-generates on first use
```

### External Dependencies

This project integrates with:
- `nn-gpt`: LLM-based architecture generation framework
- `nn-dataset`: Training and evaluation pipeline

## Usage

### Run Generator
```bash
python -m src.agents.generator
```

### Expected Output
```python
{
  "status": "success",
  "metrics_source": "eval_info.json",  # or "database"
  "has_metrics": true,
  "accuracy": 0.85,
  "model_code": "...",
  "hyperparameters": {...}
}
```

## Project Structure
```
AI_AGENT_SYSTEM/
├── src/agents/           - Agent implementations
│   ├── generator.py      - Generator agent (complete)
│   ├── predictor.py      - Predictor agent (in development)
│   └── manager.py        - Manager agent (planned)
├── nn-gpt/              - External: LLM generation framework
├── nn-dataset/          - External: Training pipeline
├── db/                  - Local database (cached results)
├── out/                 - Generated outputs (gitignored)
└── venv/                - Virtual environment (gitignored)
```

## Configuration

Generator parameters (in `src/agents/generator.py`):
- `max_new_tokens`: 12,288 (12KB - optimal for 1.3B model)
- `temperature`: 0.8 (generation diversity)
- `top_k`: 70 (sampling parameter)
- `top_p`: 0.9 (nucleus sampling)
- `nn_train_epochs`: 2 (quick evaluation)

## Known Limitations

1. **Large datasets**: Places365 (26GB) may timeout during training
2. **LLM variability**: ~30% of generations may have parsing issues
3. **GPU memory**: Requires ~8GB VRAM for model + training

These are expected behaviors in research code and do not affect the pipeline architecture.

## Future Work

- [ ] Complete predictor agent integration
- [ ] Implement manager agent for GPU coordination
- [ ] Wire agents in LangGraph state machine
- [ ] Add retry logic for failed generations
- [ ] Support for multiple LLM backends

## Development

This project was developed as part of a 10-credit research project on multi-agent AutoML systems.

**Timeline**: 2 weeks for generator agent completion

**Acknowledgments**: Built on top of ABrain's nn-gpt and nn-dataset frameworks
