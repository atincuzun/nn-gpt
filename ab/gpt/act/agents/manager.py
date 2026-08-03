"""
Manager Agent - Controls the execution flow of the pipeline.

Responsible for routing:
finetune → generate → evaluate → (predict?) → finetune → ...
"""

from typing import Dict, Any
from ab.gpt.act.agents.state import AgentState


def manager_node(state: AgentState) -> Dict[str, Any]:
    """Decide which step should run next."""

    epoch = state.get("current_epoch", 0)
    total_epochs = state.get("llm_tune_epochs", 1)
    next_action = state.get("next_action", "generate")

    # Stop condition
    if epoch >= total_epochs:
        print(f"[MANAGER] All {total_epochs} epochs complete. Ending.")
        return {"next_action": "end"}

    # Generation
    if next_action == "generate":
        print(f"[MANAGER] Epoch {epoch}: starting generation")
        return {"next_action": "generate"}

    # Evaluation (always follows generation via direct edge — manager handles it only on skip)
    if next_action == "evaluate":
        print(f"[MANAGER] Epoch {epoch}: evaluating generated models")
        return {"next_action": "evaluate"}

    # Predictor
    if next_action == "predict":
        print(f"[MANAGER] Epoch {epoch}: evaluator done → predictor running (CPU)")
        return {"next_action": "predict"}

    # Finetuning
    if next_action == "finetune":
        print(f"[MANAGER] Epoch {epoch}: finetuner gets GPU")
        return {"next_action": "finetune"}

    # Fallback
    print(f"[MANAGER] Unknown next_action '{next_action}', defaulting to generate")
    return {"next_action": "generate"}