"""Task registry and factory."""

from .math_addition import MathAdditionTask

TASK_REGISTRY = {
    "math_addition": MathAdditionTask,
}


def create_task(name: str):
    """Create a task instance by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]()
