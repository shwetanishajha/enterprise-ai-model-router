
from dataclasses import dataclass

from src.catalog import ModelProfile


@dataclass(frozen=True)
class TaskRequirements:
    """Constraints that a model must satisfy for a task."""

    task_type: str
    data_classification: str = "internal"
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 500
    max_cost_per_request: float = 1.0
    max_latency_ms: int = 5000
    minimum_reliability: float = 0.95


@dataclass(frozen=True)
class PolicyResult:
    eligible: bool
    reasons: tuple[str, ...]


def evaluate_model_policy(
    model: ModelProfile,
    task: TaskRequirements,
) -> PolicyResult:
    """Reject models that violate mandatory task requirements."""

    reasons: list[str] = []

    if task.task_type not in model.supported_tasks:
        reasons.append("Unsupported task type")

    if task.data_classification not in model.allowed_data_classifications:
        reasons.append("Data classification not permitted")

    if (
        task.estimated_input_tokens + task.estimated_output_tokens
        > model.max_context_tokens
    ):
        reasons.append("Context window exceeded")

    estimated_cost = (
        task.estimated_input_tokens
        * model.input_cost_per_million_tokens
        + task.estimated_output_tokens
        * model.output_cost_per_million_tokens
    ) / 1_000_000

    if estimated_cost > task.max_cost_per_request:
        reasons.append("Request exceeds cost budget")

    if model.expected_latency_ms > task.max_latency_ms:
        reasons.append("Latency target exceeded")

    if model.reliability_score < task.minimum_reliability:
        reasons.append("Reliability below minimum")

    return PolicyResult(
        eligible=not reasons,
        reasons=tuple(reasons),
    )
