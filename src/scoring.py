
from dataclasses import dataclass

from src.catalog import ModelProfile
from src.policy_engine import TaskRequirements


@dataclass(frozen=True)
class ScoringWeights:
    quality: float = 0.60
    cost: float = 0.15
    latency: float = 0.15
    reliability: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.quality,
            self.cost,
            self.latency,
            self.reliability,
        )

        if any(weight < 0 or weight > 1 for weight in weights):
            raise ValueError("Each weight must be between 0 and 1")

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Scoring weights must sum to 1")


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    quality_score: float
    cost_score: float
    latency_score: float
    reliability_score: float
    total_score: float


def estimate_request_cost(
    model: ModelProfile,
    task: TaskRequirements,
) -> float:
    """Estimate inference cost using configured per-million-token prices."""

    return (
        task.estimated_input_tokens * model.input_cost_per_million_tokens
        + task.estimated_output_tokens * model.output_cost_per_million_tokens
    ) / 1_000_000


def score_model(
    model: ModelProfile,
    task: TaskRequirements,
    quality_score: float,
    weights: ScoringWeights = ScoringWeights(),
) -> ModelScore:
    """Calculate a weighted score for an eligible model."""

    if not 0 <= quality_score <= 1:
        raise ValueError("Quality score must be between 0 and 1")

    cost = estimate_request_cost(model, task)

    # Lower cost is better. A zero-cost request gets the maximum score.
    cost_score = (
        1.0
        if cost == 0
        else max(0.0, 1.0 - cost / task.max_cost_per_request)
    )

    # Lower latency is better, relative to the task's latency target.
    latency_score = max(
        0.0,
        1.0 - model.expected_latency_ms / task.max_latency_ms,
    )

    reliability_score = model.reliability_score

    total = (
        weights.quality * quality_score
        + weights.cost * cost_score
        + weights.latency * latency_score
        + weights.reliability * reliability_score
    )

    return ModelScore(
        model_name=model.name,
        quality_score=quality_score,
        cost_score=cost_score,
        latency_score=latency_score,
        reliability_score=reliability_score,
        total_score=round(total, 6),
    )
