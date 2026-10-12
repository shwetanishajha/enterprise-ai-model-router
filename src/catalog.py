
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Configuration and capabilities of a candidate AI model."""

    name: str
    provider: str
    supported_tasks: tuple[str, ...]
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    expected_latency_ms: int
    reliability_score: float
    max_context_tokens: int
    allowed_data_classifications: tuple[str, ...]
    simulated: bool = True


MODEL_CATALOG: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="economy-model",
        provider="demo",
        supported_tasks=("classification", "extraction", "summarisation"),
        input_cost_per_million_tokens=0.20,
        output_cost_per_million_tokens=0.80,
        expected_latency_ms=400,
        reliability_score=0.97,
        max_context_tokens=32000,
        allowed_data_classifications=("public", "internal"),
    ),
    ModelProfile(
        name="balanced-model",
        provider="demo",
        supported_tasks=("classification", "extraction", "summarisation", "reasoning"),
        input_cost_per_million_tokens=1.50,
        output_cost_per_million_tokens=6.00,
        expected_latency_ms=900,
        reliability_score=0.99,
        max_context_tokens=128000,
        allowed_data_classifications=("public", "internal", "confidential"),
    ),
    ModelProfile(
        name="reasoning-model",
        provider="demo",
        supported_tasks=("extraction", "summarisation", "reasoning"),
        input_cost_per_million_tokens=3.00,
        output_cost_per_million_tokens=12.00,
        expected_latency_ms=1800,
        reliability_score=0.985,
        max_context_tokens=128000,
        allowed_data_classifications=("public", "internal", "confidential"),
    ),
)


def get_model_catalog() -> tuple[ModelProfile, ...]:
    """Return the configured model profiles."""
    return MODEL_CATALOG
