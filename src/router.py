
from dataclasses import dataclass

from src.catalog import get_model_catalog, ModelProfile
from src.policy_engine import TaskRequirements, evaluate_model_policy
from src.scoring import ScoringWeights, ModelScore, score_model


# Illustrative quality scores for demonstrating routing.
# These must be replaced with benchmark-derived results.
DEMO_QUALITY_SCORES = {
    "economy-model": 0.72,
    "balanced-model": 0.86,
    "reasoning-model": 0.93,
}


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: ModelProfile
    selected_score: ModelScore
    ranked_models: tuple[ModelScore, ...]
    rejected_models: dict[str, tuple[str, ...]]
    explanation: str


def select_model(
    task: TaskRequirements,
    weights: ScoringWeights = ScoringWeights(),
) -> RoutingDecision:
    """Select the highest-scoring eligible model."""

    eligible_models: list[tuple[ModelProfile, ModelScore]] = []
    rejected_models: dict[str, tuple[str, ...]] = {}

    for model in get_model_catalog():
        policy = evaluate_model_policy(model, task)

        if not policy.eligible:
            rejected_models[model.name] = policy.reasons
            continue

        quality = DEMO_QUALITY_SCORES.get(model.name)

        if quality is None:
            rejected_models[model.name] = (
                "No quality benchmark available",
            )
            continue

        result = score_model(
            model=model,
            task=task,
            quality_score=quality,
            weights=weights,
        )

        eligible_models.append((model, result))

    if not eligible_models:
        raise ValueError("No eligible model satisfies the task requirements")

    eligible_models.sort(
        key=lambda item: item[1].total_score,
        reverse=True,
    )

    selected_model, selected_score = eligible_models[0]

    explanation = (
        f"Selected {selected_model.name} with score "
        f"{selected_score.total_score:.4f}. "
        f"Evaluated {len(eligible_models)} eligible model(s); "
        f"rejected {len(rejected_models)} model(s). "
        "Quality inputs are illustrative demo values, not measured results."
    )

    return RoutingDecision(
        selected_model=selected_model,
        selected_score=selected_score,
        ranked_models=tuple(score for _, score in eligible_models),
        rejected_models=rejected_models,
        explanation=explanation,
    )
