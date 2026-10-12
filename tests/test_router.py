
import pytest

from src.policy_engine import TaskRequirements
from src.router import select_model


def test_router_selects_model_for_classification():
    task = TaskRequirements(task_type="classification")

    decision = select_model(task)

    assert decision.selected_model.name in {
        "economy-model",
        "balanced-model",
        "reasoning-model",
    }


def test_router_respects_confidential_data_policy():
    task = TaskRequirements(
        task_type="extraction",
        data_classification="confidential",
    )

    decision = select_model(task)

    assert decision.selected_model.name != "economy-model"
    assert "economy-model" in decision.rejected_models


def test_router_rejects_models_that_miss_latency_target():
    task = TaskRequirements(
        task_type="reasoning",
        max_latency_ms=500,
    )

    with pytest.raises(ValueError, match="No eligible model"):
        select_model(task)


def test_router_returns_ranked_candidates():
    task = TaskRequirements(task_type="classification")

    decision = select_model(task)

    scores = [item.total_score for item in decision.ranked_models]

    assert scores == sorted(scores, reverse=True)


def test_router_explains_its_decision():
    task = TaskRequirements(task_type="classification")

    decision = select_model(task)

    assert decision.selected_model.name in decision.explanation
    assert "illustrative" in decision.explanation
