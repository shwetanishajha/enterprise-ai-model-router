
import pytest

from src.catalog import get_model_catalog
from src.policy_engine import TaskRequirements
from src.scoring import ScoringWeights, estimate_request_cost, score_model


def test_scoring_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ScoringWeights(
            quality=0.5,
            cost=0.2,
            latency=0.2,
            reliability=0.2,
        )


def test_quality_score_must_be_between_zero_and_one():
    model = get_model_catalog()[0]
    task = TaskRequirements(task_type="classification")

    with pytest.raises(ValueError):
        score_model(model, task, quality_score=1.5)


def test_request_cost_is_calculated_correctly():
    model = get_model_catalog()[0]
    task = TaskRequirements(
        task_type="classification",
        estimated_input_tokens=1_000_000,
        estimated_output_tokens=0,
    )

    assert estimate_request_cost(model, task) == pytest.approx(0.20)


def test_score_is_between_zero_and_one():
    model = get_model_catalog()[0]
    task = TaskRequirements(task_type="classification")

    result = score_model(model, task, quality_score=0.8)

    assert 0 <= result.total_score <= 1
    assert result.model_name == "economy-model"


def test_higher_quality_increases_total_score():
    model = get_model_catalog()[0]
    task = TaskRequirements(task_type="classification")

    low = score_model(model, task, quality_score=0.5)
    high = score_model(model, task, quality_score=0.9)

    assert high.total_score > low.total_score
