
from src.catalog import get_model_catalog
from src.policy_engine import TaskRequirements, evaluate_model_policy


def test_confidential_data_excludes_economy_model():
    models = get_model_catalog()
    economy = next(m for m in models if m.name == "economy-model")

    task = TaskRequirements(
        task_type="extraction",
        data_classification="confidential",
    )

    result = evaluate_model_policy(economy, task)

    assert result.eligible is False
    assert "Data classification not permitted" in result.reasons


def test_supported_internal_task_is_eligible():
    models = get_model_catalog()
    economy = next(m for m in models if m.name == "economy-model")

    task = TaskRequirements(
        task_type="classification",
        data_classification="internal",
    )

    result = evaluate_model_policy(economy, task)

    assert result.eligible is True
    assert result.reasons == ()


def test_unsupported_task_is_rejected():
    models = get_model_catalog()
    economy = next(m for m in models if m.name == "economy-model")

    task = TaskRequirements(task_type="reasoning")

    result = evaluate_model_policy(economy, task)

    assert result.eligible is False
    assert "Unsupported task type" in result.reasons


def test_latency_constraint_is_enforced():
    models = get_model_catalog()
    reasoning = next(m for m in models if m.name == "reasoning-model")

    task = TaskRequirements(
        task_type="reasoning",
        max_latency_ms=500,
    )

    result = evaluate_model_policy(reasoning, task)

    assert result.eligible is False
    assert "Latency target exceeded" in result.reasons
