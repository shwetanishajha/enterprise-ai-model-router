
from src.catalog import get_model_catalog


def test_catalog_contains_three_models():
    models = get_model_catalog()

    assert len(models) == 3


def test_model_profiles_have_valid_metrics():
    for model in get_model_catalog():
        assert model.input_cost_per_million_tokens >= 0
        assert model.output_cost_per_million_tokens >= 0
        assert model.expected_latency_ms > 0
        assert 0 <= model.reliability_score <= 1
        assert model.max_context_tokens > 0


def test_catalog_includes_required_tasks():
    models = get_model_catalog()
    supported_tasks = {
        task
        for model in models
        for task in model.supported_tasks
    }

    assert "classification" in supported_tasks
    assert "extraction" in supported_tasks
    assert "reasoning" in supported_tasks


def test_confidential_data_has_eligible_models():
    models = get_model_catalog()

    eligible = [
        model
        for model in models
        if "confidential" in model.allowed_data_classifications
    ]

    assert len(eligible) >= 1


def test_demo_profiles_are_not_misrepresented_as_real():
    for model in get_model_catalog():
        assert model.simulated is True
        assert model.provider == "demo"
