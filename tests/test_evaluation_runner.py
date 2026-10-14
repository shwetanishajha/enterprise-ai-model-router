
from src.benchmark_runner import DEMO_ANSWERS
from src.evaluation_runner import evaluate_model
from src.evaluator import load_benchmark


def demo_generate(prompt: str) -> dict:
    """Return a deterministic fixture for the matching benchmark task."""
    for task in load_benchmark():
        if task["prompt"] == prompt:
            return {"output": DEMO_ANSWERS[task["id"]]}
    raise ValueError("Unknown benchmark prompt")


def failing_generate(prompt: str) -> dict:
    raise RuntimeError("Simulated provider failure")


def test_offline_adapter_evaluates_all_tasks():
    report = evaluate_model("offline-demo", demo_generate)

    assert report["task_count"] == 5
    assert report["successful_tasks"] == 5
    assert report["failed_tasks"] == 0


def test_offline_adapter_scores_fixtures_correctly():
    report = evaluate_model("offline-demo", demo_generate)

    assert report["aggregate_quality_score"] == 1.0


def test_failed_requests_are_recorded():
    report = evaluate_model("failing-demo", failing_generate)

    assert report["failed_tasks"] == 5
    assert report["successful_tasks"] == 0
    assert report["aggregate_quality_score"] == 0.0


def test_latency_metrics_are_recorded():
    report = evaluate_model("offline-demo", demo_generate)

    assert report["mean_latency_ms"] is not None
    assert report["p95_latency_ms"] is not None
    assert report["mean_latency_ms"] >= 0
