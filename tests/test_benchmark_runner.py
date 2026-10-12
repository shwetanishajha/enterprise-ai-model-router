
from src.benchmark_runner import run_demo_benchmark


def test_benchmark_runner_evaluates_all_tasks():
    report = run_demo_benchmark()

    assert report["task_count"] == 5
    assert len(report["results"]) == 5


def test_demo_fixtures_match_expected_answers():
    report = run_demo_benchmark()

    assert report["aggregate_quality_score"] == 1.0


def test_benchmark_results_include_task_identifiers():
    report = run_demo_benchmark()

    task_ids = {result["task_id"] for result in report["results"]}

    assert task_ids == {
        "FDD-001",
        "FDD-002",
        "FDD-003",
        "FDD-004",
        "FDD-005",
    }


def test_demo_results_are_explicitly_labelled():
    report = run_demo_benchmark()

    assert report["mode"] == "deterministic_demo"
