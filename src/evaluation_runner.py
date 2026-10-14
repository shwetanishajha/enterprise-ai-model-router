
from collections.abc import Callable
from time import perf_counter

from src.evaluator import evaluate_answer, load_benchmark

ModelCallable = Callable[[str], dict]


def evaluate_model(
    model_name: str,
    generate: ModelCallable,
    benchmark_path: str = "benchmarks/tasks.json",
) -> dict:
    """Evaluate a model adapter against the fixed benchmark dataset."""
    tasks = load_benchmark(benchmark_path)
    results = []

    for task in tasks:
        start = perf_counter()

        try:
            response = generate(task["prompt"])
            elapsed_ms = (perf_counter() - start) * 1000
            actual = response.get("output")

            if not isinstance(actual, dict):
                raise ValueError("Model output must be a JSON object")

            evaluation = evaluate_answer(
                task["expected_answer"], actual
            )

            results.append({
                "task_id": task["id"],
                "task_type": task["task_type"],
                "status": "success",
                "quality_score": evaluation["quality_score"],
                "correct_fields": evaluation["correct_fields"],
                "total_fields": evaluation["total_fields"],
                "latency_ms": round(elapsed_ms, 2),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
                "error": None,
            })

        except Exception as exc:
            elapsed_ms = (perf_counter() - start) * 1000

            results.append({
                "task_id": task["id"],
                "task_type": task["task_type"],
                "status": "failed",
                "quality_score": 0.0,
                "correct_fields": 0,
                "total_fields": len(task["expected_answer"]),
                "latency_ms": round(elapsed_ms, 2),
                "input_tokens": None,
                "output_tokens": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    total_fields = sum(r["total_fields"] for r in results)
    correct_fields = sum(r["correct_fields"] for r in results)
    successful_tasks = sum(
        r["status"] == "success" for r in results
    )

    latencies = sorted(
        r["latency_ms"]
        for r in results
        if r["status"] == "success"
    )

    p95_latency_ms = (
        latencies[
            min(len(latencies) - 1, int(len(latencies) * 0.95))
        ]
        if latencies
        else None
    )

    return {
        "model_name": model_name,
        "mode": "adapter_evaluation",
        "task_count": len(tasks),
        "successful_tasks": successful_tasks,
        "failed_tasks": len(tasks) - successful_tasks,
        "aggregate_quality_score": (
            correct_fields / total_fields if total_fields else 1.0
        ),
        "mean_latency_ms": (
            round(sum(latencies) / len(latencies), 2)
            if latencies
            else None
        ),
        "p95_latency_ms": p95_latency_ms,
        "results": results,
    }
