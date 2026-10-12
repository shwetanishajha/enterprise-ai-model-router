
import json
from pathlib import Path
from typing import Any


def load_benchmark(path: str = "benchmarks/tasks.json") -> list[dict]:
    """Load benchmark tasks from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    if not isinstance(tasks, list):
        raise ValueError("Benchmark file must contain a JSON list")

    for task in tasks:
        for field in ("id", "task_type", "prompt", "expected_answer"):
            if field not in task:
                raise ValueError(f"Benchmark task is missing field: {field}")

    return tasks


def compare_values(expected: Any, actual: Any) -> bool:
    """Compare expected and actual values, allowing small numeric tolerance."""

    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= 0.01

    if isinstance(expected, dict) and isinstance(actual, dict):
        return (
            expected.keys() == actual.keys()
            and all(compare_values(expected[key], actual[key]) for key in expected)
        )

    if isinstance(expected, list) and isinstance(actual, list):
        return (
            len(expected) == len(actual)
            and all(compare_values(a, b) for a, b in zip(expected, actual))
        )

    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().casefold() == actual.strip().casefold()

    return expected == actual


def evaluate_answer(expected: dict, actual: dict) -> dict:
    """Evaluate a structured answer using exact field and value matching."""

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError("Expected and actual answers must be dictionaries")

    correct_fields = sum(
        1
        for key, expected_value in expected.items()
        if key in actual and compare_values(expected_value, actual[key])
    )

    total_fields = len(expected)

    return {
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "quality_score": (
            correct_fields / total_fields if total_fields else 1.0
        ),
    }
