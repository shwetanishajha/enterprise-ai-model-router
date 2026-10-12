
from src.evaluator import evaluate_answer, load_benchmark


# Deterministic fixtures for testing the evaluation pipeline.
# These are NOT outputs from real AI models.
DEMO_ANSWERS = {
    "FDD-001": {
        "revenue_millions": 12.5,
        "ebitda_millions": 2.1,
    },
    "FDD-002": {
        "revenue_millions": 18.0,
        "ebitda_millions": 3.6,
    },
    "FDD-003": {
        "revenue_growth_percent": 20.0,
        "previous_ebitda_margin_percent": 20.0,
        "current_ebitda_margin_percent": 16.6667,
    },
    "FDD-004": {
        "category": "customer_concentration_risk",
    },
    "FDD-005": {
        "key_findings": [
            "revenue_growth",
            "flat_ebitda",
            "margin_decline",
        ],
    },
}


def run_demo_benchmark() -> dict:
    """Evaluate deterministic fixtures against the benchmark dataset."""

    tasks = load_benchmark()
    results = []

    for task in tasks:
        actual = DEMO_ANSWERS.get(task["id"])

        if actual is None:
            raise ValueError(f"No demo answer configured for {task['id']}")

        evaluation = evaluate_answer(task["expected_answer"], actual)

        results.append({
            "task_id": task["id"],
            "task_type": task["task_type"],
            **evaluation,
        })

    total_fields = sum(item["total_fields"] for item in results)
    correct_fields = sum(item["correct_fields"] for item in results)

    return {
        "mode": "deterministic_demo",
        "task_count": len(results),
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "aggregate_quality_score": (
            correct_fields / total_fields if total_fields else 1.0
        ),
        "results": results,
    }


if __name__ == "__main__":
    report = run_demo_benchmark()

    print(f"Mode: {report['mode']}")
    print(f"Tasks evaluated: {report['task_count']}")
    print(
        f"Aggregate fixture score: "
        f"{report['aggregate_quality_score']:.2%}"
    )

    for result in report["results"]:
        print(
            f"{result['task_id']} ({result['task_type']}): "
            f"{result['quality_score']:.0%}"
        )
