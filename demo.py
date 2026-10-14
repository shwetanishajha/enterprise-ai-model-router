
from src.policy_engine import TaskRequirements
from src.router import select_model


scenarios = [
    (
        "Routine financial data extraction",
        TaskRequirements(
            task_type="extraction",
            data_classification="internal",
            max_cost_per_request=0.05,
            max_latency_ms=2000,
        ),
    ),
    (
        "Confidential deal analysis",
        TaskRequirements(
            task_type="reasoning",
            data_classification="confidential",
            max_cost_per_request=0.05,
            max_latency_ms=3000,
        ),
    ),
    (
        "Low-latency classification",
        TaskRequirements(
            task_type="classification",
            data_classification="internal",
            max_cost_per_request=0.01,
            max_latency_ms=500,
        ),
    ),
]


for title, task in scenarios:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    try:
        decision = select_model(task)

        print(f"Selected model: {decision.selected_model.name}")
        print(f"Total score: {decision.selected_score.total_score:.4f}")
        print(f"Explanation: {decision.explanation}")

        print("\nEligible model ranking:")
        for rank, result in enumerate(decision.ranked_models, start=1):
            print(f"{rank}. {result.model_name}: {result.total_score:.4f}")

        print("\nRejected models:")
        if decision.rejected_models:
            for model_name, reasons in decision.rejected_models.items():
                print(f"- {model_name}: {', '.join(reasons)}")
        else:
            print("- None")

    except ValueError as exc:
        print(f"No model selected: {exc}")
