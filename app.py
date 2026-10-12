
import streamlit as st

from src.policy_engine import TaskRequirements
from src.router import select_model
from src.scoring import ScoringWeights


st.set_page_config(
    page_title="Enterprise AI Model Router",
    page_icon="🧭",
    layout="wide",
)

st.title("Enterprise AI Model Router")
st.caption(
    "Evidence-driven model selection across quality, cost, latency "
    "and enterprise policy constraints."
)

st.warning(
    "DEMO MODE: Model profiles, quality scores and latency values are "
    "illustrative. No live model inference is performed."
)

st.sidebar.header("Task requirements")

task_type = st.sidebar.selectbox(
    "Task type",
    ["classification", "extraction", "reasoning", "summarisation"],
)

data_classification = st.sidebar.selectbox(
    "Data classification",
    ["public", "internal", "confidential"],
)

max_cost = st.sidebar.number_input(
    "Maximum cost per request (£)",
    min_value=0.0001,
    max_value=100.0,
    value=0.05,
    format="%.4f",
)

max_latency = st.sidebar.number_input(
    "Maximum latency (ms)",
    min_value=100,
    max_value=60000,
    value=2000,
    step=100,
)

minimum_reliability = st.sidebar.slider(
    "Minimum reliability",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.01,
)

st.sidebar.header("Selection priorities")

quality_weight = st.sidebar.slider(
    "Quality weight",
    0.0, 1.0, 0.60, 0.05,
)
cost_weight = st.sidebar.slider(
    "Cost weight",
    0.0, 1.0, 0.15, 0.05,
)
latency_weight = st.sidebar.slider(
    "Latency weight",
    0.0, 1.0, 0.15, 0.05,
)
reliability_weight = st.sidebar.slider(
    "Reliability weight",
    0.0, 1.0, 0.10, 0.05,
)

weight_sum = (
    quality_weight
    + cost_weight
    + latency_weight
    + reliability_weight
)

if abs(weight_sum - 1.0) > 1e-9:
    st.sidebar.error(
        f"Weights total {weight_sum:.2f}. They must sum to 1.00."
    )
    st.info("Adjust the weights in the sidebar to continue.")
    st.stop()

task = TaskRequirements(
    task_type=task_type,
    data_classification=data_classification,
    max_cost_per_request=max_cost,
    max_latency_ms=max_latency,
    minimum_reliability=minimum_reliability,
)

weights = ScoringWeights(
    quality=quality_weight,
    cost=cost_weight,
    latency=latency_weight,
    reliability=reliability_weight,
)

try:
    decision = select_model(task, weights)

    st.subheader("Routing decision")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected model", decision.selected_model.name)
    col2.metric("Weighted score", f"{decision.selected_score.total_score:.4f}")
    col3.metric("Eligible candidates", len(decision.ranked_models))

    st.success(decision.explanation)

    st.subheader("Eligible model comparison")

    comparison_rows = []

    for score in decision.ranked_models:
        comparison_rows.append({
            "Model": score.model_name,
            "Quality": score.quality_score,
            "Cost efficiency": round(score.cost_score, 4),
            "Latency score": round(score.latency_score, 4),
            "Reliability": score.reliability_score,
            "Total score": score.total_score,
        })

    st.dataframe(
        comparison_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        {
            row["Model"]: row["Total score"]
            for row in comparison_rows
        }
    )

    st.subheader("Rejected candidates")

    if decision.rejected_models:
        for model_name, reasons in decision.rejected_models.items():
            st.write(f"**{model_name}**")
            for reason in reasons:
                st.write(f"- {reason}")
    else:
        st.write("No models were rejected.")

except ValueError as exc:
    st.error(f"No eligible model found: {exc}")
    st.info(
        "Relax a constraint or add a model that satisfies the requirements."
    )

st.divider()
st.caption(
    "Engineering note: quality values are illustrative, not benchmark-derived. "
    "The benchmark fixtures currently validate the evaluation pipeline only."
)
