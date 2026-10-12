
import pytest

from src.evaluator import (
    compare_values,
    evaluate_answer,
    load_benchmark,
)


def test_benchmark_contains_five_tasks():
    tasks = load_benchmark()

    assert len(tasks) == 5


def test_exact_structured_answer_scores_one():
    expected = {"revenue_millions": 12.5, "ebitda_millions": 2.1}

    result = evaluate_answer(expected, expected)

    assert result["quality_score"] == 1.0
    assert result["correct_fields"] == 2


def test_incorrect_field_reduces_quality_score():
    expected = {"revenue_millions": 12.5, "ebitda_millions": 2.1}
    actual = {"revenue_millions": 12.5, "ebitda_millions": 1.9}

    result = evaluate_answer(expected, actual)

    assert result["quality_score"] == 0.5


def test_numeric_tolerance_is_supported():
    assert compare_values(12.5, 12.504)
    assert not compare_values(12.5, 12.6)


def test_string_comparison_ignores_case_and_outer_whitespace():
    assert compare_values(" Revenue Growth ", "revenue growth")


def test_missing_field_is_counted_as_incorrect():
    result = evaluate_answer(
        {"revenue_millions": 12.5, "ebitda_millions": 2.1},
        {"revenue_millions": 12.5},
    )

    assert result["quality_score"] == 0.5


def test_invalid_answer_types_are_rejected():
    with pytest.raises(ValueError):
        evaluate_answer({"revenue": 12.5}, ["revenue", 12.5])
