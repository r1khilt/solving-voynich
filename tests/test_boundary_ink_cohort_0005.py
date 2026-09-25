"""Known-score controls for source-specific cohort centering."""

from scripts.register_boundary_ink_cohort_0005 import (
    cohort_scores,
    duplicate_target_control,
    rankings,
)


def toy_matrix() -> dict:
    values = {
        "a": {"a": .30, "b": .40, "c": .10},
        "b": {"a": .05, "b": .45, "c": .10},
        "c": {"a": .05, "b": .20, "c": .30},
    }
    return {target: {source: {"score": score, "sx": .8, "tx": 1,
                              "sy": .8, "ty": 1}
                     for source, score in row.items()}
            for target, row in values.items()}


def test_cohort_background_corrects_broad_template_bias() -> None:
    matrix = toy_matrix()
    scored = cohort_scores(matrix)
    ranked = rankings(matrix, scored)
    assert ranked["a"]["raw_rank"] == 2
    assert ranked["a"]["cohort_rank"] == 1
    assert ranked["a"]["cohort_margin"] > .03
    assert all(item["cohort_rank"] == 1 for item in ranked.values())


def test_duplicate_missing_target_cannot_pass_complete_gate() -> None:
    control = duplicate_target_control(toy_matrix(), ["a", "b", "c"])
    assert control["rejects_complete_claim"]
    assert control["passing_pages_after_duplicate"] < 3
