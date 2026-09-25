"""Structural and known-rule controls for EXP-0042, without manuscript validation."""

import numpy as np

from voynich.context_residual import (
    evaluate_view,
    fit_counts,
    internal_split,
    keys,
    probability_matrices,
    residual_bits,
    rows,
    synthetic,
)


def _skeleton(n_groups: int, n_words: int, prefix: str) -> list[dict]:
    return [{"leaf": f"{prefix}{i % 50}", "context": ("toy", "toy"),
             "words": [("x",)] * n_words} for i in range(n_groups)]


def test_no_future_glyph_in_features() -> None:
    template = _skeleton(1, 4, "toy")
    template[0]["words"] = [("a", "z"), ("b", "z"), ("c", "z"), ("d", "z")]
    first = rows(template)[0]
    other = _skeleton(1, 4, "toy")
    other[0]["words"] = [("a", "z"), ("b", "other"), ("c", "z"), ("d", "z")]
    second = rows(other)[0]
    assert first.target == second.target == "b"
    for family in ("edge", "form", "history", "layout", "all"):
        assert keys(first, family) == keys(second, family)


def test_leaf_split_and_normalized_baseline() -> None:
    train = synthetic(_skeleton(200, 12, "train"), 9001, 0.85)
    fit, dev = internal_split(train)
    assert {group["leaf"] for group in fit}.isdisjoint({group["leaf"] for group in dev})
    model = fit_counts(fit)
    baseline, _, labels = probability_matrices(rows(dev), model)
    assert np.max(np.abs(np.exp(baseline).sum(axis=1) - 1)) < 1e-12
    assert np.array_equal(residual_bits(baseline, labels, np.zeros_like(baseline), 0),
                          np.zeros(len(labels)))


def test_known_rule_and_iid_calibration_on_fresh_toy_streams() -> None:
    train_template = _skeleton(200, 12, "train")
    held_template = _skeleton(40, 12, "held")
    positive_train = synthetic(train_template, 9001, 0.85)
    negative_train = synthetic(train_template, 9001, 0.0)
    assert all(a[1:-1] == b[1:-1] for ga, gb in zip(positive_train, negative_train)
               for a, b in zip(ga["words"], gb["words"]))
    positive = evaluate_view(positive_train, synthetic(held_template, 9011, 0.85), None)
    negative = evaluate_view(negative_train, synthetic(held_template, 9011, 0.0), None)
    assert positive["richer_minus_edge_bits_per_pair"] > 0.10
    assert abs(negative["richer_minus_edge_bits_per_pair"]) < 0.03
