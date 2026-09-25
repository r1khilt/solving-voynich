import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "naibbe_evaluation", Path(__file__).resolve().parents[1] / "scripts/evaluate_naibbe001.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def simple_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a):
        new = [i + 1]
        for j, y in enumerate(b):
            new.append(min(new[-1] + 1, row[j + 1] + 1, row[j] + (x != y)))
        row = new
    return row[-1]


def test_bit_parallel_edit_distance_matches_independent_dynamic_program():
    rng = np.random.default_rng(18731)
    for _ in range(500):
        a = "".join(rng.choice(list("abc"), int(rng.integers(0, 50))))
        b = "".join(rng.choice(list("abc"), int(rng.integers(0, 50))))
        assert module.edit_distance(a, b) == simple_distance(a, b)


def test_weighted_key_accuracy_counts_true_key_not_arbitrary_id_order():
    values = module.metrics(["aaab"], ["aaab"], [[(0,)]],
                            np.array([0, 1, 2]), np.array([0, 2, 1]), "abc")
    assert values["key_frequency_weighted_accuracy"] == .75
    assert values["cer"] == 0
    assert values["key_types_correct"] == 1
