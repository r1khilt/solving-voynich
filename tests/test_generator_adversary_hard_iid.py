"""Unit tests for EXP-0025 harder-iid generator-adversary helpers."""

from __future__ import annotations

import numpy as np

from voynich.generator_adversary import FLOORS, OPT_METRICS, metric_verdicts
from voynich.generator_adversary_hard_iid import (
    EXPERIMENT_ID,
    IID_NAME,
    IID_WORD_LEN,
    decide_pass,
    generate_iid_uniform_fixedlen,
)


def test_experiment_id_and_control_name():
    assert EXPERIMENT_ID == "EXP-0025"
    assert IID_NAME == "iid_uniform_fixedlen"
    assert IID_WORD_LEN == 8


def test_uniform_fixedlen_emits_shaped_pages():
    shapes = [
        {"page_id": "p1", "L": "A", "line_word_counts": [3, 2]},
        {"page_id": "p2", "L": "B", "line_word_counts": [1]},
    ]
    rng = np.random.default_rng(0)
    pages = generate_iid_uniform_fixedlen(rng, shapes, word_len=8)
    assert len(pages) == 2
    words = pages[0]["text"].split()
    assert len(words) == 5
    assert all(len(w) == 8 and w.isalpha() and w.islower() for w in words)


def test_decide_pass_modes_include_surface_unmatchable():
    assert decide_pass(4, 3, 1)["mode"] == "iid_control_failed"
    assert decide_pass(4, 3, 1)["hyp005_status"] == "untested"
    assert decide_pass(2, 3, 3)["mode"] == "surface_unmatchable"
    assert decide_pass(2, 3, 3)["hyp005_status"] == "open_not_killed"
    assert decide_pass(4, 1, 3)["mode"] == "heldout_also_matched"
    ok = decide_pass(3, 2, 3)
    assert ok["passed"] is True
    assert ok["hyp005_status"] == "open_compatible"


def test_floors_unchanged_from_exp0019():
    assert FLOORS["zipf_slope_top500"] == 0.15
    assert FLOORS["adjacent_repeat_rate"] == 0.005
    assert len(OPT_METRICS) == 4
    train = {m: 0.5 for m in OPT_METRICS}
    val = {m: 0.5 for m in OPT_METRICS}
    gen = {m: 0.5 for m in OPT_METRICS}
    assert metric_verdicts(train, val, gen, OPT_METRICS)["n_match"] == 4
