"""Typed decoder-program helpers (no Finnish scoring)."""

from voynich.decoder_programs import (
    LENGTH_PENALTY,
    OPS,
    enumerate_programs,
    readable,
    val_gates_ok,
)


def test_length_penalty_frozen():
    assert LENGTH_PENALTY == 0.02


def test_enumerate_keeps_all_length1_and_2():
    programs = enumerate_programs()
    length1 = [p for p in programs if len(p) == 1]
    length2 = [p for p in programs if len(p) == 2]
    assert len(length1) == len(OPS)
    assert len(length2) == len(OPS) * (len(OPS) - 1)
    assert len(programs) <= 200
    assert readable(("exact_count_neural",)) == "exact_count_neural"


def test_val_gates_reject_delete_nothing():
    assert not val_gates_ok(
        {"null_recall": 0.6, "null_precision": 0.6, "pred_null_rate": 0.0}
    )
    assert val_gates_ok(
        {"null_recall": 0.6, "null_precision": 0.6, "pred_null_rate": 0.3}
    )
