"""EXP-0023 helpers (no Finnish scoring)."""

from voynich.copy_mutate_transfer import (
    FILLER_FAMILIES,
    FROZEN_MATCHED_RANDOM_RECON,
    SEARCH_SEED,
    apply_exp0014_gates,
    enumerate_programs,
)
from voynich.decoder_programs import LENGTH_PENALTY, OPS
from voynich.exact_count_decode import FROZEN_MATCHED_RANDOM_RECON as GATE


def test_frozen_gate_unchanged():
    assert FROZEN_MATCHED_RANDOM_RECON == GATE == 0.2043264147237504


def test_filler_set_matches_exp0014b():
    assert FILLER_FAMILIES == ("random_char", "periodic", "copy_mutate")


def test_search_seed_and_catalog():
    assert SEARCH_SEED == 4023
    assert LENGTH_PENALTY == 0.02
    programs = enumerate_programs()
    assert len(programs) <= 200
    assert len([p for p in programs if len(p) == 1]) == len(OPS)


def test_gates_reject_below_frozen_recon():
    ok, reasons = apply_exp0014_gates(
        {
            "null_recall": 0.6,
            "null_precision": 0.6,
            "pred_null_rate": 0.3,
            "recon_acc": 0.1848,
        },
        {"mask_f1": 0.4},
    )
    assert not ok
    assert any("frozen_matched_random" in r for r in reasons)


def test_gates_accept_above_frozen_recon():
    ok, reasons = apply_exp0014_gates(
        {
            "null_recall": 0.6,
            "null_precision": 0.6,
            "pred_null_rate": 0.3,
            "recon_acc": 0.21774,
        },
        {"mask_f1": 0.4},
    )
    assert ok
    assert reasons == []
