from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.audit_joint_key_order001 import audit_rows
from voynich.joint_key_order_controls import controlled_episode, score_episode, summarize
from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig, unit_pool
from voynich.joint_key_training import EpisodeSampler
from voynich.recurrent_latin_source import ALPHABET, encode


def episode():
    texts = [ALPHABET*20]
    return texts, EpisodeSampler(texts).make(
        [{"segment": 0, "start": 3, "length": 64}, {"segment": 0, "start": 60, "length": 100}],
        np.random.default_rng(37).integers(42, size=23).tolist())


@pytest.mark.parametrize("condition", ["unit_shuffle", "glyph_shuffle"])
def test_controls_preserve_counts_names_prefix_and_gold_reencoding(condition):
    texts, own = episode()
    changed, detail = controlled_episode(own, texts, condition=condition, seed=42)
    assert controlled_episode(own, texts, condition=condition, seed=42) == (changed, detail)
    assert changed[1] == own[1]
    assert changed[2]["used_rows"] == own[2]["used_rows"]
    assert changed[2]["counterfactual"] is True
    assert changed[2]["original_episode_metadata"] == own[2]
    assert detail["gold_segmentation_only_for_control_construction"]
    pool = unit_pool(6)
    for old, new, spec, window in zip(own[0], changed[0], detail["records"], own[2]["windows"], strict=True):
        cut = spec["prefix_cipher_glyphs"]
        assert Counter(new) == Counter(old) and len(old) == len(new)
        assert tuple(dict.fromkeys(new)) == tuple(dict.fromkeys(old))
        assert new[:cut] == old[:cut]
        assert spec["changed_glyph_positions"] == sum(a != b for a, b in zip(old, new, strict=True)) > 0
        source = encode(texts[window["segment"]])[window["start"]:window["start"]+window["length"]]
        units = [pool[own[1][int(c)]] for c in source]
        # Entire source units precede the cutoff; no partial digram anchor.
        assert sum(map(len, units[:spec["prefix_source_units"]])) == cut
        if condition == "unit_shuffle":
            rng = np.random.default_rng(42)
            # Unit-multiset source frequency is unchanged, unlike glyph shuffle.
            suffix = units[spec["prefix_source_units"]:]
            if old is own[0][0]:
                expected = tuple(c for i in rng.permutation(len(suffix)) for c in suffix[i])
                assert new[cut:] == expected


def test_missing_glyphs_and_unchangeable_suffix_are_retained():
    texts = [ALPHABET*20]
    own = EpisodeSampler(texts).make([{"segment": 0, "start": 0, "length": 64}]*2, [0]*23)
    changed, detail = controlled_episode(own, texts, condition="glyph_shuffle", seed=9)
    assert changed[0] == own[0]
    assert all(r["changed_glyph_positions"] == 0 for r in detail["records"])
    assert own[2]["unseen_symbols"]  # No invented glyph is added by the control.


def test_control_refuses_wrong_gold_source_or_key():
    texts, own = episode()
    wrong = list(own)
    wrong[1] = [0]*23
    with pytest.raises(ValueError, match="re-encode"):
        controlled_episode(wrong, texts, condition="unit_shuffle", seed=2)
    with pytest.raises(ValueError, match="Declared"):
        controlled_episode(own, texts, condition="other", seed=2)


def tiny():
    torch.manual_seed(11)
    return JointKeyProposal(KeyProposalConfig(width=8, heads=2, encoder_layers=1,
        decoder_layers=1)).double().eval()


def test_real_forward_is_independent_of_diagnostic_mask_and_replays_reference():
    _, own = episode()
    model = tiny()
    a = score_episode(model, own)
    other = deepcopy(own)
    other[2]["used_rows"] = []
    b = score_episode(model, other, saved_original={"key": a["greedy_key"],
        "joint_logq": a["joint_logq"], "greedy_logq": a["greedy_logq"]})
    assert (a["joint_logq"], a["row_logq"], a["greedy_key"]) == (b["joint_logq"], b["row_logq"], b["greedy_key"])
    assert b["used_rows"] == 0 and max(b["numeric"].values()) < 1e-10
    with pytest.raises(ValueError, match="CPUfloat64"):
        score_episode(model.float(), own)


def artificial_grid():
    rows = []
    for step in (0, 20000):
        for i in range(2):
            for condition in ("original", "unit_shuffle", "glyph_shuffle"):
                delta = .001 if step == 0 else .1
                rows.append({"checkpoint": step, "episode": i, "condition": condition,
                    "joint_logq": -23*(4+(delta if condition != "original" else 0)),
                    "greedy_key": [0]*23, "used_correct_rows": 1, "used_rows": 2, "whole_key_exact": False})
    return rows


def test_paired_direction_bootstrap_and_declared_decision():
    rows = artificial_grid()
    result = summarize(rows)
    assert result["exploratory_unit_order_diagnostic"] == "SUPPORTED"
    assert result["selected_minus_initial_unit_order_effect"] == pytest.approx(.099)
    assert result["checkpoints"]["20000"]["comparisons"]["unit_shuffle"]["mean_control_minus_original_nats_per_row"] == pytest.approx(.1)
    for row in rows:
        row["joint_logq"] = -92
    assert summarize(rows)["exploratory_unit_order_diagnostic"] == "NOT_SUPPORTED"
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows+[rows[0]])
    with pytest.raises(ValueError, match="Complete"):
        summarize(rows[:-1])


def audit_grid():
    texts, own = episode()
    truth, used = list(own[1]), own[2]["used_rows"]
    saved = {s: {"joint_logq": [-92.], "free_running_greedy_keys": [truth]} for s in (0, 20000)}
    rows = []
    for step in (0, 20000):
        for condition in ("original", "unit_shuffle", "glyph_shuffle"):
            construction = None if condition == "original" else controlled_episode(own, texts, condition=condition, seed=72341)[1]
            row = {"checkpoint": step, "episode": 0, "condition": condition, "construction": construction,
                "joint_logq": -92., "row_logq": [-4.]*23, "greedy_key": truth,
                "greedy_logq": -90., "correct_rows": 23, "used_correct_rows": len(used),
                "used_rows": len(used), "whole_key_exact": True}
            if condition == "original":
                row["numeric"] = {"true_logq_delta": 0., "saved_greedy_logq_delta": 0.,
                    "saved_greedy_max_cpu_deficit": 0., "cpu_vs_mps_greedy_changed_rows": 0}
            rows.append(row)
    return rows, [own], texts, saved


@pytest.mark.parametrize("corrupt,match", [("order", "grid"), ("construction", "construction"),
                                         ("probability", "probability"), ("used", "diagnostics")])
def test_audit_refuses_grid_substitutions_and_incorrect_accounting(corrupt, match):
    rows, episodes, texts, saved = audit_grid()
    assert audit_rows(rows, episodes, texts, saved)["episodes"] == 1
    if corrupt == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif corrupt == "construction":
        rows[1]["construction"]["seed"] += 1
    elif corrupt == "probability":
        rows[1]["row_logq"][0] += 1
    else:
        rows[0]["used_correct_rows"] -= 1
    with pytest.raises(ValueError, match=match):
        audit_rows(rows, episodes, texts, saved)
