"""Synthetic metric fixtures only: no corpus, training run or manuscript score."""

import itertools
import json
import math

import numpy as np
import pytest

from voynich.latent_recovery import (
    apply_pass_rule_v13, bigram_bits, evaluate_masks, f1_binary, fit_rank_bigram,
    matched_random_baseline, pred_bits_of_selected, rank_bucket_sequence,
    recon_accuracy, recon_edit_similarity,
)
from voynich.recovery_metrics import (
    edit_distance, fit_fixed_rank_bigram, fixed_rank_deletion_diagnostic,
    frozen_rank_encoding, reconstruction_diagnostics,
)


def sample(text="axbxcxd", mask=(1, 0, 1, 0, 1, 0, 1), target="abcd"):
    return {"text": text, "mask": list(mask), "ciphered": target}


def test_single_deletion_separates_positional_alignment_and_exact_recovery():
    text = "abcdefghij"
    mask = np.ones(len(text), dtype=int)
    mask[2] = 0
    row = reconstruction_diagnostics(text, text, mask)
    assert recon_accuracy(text, text, mask) == pytest.approx(.2)
    assert row["aligned_similarity"] == .9
    assert row["aligned_similarity"] == recon_edit_similarity(text, text, mask)
    assert row["edit_distance"] == 1
    assert not row["exact_sequence_match"]
    assert (row["input_count"], row["target_count"], row["retained_count"], row["deleted_count"]) == (10, 10, 9, 1)


@pytest.mark.parametrize("target,text,mask,distance,similarity,exact", [
    ("abc", "axbc", [1, 1, 1, 1], 1, .75, False),
    ("abc", "axbc", [1, 0, 1, 1], 0, 1., True),
    ("abc", "abd", [1, 1, 1], 1, 1 - 1 / 3, False),
    ("", "", [], 0, 1., True),
    ("", "aaa", [0, 0, 0], 0, 1., True),
    ("abc", "abc", [0, 0, 0], 3, 0., False),
])
def test_reconstruction_edge_cases(target, text, mask, distance, similarity, exact):
    row = reconstruction_diagnostics(target, text, mask)
    assert row["edit_distance"] == distance
    assert row["aligned_similarity"] == similarity
    assert row["exact_sequence_match"] is exact


def test_exact_surface_recovery_does_not_imply_position_localization():
    row = reconstruction_diagnostics("aa", "aaaa", [0, 0, 1, 1])
    assert row["exact_sequence_match"]  # Gold positions could instead be [1,1,0,0].
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("sitting", "kitten") == 3


@pytest.mark.parametrize("mask", [[1], [1, .5], [[1, 0]], [1, 2]])
def test_new_mask_api_rejects_silent_truncation_and_nonbinary_values(mask):
    with pytest.raises(ValueError, match="binary"):
        reconstruction_diagnostics("a", "ab", mask)


def test_ordinary_and_random_reports_have_comparable_alignment_exactness_and_counts():
    samples = [sample()]
    legacy = fit_rank_bigram(["abcd abcd", "bcd abc"])
    normal = evaluate_masks(samples, [np.array(samples[0]["mask"])], legacy)
    random = matched_random_baseline(samples, legacy, n_seeds=3)
    assert normal["recon_edit_sim"] == normal["recon_exact_match_rate"] == 1.
    assert set(normal["reconstruction_diagnostics"]) == set(random["reconstruction_diagnostics"])
    assert normal["reconstruction_diagnostics"]["retained_count_mean"] == 4
    assert random["reconstruction_diagnostics"]["retained_count_mean"] == 4
    assert random["matched_count_source"] == "gold_signal_mask"


def legacy_random_reference(samples, rank_model, n_seeds):
    """Pre-audit draw order and scalar formulas, deliberately no new helpers."""
    fs, rs, gs = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        pf, pr, pg = [], [], []
        for s in samples:
            true = np.array(s["mask"], dtype=int)
            pred = np.ones(len(true), dtype=int)
            count = int((true == 0).sum())
            if count:
                pred[rng.choice(len(true), size=count, replace=False)] = 0
            pf.append(f1_binary(true, pred))
            pr.append(recon_accuracy(s["ciphered"], s["text"], pred))
            pg.append(bigram_bits(rank_bucket_sequence(s["text"]), rank_model)
                      - pred_bits_of_selected(s["text"], pred, rank_model))
        fs.append(float(np.mean(pf)))
        rs.append(float(np.mean(pr)))
        gs.append(float(np.mean(pg)))
    return {"mask_f1": float(np.mean(fs)), "mask_f1_std": float(np.std(fs)),
            "recon_acc": float(np.mean(rs)), "pred_bits_gain": float(np.mean(gs))}


def test_existing_random_draws_and_all_legacy_baseline_scalars_are_unchanged():
    samples = [sample(), sample("aaaba", [1, 0, 1, 0, 1], "aaa")]
    legacy = fit_rank_bigram(["abc abc", "cba bcc"])
    old = legacy_random_reference(samples, legacy, 7)
    new = matched_random_baseline(samples, legacy, n_seeds=7)
    assert {key: new[key] for key in old} == old
    reference_masks = [np.array([1, 1, 0, 0, 0, 0, 0]), np.ones(5, dtype=int)]
    matched = matched_random_baseline(samples, legacy, n_seeds=7, reference_masks=reference_masks)
    assert matched["matched_count_source"] == "supplied_prediction_mask"
    assert matched["reconstruction_diagnostics"]["retained_count_mean"] == 3.5


def test_new_diagnostics_cannot_rescue_an_old_failed_pass_rule():
    values = {"mask_f1": .9, "mask_acc": .9, "null_recall": .8, "null_precision": .8,
              "pred_null_rate": .3, "recon_acc": .19, "pred_bits_gain": 0.,
              "recon_edit_sim": 1., "recon_exact_match_rate": 1.}
    result = apply_pass_rule_v13(values, values, {"mask_acc": .7},
                               {"recon_acc": .20}, {"mask_f1": .1})
    assert result["passed"] is False


def test_rank_labels_do_not_change_after_selection_and_renaming():
    text = "aaaabbbc"
    keep = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    full = np.array(frozen_rank_encoding(text))
    assert full[keep].tolist() == [0, 2, 2, 2, 5]
    assert rank_bucket_sequence("abbbc") == [2, 0, 0, 0, 5]
    renamed = text.translate(str.maketrans("abc", "γζα"))
    assert frozen_rank_encoding(renamed) == tuple(full)
    assert frozen_rank_encoding("ab ba") == tuple(rank_bucket_sequence("ab ba"))


def test_normalized_model_covers_unseen_rows_first_symbol_and_all_fixed_length_strings():
    model = fit_fixed_rank_bigram(["aaaa", "aa"], training_source="toy-independent-fit", n_buckets=2)
    assert np.allclose(np.asarray(model.transition).sum(1), 1)
    assert sum(model.initial) == pytest.approx(1)
    for length in range(4):
        mass = sum(2 ** -model.score(seq)["negative_log2_probability"]
                   for seq in itertools.product(range(3), repeat=length))
        assert mass == pytest.approx(1., abs=1e-12)
    singleton = model.score([2])  # Space was absent from training, but is supported.
    assert singleton["initial_symbol_scored"] and singleton["transition_count"] == 0
    assert singleton["bits_per_encoded_symbol"] == -math.log2(model.initial[2])
    assert model.score([])["bits_per_encoded_symbol"] is None
    assert model.score([])["status"] == "no_observations"
    with pytest.raises(ValueError, match="vocabulary"):
        model.score([3])


def test_count_controls_are_reproducible_and_do_not_refit_or_mutate_the_model():
    model = fit_fixed_rank_bigram(["aabb cc", "abc abc"], training_source="toy-fit")
    before = model.metadata()
    text = "aaaabbbbcccc"
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    first = fixed_rank_deletion_diagnostic(text, mask, model, n_random=6, seed=7)
    second = fixed_rank_deletion_diagnostic(text, mask, model, n_random=6, seed=7)
    assert first == second
    assert before == model.metadata()
    for control in first["controls"].values():
        assert control["retained_count_each"] == 6
        assert control["transition_count_each"] == 5
        assert control["replicates"] == 6
    json.dumps(first, allow_nan=False)


def test_histogram_control_exposes_selecting_only_easy_symbols():
    model = fit_fixed_rank_bigram(["aaaaab", "aaaac"], training_source="toy-fit")
    row = fixed_rank_deletion_diagnostic("aaaaaabbbccc", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], model)
    histogram = row["controls"]["same_retained_count_and_bucket_histogram"]
    assert histogram["random_minus_selected_bits_per_encoded_symbol"] == pytest.approx(0., abs=1e-14)
    assert histogram["std_bits_per_encoded_symbol"] == pytest.approx(0., abs=1e-14)
    assert "passed" not in row


@pytest.mark.parametrize("text,mask", [("", []), ("abcde", [0] * 5),
                                      ("abcde", [1, 0, 0, 0, 0]), ("abcde", [1] * 5)])
def test_empty_short_and_no_deletion_comparisons_are_not_evidence(text, mask):
    model = fit_fixed_rank_bigram(["aa bb"], training_source="toy-fit")
    row = fixed_rank_deletion_diagnostic(text, mask, model)
    assert all(not c["eligible_for_descriptive_comparison"] for c in row["controls"].values())
    json.dumps(row, allow_nan=False)


def test_new_proxy_is_opt_in_and_cannot_change_historical_scores():
    samples = [sample()]
    masks = [np.array([1, 1, 0, 1, 1, 1, 0])]
    legacy = fit_rank_bigram(["abcd abcd"])
    fixed = fit_fixed_rank_bigram(["ab cd ab cd"], training_source="toy-fit")
    before = evaluate_masks(samples, masks, legacy)
    after = evaluate_masks(samples, masks, legacy, fixed_rank_model=fixed, diagnostic_random_replicates=3)
    assert "fixed_rank_deletion_diagnostics" not in before
    assert {key: after[key] for key in before} == before
    assert len(after["fixed_rank_deletion_diagnostics"]["per_sample"]) == 1


def test_reports_reject_missing_predictions_and_bad_control_counts():
    legacy = fit_rank_bigram(["abc"])
    with pytest.raises(ValueError, match="one prediction"):
        evaluate_masks([sample()], [], legacy)
    with pytest.raises(ValueError, match="positive"):
        matched_random_baseline([sample()], legacy, n_seeds=0)
    with pytest.raises(ValueError, match="one reference"):
        matched_random_baseline([sample()], legacy, reference_masks=[])
