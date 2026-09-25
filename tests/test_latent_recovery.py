"""Unit tests for EXP-0011 latent recovery data and metrics (no holdout scores)."""

import numpy as np
import pytest

from voynich.latent_recovery import (
    CIPHER_POOL,
    PRIMARY_FILLER_RATE,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    WORLD_D,
    apply_pass_rule,
    classical_null_mask,
    f1_binary,
    insert_nulls,
    make_sample,
    TinySignalModel,
    build_vocab,
    sample_cipher,
    apply_cipher,
)


def test_cipher_alphabet_excludes_latin_letters():
    assert not set(CIPHER_POOL) & set("abcdefghijklmnopqrstuvwxyz")


def test_world_c_mask_rate_near_target():
    rng = np.random.default_rng(0)
    alphabet = "".join(list(CIPHER_POOL)[:36])
    sample = make_sample("the quick brown fox jumps over the lazy dog " * 4, rng, WORLD_C, PRIMARY_FILLER_RATE, alphabet)
    assert len(sample["text"]) == len(sample["mask"]) == 128
    rate = 1 - (sum(sample["mask"]) / len(sample["mask"]))
    assert abs(rate - PRIMARY_FILLER_RATE) < 0.12


def test_all_worlds_fixed_length():
    rng = np.random.default_rng(9)
    alphabet = "".join(list(CIPHER_POOL)[:36])
    for world in (WORLD_A, WORLD_B, WORLD_C, WORLD_D):
        sample = make_sample("gallia est omnis divisa in partes tres " * 3, rng, world, 0.3, alphabet)
        assert len(sample["text"]) == 128
        assert len(sample["mask"]) == 128


def test_world_a_all_signal_latin():
    rng = np.random.default_rng(1)
    sample = make_sample("hello world this is a test of pure language text here", rng, WORLD_A, 0.3)
    assert all(m == 1 for m in sample["mask"])
    assert set(sample["text"]) <= set("abcdefghijklmnopqrstuvwxyz ")


def test_world_d_all_null():
    rng = np.random.default_rng(2)
    alphabet = "".join(list(CIPHER_POOL)[:36])
    sample = make_sample("ignored", rng, WORLD_D, 0.3, alphabet)
    assert all(m == 0 for m in sample["mask"])


def test_world_b_no_nulls_but_not_latin():
    rng = np.random.default_rng(3)
    alphabet = "".join(list(CIPHER_POOL)[:36])
    sample = make_sample("gallia est omnis divisa in partes tres quarum unam", rng, WORLD_B, 0.3, alphabet)
    assert all(m == 1 for m in sample["mask"])
    # Ciphered text should mostly avoid latin letters (space allowed)
    letters = [c for c in sample["text"] if c != " "]
    assert letters and not all(c in "abcdefghijklmnopqrstuvwxyz" for c in letters)


def test_insert_nulls_preserves_signal_order():
    rng = np.random.default_rng(4)
    alphabet = "".join(list(CIPHER_POOL)[:30])
    ciphered = "αβγδεζηθικλμνξοπρστυ"
    noisy, mask, _ = insert_nulls(ciphered, rng, 0.3, alphabet)
    recovered = "".join(ch for ch, m in zip(noisy, mask) if m == 1)
    assert recovered == ciphered


@pytest.mark.parametrize(
    "family", ("random_char", "random_pseudoword", "copy_mutate", "periodic", "shift_phase", "stateful")
)
@pytest.mark.parametrize("rate", (0.0, 0.3, 0.7))
def test_insert_nulls_character_alignment(family, rate):
    alphabet = "".join(list(CIPHER_POOL)[:36])
    ciphered = "αβγδε ζηθικ λμνξο πρστυ " * 5
    expected_nulls = round(rate / max(1e-6, 1 - rate) * len(ciphered))
    for seed in range(10):
        noisy, mask, families = insert_nulls(
            ciphered,
            np.random.default_rng(seed),
            rate,
            alphabet,
            filler_families=(family,),
        )
        assert len(noisy) == len(mask) == len(families) == len(ciphered) + expected_nulls
        assert mask.count(0) == expected_nulls
        assert "".join(ch for ch, keep in zip(noisy, mask) if keep) == ciphered
        assert all(tag == family for tag, keep in zip(families, mask) if not keep)


def test_length_normalization_rejects_misaligned_mask():
    from voynich.latent_recovery import _fit_len

    with pytest.raises(ValueError, match="aligned"):
        _fit_len("abc", [1, 1], length=2)


def test_easy_filler_families_only():
    from voynich.latent_recovery import EASY_FILLER_FAMILIES, parse_filler_families

    assert parse_filler_families("random_char,periodic") == EASY_FILLER_FAMILIES
    rng = np.random.default_rng(5)
    alphabet = "".join(list(CIPHER_POOL)[:30])
    for _ in range(20):
        sample = make_sample(
            "the quick brown fox jumps over the lazy dog " * 4,
            rng,
            WORLD_C,
            0.3,
            alphabet,
            filler_families=EASY_FILLER_FAMILIES,
        )
        assert sample["filler_family"] in EASY_FILLER_FAMILIES


def test_param_count_under_100k():
    vocab = build_vocab()
    model = TinySignalModel(len(vocab))
    assert model.param_count() < 100_000


def test_classical_mask_length():
    mask = classical_null_mask("αβγ αβγ δεζ δεζ ηθηθ", 0.3)
    assert len(mask) == len("αβγ αβγ δεζ δεζ ηθηθ")


def test_classical_hsmm_mask_length():
    from voynich.latent_recovery import classical_hsmm_null_mask

    text = "αβγ αβγ δεζ δεζ ηθηθ αααα"
    mask = classical_hsmm_null_mask(text, 0.3)
    assert len(mask) == len(text)
    assert set(mask.tolist()) <= {0, 1}


def test_pass_rule_v12_blocks_delete_nothing():
    from voynich.latent_recovery import apply_pass_rule_v12

    # High signal F1 / delete-nothing style neural
    neural = {
        "mask_f1": 0.83,
        "mask_acc": 0.71,
        "recon_acc": 0.07,
        "pred_bits_gain": 0.0,
        "null_recall": 0.04,
        "null_precision": 0.5,
        "pred_null_rate": 0.03,
    }
    classical = {
        "mask_f1": 0.72,
        "mask_acc": 0.60,
        "recon_acc": 0.18,
        "pred_bits_gain": 0.01,
        "null_recall": 0.32,
        "null_precision": 0.31,
        "pred_null_rate": 0.30,
    }
    majority = {"mask_f1": 0.83, "mask_acc": 0.71, "recon_acc": 0.07, "pred_bits_gain": 0.0}
    random_b = {"mask_f1": 0.71, "mask_acc": 0.6, "recon_acc": 0.20, "pred_bits_gain": -0.03}
    vocab_b = {"mask_f1": 0.42, "mask_acc": 0.5, "recon_acc": 0.05, "pred_bits_gain": 1.0}
    decision = apply_pass_rule_v12(neural, classical, majority, random_b, vocab_b)
    assert decision["passed"] is False


def test_pass_rule_vocab_cheat_fails():
    neural = {"mask_f1": 0.9, "mask_acc": 0.9, "recon_acc": 0.9, "pred_bits_gain": 1.0}
    classical = {"mask_f1": 0.2, "mask_acc": 0.5, "recon_acc": 0.2, "pred_bits_gain": 0.0}
    majority = {"mask_f1": 0.8, "mask_acc": 0.7, "recon_acc": 0.5, "pred_bits_gain": 0.0}
    random_b = {"mask_f1": 0.7, "mask_acc": 0.6, "recon_acc": 0.5, "pred_bits_gain": 0.1}
    vocab_b = {"mask_f1": 0.8, "mask_acc": 0.7, "recon_acc": 0.5, "pred_bits_gain": 0.0}  # too strong
    decision = apply_pass_rule(neural, classical, majority, random_b, vocab_b)
    assert decision["passed"] is False


def test_f1_perfect():
    y = np.array([1, 1, 0, 0, 1])
    assert f1_binary(y, y) == 1.0


def test_recon_acc_definition_prefix_length_pen():
    from voynich.latent_recovery import recon_accuracy

    # Exact match
    assert recon_accuracy("abc", "axbxc", np.array([1, 0, 1, 0, 1])) == 1.0
    # Length mismatch penalizes
    score = recon_accuracy("abcd", "axbxc", np.array([1, 0, 1, 0, 1]))
    assert 0.0 < score < 1.0


def test_ctc_deletion_loss_runs():
    import torch
    from voynich.latent_recovery import build_vocab, ctc_deletion_loss, encode

    vocab = build_vocab()
    text = "αβγδε"
    x = torch.tensor([encode(text, vocab)], dtype=torch.long)
    # Prefer keeping all positions
    mask_logit = torch.full((1, len(text)), 3.0, requires_grad=True)
    target = encode(text, vocab)
    loss = ctc_deletion_loss(mask_logit, x, [target], len(vocab))
    assert torch.isfinite(loss)
    loss.backward()
    assert mask_logit.grad is not None


def test_pass_rule_v13_requires_recon_above_random():
    from voynich.latent_recovery import apply_pass_rule_v13

    neural = {
        "mask_f1": 0.7,
        "mask_acc": 0.75,
        "recon_acc": 0.18,
        "pred_bits_gain": 0.0,
        "null_recall": 0.55,
        "null_precision": 0.55,
        "pred_null_rate": 0.30,
    }
    classical = {
        "mask_f1": 0.6,
        "mask_acc": 0.6,
        "recon_acc": 0.10,
        "pred_bits_gain": 0.0,
        "null_recall": 0.4,
        "null_precision": 0.4,
        "pred_null_rate": 0.30,
    }
    majority = {"mask_f1": 0.83, "mask_acc": 0.71}
    random_b = {"mask_f1": 0.71, "recon_acc": 0.20, "pred_bits_gain": -0.03}
    vocab_b = {"mask_f1": 0.42}
    decision = apply_pass_rule_v13(neural, classical, majority, random_b, vocab_b)
    assert decision["passed"] is False
    # Same null gates but recon above random → pass
    neural2 = {**neural, "recon_acc": 0.25}
    decision2 = apply_pass_rule_v13(neural2, classical, majority, random_b, vocab_b)
    assert decision2["passed"] is True
    assert decision2["winner"] == "neural"


def test_homophone_cipher_runs():
    rng = np.random.default_rng(5)
    alphabet = "".join(list(CIPHER_POOL)[:40])
    spec = sample_cipher(rng, alphabet)
    out = apply_cipher("caesar gallia belgae", spec, rng)
    assert isinstance(out, str) and len(out) > 0
