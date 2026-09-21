"""Unit tests for EXP-0011 latent recovery data and metrics (no holdout scores)."""

import numpy as np

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


def test_homophone_cipher_runs():
    rng = np.random.default_rng(5)
    alphabet = "".join(list(CIPHER_POOL)[:40])
    spec = sample_cipher(rng, alphabet)
    out = apply_cipher("caesar gallia belgae", spec, rng)
    assert isinstance(out, str) and len(out) > 0
