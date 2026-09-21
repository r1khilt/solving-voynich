"""Tests for EXP-0015 romanization and exact-count decode (no holdout scores)."""

import numpy as np

from voynich.multilang_recovery import exact_count_masks, is_indo_european, language_passes
from voynich.romanize import fold_latin, romanize


def test_fold_latin_strips_diacritics():
    assert fold_latin("Việt Nam") == "viet nam"
    assert "á" not in fold_latin("café")


def test_cyrillic_romanize_has_letters():
    out = romanize("привет мир", "cyrillic")
    assert "privet" in out.replace(" ", "") or "p" in out
    assert set(out) <= set("abcdefghijklmnopqrstuvwxyz ")


def test_greek_romanize():
    out = romanize("αβγδε", "greek")
    assert out.startswith("abg") or "a" in out
    assert set(out) <= set("abcdefghijklmnopqrstuvwxyz ")


def test_hebrew_romanize():
    out = romanize("שלום", "hebrew")
    assert len(out) >= 3
    assert set(out) <= set("abcdefghijklmnopqrstuvwxyz ")


def test_exact_count_keep_count():
    probs = [np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])]
    # L=10, rate 0.30 → n_keep = round(7.0) = 7
    masks = exact_count_masks(probs, filler_rate=0.30)
    assert masks[0].sum() == 7
    # Highest scores kept
    assert masks[0][1] == 1 and masks[0][2] == 1


def test_exact_count_ties_prefer_earlier_index():
    probs = [np.array([0.5, 0.5, 0.5, 0.5])]
    masks = exact_count_masks(probs, filler_rate=0.50)
    # n_keep = round(2.0) = 2; stable argsort on equal scores keeps earlier indices
    assert masks[0].tolist() == [1, 1, 0, 0]


def test_language_pass_requires_beat_random():
    neural = {
        "null_recall": 0.6,
        "null_precision": 0.6,
        "pred_null_rate": 0.3,
        "recon_acc": 0.20,
    }
    random_b = {"recon_acc": 0.21}
    vocab_b = {"recon_acc": 0.05}
    assert language_passes(neural, random_b, vocab_b)["pass"] is False
    neural2 = {**neural, "recon_acc": 0.25}
    assert language_passes(neural2, random_b, vocab_b)["pass"] is True


def test_vocab_cheat_fails_language():
    neural = {
        "null_recall": 0.6,
        "null_precision": 0.6,
        "pred_null_rate": 0.3,
        "recon_acc": 0.22,
    }
    random_b = {"recon_acc": 0.20}
    vocab_b = {"recon_acc": 0.25}
    assert language_passes(neural, random_b, vocab_b)["pass"] is False


def test_ie_family_helper():
    assert is_indo_european("indo_european_germanic")
    assert is_indo_european("indo_aryan")
    assert not is_indo_european("uralic")
    assert not is_indo_european("isolate")
