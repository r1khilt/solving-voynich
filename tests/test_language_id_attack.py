"""Unit tests for EXP-0022 language-ID attack helpers (no Voynich score inspection)."""

import numpy as np

from voynich.language_id_attack import (
    BigramLM,
    clean_text,
    decide,
    encode_stream,
    letter_prefix,
    mono_attack,
    preferred_languages,
    scrambled_glyph,
    zipf_length_null,
)


def test_clean_and_prefix_preserve_letter_budget():
    text = clean_text("Hello, World!\nAGAIN  test")
    assert "hello" in text
    assert "," not in text
    prefixed = letter_prefix(text + " " + "abcde" * 5000, budget=100)
    assert sum(ch.isalpha() for ch in prefixed) == 100


def test_scrambled_preserves_mask_and_type_inventory():
    text = "qo qokeedy qokain\nchol daiin"
    out = scrambled_glyph(text, seed=4022)
    assert len(out) == len(text)
    assert [ch for ch in text if not ch.isalpha()] == [ch for ch in out if not ch.isalpha()]
    assert len({ch for ch in text if ch.isalpha()}) == len({ch for ch in out if ch.isalpha()})


def test_zipf_null_preserves_multiset_and_mask():
    text = "qo qokeedy qokain\nchol daiin"
    out = zipf_length_null(text, seed=4024)
    assert sorted(ch for ch in text if ch.isalpha()) == sorted(ch for ch in out if ch.isalpha())
    assert [i for i, ch in enumerate(text) if not ch.isalpha()] == [
        i for i, ch in enumerate(out) if not ch.isalpha()
    ]


def test_mono_improves_on_frequency_init_for_identity_cipher():
    plain = ("the quick brown fox jumps over the lazy dog " * 40).strip()
    lm = BigramLM(plain)
    from voynich.language_id_attack import cipher_bigram_pairs, frequency_rank_table, score_table

    ids = encode_stream(plain)
    prev, curr = cipher_bigram_pairs(ids)
    init = frequency_rank_table(ids, lm.unigram)
    init_bits = score_table(lm, prev, curr, init)
    attack_bits = mono_attack(ids, lm, np.random.default_rng(1))
    assert attack_bits <= init_bits + 1e-9
    identity = np.arange(27, dtype=np.int16)
    true_bits = score_table(lm, prev, curr, identity)
    assert attack_bits < true_bits + 0.05  # identity cipher; hill-climb should reach near-truth


def test_preference_and_decide_modes():
    assert preferred_languages({"a": 1.0, "b": 1.05, "c": 2.0}, margin=0.02) == ["a"]
    assert preferred_languages({"a": 1.0, "b": 1.01, "c": 2.0}, margin=0.02) == []
    inv = decide(
        {
            "true_eva": ["eng"],
            "scrambled_glyph": [],
            "section_shuffled": [],
            "zipf_length_null": ["heb"],
        }
    )
    assert inv["mode"] == "hebrew_latin_null_preferred"
    lead = decide(
        {
            "true_eva": ["fin"],
            "scrambled_glyph": [],
            "section_shuffled": [],
            "zipf_length_null": [],
        }
    )
    assert lead["mode"] == "lead_open" and lead["leads"] == ["fin"]
    rej = decide(
        {
            "true_eva": ["eng"],
            "scrambled_glyph": ["eng"],
            "section_shuffled": [],
            "zipf_length_null": [],
        }
    )
    assert rej["mode"] == "simple_language_id_rejected" and rej["pass"] is True
