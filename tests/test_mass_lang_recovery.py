"""Tests for EXP-0016 mass-language recovery (no holdout scores)."""

from voynich.mass_lang_recovery import (
    apply_pass_rule_v16,
    detect_script,
    is_indo_european,
    romanize_auto,
    stratified_train_subset,
)
from voynich.multilang_recovery import exact_count_masks
import numpy as np


def test_detect_script_latin_cyrillic():
    assert detect_script("hello world") == "latin"
    assert detect_script("привет мир") == "cyrillic"


def test_romanize_auto_skips_cjk():
    text, scheme, ok = romanize_auto("日本語の文章です" * 20, None)
    assert ok is False
    assert "cjk" in scheme or "skip" in scheme


def test_romanize_auto_arabic():
    text, scheme, ok = romanize_auto("اللغة العربية " * 40, "Arab")
    assert ok
    assert set(text) <= set("abcdefghijklmnopqrstuvwxyz ")
    assert len(text) >= 50


def test_exact_count_still_stable():
    probs = [np.array([0.1, 0.9, 0.8, 0.2])]
    masks = exact_count_masks(probs, 0.5)
    assert masks[0].sum() == 2


def test_stratified_subset_covers_families():
    pool = {
        f"l{i}": {"family": f"fam{i % 5}"} for i in range(100)
    }
    # attach required keys
    for k, v in pool.items():
        v["iso6393"] = k
    sel = stratified_train_subset(pool, n=40, seed=4016)
    assert len(sel) == 40
    fams = {pool[i]["family"] for i in sel}
    assert len(fams) == 5


def test_pass_rule_requires_100_and_non_ie():
    per = {}
    for i in range(120):
        if i < 24:
            fam = "Indo-European"
        elif i < 48:
            fam = "Austronesian"
        elif i < 72:
            fam = "Atlantic-Congo"
        elif i < 96:
            fam = "Sino-Tibetan"
        else:
            fam = "Dravidian"
        per[f"l{i}"] = {
            "family": fam,
            "neural": {
                "recon_acc": 0.30,
                "null_recall": 0.6,
                "null_precision": 0.6,
                "pred_null_rate": 0.3,
            },
            "baselines": {
                "matched_random": {"recon_acc": 0.20},
                "vocab_filter": {"recon_acc": 0.05},
            },
            "beats_matched_random": True,
        }
    d = apply_pass_rule_v16(per)
    assert d["passed"] is True
    assert d["n_beat_matched_random"] == 120
    assert d["n_passer_families"] >= 5
    assert d["non_ie_passer_fraction"] >= 0.5


def test_ie_helper():
    assert is_indo_european("Indo-European")
    assert not is_indo_european("Austronesian")
