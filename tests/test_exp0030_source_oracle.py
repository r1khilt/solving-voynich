"""Controls for the independent pre-null-source replay check."""

from scripts.audit_exp0030_source_oracle import alignment_checks


def test_alignment_check_accepts_true_signal_subsequence():
    checks = alignment_checks("abc", "axbc", [1, 0, 1, 1], ["", "copy_mutate", "", ""], "abc")
    assert all(checks.values())


def test_alignment_check_rejects_shifted_or_extra_null():
    checks = alignment_checks("abc", "axxbc", [1, 0, 1, 1], ["", "copy_mutate", "", ""], "axc")
    assert not checks["lengths_match"]
    assert not checks["full_inverse_matches_source"]
    assert not checks["window_target_is_source_prefix"]
