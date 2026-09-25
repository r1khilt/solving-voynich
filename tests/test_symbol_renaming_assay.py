"""Coupled generator and representation invariance controls for EXP-0031."""

import numpy as np
import pytest

from voynich.latent_recovery import CIPHER_POOL, copy_aware_features, insert_nulls, recon_accuracy
from voynich.symbol_renaming_assay import relabel_text


@pytest.mark.parametrize("family", ("random_char", "periodic", "copy_mutate"))
@pytest.mark.parametrize("seed", (0, 1, 17, 42))
def test_corrected_channel_commutes_with_symbol_renaming(family: str, seed: int) -> None:
    alphabet = CIPHER_POOL[:36]
    ciphered = "αβγ δεζη αβγ θικλ " * 6
    row_index, rep = 13, 7
    original, mask, tags = insert_nulls(
        ciphered, np.random.default_rng(seed), 0.30, alphabet, (family,)
    )
    renamed, renamed_mask, renamed_tags = insert_nulls(
        relabel_text(ciphered, row_index, rep),
        np.random.default_rng(seed),
        0.30,
        relabel_text(alphabet, row_index, rep),
        (family,),
    )
    assert renamed == relabel_text(original, row_index, rep)
    assert renamed_mask == mask
    assert renamed_tags == tags
    assert np.array_equal(copy_aware_features(original), copy_aware_features(renamed))
    predicted_mask = np.random.default_rng(1000 + seed).integers(0, 2, len(mask))
    assert recon_accuracy(
        "".join(ch for ch, keep in zip(original, mask, strict=True) if keep),
        original,
        predicted_mask,
    ) == recon_accuracy(
        "".join(ch for ch, keep in zip(renamed, mask, strict=True) if keep),
        renamed,
        predicted_mask,
    )


def test_relabel_is_full_pool_bijection_and_preserves_spaces() -> None:
    text = CIPHER_POOL + " " + CIPHER_POOL[:10]
    renamed = relabel_text(text, 3, 1)
    assert len(set(renamed[: len(CIPHER_POOL)])) == len(CIPHER_POOL)
    assert renamed[len(CIPHER_POOL)] == " "
    assert renamed[-10:] == renamed[:10]
    assert relabel_text(text, 3, 0) == text
    assert np.array_equal(copy_aware_features(text), copy_aware_features(renamed))
