"""Symmetry and model-capacity controls for the registered EXP-0032 comparison."""

import numpy as np
import torch

from voynich.exp0032_model import ContextMaskModel, canonical_token_ids, raw_token_ids
from voynich.latent_recovery import CIPHER_POOL, copy_aware_features


def test_canonical_encoding_ignores_all_cipher_glyph_names() -> None:
    rng = np.random.default_rng(70)
    alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
    text = alphabet[:12] + " " + alphabet[5:12] + " " + alphabet[:12] + " "
    for seed in range(20):
        permutation = np.random.default_rng(seed).permutation(len(CIPHER_POOL))
        mapping = {ch: CIPHER_POOL[int(permutation[i])] for i, ch in enumerate(CIPHER_POOL)}
        renamed = "".join(mapping.get(ch, ch) for ch in text)
        assert canonical_token_ids(text) == canonical_token_ids(renamed)
        assert np.array_equal(copy_aware_features(text), copy_aware_features(renamed))
        assert raw_token_ids(text) != raw_token_ids(renamed)


def test_canonical_preserves_known_plaintext_and_space_identity() -> None:
    assert canonical_token_ids("ab ba") == raw_token_ids("ab ba")
    assert canonical_token_ids("αβ α") != canonical_token_ids("αα β")


def test_matched_large_model_has_deterministic_invariant_eval() -> None:
    torch.manual_seed(123)
    model = ContextMaskModel().eval()
    assert 10_000_000 < sum(p.numel() for p in model.parameters()) < 12_000_000
    original = (CIPHER_POOL[:16] + " " + CIPHER_POOL[5:15] + " ") * 5
    original = original[:128].ljust(128)
    permutation = np.random.default_rng(10).permutation(len(CIPHER_POOL))
    mapping = {ch: CIPHER_POOL[int(permutation[i])] for i, ch in enumerate(CIPHER_POOL)}
    changed = "".join(mapping.get(ch, ch) for ch in original)
    x0 = torch.tensor([canonical_token_ids(original)])
    x1 = torch.tensor([canonical_token_ids(changed)])
    f0 = torch.tensor(copy_aware_features(original)[None])
    f1 = torch.tensor(copy_aware_features(changed)[None])
    with torch.no_grad():
        y0, w0 = model(x0, f0)
        y1, w1 = model(x1, f1)
    assert torch.equal(y0, y1)
    assert torch.equal(w0, w1)
