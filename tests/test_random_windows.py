"""Random contexts cover old boundaries without changing deterministic scoring."""

import json

import pytest
import torch

from voynich.runtime import PageWindows
from voynich.tokenizer import EVATokenizer
from voynich.train import train


def corpus(tmp_path):
    texts = ["abcdefghijklmnop", "x"]
    tokenizer = EVATokenizer.fit(texts)
    tokenizer.save(tmp_path / "tokenizer.json")
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text("".join(
            json.dumps({"page_id": str(i), "split": split, "text": text}) + "\n"
            for i, text in enumerate(texts)))
    return tokenizer


def test_random_offsets_cover_every_full_start_including_last_and_cross_old_boundaries(tmp_path):
    corpus(tmp_path)
    data = PageWindows(tmp_path, "train", 4)
    windows = data.sample_random_windows(2000, torch.Generator().manual_seed(431))
    long = [w for w in windows if w.page_id == "0"]
    last_start = len(data.sequences["0"]) - 1 - 4
    assert {w.start for w in long} == set(range(last_start + 1))
    # The input at the old start 4 is sometimes given its preceding context.
    assert any(w.start < 4 < w.start + w.length for w in long)
    assert all(w.sequence is data.sequences[w.page_id] for w in windows)
    short = [w for w in windows if w.page_id == "1"]
    assert short and all(w.start == 0 and w.length == 2 for w in short)


def test_random_offsets_rng_replay_and_batch_alignment(tmp_path):
    tokenizer = corpus(tmp_path)
    data = PageWindows(tmp_path, "train", 4)
    rng = torch.Generator().manual_seed(81)
    saved = rng.get_state()
    windows = data.sample_random_windows(32, rng)
    rng.set_state(saved)
    x, targets, pages = data.random_offset_batch(32, "cpu", rng, horizons=(1, 2, 3))
    for row, w in enumerate(windows):
        assert pages[row] == w.page_id
        assert x[row, :w.length].tolist() == w.sequence[w.start:w.start+w.length]
        assert x[row, w.length:].eq(tokenizer.pad_id).all()
        for h in targets:
            for t in range(4):
                j = w.start + t + h
                expected = w.sequence[j] if t < w.length and j < len(w.sequence) else -100
                assert targets[h][row, t].item() == expected
    assert any(targets[1].eq(tokenizer.eos_id).flatten())


def test_page_sampling_uses_length_weights_independent_of_context(tmp_path):
    corpus(tmp_path)
    a, b = PageWindows(tmp_path, "train", 2), PageWindows(tmp_path, "train", 12)
    assert torch.equal(a._page_weights, b._page_weights)
    assert a._page_weights.tolist() == [17, 2]


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_sizes_rejected(tmp_path, size):
    corpus(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        PageWindows(tmp_path, "train", 4).sample_random_windows(size, torch.Generator())


def test_random_sampling_never_changes_validation_or_fixed_windows(tmp_path):
    corpus(tmp_path)
    data = PageWindows(tmp_path, "train", 4)
    before = data.batch(range(len(data.windows)), "cpu")
    data.random_offset_batch(32, "cpu", torch.Generator().manual_seed(14))
    after = data.batch(range(len(data.windows)), "cpu")
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1][1], after[1][1])
    with pytest.raises(ValueError, match="training-only"):
        PageWindows(tmp_path, "validation", 4).sample_random_windows(2, torch.Generator())


def test_unknown_sampler_rejected_before_data_or_output_access(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"training": {"window_sampling": "invented"}}))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="window_sampling"):
        train(config, tmp_path / "absent", out, threads=1)
    assert not out.exists()
