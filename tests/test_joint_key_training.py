import copy
import hashlib
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from voynich.joint_key_proposal import JointKeyProposal, KeyProposalConfig, unit_pool
from voynich.joint_key_training import (EpisodeSampler, dictionary_code, metadata_bytes,
    pack_episodes, schedule, validation)
from voynich.recurrent_latin_source import ALPHABET


def sampler():
    return EpisodeSampler([ALPHABET*30, ALPHABET[::-1]*25])


def tiny():
    torch.manual_seed(72301)
    return JointKeyProposal(KeyProposalConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1)).double().eval()


def test_boundary_literal_canonical_deterministic_episode_replay_and_no_plaintext_input():
    source = sampler()
    a = source.sample(np.random.default_rng(72303))
    b = source.sample(np.random.default_rng(72303))
    assert a == b and len(a[0]) == 2
    replay = source.make(a[2]["windows"], a[2]["raw_indices"])
    assert replay == a
    units = unit_pool(6)
    observed = []
    for spec in a[2]["windows"]:
        text = source.records[spec["segment"]][spec["start"]:spec["start"]+spec["length"]]
        observed.append("".join("ABCDEF"[i] for c in text for i in units[a[2]["raw_indices"][int(c)]]))
        assert 64 <= len(text) <= 224
    assert a[2]["cipher_sha256"] == hashlib.sha256("\n".join(observed).encode()).hexdigest()
    four, keys = pack_episodes([a])
    two, _ = pack_episodes([a], duplicate=False)
    assert four.shape == (1, 4, 448) and keys.shape == (1, 23)
    assert torch.equal(four[:, :2], two) and torch.equal(four[:, :2], four[:, 2:])
    assert not any(k in a[2] for k in ("plaintext", "source_length_input", "segmentation"))


def test_duplicate_record_pair_invariance_includes_parameter_gradients_and_whole_key_law():
    episode = sampler().sample(np.random.default_rng(72305))
    four, keys = pack_episodes([episode])
    # Short actual prefix controls numerical cost; max448 budget still enforced.
    four = four[:, :, :9]
    two = four[:, :2]
    model = tiny()
    comparison = copy.deepcopy(model)
    original, duplicated = model(two, keys), comparison(four, keys)
    assert torch.allclose(original, duplicated, atol=2e-12, rtol=0)
    F.cross_entropy(original.reshape(-1, 42), keys.reshape(-1)).backward()
    F.cross_entropy(duplicated.reshape(-1, 42), keys.reshape(-1)).backward()
    for a, b in zip(model.parameters(), comparison.parameters(), strict=True):
        assert a.grad is not None and torch.allclose(a.grad, b.grad, atol=2e-12, rtol=0)
    with torch.inference_mode():
        a, am = model.encode_records(two)
        b, bm = model.encode_records(four)
        assert torch.allclose(model.joint_log_probability(a, am, keys), model.joint_log_probability(b, bm, keys), atol=2e-12, rtol=0)


@pytest.mark.parametrize("kind", ["raw", "canonical"])
def test_explicit_validation_key_exclusion_before_return(kind):
    seed = 72307
    source = sampler()
    initial = source.sample(np.random.default_rng(seed))
    kwargs = {"forbidden_raw" if kind == "raw" else "forbidden_canonical":
              [dictionary_code(initial[2]["raw_indices"] if kind == "raw" else initial[1])]}
    restricted = EpisodeSampler([ALPHABET*30, ALPHABET[::-1]*25], **kwargs)
    accepted = restricted.sample(np.random.default_rng(seed))
    assert restricted.rejected_keys == 1
    assert accepted[2]["windows"] == initial[2]["windows"]
    assert accepted[2]["raw_indices"] != initial[2]["raw_indices"]
    assert dictionary_code(accepted[1]) != dictionary_code(initial[1])


def test_validation_uses_full_row_density_and_free_running_diagnostics_no_target_mask():
    source = EpisodeSampler(["a"*800])
    episode = source.make([{"segment": 0, "start": 0, "length": 64}]*2, [0]*23)
    model = tiny()
    score = validation(model, [episode], device="cpu", batch=1)
    assert score["rows"] == 23 and score["used_mask_is_diagnostic_only"]
    assert score["used_rows"] == 1
    assert score["mean_nats_per_row"] == pytest.approx(-math.fsum(score["joint_logq"])/23)
    records, keys = pack_episodes([episode])
    with torch.inference_mode():
        memory, mask = model.encode_records(records)
        assert score["joint_logq"][0] == pytest.approx(float(model.joint_log_probability(memory, mask, keys)), abs=2e-12)
        greedy = torch.tensor(score["free_running_greedy_keys"])
        assert score["greedy_logq"][0] == pytest.approx(float(model.joint_log_probability(memory, mask, greedy)), abs=2e-12)


def test_constant_source_lengths_are_not_a_teacher_feature_and_schedule_boundaries():
    source, rng = sampler(), np.random.default_rng(72309)
    lengths = {source.draw_window(rng)["length"] for _ in range(100)}
    assert min(lengths) >= 64 and max(lengths) <= 224 and len(lengths) > 40
    assert schedule(1) == 3e-4/200 and schedule(200) == 3e-4
    assert schedule(20000) == pytest.approx(3e-5)
    with pytest.raises(ValueError):
        schedule(0)
    with pytest.raises(ValueError):
        EpisodeSampler(["a"*223])
    with pytest.raises(ValueError):
        dictionary_code([0]*22)
    row = {"step": 1, "episodes": []}
    assert metadata_bytes(row) == metadata_bytes({"episodes": [], "step": 1})
