"""Independent alignment and split-isolation audits; no scientific final scoring."""

import itertools
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from voynich import episodic_campaign as campaign
from voynich.episodic_data import canonical_raw_probs, canonicalize, make_task, oracle_next
from voynich.episodic_models import ModelSpec
from voynich.runtime import digest


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float64))

    def forward(self, tokens):
        return self.logits[:len(tokens), :tokens.shape[1]]


class TokenTable(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float64))

    def forward(self, tokens):
        return self.logits[tokens]


def test_prepare_uses_disjoint_parameter_seeds_visible_only_arrays_and_frozen_hashes(tmp_path, monkeypatch):
    calls = []

    def recording_task(family, seed):
        task = make_task(family, seed)
        calls.append((family, seed, task["task_id"]))
        return task

    monkeypatch.setattr(campaign, "make_task", recording_task)
    config = {"steps": 3, "batch_size": 4, "context": 8, "data_seed": 128001,
              "dev_tasks_per_family": 2, "test_tasks_per_family": 2, "sequences_per_task": 3}
    manifest = campaign.prepare(config, tmp_path)
    assert len(calls) == 32 + 12 + 8 + 12
    assert len({seed for _, seed, _ in calls}) == len(calls)
    assert len({task_id for _, _, task_id in calls}) == len(calls)
    assert manifest["config"] == config and manifest["training_reads_oracles"] is False
    assert manifest["fresh_parameter_seed_range"] == [228001, 228012]
    for name, expected in manifest["sha256"].items():
        assert digest(tmp_path / name) == expected
    for regime in ("fixed", "fresh"):
        data = np.load(tmp_path / f"train_{regime}.npy")
        assert data.shape == (12, 9) and data.dtype == np.uint8
        assert data.min() >= 0 and data.max() < 4
    for split, family_count in (("development", 4), ("confirmation", 6)):
        data = np.load(tmp_path / f"{split}.npy")
        groups = np.load(tmp_path / f"{split}_groups.npy")
        tasks = json.loads((tmp_path / f"{split}_tasks.json").read_text())
        assert data.shape == (family_count * 2 * 3, 9)
        np.testing.assert_array_equal(np.bincount(groups), np.full(family_count * 2, 3))
        assert set(t["task_id"] for t in tasks) == set(manifest["task_ids"][split])
        if split == "development":
            assert {t["family"] for t in tasks} == set(campaign.FAMILIES)
    with pytest.raises(FileExistsError):
        campaign.prepare(config, tmp_path)


def test_registered_parameter_seed_ranges_have_no_overlap():
    from pathlib import Path
    root = Path(campaign.__file__).resolve().parents[2]
    config = json.loads((root / "configs/exp0012.json").read_text())
    seed = config["data_seed"]
    fixed = set(range(seed + 1000, seed + 1032))
    fresh = set(range(seed + 100000, seed + 100000 + config["steps"] * config["batch_size"]))
    development = {seed + 20000000 + i * 1000 + j for i in range(4)
                   for j in range(config["dev_tasks_per_family"])}
    confirmation = {seed + 30000000 + i * 1000 + j for i in range(6)
                    for j in range(config["test_tasks_per_family"])}
    explicit = {seed + 40000000 + i * 1000 + j for i in range(6)
                for j in range(config["explicit_tasks_per_family"])}
    pools = [fixed, fresh, development, confirmation, explicit]
    for left, right in itertools.combinations(pools, 2):
        assert left.isdisjoint(right)
    assert len(development) == 4 * config["dev_tasks_per_family"]
    assert len(confirmation) == 6 * config["test_tasks_per_family"]


@pytest.mark.parametrize("order", [0, 1, 2])
def test_online_baseline_counts_only_observed_transitions_and_aligns_targets(order):
    sequence = np.array([2, 2, 0, 2, 2, 0, 3, 1, 2])
    expected = []
    for position in range(1, len(sequence)):
        context = tuple(sequence[max(0, position - order):position]) if order else ()
        counts = np.full(4, 0.5)
        for observed in range(position):
            historical = tuple(sequence[max(0, observed - order):observed]) if order else ()
            if historical == context:
                counts[sequence[observed]] += 1
        expected.append(-math.log2(counts[sequence[position]] / counts.sum()))
    actual = campaign.online_baseline(sequence, order)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    for stop in range(2, len(sequence)):
        np.testing.assert_array_equal(actual[:stop - 1], campaign.online_baseline(sequence[:stop], order))


def test_oracle_losses_score_each_next_target_after_its_prefix():
    task = make_task("pair_parity", 128501)
    sequence = np.array([2, 2, 0, 2, 1, 3, 1])
    expected = [-math.log2(oracle_next(task, sequence[:t])[sequence[t]]) for t in range(1, len(sequence))]
    np.testing.assert_allclose(campaign.oracle_losses(task, sequence), expected, atol=1e-14)
    # The first reported loss is P(x1|x0), not P(x0), and warmup 2 starts at x3.
    assert campaign.oracle_losses(task, sequence)[2] == pytest.approx(
        -math.log2(oracle_next(task, sequence[:3])[sequence[3]]))


def test_raw_and_canonical_training_score_identical_observed_target_probabilities():
    raw = np.array([[2, 2, 0, 2, 1, 3], [0, 1, 0, 3, 0, 2]])
    _, counts, inverse = canonicalize(raw[:, :-1])
    logits = np.random.default_rng(129001).normal(size=(2, 5, 5))
    probability = canonical_raw_probs(logits, counts, inverse)
    canonical_model = FixedLogits(logits)
    raw_model = FixedLogits(np.log(probability))
    for warmup in (0, 1, 3):
        canonical_loss = campaign.loss_on(canonical_model, raw, ModelSpec(canonical=True), "cpu", warmup)
        raw_loss = campaign.loss_on(raw_model, raw, ModelSpec(canonical=False), "cpu", warmup)
        expected = -np.log(np.take_along_axis(probability, raw[:, 1:, None], -1)[..., 0])[:, warmup:].mean()
        assert float(canonical_loss) == pytest.approx(expected, abs=1e-14)
        assert float(raw_loss) == pytest.approx(expected, abs=1e-14)


def test_canonical_loss_includes_cost_of_selecting_an_unseen_raw_id():
    # At prefix [2], rank0 and NEW each have probability 1/2. There are 3 unseen IDs.
    model = FixedLogits(np.zeros((1, 1, 5)))
    raw = np.array([[2, 0]])
    loss = campaign.loss_on(model, raw, ModelSpec(canonical=True), "cpu", 0)
    assert float(loss) == pytest.approx(math.log(6))
    assert float(loss) != pytest.approx(math.log(2))  # Event-only scoring would be unfair.


def test_neural_joint_matches_independent_lexicographic_markov_enumeration():
    transition = np.array([[.1, .2, .3, .4], [.4, .2, .3, .1], [.25, .15, .5, .1], [.6, .1, .1, .2]])
    model = TokenTable(np.log(transition))
    prefixes = np.array([[0, 1], [3, 2]])
    for horizon in range(4):
        actual = campaign.neural_joint(model, ModelSpec(), prefixes, horizon, "cpu")
        expected = []
        for prefix in prefixes:
            row = []
            for future in itertools.product(range(4), repeat=horizon):
                probability, previous = 1.0, prefix[-1]
                for symbol in future:
                    probability *= transition[previous, symbol]
                    previous = symbol
                row.append(probability)
            expected.append(row)
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)
        np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-14)


def test_canonical_joint_updates_new_symbol_maps_separately_on_each_future_branch():
    logits = np.random.default_rng(129501).normal(size=(4, 5))
    model = TokenTable(logits)
    prefixes = np.array([[2, 2], [3, 1]])
    actual = campaign.neural_joint(model, ModelSpec(canonical=True), prefixes, 3, "cpu")
    expected = []
    for prefix in prefixes:
        row = []
        for future in itertools.product(range(4), repeat=3):
            probability, context = 1.0, list(prefix)
            for symbol in future:
                ranks, counts, inverse = canonicalize(np.array([context]))
                p = canonical_raw_probs(logits[ranks[0, -1]], counts[0, -1], inverse[0, -1])
                probability *= p[symbol]
                context.append(symbol)
            row.append(probability)
        expected.append(row)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-13)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-14)


@pytest.mark.parametrize("change,message", [
    ("config", "config differs"),
    ("manifest", "different data manifest"),
    ("checkpoint", "checkpoint hash mismatch"),
    ("source", "source changed"),
])
def test_final_evaluation_rejects_changed_frozen_inputs_before_reading_confirmation(tmp_path, monkeypatch,
                                                                               change, message):
    # Deliberately fake checkpoint bytes: no model fitting or scientific pool is created.
    config = {"conditions": [{"name": "test"}], "model_seeds": [99], "warmup": 1}
    manifest = {"config": config, "sha256": {}}
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    run = tmp_path / "runs" / "test-s99"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"frozen-engineering-fixture")
    source = {"src/voynich/fake.py": "frozen-source"}
    summary = {"data_manifest_sha256": digest(manifest_path),
               "best_checkpoint_sha256": digest(run / "best.pt"),
               "provenance": {"source_hashes": source}}
    if change == "config":
        config = {**config, "warmup": 2}
    elif change == "manifest":
        summary["data_manifest_sha256"] = "changed"
    elif change == "checkpoint":
        (run / "best.pt").write_bytes(b"unfrozen-engineering-fixture")
    elif change == "source":
        source = {"src/voynich/fake.py": "changed-source"}
    (run / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(campaign, "provenance", lambda: {"source_hashes": source})

    def forbid_loading(*args, **kwargs):
        raise AssertionError("a confirmation array was read before frozen inputs were verified")

    monkeypatch.setattr(campaign.np, "load", forbid_loading)
    with pytest.raises(ValueError, match=message):
        campaign.evaluate(config, tmp_path, "cpu")
