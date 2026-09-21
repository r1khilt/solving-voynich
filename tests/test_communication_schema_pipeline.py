from dataclasses import replace
import json

import pytest
import torch

from voynich.communication.denoiser import ConditionalDenoiser, DenoiserConfig
from voynich.communication.pipeline import (
    HELD_OUT,
    infer,
    make_episode,
    project_consistency,
    verify_candidate,
)
from voynich.communication.schema import Anchor, JointLayout, Observation


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def layout():
    return JointLayout(24, 128, 4)


def test_observation_schema_rejects_gold_and_seed_side_channels(layout):
    e = make_episode(9, "train", layout, anchor_count=2)
    record = e.observation.to_dict()
    assert Observation.from_dict(json.loads(json.dumps(record))) == e.observation
    assert str(e.world.config.seed) not in json.dumps(record)
    for forbidden in ["inverse_key", "seed", "targets", "plaintext", "grammar"]:
        with pytest.raises(ValueError):
            Observation.from_dict({**record, forbidden: []})
    with pytest.raises(ValueError):
        replace(e.observation, anchors=(Anchor(2, 2, e.observation.source_group),))


def test_structural_and_semantic_gold_compiles_for_diverse_worlds(layout):
    for split in ("train", "validation", "test"):
        for family in ("procedure", "taxonomy", "copy"):
            for i in range(5):
                e = make_episode(i, split, layout, family=family)
                h = layout.unpack(e.clean, e.observation)
                report = verify_candidate(e.observation, h)
                assert report["valid"], report
                assert report["plaintext"] == list(e.world.plaintext)
                json.dumps(report, allow_nan=False)
                if split == "train":
                    assert (e.world.config.grammar, e.world.config.morphology) not in HELD_OUT.values()
                else:
                    assert (e.world.config.grammar, e.world.config.morphology) == HELD_OUT[split]


def test_fresh_split_keys_and_fingerprints(layout):
    worlds = [make_episode(i, s, layout) for s in ("train", "validation", "test") for i in range(20)]
    assert len({e.world.config.seed for e in worlds}) == len(worlds)
    assert len({e.world.inverse_key for e in worlds}) == len(worlds)
    assert len({e.observation.digest for e in worlds}) == len(worlds)


def test_tensor_conditions_do_not_contain_hidden_metadata(layout):
    e = make_episode(7, "train", layout)
    other = replace(e.observation, document_id="different", source_group="different", split="validation")
    a, b = layout.tensors([e.observation]), layout.tensors([other])
    for name in a:
        assert torch.equal(a[name], b[name])
    present = set(e.observation.symbols)
    for symbol in range(2, 26):
        if symbol not in present:
            assert e.clean[symbol - 2] == 0
            assert a["fixed"][0, symbol - 2] == 0


def test_no_unlicensed_deletion_or_anchor_override(layout):
    e = make_episode(4, "train", layout, anchor_count=2)
    h = layout.unpack(e.clean, e.observation)
    altered = {**h, "keep": tuple(not k if i == 0 else k for i, k in enumerate(h["keep"]))}
    assert not verify_candidate(e.observation, altered)["valid"]
    tokens = e.clean.clone()
    tokens[e.observation.anchors[0].observed - 2] = 2
    with pytest.raises(ValueError):
        layout.unpack(tokens, e.observation)


def test_reencoding_is_insufficient_without_grammar(layout):
    e = make_episode(11, "train", layout, family="procedure")
    h = layout.unpack(e.clean, e.observation)
    wrong = {**h, "boundaries": tuple(False for _ in h["boundaries"])}
    report = verify_candidate(e.observation, wrong)
    assert report["exact_reencoding"] and not report["valid"]
    assert "declared_grammar_rejected" in report["issues"]


def test_projection_logs_repairs_and_keeps_anchors(layout):
    e = make_episode(12, "train", layout, anchor_count=2)
    tokens = e.clean.clone()
    tokens[layout.keep_slice.start] = 5 - int(tokens[layout.keep_slice.start])
    projected, changes = project_consistency(tokens, e.observation, layout)
    assert changes and torch.equal(projected, e.clean)


def test_joint_sampler_integrates_and_abstains_without_semantic_support(layout):
    torch.manual_seed(3)
    model = ConditionalDenoiser(
        DenoiserConfig(
            layout.vocab_size,
            layout.condition_vocab_size,
            layout.latent_length,
            layout.condition_length,
            width=16,
            layers=1,
            heads=2,
        )
    )
    e = make_episode(0, "validation", layout, anchor_count=2)
    report = infer(model, e.observation, layout, candidates=2, steps=3, seed=8)
    assert report["trace_steps"] == 3
    assert report["status"] in ("abstain", "candidate_hypotheses")
    assert len(report["candidates"]) > 0
    for candidate in report["candidates"]:
        for anchor in e.observation.anchors:
            assert candidate["hypothesis"]["inverse_key"][anchor.observed - 2] == anchor.canonical
    json.dumps(report, allow_nan=False)


def test_backend_final_holdout_guard_requires_explicit_boolean_override(layout):
    torch.manual_seed(3)
    model = ConditionalDenoiser(
        DenoiserConfig(
            layout.vocab_size,
            layout.condition_vocab_size,
            layout.latent_length,
            layout.condition_length,
            width=16,
            layers=1,
            heads=2,
        )
    )
    # Relabel a validation fixture only; do not generate or score a final holdout.
    observation = make_episode(7, "validation", layout, anchor_count=2).observation
    relabeled = replace(observation, split="test")
    calls = []
    hook = model.register_forward_pre_hook(lambda _model, _args: calls.append(True))
    try:
        with pytest.raises(ValueError, match="Final holdout inference requires"):
            infer(model, relabeled, layout, candidates=1, steps=1, compiler_budget=0)
        for invalid in (1, 0, "true", None):
            with pytest.raises(ValueError, match="allow_test must be a boolean"):
                infer(model, relabeled, layout, allow_test=invalid)
        assert calls == []
        ordinary = infer(model, observation, layout, candidates=1, steps=1, compiler_budget=0, seed=8)
        explicit = infer(
            model, relabeled, layout, candidates=1, steps=1, compiler_budget=0, seed=8, allow_test=True,
        )
        assert calls
        assert explicit["candidates"] == ordinary["candidates"]
    finally:
        hook.remove()


def test_bad_capacities_and_truncation_refused(layout):
    e = make_episode(0, "train", layout)
    with pytest.raises(ValueError):
        JointLayout(24, 2).tensors([e.observation])
    with pytest.raises(ValueError):
        Observation("x", (1, 2), 24, source_group="s")
    with pytest.raises(ValueError):
        make_episode(1_000_000, "train", layout)
    with pytest.raises(ValueError):
        layout.pack(e.world.inverse_key, [0.0], [1], grammar="SOV", morphology="none", family="copy")
