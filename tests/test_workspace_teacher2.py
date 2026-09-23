"""CPU-only generator, scoring and numerical checks; no optimizer run."""

import gzip
import json
import random

import pytest
import torch

from voynich.workspace.teacher2_tasks import (
    KEYS, OBJECTS, evaluation_suite, family_signatures, sample_episode,
    sample_factorial, schedule, symbolic_oracle, training_batch,
)
from voynich.workspace.teacher2_train import (
    Config, decide, evaluate, model, numerical_qualification, task_loss,
)


def test_fresh_stream_partitions_and_matched_null_inputs():
    for step in (0, 1499, 1500, 2999, 3000, 4999):
        base, base_y = training_batch(777 + step, 80, arm="baseline", step=step)
        curriculum, curriculum_y = training_batch(777 + step, 80, arm="curriculum", step=step)
        null, null_y = training_batch(777 + step, 80, arm="null", step=step)
        assert curriculum == null
        assert all(ep.f_partition == ep.g_partition == "train" for ep in base + curriculum)
        assert all(symbolic_oracle(ep.tokens) == ep.answer == y
                   for ep, y in zip(base, base_y, strict=True))
        assert all(symbolic_oracle(ep.tokens) == ep.answer == y
                   for ep, y in zip(curriculum, curriculum_y, strict=True))
        assert all(y == ep.answer for ep, y in zip(null, null_y, strict=True)
                   if ep.task != "composed")
    assert schedule("baseline", 0) == schedule("baseline", 4999)
    assert dict(schedule("curriculum", 0)).get("composed", 0) == 0
    assert dict(schedule("curriculum", 4999))["composed"] == .60
    with pytest.raises(ValueError):
        schedule("curriculum", 5000)


@pytest.mark.parametrize("f,g", [("train", "train"), ("holdout", "train"),
                                      ("train", "holdout"), ("holdout", "holdout")])
def test_all_partitions_and_oracle(f, g):
    ep = sample_episode(random.Random(111), f_partition=f, g_partition=g, task="composed")
    assert (ep.f_partition, ep.g_partition) == (f, g)
    assert symbolic_oracle(ep.tokens) == ep.answer


def test_query_reversals_and_factorial_are_nontrivial():
    suite = evaluation_suite(62311, 16)
    for name in ("first_hop_pairs_holdout", "direct_pairs_holdout"):
        rows = suite[name]
        assert len(rows) == 16
        for i in range(0, len(rows), 2):
            a, b = rows[i:i + 2]
            assert a.tokens[:12] == b.tokens[:12]
            assert a.tokens[12] != b.tokens[12]
            assert a.answer != b.answer
            assert family_signatures(a) == family_signatures(b)
    quartet = sample_factorial(random.Random(401))
    assert len({ep.answer for ep in quartet}) == 4
    assert all(ep.f_partition == ep.g_partition == "holdout" for ep in quartet)
    assert quartet[0].tokens[6:] == quartet[1].tokens[6:]
    assert quartet[0].tokens[:6] == quartet[2].tokens[:6]
    assert all(symbolic_oracle(ep.tokens) == ep.answer for ep in quartet)


def test_teacher2_family_holdout_and_prediction_archive(tmp_path):
    suite = evaluation_suite(62311, 8)
    seen_f, seen_g = set(), set()
    for seed in range(100):
        episodes, _ = training_batch(50000 + seed, 32, arm="curriculum", step=3800)
        for ep in episodes:
            f, g = family_signatures(ep)
            seen_f.add(f)
            seen_g.add(g)
    heldout = suite["composed_holdout_holdout"] + suite["factorial"]
    assert all(family_signatures(ep)[0] not in seen_f and
               family_signatures(ep)[1] not in seen_g for ep in heldout)
    scores, digest = evaluate(model(), suite, "cpu", 0, "curriculum", tmp_path)
    archive = tmp_path / "predictions-rep0-curriculum.json.gz"
    assert len(digest) == 64 and archive.is_file()
    saved = json.loads(gzip.decompress(archive.read_bytes()))
    assert saved["factorial"][0]["answer"] == suite["factorial"][0].answer
    assert saved["factorial"][0]["f_family"] == [list(part) for part in
                                                        family_signatures(suite["factorial"][0])[0]]
    assert scores["factorial"]["total_groups"] == 4
    assert scores["first_hop_pairs_holdout"]["total_groups"] == 4


def test_mixed_key_object_loss_and_cpu_numerical_gate():
    first = sample_episode(random.Random(31), f_partition="train", g_partition="train",
                           task="first_hop")
    composed = sample_episode(random.Random(32), f_partition="train", g_partition="train",
                              task="composed")
    logits = torch.zeros(2, 45, requires_grad=True)
    loss = task_loss(logits, [first, composed], [first.answer, composed.answer], "cpu")
    assert abs(float(loss.item()) - torch.log(torch.tensor(12.0)).item()) < 1e-6
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert first.answer in KEYS and composed.answer in OBJECTS
    assert model().parameter_count == 668032
    numerical = numerical_qualification("cpu")
    assert numerical["max_cache_logit_error"] < .002
    Config().validate()
    with pytest.raises(ValueError):
        Config(max_seconds=1801).validate()


def test_decision_requires_both_replicates_and_matched_advantage():
    cells = ("first_hop_holdout", "first_hop_pairs_holdout", "direct_train",
             "direct_holdout", "direct_pairs_holdout", "composed_holdout_holdout",
             "composed_holdout_train", "composed_train_holdout", "factorial", "copy")
    positive = {cell: {"accuracy": .99, "group_accuracy": .9} for cell in cells}
    baseline = {cell: {"accuracy": .5, "group_accuracy": .4} for cell in cells}
    null = {cell: {"accuracy": .2, "group_accuracy": .05} for cell in cells}
    scores = {str(rep): {"curriculum": positive, "baseline": baseline, "null": null}
              for rep in range(2)}
    assert decide(scores)["verdict"] == "qualified"
    scores["1"]["curriculum"] = {**positive, "composed_holdout_holdout": {"accuracy": .70}}
    assert decide(scores)["verdict"] == "not_qualified"
