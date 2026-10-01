import itertools
import json

import pytest
import torch
from torch.nn import functional as F

from voynich.joint_key_proposal import (JointKeyProposal, KeyProposalConfig, canonicalize_records,
                                      canonical_key_indices, rebind_key_orbit, unit_pool)


def tiny(rows=3):
    torch.manual_seed(72101)
    return JointKeyProposal(KeyProposalConfig(source_rows=rows, glyphs=2, width=8, heads=2,
                            encoder_layers=2, decoder_layers=2, max_record_glyphs=12, max_records=3)).double().eval()


def test_full_joint_distribution_normalizes_and_sequential_conditioning_matches_all_legal_keys():
    model = tiny(2)
    records = torch.tensor([[[0, 1, 0], [1, 0, 2]]])
    keys = torch.tensor(list(itertools.product(range(6), repeat=2)))
    with torch.inference_mode():
        memory, mask = model.encode_records(records)
        for order in (torch.tensor([[0, 1]]), torch.tensor([[1, 0]])):
            expanded_order = order.expand(36, -1)
            score = model.joint_log_probability(memory.expand(36, -1, -1), mask.expand(36, -1), keys, expanded_order)
            assert score.exp().sum().item() == pytest.approx(1., abs=2e-12)
            first = model.decode_partial(memory, mask, torch.empty((1, 0), dtype=torch.long), order)
            for index, key in enumerate(keys):
                targets = key[order[0]]
                second = model.decode_partial(memory, mask, targets[:1][None], order)
                expected = F.log_softmax(first[0, 0], -1)[targets[0]]+F.log_softmax(second[0, 1], -1)[targets[1]]
                assert score[index].item() == pytest.approx(expected.item(), abs=2e-12)


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1]])
def test_current_and_future_supervised_labels_never_leak_into_earlier_choices(order):
    model = tiny()
    records = torch.tensor([[[0, 1, 0], [1, 0, 2]]])
    key = torch.tensor([[0, 1, 2]])
    order = torch.tensor([order])
    original = model(records, key, order)
    for step in range(3):
        changed = key.clone()
        changed[0, order[0, step]] = (changed[0, order[0, step]]+3)%6
        after = model(records, changed, order)
        assert torch.equal(original[:, :step+1], after[:, :step+1])
        if step < 2:
            assert torch.max(torch.abs(original[:, step+1:]-after[:, step+1:])).item() > 1e-5


def test_record_permutation_padding_and_batch_isolation_preserve_conditioning():
    model = tiny()
    records = torch.tensor([[[0, 1, 0], [1, 0, 2]]])
    key = torch.tensor([[0, 1, 2]])
    with torch.inference_mode():
        original = model(records, key)
        reversed_records = model(records.flip(1), key)
        padded = model(F.pad(records, (0, 4), value=2), key)
        mixed_records = torch.cat((records, torch.tensor([[[1, 1, 0], [0, 0, 1]]])), 0)
        mixed = model(mixed_records, key.expand(2, -1))
    assert torch.allclose(original, reversed_records, atol=2e-12, rtol=0)
    assert torch.allclose(original, padded, atol=2e-12, rtol=0)
    assert torch.allclose(original, mixed[:1], atol=2e-12, rtol=0)
    assert not torch.allclose(mixed[:1], mixed[1:])


def test_proposal_log_law_teacher_replay_greedy_trace_reproducibility_and_duplicate_rows_allowed():
    model = tiny()
    records = torch.tensor([[[0, 1, 0]]])
    with torch.inference_mode():
        memory, mask = model.encode_records(records)
        order = torch.tensor([[2, 0, 1]])
        first, logq = model.propose(memory, mask, samples=20, generator=torch.Generator().manual_seed(72109), order=order)
        second, again = model.propose(memory, mask, samples=20, generator=torch.Generator().manual_seed(72109), order=order)
        replay = model.joint_log_probability(memory.expand(20, -1, -1), mask.expand(20, -1), first[0], order.expand(20, -1))
        assert torch.equal(first, second) and torch.equal(logq, again)
        assert torch.allclose(logq[0], replay, atol=2e-12, rtol=0)
        trace = []
        greedy, score = model.propose(memory, mask, greedy=True, after_step=trace.append)
        assert [r["choice"] for r in trace] == [1, 2, 3]
        assert greedy.shape == (1, 1, 3) and score.shape == (1, 1)
        # Every all-equal assignment is legal and receives a finite probability.
        duplicates = torch.tensor([[i]*3 for i in range(6)])
        assert torch.isfinite(model.joint_log_probability(memory.expand(6, -1, -1), mask.expand(6, -1), duplicates)).all()
        json.dumps({"keys": first.tolist(), "logq": logq.tolist()}, allow_nan=False)


def test_training_full_row_loss_gradients_and_independent_layer_initializations():
    model = tiny().train()
    records = torch.tensor([[[0, 1, 0]]])
    key = torch.tensor([[0, 1, 2]])
    logits = model(records, key)
    loss = F.cross_entropy(logits.reshape(-1, 6), key.reshape(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.equal(model.encoder[0].linear1.weight, model.encoder[1].linear1.weight)
    assert not torch.equal(model.decoder[0].linear1.weight, model.decoder[1].linear1.weight)
    # A genuine finite-difference check on a learned decoder coefficient.
    parameter = model.output.weight
    expected = parameter.grad[0, 0].item()
    original = parameter[0, 0].item()
    with torch.no_grad():
        values = []
        for delta in (1e-5, -1e-5):
            parameter[0, 0] = original+delta
            values.append(F.cross_entropy(model(records, key).reshape(-1, 6), key.reshape(-1)).item())
        parameter[0, 0] = original
    assert (values[0]-values[1])/2e-5 == pytest.approx(expected, abs=2e-8)


def test_all_glyph_renamings_preserve_observed_canonical_inputs_and_complete_rebound_orbits():
    alphabet, records, key = "abcd", ("ababa", "ba"), ("a", "cd", "dc")
    canonical = canonicalize_records(records, alphabet)
    indices = canonical_key_indices(key, canonical)
    original_orbit = set(rebind_key_orbit(indices, canonical))
    assert len(original_orbit) == 2
    for permutation in itertools.permutations(alphabet):
        table = dict(zip(alphabet, permutation, strict=True))
        renamed = canonicalize_records(tuple("".join(table[c] for c in r) for r in records), permutation)
        assert renamed.records == canonical.records
        expected = {tuple("".join(table[c] for c in u) for u in k) for k in original_orbit}
        assert set(rebind_key_orbit(indices, renamed)) == expected
    with pytest.raises(RuntimeError, match="budget"):
        rebind_key_orbit(indices, canonical, max_permutations=1)
    # A stabilizer is deduplicated, not multiplied into reader prior mass.
    assert len(rebind_key_orbit((0, 0), canonical)) == 1
    assert len(unit_pool(6)) == 42


@pytest.mark.parametrize("fault", ["allpadding", "prefixpadding", "glyph", "length", "key", "order"])
def test_invalid_support_budget_key_or_order_is_explicit_failure(fault):
    model = tiny()
    records, keys, order = torch.tensor([[[0, 1, 2]]]), torch.tensor([[0, 1, 2]]), None
    if fault == "allpadding":
        records[:] = 2
    elif fault == "prefixpadding":
        records[:] = torch.tensor([2, 0, 1])
    elif fault == "glyph":
        records[0, 0, 0] = 3
    elif fault == "length":
        records = torch.zeros((1, 1, 13), dtype=torch.long)
    elif fault == "key":
        keys[0, 0] = 6
    else:
        order = torch.tensor([[0, 0, 1]])
    with pytest.raises(ValueError):
        model(records, keys, order)


def test_invalid_config_and_training_mode_sampling_fail():
    with pytest.raises(ValueError):
        KeyProposalConfig(width=7, heads=2)
    with pytest.raises(ValueError):
        canonicalize_records(("ax",), "ab")
    model = tiny()
    memory, mask = model.encode_records(torch.tensor([[[0]]]))
    model.train()
    with pytest.raises(ValueError, match="Eval"):
        model.propose(memory, mask)
    model.eval()
    with pytest.raises(ValueError):
        model.propose(memory, mask, greedy=True, samples=2)
