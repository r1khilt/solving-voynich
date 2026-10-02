import itertools
import math
from fractions import Fraction

import pytest
import torch

from voynich.source_action_proposal import ReadingEnvironment, ReadingState, SourceActionConfig, SourceActionProposal


def tiny_model():
    torch.manual_seed(9181)
    return SourceActionProposal(SourceActionConfig(rows=2, glyphs=2, width=16, heads=2,
        encoder_layers=2, decoder_layers=2, max_glyphs=8, max_records=2)).double()


def leaves(env, state=None):
    state = env.initial if state is None else state
    if env.selected_record(state) is None:
        yield state
    else:
        for action in env.legal_actions(state):
            yield from leaves(env, env.advance(state, action))


def test_exact_shared_key_evidence_and_unique_action_order():
    records = [r for n in (1, 2) for r in itertools.product(range(2), repeat=n)]
    for panel in itertools.product(records, repeat=2):
        env = ReadingEnvironment(panel, rows=2, glyphs=2)
        terminal = list(leaves(env))
        assert len({(s.key, s.texts) for s in terminal}) == len(terminal)
        trace_evidence = sum((Fraction(1, 4)**2*Fraction(3, 8)**sum(map(len, s.texts))
                              *Fraction(1, 6)**sum(k >= 0 for k in s.key) for s in terminal), Fraction(0))
        # Independently enumerate FULL dictionaries and source strings; unused
        # completions are integrated by enumeration, not by the trace formula.
        full_evidence = Fraction(0)
        for key in itertools.product(range(6), repeat=2):
            options = []
            for record in panel:
                options.append([text for n in range(1, len(record)+1)
                    for text in itertools.product(range(2), repeat=n)
                    if tuple(g for a in text for g in env.pool[key[a]]) == record])
            for texts in itertools.product(*options):
                full_evidence += Fraction(1, 36)*Fraction(1, 4)**2*Fraction(3, 8)**sum(map(len, texts))
                snapshots, actions, state = env.teaching_trace(texts, key)
                assert state.texts == texts and len(actions) == sum(map(len, texts))
                assert snapshots[0] == env.initial
        assert full_evidence == trace_evidence


def test_binding_memory_not_injective_and_no_future_slots():
    env = ReadingEnvironment(((0, 0), (0,)), rows=3, glyphs=2)
    trace = env.teaching_trace(((0, 1), (1,)), (0, 0, 5))
    assert trace[2].key == (0, 0, -1)
    assert trace[0][0].key == (-1, -1, -1)
    assert trace[0][1].key == (0, -1, -1)
    with pytest.raises(ValueError, match="future"):
        env.legal_actions(ReadingState((0, -1, 5), (1, 0), ((0,), ())))
    with pytest.raises(ValueError, match="literally"):
        env.validate(ReadingState((1, -1, -1), (1, 0), ((0,), ())))


def test_dead_end_is_possible_and_not_repaired():
    env = ReadingEnvironment(((0, 1),), rows=1, glyphs=2)
    state = env.advance(env.initial, 0)
    assert env.selected_record(state) == 0 and env.legal_actions(state) == ()
    with pytest.raises(ValueError):
        env.advance(state, 0)


def test_action_proposal_mass_includes_failed_prefixes():
    def mass(env, state):
        if env.selected_record(state) is None:
            return Fraction(1), Fraction(0)
        actions = env.legal_actions(state)
        if not actions:
            return Fraction(0), Fraction(1)
        children = [mass(env, env.advance(state, a)) for a in actions]
        return tuple(sum((p[i] for p in children), Fraction(0))/len(actions) for i in (0, 1))
    failed = 0
    for n in (1, 2, 3):
        for record in itertools.product(range(2), repeat=n):
            env = ReadingEnvironment((record,), rows=1, glyphs=2)
            success, failure = mass(env, env.initial)
            assert success+failure == 1
            failed += failure > 0
    assert failed > 0


def test_future_targets_and_unused_gold_key_cannot_change_earlier_logits():
    model = tiny_model().eval()
    env = ReadingEnvironment(((0, 0, 0),), rows=2, glyphs=2)
    # Different legal continuations, same observation and first action.
    a = env.teaching_trace(((0, 0, 0),), (0, 5))
    b = env.teaching_trace(((0, 1, 1),), (0, 0))
    pa, pb = model.pack([env], [a]), model.pack([env], [b])
    assert all(torch.equal(x[:, :2], y[:, :2]) for x, y in zip(pa[1:-1], pb[1:-1], strict=True))
    la, lb = model(pa), model(pb)
    torch.testing.assert_close(la[:, :2], lb[:, :2], rtol=0, atol=1e-12)
    c = env.teaching_trace(((0, 0, 0),), (0, 4))
    pc = model.pack([env], [c])
    assert all(torch.equal(x, y) for x, y in zip(pa, pc, strict=True))
    forged = (a[0][:-1]+(b[0][-1],), a[1], a[2])
    with pytest.raises(ValueError, match="preceding"):
        model.pack([env], [forged])


def test_whole_path_loss_padding_gradients_and_legal_normalization():
    model = tiny_model()
    a = ReadingEnvironment(((0, 0, 0),), rows=2, glyphs=2)
    b = ReadingEnvironment(((1,),), rows=2, glyphs=2)
    ta, tb = a.teaching_trace(((0, 1, 0),), (0, 0)), b.teaching_trace(((1,),), (0, 1))
    packed = model.pack([a, b], [ta, tb])
    loss = model.loss(packed)
    separate = (model.loss(model.pack([a], [ta]))+model.loss(model.pack([b], [tb])))/2
    torch.testing.assert_close(loss, separate, rtol=0, atol=1e-12)
    probs = model(packed).softmax(-1)
    assert torch.equal(probs[~packed[5]], torch.zeros_like(probs[~packed[5]]))
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.binding.weight.grad.abs().sum() > 0 and model.glyph.weight.grad.abs().sum() > 0


def test_rollout_density_matches_teacher_trace_or_reports_failed_prefix():
    model = tiny_model()
    env = ReadingEnvironment(((0, 1, 0), (1,)), rows=2, glyphs=2)
    for seed in range(8):
        result = model.propose(env, seed=seed, temperature=.8)
        assert model.training
        state, snapshots = env.initial, []
        for action in result['actions']:
            snapshots.append(state)
            state = env.advance(state, action)
        assert state == result['state']
        if result['status'] == 'complete_path':
            trace = (tuple(snapshots), result['actions'], state)
            packed = model.pack([env], [trace])
            logits = model(packed)[0].detach().double()/.8
            expected = logits.log_softmax(-1)[range(len(result['actions'])), list(result['actions'])].sum().item()
            assert abs(expected-result['path_log_probability']) < 1e-12
        else:
            assert result['status'] == 'dead_end' and env.legal_actions(state) == ()
        assert math.isfinite(result['path_log_probability'])


@pytest.mark.parametrize('bad', [((True,),), ((2,),), ((),)])
def test_invalid_observation(bad):
    with pytest.raises(ValueError):
        ReadingEnvironment(bad, rows=2, glyphs=2)
