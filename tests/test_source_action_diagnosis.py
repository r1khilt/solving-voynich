import pytest
import torch

from dataclasses import replace
from voynich.source_action_diagnosis import action_groups, binding_packet, first_deviation
from voynich.source_action_proposal import ReadingEnvironment
from tests.test_source_action_proposal import tiny_model
from scripts.run_source_action_diag001 import score_logits


def fixture():
    env = ReadingEnvironment(((0, 1, 0), (1, 0)), rows=2, glyphs=2)
    trace = env.teaching_trace(((0, 1, 0), (1, 0)), (0, 1))
    model = tiny_model().eval()
    return model, env, trace, model.pack([env], [trace])


@pytest.mark.parametrize('mode', ['sham', 'erase', 'rotate'])
def test_only_binding_features_change(mode):
    model, _, _, packed = fixture()
    altered = binding_packet(packed, rows=2, glyphs=2, mode=mode)
    assert torch.equal(packed[1][:, 0], altered[1][:, 0])
    for i in range(8):
        if i != 1:
            assert altered[i] is packed[i]
    if mode == 'sham':
        assert torch.equal(packed[1], altered[1])
        with torch.no_grad():
            assert torch.equal(model(packed), model(altered))
    if mode == 'rotate':
        units, changed = packed[1]%7, altered[1]%7
        assert torch.equal(units > 0, changed > 0)
        assert torch.equal(units.sort(-1).values, changed.sort(-1).values)


def test_binding_off_control_is_insensitive_to_all_interventions():
    model, _, _, packed = fixture()
    model.config = replace(model.config, binding_input=False)
    with torch.no_grad():
        expected = model(packed)
        for mode in ('sham', 'erase', 'rotate'):
            assert torch.equal(expected, model(binding_packet(packed, rows=2, glyphs=2, mode=mode)))


def test_padding_slots_untouched_and_illegal_row_ids_refused():
    model, env, trace, _ = fixture()
    shorter = ReadingEnvironment(((0,), (1,)), rows=2, glyphs=2)
    short_trace = shorter.teaching_trace(((0,), (1,)), (0, 1))
    packed = model.pack([env, shorter], [trace, short_trace])
    changed = binding_packet(packed, rows=2, glyphs=2, mode='rotate')
    assert torch.equal(changed[1][~packed[-2]], packed[1][~packed[-2]])
    bad = packed[1].clone()
    bad[0, 0, 1] = 0
    with pytest.raises(ValueError, match='identities'):
        binding_packet((packed[0], bad, *packed[2:]), rows=2, glyphs=2, mode='erase')


def test_first_deviation_distinguishes_wrong_inventory_from_duplicate_label():
    env = ReadingEnvironment(((0, 1),), rows=2, glyphs=2)
    trace = env.teaching_trace(((0,),), (3, 0))
    wrong = first_deviation(env, trace, {'actions': (0,)}, (3, 0))
    assert wrong['introduced_wrong_dictionary_binding'] and wrong['wrong_unit_length']
    dup = ReadingEnvironment(((0,),), rows=2, glyphs=2)
    truth = dup.teaching_trace(((0,),), (0, 0))
    wrong = first_deviation(dup, truth, {'actions': (2,)}, (0, 0))
    assert wrong['wrong_source_row'] and not wrong['introduced_wrong_dictionary_binding']
    assert first_deviation(dup, truth, {'actions': truth[1]}, (0, 0))['exact_actions']
    with pytest.raises(ValueError):
        first_deviation(dup, truth, {'actions': ()}, (0, 0))
    assert action_groups(truth) == ('first_binding',)


def test_unsupported_intervention_refused():
    _, _, _, packed = fixture()
    with pytest.raises(ValueError):
        binding_packet(packed, rows=2, glyphs=2, mode='repair')


def test_score_groups_use_fixed_targets_and_count_forced_legal_actions():
    logits = torch.tensor([[0., 0., -torch.inf], [1., -torch.inf, -torch.inf],
                           [0., 2., -torch.inf]], dtype=torch.float64)
    targets = torch.tensor([1, 0, 1])
    result, logq, correct = score_logits(logits, targets, ('first_binding', 'reuse', 'reuse'))
    assert correct == [False, True, True]  # Fixed lowest-index argmax tie.
    assert result['groups']['first_binding']['actions'] == 1
    assert result['groups']['reuse']['actions'] == 2
    assert result['groups']['reuse']['correct_argmax'] == 2
    assert result['groups']['reuse']['forced_actions'] == 1
    assert result['groups']['reuse']['nonforced_correct_argmax'] == 1
    assert logq[1] == 0  # Deterministic legal-mask assistance is explicitly retained.
    assert abs(result['whole_path_nll']+sum(logq)) < 1e-14
