"""Independent full-dictionary sums and exact finite MH kernels."""

import itertools
import math
from fractions import Fraction as F

import pytest

from voynich.joint_reading_mh import independence_log_acceptance, joint_reading_log_weight
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def proposal_law(env, *, weighted=False):
    complete, failed = {}, F(0)
    agenda = [(env.initial, F(1))]
    while agenda:
        state, probability = agenda.pop()
        if env.selected_record(state) is None:
            assert state not in complete  # Unique deterministic action schedule.
            complete[state] = probability
            continue
        actions = env.legal_actions(state)
        if not actions:
            failed += probability
            continue
        weights = [1+(a//2+1)*(a%2+1)+sum(state.offsets) if weighted else 1 for a in actions]
        denominator = sum(weights)
        agenda.extend((env.advance(state, a), probability*F(w, denominator))
                      for a, w in zip(actions, weights, strict=True))
    assert sum(complete.values(), failed) == 1
    return complete, failed


def source_probability(texts, contextual):
    probability = F(1)
    rows = ((F(2, 3), F(1, 3)), (F(1, 4), F(3, 4)), (F(3, 4), F(1, 4)))
    for text in texts:
        probability *= F(1, 3)*F(2, 3)**len(text)  # One EOS per record.
        context = 0  # Reset each record.
        for letter in text:
            probability *= rows[context if contextual else 0][letter]
            context = letter+1
    return probability


def full_dictionary_law(env, contextual):
    """Enumerate full keys and source strings without action enumeration."""
    result = {}
    for key in itertools.product(range(len(env.pool)), repeat=env.rows):
        options = []
        for record in env.records:
            options.append([text for n in range(1, len(record)+1)
                            for text in itertools.product(range(env.rows), repeat=n)
                            if tuple(g for a in text for g in env.pool[key[a]]) == record])
        for texts in itertools.product(*options):
            result[key, texts] = source_probability(texts, contextual)*F(1, len(env.pool))**env.rows
    return result


def kernel(target, proposal):
    """Independent rational transition construction, failures on diagonal."""
    result = {}
    for x, tx in target.items():
        row = {y: min(qy, ty*proposal[x]/tx) for y, ty in target.items()
               if y != x for qy in (proposal[y],)}
        row[x] = 1-sum(row.values(), F(0))
        assert all(p >= 0 for p in row.values()) and sum(row.values(), F(0)) == 1
        result[x] = row
    return result


def finite_panel(panel, contextual, weighted):
    env = ReadingEnvironment(panel, rows=2, glyphs=2)
    proposal, failed = proposal_law(env, weighted=weighted)
    target = {s: source_probability(s.texts, contextual)*F(1, 6)**sum(k >= 0 for k in s.key)
              for s in proposal}
    full = full_dictionary_law(env, contextual)
    collapsed, full_proposal, key_proposal = {}, {}, {}
    unused_witnesses = duplicate_witnesses = 0
    for (key, texts), value in full.items():
        used = set(itertools.chain.from_iterable(texts))
        s = ReadingState(tuple(k if a in used else -1 for a, k in enumerate(key)),
                         tuple(map(len, env.records)), texts)
        collapsed[s] = collapsed.get(s, F(0))+value
        full_proposal[key, texts] = proposal[s]*F(1, 6)**(env.rows-len(used))
        key_proposal[key] = key_proposal.get(key, F(0))+full_proposal[key, texts]
        assert value/full_proposal[key, texts] == target[s]/proposal[s]
        unused_witnesses += len(used) < env.rows
        duplicate_witnesses += len(used) == 2 and key[0] == key[1]
    assert collapsed == target and sum(full_proposal.values(), failed) == 1
    total = sum(target.values(), F(0))
    transition = kernel(target, proposal)
    conditioned = kernel(target, {s: q/(1-failed) for s, q in proposal.items()})
    full_transition = kernel(full, full_proposal)
    # Actual candidates still follow full_proposal. A key marginal repeated
    # across readings is NOT a normalized joint candidate law.
    key_marginal_balance_failures = 0
    for x, tx in full.items():
        for y, ty in full.items():
            assert tx*full_transition[x][y] == ty*full_transition[y][x]
            if x != y:
                axy = min(F(1), ty*key_proposal[x[0]]/(tx*key_proposal[y[0]]))
                ayx = min(F(1), tx*key_proposal[y[0]]/(ty*key_proposal[x[0]]))
                key_marginal_balance_failures += tx*full_proposal[y]*axy != ty*full_proposal[x]*ayx
        assert sum((ty*full_transition[y][x] for y, ty in full.items()), F(0)) == tx
    epsilon = min(q*total/target[s] for s, q in proposal.items())
    assert 0 < epsilon <= 1-failed
    # Failed draws have weight zero; the complete defective proposal still
    # yields an unbiased evidence estimator. Efficiency is a SEPARATE issue.
    first_moment = sum((q*target[s]/q for s, q in proposal.items()), F(0))
    second_moment = sum((target[s]**2/q for s, q in proposal.items()), F(0))
    population_ess_fraction = total**2/second_moment
    assert first_moment == total and 0 < population_ess_fraction <= 1-failed
    max_delta, omitted_prior_failures = 0., 0
    bad_target = {s: source_probability(s.texts, contextual) for s in target}
    bad = kernel(bad_target, proposal)
    for x, tx in target.items():
        log_wx = joint_reading_log_weight(env, x,
            source_log_probability=math.log(float(source_probability(x.texts, contextual))),
            proposal_log_probability=math.log(float(proposal[x])))
        for y, ty in target.items():
            assert tx*transition[x][y] == ty*transition[y][x]
            assert tx*conditioned[x][y] == ty*conditioned[y][x]
            if x != y:
                assert conditioned[x][y] == transition[x][y]/(1-failed)
            assert transition[x][y] >= epsilon*ty/total
            omitted_prior_failures += tx*bad[x][y] != ty*bad[y][x]
            log_wy = math.log(float(ty/proposal[y]))
            expected = min(F(1), (ty/proposal[y])/(tx/proposal[x]))
            delta = abs(math.exp(independence_log_acceptance(log_wx, log_wy))-float(expected))
            assert delta <= 2e-12
            max_delta = max(max_delta, delta)
    for y, ty in target.items():
        assert sum((tx*transition[x][y] for x, tx in target.items()), F(0)) == ty
    # A key marginal sums DIFFERENT reading paths: no path-to-key density shortcut.
    key_density_witnesses = sum(full_proposal[z] != key_proposal[z[0]] for z in full_proposal)
    inventories = {tuple(sorted(env.pool[k] for k in s.key if k >= 0)) for s in target}
    return {'states': len(target), 'full_joint_states': len(full), 'balance_pairs': len(target)**2,
            'full_balance_pairs': len(full)**2, 'key_marginal_balance_failures': key_marginal_balance_failures,
            'failure_mass': str(failed), 'minorization': str(epsilon),
            'population_ess_fraction': str(population_ess_fraction),
            'unused_witnesses': unused_witnesses, 'duplicate_witnesses': duplicate_witnesses,
            'key_density_witnesses': key_density_witnesses,
            'omitted_prior_balance_failures': omitted_prior_failures,
            'distinct_used_inventories': len(inventories), 'max_acceptance_delta': max_delta}


@pytest.mark.parametrize('contextual', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_exact_joint_target_proposal_and_balance(contextual, weighted):
    value = finite_panel(((0, 0), (0,)), contextual, weighted)
    assert value['unused_witnesses'] and value['duplicate_witnesses']
    assert value['key_density_witnesses'] and value['omitted_prior_balance_failures']
    assert value['key_marginal_balance_failures']
    assert value['distinct_used_inventories'] > 1


def test_dead_ends_are_self_loops_not_refilled():
    env = ReadingEnvironment(((0, 1),), rows=1, glyphs=2)
    q, failed = proposal_law(env)
    assert failed == F(1, 2) and sum(q.values()) == F(1, 2)
    s = next(iter(q))
    assert kernel({s: F(1)}, q)[s][s] == 1
    with pytest.raises(ValueError, match='complete'):
        joint_reading_log_weight(env, env.advance(env.initial, 0),
                                 source_log_probability=-1, proposal_log_probability=-1)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True, '0', .01])
def test_invalid_density_refused(value):
    env = ReadingEnvironment(((0,),), rows=1, glyphs=1)
    s = env.advance(env.initial, 0)
    for argument in ('source_log_probability', 'proposal_log_probability'):
        args = dict(source_log_probability=-1, proposal_log_probability=-1)
        args[argument] = value
        with pytest.raises(ValueError):
            joint_reading_log_weight(env, s, **args)


@pytest.mark.parametrize('kwargs', [{'temperature': 0}, {'temperature': .5},
                                  {'temperature': True}, {'uniform_row_prior': False},
                                  {'uniform_row_prior': 1}])
def test_unsupported_targets_refused(kwargs):
    env = ReadingEnvironment(((0,),), rows=1, glyphs=1)
    with pytest.raises(ValueError):
        joint_reading_log_weight(env, env.advance(env.initial, 0),
                                 source_log_probability=-1, proposal_log_probability=-1, **kwargs)


def test_log_acceptance_extremes_and_invalid_weights():
    assert independence_log_acceptance(-10000, 10000) == 0
    assert independence_log_acceptance(10000, -10000) == -20000
    for bad in (True, math.nan, math.inf, -math.inf, '0'):
        with pytest.raises(ValueError):
            independence_log_acceptance(bad, 0)
    with pytest.raises(ArithmeticError):
        independence_log_acceptance(-1e308, 1e308)
