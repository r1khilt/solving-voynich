"""Known-answer-only obstructions to global label permutation, never a proposal.

Inputs are complete literal visited-only states of the same environment. This
diagnostic must run only after candidate sealing; it supplies no sampling law.
"""
from collections import Counter

from voynich.source_action_proposal import ReadingEnvironment


def label_orbit_obstructions(env, candidate, known):
    if not isinstance(env, ReadingEnvironment):
        raise ValueError('Shared literal environment required')
    for state in (candidate, known):
        env.validate(state)
        if env.selected_record(state) is not None:
            raise ValueError('Two complete literal readings required')
    source_lengths_match = tuple(map(len, candidate.texts)) == tuple(map(len, known.texts))
    unit_lengths_match = source_lengths_match and all(
        len(env.pool[candidate.key[a]]) == len(env.pool[known.key[b]])
        for x, y in zip(candidate.texts, known.texts, strict=True)
        for a, b in zip(x, y, strict=True))
    forward, reverse = {}, {}
    pattern_matches = source_lengths_match
    if source_lengths_match:
        for x, y in zip(candidate.texts, known.texts, strict=True):
            for a, b in zip(x, y, strict=True):
                if (a in forward and forward[a] != b) or (b in reverse and reverse[b] != a):
                    pattern_matches = False
                forward[a], reverse[b] = b, a
    reachable = bool(source_lengths_match and unit_lengths_match and pattern_matches)
    available = Counter(k for k in candidate.key if k >= 0)
    needed = Counter(k for k in known.key if k >= 0)
    match_ceiling = sum(min(available[k], count) for k, count in needed.items())
    return {'source_lengths_match': source_lengths_match, 'unit_lengths_match': unit_lengths_match,
        'cross_record_equality_pattern_matches': pattern_matches, 'label_orbit_reachable': reachable,
        'length_edit_lower_bound': sum(abs(len(x)-len(y)) for x, y in zip(
            candidate.texts, known.texts, strict=True)), 'binding_match_ceiling': match_ceiling,
        'known_used_rows': sum(needed.values()), 'inventory_can_cover_known_used_rows': match_ceiling == sum(
            needed.values()), 'permutation_of_used_rows': sorted(map(list, forward.items())) if reachable else None,
        'oracle_only_no_proposal_or_density': True}
