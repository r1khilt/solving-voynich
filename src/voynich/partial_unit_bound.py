"""Path-marginal relaxation for partial deterministic emission dictionaries.

Unassigned rows may select a legal unit at EACH occurrence, with coefficient
one. This is an upper-bound operator, not a normalized probabilistic channel.
"""
import math


def family(partial, pool):
    partial, pool = tuple(partial), tuple(pool)
    if (not partial or not pool or len(set(pool)) != len(pool)
            or any(not isinstance(u, str) or not u for u in pool)
            or any(u is not None and u not in pool for u in partial)):
        raise ValueError('Invalid partial dictionary or legal unit pool')
    return partial, pool


def minimum_code_bits(partial, pool, context):
    partial, pool = family(partial, pool)

    def width(cardinality):
        if type(cardinality) is not int or cardinality < 1:
            raise ValueError('Invalid coding cardinality')
        return (cardinality-1).bit_length()

    glyphs, alphabet = tuple(context['glyph_alphabet']), tuple(context['source_alphabet'])
    if (len(alphabet) != len(partial) or not glyphs or not alphabet
            or len(set(glyphs)) != len(glyphs) or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or len(c) != 1 for c in (*alphabet, *glyphs))
            or type(context['max_emission_length']) is not int
            or any(set(u)-set(glyphs) or len(u) > context['max_emission_length'] for u in pool)):
        raise ValueError('Coding context and declared family differ')
    shortest = min(map(len, pool))
    return (width(context.get('source_count', 1)) + width(context['max_states'])
            + len(partial)*(width(context['max_alternatives']) + width(context['max_emission_length']))
            + sum(shortest if unit is None else len(unit) for unit in partial)*width(len(glyphs)))


def relaxed_record(source, partial, pool, observed, rho, *, max_nodes=500_000, max_edges=2_000_000):
    partial, pool = family(partial, pool)
    if (len(partial) != len(source.alphabet) or not isinstance(observed, str)
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or any(type(v) is not int or v < 1 for v in (max_nodes, max_edges))):
        raise ValueError('Invalid relaxed inference settings')
    allowed = [(u,) if u is not None else pool for u in partial]
    graph = [{} for _ in range(len(observed)+1)]
    graph[0][source.state('')] = 0.
    nodes, edges = 1, 0
    transitions, checked_rows = {}, set()
    cont = math.log1p(-rho)
    for offset in range(len(observed)):
        matches = [(letter, offset+len(unit)) for letter, options in enumerate(allowed)
                   for unit in options if observed.startswith(unit, offset)]
        for state, prefix in graph[offset].items():
            row = source.probabilities[state]
            if state not in checked_rows:
                if (len(row) != len(partial) or any(not math.isfinite(p) or p < 0 or p > 1 for p in row)
                        or abs(math.fsum(row)-1) > 1e-12):
                    raise ValueError('Source probability row is not normalized')
                checked_rows.add(state)
            for letter, end in matches:
                if row[letter] == 0:
                    continue
                key = state, letter
                if key not in transitions:
                    transitions[key] = source.step(state, letter), cont+math.log(row[letter])
                following, weight = transitions[key]
                edges += 1
                if edges > max_edges:
                    raise RuntimeError('Relaxed edge cap; no pruning')
                score = prefix + weight
                if following not in graph[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('Relaxed node cap; no pruning')
                    graph[end][following] = score
                else:
                    other = graph[end][following]
                    high, low = max(score, other), min(score, other)
                    graph[end][following] = high+math.log1p(math.exp(low-high))
        graph[offset].clear()
    values = list(graph[-1].values())
    high = max(values, default=-math.inf)
    total = high if high == -math.inf else high+math.log(math.fsum(math.exp(v-high) for v in values))+math.log(rho)
    return {'log_likelihood_upper_bound': None if total == -math.inf else total,
            'nodes': nodes, 'edges': edges, 'normalized_channel': False}


def partial_fit_bound(source, partial, pool, records, rho, context, *, max_nodes=500_000, max_edges=2_000_000):
    partial, pool = family(partial, pool)
    records = tuple(records)
    if not records:
        raise ValueError('At least one record required for fitting bound')
    code = minimum_code_bits(partial, pool, context)
    rows = [relaxed_record(source, partial, pool, record, rho, max_nodes=max_nodes, max_edges=max_edges)
            for record in records]
    logs = [row['log_likelihood_upper_bound'] for row in rows]
    total = None if None in logs else math.fsum(logs)-code*math.log(2)
    return {'log_objective_upper_bound': total, 'minimum_model_bits': code, 'records': rows,
            'normalized_channel': False, 'interval_arithmetic_certified': False}
