"""Markov-specific shared-key DAG: sum histories only at sufficient states.

Literal observations, iid once-binding key prior, geometric source lengths.
Beam/guide ordering can lose support; they never change original edge weights.
This reference implementation is for correctness and bounded engineering.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from voynich.guided_source_particles import IidSuffixGuide
from voynich.source_key_particles import validate_source
from voynich.source_prefix_inverse import logadd, logsum


@dataclass(frozen=True, order=True)
class State:
    offsets: tuple
    contexts: tuple
    key: tuple
    path_id: int = 0


def selected_record(offsets, lengths, schedule):
    unfinished = [r for r, (i, n) in enumerate(zip(offsets, lengths, strict=True)) if i < n]
    if not unfinished:
        return None
    if schedule == "sequential":
        return unfinished[0]
    # Integer cross products avoid floating ties in relative glyph progress.
    best = unfinished[0]
    for candidate in unfinished[1:]:
        if offsets[candidate]*lengths[best] < offsets[best]*lengths[candidate]:
            best = candidate
    return best


def remaining_log_bound(state, lengths, rho):
    """Sum the entire possible geometric length interval, not a single EOS."""
    value, continuation = 0., math.log1p(-rho)
    for offset, length in zip(state.offsets, lengths, strict=True):
        remaining = length-offset
        if remaining:
            low, high = (remaining+1)//2, remaining
            value += low*continuation+math.log(-math.expm1((high-low+1)*continuation))
    return value


def search_state_lattice(cipher, probabilities, transitions, *, glyphs=6, rho=1/225,
                         width=4096, max_expanded=1_000_000, max_generated=40_000_000,
                         max_active=500_000, max_terminals=4096, merge=True,
                         schedule="balanced", guidance="none", guide_prewidth=None,
                         observe=None):
    """No plaintext/key/boundary inputs. Only a validated finite Markov source.

    Width is applied after ALL incoming edges to a glyph layer arrive. The
    unmerged control carries unique path IDs. Terminal output truncation does
    not discard its already accumulated evidence. Bounds use float arithmetic.
    """
    validate_source(probabilities, transitions)
    cipher = tuple(map(tuple, cipher))
    if (type(glyphs) is not int or not 1 <= glyphs <= 6 or not 1 <= len(cipher) <= 2
            or any(not c or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or any(type(v) is not int or v < 1 for v in (width, max_expanded, max_generated, max_active, max_terminals))
            or type(merge) is not bool or schedule not in ("sequential", "balanced") or guidance not in ("none", "iid")
            or probabilities.shape[1] > 23):
        raise ValueError("Bounded one/two-record literal Markov configuration required")
    rows, lengths = probabilities.shape[1], tuple(map(len, cipher))
    guide_prewidth = 4*width if guide_prewidth is None else guide_prewidth
    if type(guide_prewidth) is not int or guide_prewidth < 1:
        raise ValueError("Positive preliminary guide beam required")
    units = tuple(tuple(u) for n in (1, 2) for u in itertools.product(range(glyphs), repeat=n))
    penalty, cont, stop = math.log(len(units)), math.log1p(-rho), math.log(rho)
    initial = State((0,)*len(cipher), (0,)*len(cipher), (-1,)*rows)
    layers, terminal = {0: {initial: (0., 0.)}}, {}
    expanded = generated = merges = pruned = serial = 0
    maximum_active, maximum_layer, dropped_upper = 1, 1, -math.inf
    guide = IidSuffixGuide(cipher, probabilities[0], glyphs=glyphs, rho=rho) if guidance == "iid" else None
    trace, reason = [], "frontier_exhausted"
    horizon = sum(lengths)
    for rank in range(horizon+1):
        here = layers.pop(rank, {})
        if not here:
            continue
        maximum_layer = max(maximum_layer, len(here))
        if rank == horizon:
            for state, (mass, best) in here.items():
                # An unmerged terminal can share a used-key cylinder with
                # other histories. Sum that group without confusing it with
                # a full-key posterior event.
                old_mass, old_best = terminal.get(state.key, (-math.inf, -math.inf))
                terminal[state.key] = (logadd(old_mass, mass), max(old_best, best))
            continue
        states = list(here)
        if guide is None:
            priorities = {s: here[s][0]+remaining_log_bound(s, lengths, rho) for s in states}
        else:
            states.sort(key=lambda s: (-(here[s][0]+remaining_log_bound(s, lengths, rho)), s))
            for state in states[guide_prewidth:]:
                dropped_upper = logadd(dropped_upper, here[state][0]+remaining_log_bound(state, lengths, rho))
                pruned += 1
            states = states[:guide_prewidth]
            offsets = np.array([s.offsets for s in states], dtype=np.int32)
            future = guide(np.array([s.key for s in states], dtype=np.int32), offsets,
                           offsets == np.array(lengths))
            priorities = {s: here[s][0]+float(h) for s, h in zip(states, future, strict=True)}
        ordered = sorted(states, key=lambda s: (-priorities[s], s))
        for state in ordered[width:]:
            dropped_upper = logadd(dropped_upper, here[state][0]+remaining_log_bound(state, lengths, rho))
            pruned += 1
        ordered = ordered[:width]
        # Stop before an entire layer, so no partially expanded parent needs
        # double-counting or an invented remainder probability.
        if expanded+len(ordered) > max_expanded or generated+2*rows*len(ordered) > max_generated:
            layers[rank] = {s: here[s] for s in ordered}
            reason = "expansion_cap" if expanded+len(ordered) > max_expanded else "generated_preflight_cap"
            break
        for state in ordered:
            mass, best = here[state]
            record = selected_record(state.offsets, lengths, schedule)
            offset, context = state.offsets[record], state.contexts[record]
            expanded += 1
            for row, probability in enumerate(probabilities[context]):
                if probability == 0:
                    continue
                bound = state.key[row]
                choices = (bound,) if bound >= 0 else tuple(
                    units.index(cipher[record][offset:offset+n]) for n in (1, 2) if offset+n <= lengths[record])
                for code in choices:
                    unit = units[code]
                    if cipher[record][offset:offset+len(unit)] != unit:
                        continue
                    offsets, contexts, key = list(state.offsets), list(state.contexts), list(state.key)
                    offsets[record] += len(unit)
                    contexts[record] = int(transitions[context, row])
                    edge = cont+math.log(float(probability))
                    if bound < 0:
                        key[row], edge = code, edge-penalty
                    if offsets[record] == lengths[record]:
                        contexts[record], edge = 0, edge+stop
                    generated, serial = generated+1, serial+1
                    child = State(tuple(offsets), tuple(contexts), tuple(key), 0 if merge else serial)
                    next_rank = sum(offsets)
                    target = layers.setdefault(next_rank, {})
                    if child in target:
                        old_mass, old_best = target[child]
                        target[child] = (logadd(old_mass, mass+edge), max(old_best, best+edge))
                        merges += 1
                    else:
                        target[child] = (mass+edge, best+edge)
                    active = sum(map(len, layers.values()))
                    maximum_active = max(maximum_active, active)
                    if active > max_active:
                        raise MemoryError("Unpruned pending DAG states exceed declared allocation cap")
        trace.append({"glyph_layer": rank, "incoming_states": len(here), "retained_states": len(ordered),
            "expanded": expanded, "generated": generated, "merged_arrivals": merges,
            "pruned_states": pruned, "pending_states": sum(map(len, layers.values()))})
        if observe is not None:
            observe(trace[-1])
    found = logsum(mass for mass, _ in terminal.values())
    unresolved = logsum(mass+remaining_log_bound(state, lengths, rho)
        for layer in layers.values() for state, (mass, _) in layer.items())
    lost = logadd(dropped_upper, unresolved)
    ordered_terminal = sorted(terminal.items(), key=lambda item: (-item[1][0], item[0]))
    selected = ordered_terminal[:max_terminals]
    return {"terminals": [{"used_key": key, "log_mass": mass, "best_leaf_log_mass": best}
                          for key, (mass, best) in selected],
            "terminal_states": len(terminal), "terminal_output_truncated": len(selected) < len(terminal),
            "found_log_mass": found, "returned_terminal_log_mass": logsum(value[0] for _, value in selected),
            "unresolved_log_mass_upper": lost, "evidence_log_upper": logadd(found, lost),
            "complete_search": pruned == 0 and not layers,
            "interval_arithmetic_certificate": False, "full_key_posterior_claimed": False,
            "stop_reason": reason, "expanded": expanded, "generated": generated,
            "merged_arrivals": merges, "pruned_states": pruned,
            "maximum_active_states": maximum_active, "maximum_unpruned_layer": maximum_layer,
            "guide_tables_built": 0 if guide is None else guide.tables_built, "trace": trace,
            "merge": merge, "schedule": schedule, "guidance": guidance,
            "guide_prewidth": guide_prewidth,
            "scope": "Markov sufficient-state history sums with once-binding prior. Finite beam loses mass; no historical recovery/reading MAP/full-key posterior claim."}
