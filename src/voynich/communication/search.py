"""Finite beam proposals coupling grammar phases to a document-global cipher key.

This is approximate compiler-guided search, not complete enumeration or posterior
sampling. It receives only observations, explicit anchors and optional model scores.
The caller must independently check semantic execution and rank surviving proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .schema import FAMILIES, GRAMMARS, MORPHOLOGIES, Observation


@dataclass(frozen=True)
class _Beam:
    phase: int
    key: tuple[int, ...]
    keep: tuple[bool, ...]
    boundaries: tuple[bool, ...]
    score: float
    nulls: int


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _score_values(values, length, name):
    if values is None:
        return (0.0,) * length
    try:
        result = tuple(float(x) for x in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if len(result) != length or any(math.isnan(x) or x > 0 for x in result):
        raise ValueError(f"{name} needs {length} nonpositive log scores (negative infinity is allowed)")
    return result


def _key_scores(values, alphabet):
    if values is None:
        return ((0.0,) * (alphabet + 2),) * alphabet
    if len(values) != alphabet:
        raise ValueError("key_log_probs must have one row per observed alphabet symbol")
    return tuple(_score_values(row, alphabet + 2, "key_log_probs row") for row in values)


def _template(grammar, morphology):
    template = []
    for role in grammar:
        if role == "V":
            template.append(((6, 7, 8), True))
            continue
        stem, marker = (2, 3, 4, 5), (9 if role == "S" else 10,)
        if morphology == "none":
            template.append((stem, True))
        elif morphology == "prefix":
            template.extend(((marker, True), (stem, False)))
        else:
            template.extend(((stem, True), (marker, False)))
    return tuple(template)


def _run_grammar(observation, grammar, morphology, family, scores, *, beam_size, max_expansions,
                 max_null_fraction, max_candidates):
    template = _template(grammar, morphology)
    initial_key = [-1] * observation.alphabet_size
    maximum = 8 if morphology == "none" else 10
    for anchor in observation.anchors:
        if anchor.canonical > maximum:
            return {"candidates": [], "expansions": 0, "budget_exhausted": False,
                    "all_observations_processed": False, "beam_pruned": False,
                    "reason": "anchor_outside_declared_inventory"}
        initial_key[anchor.observed - 2] = anchor.canonical
    beam = [_Beam(0, tuple(initial_key), (), (), 0.0, 0)]
    max_nulls = math.floor(len(observation.symbols) * max_null_fraction + 1e-12)
    expansions, exhausted, pruned, processed = 0, False, False, 0
    # An explicit weak channel preference, not a calibrated text likelihood.
    keep_score, null_score = math.log(0.85), math.log(0.15)
    for position, observed in enumerate(observation.symbols):
        children = {}
        for parent in beam:
            existing = parent.key[observed - 2]
            allowed, starts_word = template[parent.phase]
            options = (existing,) if existing >= 0 else (0, *allowed)
            for canonical in options:
                if canonical and canonical not in allowed:
                    continue
                if canonical == 0 and parent.nulls >= max_nulls:
                    continue
                if expansions >= max_expansions:
                    exhausted = True
                    break
                expansions += 1
                assigned_score = scores[observed - 2][canonical] if existing < 0 else 0.0
                if assigned_score == -math.inf:
                    continue
                key = parent.key
                if existing < 0:
                    changed = list(key)
                    changed[observed - 2] = canonical
                    key = tuple(changed)
                retained = canonical != 0
                phase = (parent.phase + int(retained)) % len(template)
                score = parent.score + assigned_score + (keep_score if retained else null_score)
                if score == -math.inf:
                    continue
                child = _Beam(phase, key, (*parent.keep, retained),
                              (*parent.boundaries, retained and starts_word),
                              score,
                              parent.nulls + int(not retained))
                # A key and phase determine the retained history on this prefix.
                fingerprint = (phase, key)
                old = children.get(fingerprint)
                if old is None or child.score > old.score:
                    children[fingerprint] = child
            if exhausted:
                break
        next_beam = sorted(children.values(), key=lambda state: (-state.score, state.key, state.phase))
        pruned |= len(next_beam) > beam_size
        beam = next_beam[:beam_size]
        processed = position + 1
        if exhausted or not beam:
            break
    candidates = []
    if processed == len(observation.symbols):
        for state in beam:
            if state.phase != 0 or not any(state.keep):
                continue
            hypothesis = {"inverse_key": tuple(max(value, 0) for value in state.key), "keep": state.keep,
                          "boundaries": state.boundaries, "grammar": grammar, "morphology": morphology,
                          "family": family,
                          "unresolved_key_symbols": tuple(i + 2 for i, value in enumerate(state.key) if value < 0)}
            candidates.append({"hypothesis": hypothesis, "compiler_score": state.score})
            if len(candidates) >= max_candidates:
                break
    return {"candidates": candidates, "expansions": expansions, "budget_exhausted": exhausted,
            "all_observations_processed": processed == len(observation.symbols), "beam_pruned": pruned,
            "processed_positions": processed}


def search_grammar(
    observation, *, grammar, morphology, family="procedure", key_log_probs=None,
    beam_size=32, max_expansions=20000, max_candidates=16, max_null_fraction=0.7,
):
    """Search one declared synthetic grammar; no semantic trace or gold is read."""
    _validate_request(observation, beam_size, max_expansions, max_candidates, max_null_fraction)
    if grammar not in GRAMMARS or morphology not in MORPHOLOGIES or family not in FAMILIES:
        raise ValueError("unsupported grammar/morphology/family")
    scores = _key_scores(key_log_probs, observation.alphabet_size)
    result = _run_grammar(observation, grammar, morphology, family, scores, beam_size=beam_size,
                          max_expansions=max_expansions, max_candidates=max_candidates,
                          max_null_fraction=max_null_fraction)
    return {**result, "grammar": grammar, "morphology": morphology, "family": family,
            "beam_size": beam_size, "max_expansions": max_expansions,
            "score_kind": "key-score sum plus a fixed 0.15-null channel preference; not posterior evidence",
            "claim": "approximate syntactic proposals; semantic verification still required"}


def _validate_request(observation, beam_size, max_expansions, max_candidates, max_null_fraction):
    if not isinstance(observation, Observation):
        raise ValueError("compiler search requires an Observation, not a gold world")
    if observation.alphabet_size < 9:
        raise ValueError("synthetic compiler needs an alphabet supporting canonical symbols through 10")
    _positive(beam_size, "beam_size", 1024)
    _positive(max_expansions, "max_expansions", 1_000_000)
    _positive(max_candidates, "max_candidates", 1024)
    if type(max_null_fraction) not in (float, int) or not math.isfinite(max_null_fraction) or not 0 <= max_null_fraction < 1:
        raise ValueError("max_null_fraction must be finite and in [0, 1)")


def compiler_search(
    observation, *, key_log_probs=None, global_log_probs=None, beam_size=32,
    max_expansions=20000, max_candidates=16, max_null_fraction=0.7,
):
    """Share a strict expansion budget across ranked grammar/morphology hypotheses.

    Key scores have shape [alphabet_size, alphabet_size+2], indexed by canonical
    value 0(null), 2..alphabet_size+1; category 1 is unused. Optional global scores
    are a dictionary with grammar/morphology/family vectors in schema tuple order.
    Surface search is shared by families, then each result is offered under each
    family. No family gets semantic credit until an independent executor checks it.

    Each remaining combination receives a fair share of the remaining budget.
    Beam widths are reduced when that share is small relative to document length;
    this is recorded explicitly. Search may correctly return no complete proposal.
    """
    _validate_request(observation, beam_size, max_expansions, max_candidates, max_null_fraction)
    scores = _key_scores(key_log_probs, observation.alphabet_size)
    names = {"grammar": GRAMMARS, "morphology": MORPHOLOGIES, "family": FAMILIES}
    if global_log_probs is not None and set(global_log_probs) != set(names):
        raise ValueError("global_log_probs must have grammar, morphology, and family vectors")
    globals_ = {name: _score_values(None if global_log_probs is None else global_log_probs[name], len(values), name)
                for name, values in names.items()}
    combinations = sorted(
        [(globals_["grammar"][gi] + globals_["morphology"][mi], grammar, morphology)
         for gi, grammar in enumerate(GRAMMARS) for mi, morphology in enumerate(MORPHOLOGIES)],
        key=lambda row: (-row[0], GRAMMARS.index(row[1]), MORPHOLOGIES.index(row[2])),
    )
    combinations = [row for row in combinations if row[0] != -math.inf]
    candidates, reports = [], []
    expansions = 0
    for index, (global_score, grammar, morphology) in enumerate(combinations):
        remaining = max_expansions - expansions
        if remaining < 1:
            break
        share = max(1, remaining // (len(combinations) - index))
        width = min(beam_size, max(1, share // max(1, len(observation.symbols) * 3)))
        result = _run_grammar(observation, grammar, morphology, "procedure", scores, beam_size=width,
                              max_expansions=share, max_candidates=max_candidates,
                              max_null_fraction=max_null_fraction)
        expansions += result["expansions"]
        reports.append({"grammar": grammar, "morphology": morphology, "allocated_expansions": share,
                        "effective_beam_size": width, **{k: v for k, v in result.items() if k != "candidates"}})
        for candidate in result["candidates"]:
            for fi, family in enumerate(FAMILIES):
                family_score = globals_["family"][fi]
                if family_score == -math.inf:
                    continue
                candidates.append({"hypothesis": {**candidate["hypothesis"], "family": family},
                                   "compiler_score": candidate["compiler_score"] + global_score + family_score})
    candidates.sort(key=lambda row: (-row["compiler_score"], row["hypothesis"]["inverse_key"],
                                     row["hypothesis"]["grammar"], row["hypothesis"]["morphology"],
                                     row["hypothesis"]["family"]))
    return {"candidates": candidates[:max_candidates], "expansions": expansions,
            "max_expansions": max_expansions, "beam_size": beam_size,
            "budget_exhausted": expansions >= max_expansions or any(r["budget_exhausted"] for r in reports),
            "combinations_attempted": len(reports), "combination_reports": reports,
            "candidate_count_before_limit": len(candidates),
            "score_kind": "key/global score sum plus fixed 0.15-null preference; not posterior evidence",
            "claim": "bounded approximate compiler proposals, without semantic or historical identification"}
