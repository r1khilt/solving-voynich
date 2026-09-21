"""Gold-separated diagnostics for frozen joint-inference proposals on validation.

Candidate selection uses the inference engine's own ordering. Gold is consulted
only AFTER the first candidate is fixed. Exact IDs remain gauge-dependent: entity
renamings can preserve all observations, so lexical accuracy is not identifiability.
"""

from __future__ import annotations

import hashlib
import random

from .pipeline import episode_identity, infer, make_episode, verify_candidate
from .schema import FAMILIES, GRAMMARS, MORPHOLOGIES, object_digest
from .worlds import decode


def _seed(seed: int, index: int, domain: str) -> int:
    return int.from_bytes(hashlib.sha256(f"joint-eval-v1:{seed}:{index}:{domain}".encode()).digest()[:8], "big")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _f1(counts):
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def _binary_counts(predicted, expected):
    if len(predicted) != len(expected):
        raise ValueError("prediction/gold mask length mismatch")
    return [sum(p and g for p, g in zip(predicted, expected)),
            sum(p and not g for p, g in zip(predicted, expected)),
            sum(not p and g for p, g in zip(predicted, expected))]


def normalized_levenshtein(left, right):
    """1 - edit_distance/max_length; both empty=1, one empty=0."""
    if not left and not right:
        return 1.0
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, target in enumerate(right, 1):
        current = [row]
        for col, source in enumerate(left, 1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (source != target)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def baseline_hypothesis(observation, *, seed: int, kind: str):
    """A legal global channel drawn without a gold world, or a deletion falsifier.

    Random keys use one/two homophones per canonical class and fill the remaining
    alphabet with nulls; explicit anchors are fixed. Grammar, family and boundaries
    are guesses. Unobserved unanchored slots stay unresolved. The all-null baseline
    obeys anchor exceptions and is expected to fail the explanation null budget.
    """
    if kind not in ("random_key", "all_null"):
        raise ValueError("unknown baseline")
    rng = random.Random(seed)
    minimum_max = max((a.canonical for a in observation.anchors), default=2)
    choices = [m for m in MORPHOLOGIES if (8 if m == "none" else 10) >= minimum_max]
    if not choices:
        raise ValueError("anchors exceed the synthetic parser inventory")
    morphology = rng.choice(choices)
    maximum = 8 if morphology == "none" else 10
    key = [0] * observation.alphabet_size
    anchor_counts = {}
    for anchor in observation.anchors:
        anchor_counts[anchor.canonical] = anchor_counts.get(anchor.canonical, 0) + 1
        key[anchor.observed - 2] = anchor.canonical
    if kind == "random_key":
        copies = max(max(anchor_counts.values(), default=1), rng.choice((1, 2)))
        if (maximum - 1) * copies > observation.alphabet_size:
            raise ValueError("baseline anchor/homophone budget exceeds alphabet")
        bag = [canonical for canonical in range(2, maximum + 1)
               for _ in range(copies - anchor_counts.get(canonical, 0))]
        bag += [0] * (observation.alphabet_size - len(observation.anchors) - len(bag))
        rng.shuffle(bag)
        anchored = {a.observed for a in observation.anchors}
        for slot in range(observation.alphabet_size):
            if slot + 2 not in anchored:
                key[slot] = bag.pop()
    known = set(observation.symbols) | {a.observed for a in observation.anchors}
    for index in range(len(key)):
        if index + 2 not in known:
            key[index] = 0
    keep = tuple(key[x - 2] != 0 for x in observation.symbols)
    boundaries = [retained and (kind == "all_null" or rng.random() < 0.6) for retained in keep]
    if any(keep):
        boundaries[keep.index(True)] = True
    return {"inverse_key": tuple(key), "keep": keep, "boundaries": tuple(boundaries),
            "grammar": rng.choice(GRAMMARS), "morphology": morphology, "family": rng.choice(FAMILIES),
            "unresolved_key_symbols": tuple(i + 2 for i in range(len(key)) if i + 2 not in known)}


def score_selected(episode, inference_result):
    """Score rank zero, even when another candidate would match gold better."""
    candidates = inference_result["candidates"]
    selected = candidates[0] if candidates else None
    gold, observation = episode.world, episode.observation
    anchored = {a.observed for a in observation.anchors}
    unanchored = sorted(set(observation.symbols) - anchored)
    nonnull = [x for x in unanchored if gold.inverse_key[x - 2] != 0]
    hypothesis = selected["hypothesis"] if selected else None
    n = len(observation.symbols)
    if hypothesis is None:
        key = (None,) * observation.alphabet_size
        keep, boundaries, plaintext = (False,) * n, (False,) * n, ()
    else:
        key = tuple(hypothesis["inverse_key"])
        keep, boundaries = tuple(hypothesis["keep"]), tuple(hypothesis["boundaries"])
        plaintext = decode(observation.symbols, key).plaintext
    key_correct = sum(key[x - 2] == gold.inverse_key[x - 2] for x in unanchored)
    nonnull_correct = sum(key[x - 2] == gold.inverse_key[x - 2] for x in nonnull)
    counts = {"observed_unanchored_key_correct": key_correct, "observed_unanchored_key_total": len(unanchored),
              "nonnull_key_correct": nonnull_correct, "nonnull_key_total": len(nonnull),
              "keep": _binary_counts(keep, gold.keep), "boundary": _binary_counts(boundaries, gold.word_boundary),
              "valid_candidates": sum(bool(x["verification"]["valid"]) for x in candidates),
              "candidates": len(candidates)}
    return {"selected_id": selected["id"] if selected else None,
            "selected_rank": 0 if selected else None,
            "observed_unanchored_key_accuracy": _ratio(key_correct, len(unanchored)),
            "nonnull_key_accuracy": _ratio(nonnull_correct, len(nonnull)),
            "keep_f1": _f1(counts["keep"]), "boundary_f1": _f1(counts["boundary"]),
            "decoded_similarity": normalized_levenshtein(plaintext, gold.plaintext),
            "grammar_accuracy": int(hypothesis is not None and hypothesis["grammar"] == gold.config.grammar),
            "morphology_accuracy": int(hypothesis is not None and hypothesis["morphology"] == gold.config.morphology),
            "family_accuracy": int(hypothesis is not None and hypothesis["family"] == gold.config.family),
            "selected_valid": bool(selected and selected["verification"]["valid"]),
            "abstained": inference_result["status"] == "abstain", "counts": counts}


def aggregate_scores(rows):
    """Micro counts for keys/masks/candidates; macro means for document diagnostics."""
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    totals = {}
    for name in ("observed_unanchored_key_correct", "observed_unanchored_key_total", "nonnull_key_correct",
                 "nonnull_key_total", "valid_candidates", "candidates"):
        totals[name] = sum(row["counts"][name] for row in rows)
    for name in ("keep", "boundary"):
        totals[name] = [sum(row["counts"][name][i] for row in rows) for i in range(3)]
    result = {"documents": len(rows), "counts": totals,
              "observed_unanchored_key_accuracy": _ratio(totals["observed_unanchored_key_correct"],
                                                         totals["observed_unanchored_key_total"]),
              "nonnull_key_accuracy": _ratio(totals["nonnull_key_correct"], totals["nonnull_key_total"]),
              "keep_f1": _f1(totals["keep"]), "boundary_f1": _f1(totals["boundary"]),
              "valid_candidate_fraction": _ratio(totals["valid_candidates"], totals["candidates"])}
    for name in ("decoded_similarity", "grammar_accuracy", "morphology_accuracy", "family_accuracy",
                 "selected_valid", "abstained"):
        result[name + ("_fraction" if name in ("selected_valid", "abstained") else "")] = (
            sum(row[name] for row in rows) / len(rows)
        )
    return result


def evaluate_joint(
    model, layout, *, seed=41021, count=32, anchor_count=0, candidates=4, steps=12,
    evaluate_baselines=True, untrained_model=None,
):
    """Evaluate a supplied frozen model on VALIDATION ONLY; no test-switch exists.

    These metrics are diagnostics of synthetic supervision, with no pass/fail or
    decipherment conclusion. Invalid selected proposals are still scored and are
    separately marked abstained; there is no oracle filtering or best-of-k score.
    """
    for name, value, minimum in (("seed", seed, 0), ("count", count, 1), ("anchor_count", anchor_count, 0),
                                 ("candidates", candidates, 1), ("steps", steps, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {name}")
    if type(evaluate_baselines) is not bool:
        raise ValueError("evaluate_baselines must be boolean")
    methods = {"neural": []}
    if untrained_model is not None:
        methods["untrained"] = []
    if evaluate_baselines:
        methods.update(random_key=[], all_null_except_anchors=[])
    worlds, identities = [], []
    for index in range(count):
        episode = make_episode(index, "validation", layout, anchor_count=anchor_count, seed=seed)
        identities.append(episode_identity(episode))
        world_row = {"index": index, "observation_sha256": episode.observation.digest,
                     "family": episode.world.config.family, "anchor_count": len(episode.observation.anchors),
                     "methods": {}}
        for name, candidate_model in (("neural", model), ("untrained", untrained_model)):
            if candidate_model is None:
                continue
            result = infer(candidate_model, episode.observation, layout, candidates=candidates, steps=steps,
                           seed=_seed(seed, index, "neural"))
            score = score_selected(episode, result)
            methods[name].append(score)
            world_row["methods"][name] = score
        if evaluate_baselines:
            for kind, name in (("random_key", "random_key"), ("all_null", "all_null_except_anchors")):
                hypothesis = baseline_hypothesis(episode.observation, seed=_seed(seed, index, kind), kind=kind)
                verification = verify_candidate(episode.observation, hypothesis)
                result = {"candidates": [{"id": f"baseline:{kind}:{index}", "hypothesis": hypothesis,
                                           "verification": verification}],
                          "status": "candidate_hypotheses" if verification["valid"] else "abstain"}
                score = score_selected(episode, result)
                methods[name].append(score)
                world_row["methods"][name] = score
        worlds.append(world_row)
    return {"schema_version": 1, "split": "validation", "seed": seed, "count": count,
            "anchor_count": anchor_count, "candidates": candidates, "denoising_steps": steps,
            "input_sha256": object_digest(identities),
            "selection": "first candidate in model ranking, before any gold comparison",
            "metrics": {name: aggregate_scores(rows) for name, rows in methods.items()}, "worlds": worlds,
            "limitations": ["Synthetic validation only; no historical or scientific pass claim",
                            "Exact key and lexical scores are identity-gauge dependent",
                            "Grammar/morphology combination is held out, not every component mechanism",
                            "Oracle anchors are explicit synthetic correspondence supervision",
                            "Invalid rank-zero proposals are scored and separately marked abstained"]}
