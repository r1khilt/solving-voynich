"""Independent, explicitly gated DEV004 source/inference/decision replay.

No production estimator, inference, distance, coding or runner functions are
imported. Importing performs no artifact reads. The CLI requires both a frozen
revision and --after-evaluation; ordinary unit tests use in-memory hand fixtures.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import resource
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-004"
CASE_NAMES = ("B-key1", "B-key1-shuffle", "B-key2", "B-key2-shuffle")
GRID = ((0, 1.), *((order, tau) for order in (1, 2, 3) for tau in (.25, 1., 4., 16., 64., 256.)))
MANIFEST = "data/manifests/blind_channel_dev001.json"
CORPUS = "data/manifests/blind_channel_development_corpora.json"
SELECTION = f"results/{EXPERIMENT}/source_selection.json"
PREDICTIONS = f"results/{EXPERIMENT}/prediction_manifest.json"
EVALUATION = f"results/{EXPERIMENT}/evaluation.json"
TOLERANCE = 1e-7


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def close(actual, expected, label: str, tolerance: float = TOLERANCE) -> float:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label}: expected null")
        return 0.
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual) or abs(actual - expected) > tolerance):
        raise ValueError(f"{label}: numerical mismatch")
    return abs(actual - expected)


def body_records(payload: dict, alphabet: tuple[str, ...]) -> list[str]:
    text, boundaries = payload["text"], payload["body_boundaries"]
    if (not isinstance(text, str) or set(text) - set(alphabet) or not boundaries
            or any(type(end) is not int for end in boundaries)
            or boundaries != sorted(set(boundaries)) or boundaries[0] <= 0 or boundaries[-1] != len(text)):
        raise ValueError("Invalid source body boundaries or alphabet")
    return [text[start:end] for start, end in zip([0, *boundaries[:-1]], boundaries, strict=True)]


def manual_source(records: list[str], alphabet: tuple[str, ...], order: int, tau: float) -> dict:
    """Count complete ngrams by depth, then normalize suffix-interpolated rows."""
    if (type(order) is not int or not 0 <= order <= 3 or not records or not alphabet
            or len(set(alphabet)) != len(alphabet) or any(len(char) != 1 for char in alphabet)
            or not math.isfinite(tau) or tau <= 0
            or any(not record or set(record) - set(alphabet) for record in records)):
        raise ValueError("Invalid manual source-estimation inputs")
    grams = [Counter() for _ in range(order + 1)]
    for depth in range(order + 1):
        for record in records:
            grams[depth].update(record[start:start + depth + 1] for start in range(len(record) - depth))
    count = sum(grams[0].values())
    rows = {"": {char: (grams[0][char] + .5) / (count + .5 * len(alphabet)) for char in alphabet}}
    for depth in range(1, order + 1):
        for chars in itertools.product(alphabet, repeat=depth):
            context = "".join(chars)
            counts = [grams[depth][context + char] for char in alphabet]
            total = sum(counts)
            lower = rows[context[1:]]
            rows[context] = {char: (n + tau * lower[char]) / (total + tau)
                             for char, n in zip(alphabet, counts, strict=True)}
    return {"schema_version": 1, "alphabet": list(alphabet), "order": order, "probabilities": rows}


def plaintext_bits(source: dict, records: list[str]) -> float:
    """Count complete-order grams and explicit short record prefixes separately."""
    order, rows = source["order"], source["probabilities"]
    events = Counter()
    for record in records:
        for position, char in enumerate(record):
            start = max(0, position - order)
            events[record[start:position], char] += 1
    return -math.fsum(count * math.log2(rows[context][char]) for (context, char), count in events.items())


def validate_source(source: dict) -> None:
    alphabet, order = tuple(source["alphabet"]), source["order"]
    if (source.get("schema_version") != 1 or type(order) is not int or not 0 <= order <= 3
            or not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(char, str) or len(char) != 1 for char in alphabet)):
        raise ValueError("Invalid source metadata")
    contexts = {"".join(chars) for depth in range(order + 1) for chars in itertools.product(alphabet, repeat=depth)}
    if set(source["probabilities"]) != contexts:
        raise ValueError("Incomplete source context inventory")
    for row in source["probabilities"].values():
        if (set(row) != set(alphabet) or any(isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1
                                           for p in row.values()) or abs(math.fsum(row.values()) - 1) > 1e-12):
            raise ValueError("Invalid normalized source row")


def source_selection_audit(corpus: dict, selection: dict, payloads: dict, *, expected_author_characters=50_000) -> dict:
    alphabet = tuple(corpus["alphabet"])
    bodies = {}
    for author, role in (("caesar", "P1"), ("virgil", "P2")):
        spec = corpus["sources"][author]
        if spec["role"] != role:
            raise ValueError("Source role changed")
        raw = payloads[spec["derived_path"]]
        if digest(raw["text"].encode()) != spec["selected_text_sha256"]:
            raise ValueError("Selected source text hash mismatch")
        bodies[author] = body_records(raw, alphabet)
        if sum(map(len, bodies[author])) != expected_author_characters:
            raise ValueError("Source length differs from registered count")
    if (selection["source_authors_accessed"] != ["caesar", "virgil"]
            or selection["source_characters_per_author"] != expected_author_characters):
        raise ValueError("Source-selection provenance mismatch")
    candidates = []
    for order, tau in GRID:
        source = manual_source(bodies["caesar"], alphabet, order, tau)
        candidates.append({"order": order, "tau": tau,
                           "validation_bits_per_character": plaintext_bits(source, bodies["virgil"])
                           / expected_author_characters})
    if len(selection["candidates"]) != len(GRID):
        raise ValueError("Source candidate inventory mismatch")
    deltas = []
    for actual, expected in zip(selection["candidates"], candidates, strict=True):
        if (actual["order"], actual["tau"]) != (expected["order"], expected["tau"]):
            raise ValueError("Source candidate order/configuration mismatch")
        deltas.append(close(actual["validation_bits_per_character"], expected["validation_bits_per_character"],
                            "source validation bits", 1e-10))
    chosen = min(candidates, key=lambda row: row["validation_bits_per_character"])
    if (selection["selected"]["order"], selection["selected"]["tau"]) != (chosen["order"], chosen["tau"]):
        raise ValueError("Global source selection mismatch")
    close(selection["selected"]["validation_bits_per_character"], chosen["validation_bits_per_character"],
          "selected validation bits", 1e-10)
    if selection["primary_model"] != f"order{chosen['order']}":
        raise ValueError("Primary source label mismatch")
    models = payloads[selection["models"]["path"]]["models"]
    if set(models) != {f"order{order}" for order in range(4)} or set(selection["selected_per_order"]) != set("0123"):
        raise ValueError("Selected source inventory mismatch")
    probability_delta, probability_count = 0., 0
    for order in range(4):
        expected = min((row for row in candidates if row["order"] == order),
                       key=lambda row: row["validation_bits_per_character"])
        selected = selection["selected_per_order"][str(order)]
        if (selected["order"], selected["tau"]) != (order, expected["tau"]):
            raise ValueError("Per-order source selection mismatch")
        close(selected["validation_bits_per_character"], expected["validation_bits_per_character"],
              "per-order validation bits", 1e-10)
        rebuilt = manual_source(bodies["caesar"] + bodies["virgil"], alphabet, order, expected["tau"])
        actual = models[f"order{order}"]
        validate_source(actual)
        if actual["order"] != order or actual["alphabet"] != list(alphabet):
            raise ValueError("Selected refit source metadata mismatch")
        for context, row in rebuilt["probabilities"].items():
            for char, value in row.items():
                probability_delta = max(probability_delta, close(actual["probabilities"][context][char], value,
                                                                 "refit probability", 1e-12))
                probability_count += 1
    old = payloads[selection["old_source"]["path"]]["source_model"]
    rebuilt = manual_source(bodies["caesar"] + bodies["virgil"], alphabet, 1, 64.)
    validate_source(old)
    if old["order"] != 1 or old["alphabet"] != list(alphabet):
        raise ValueError("Old source metadata mismatch")
    regression_delta = max(close(old["probabilities"][h][a], p, "old-source tau64 regression", 1e-12)
                           for h, row in rebuilt["probabilities"].items() for a, p in row.items())
    close(selection["old_source_regression_max_delta"], regression_delta, "reported old-source delta", 1e-12)
    return {"candidates_checked": len(candidates), "candidate_scores": candidates,
            "selected": chosen, "refit_probabilities_checked": probability_count,
            "maximum_selection_bits_per_character_delta": max(deltas),
            "maximum_refit_probability_delta": probability_delta, "old_source_regression_delta": regression_delta}


def channel_units(channel: dict, alphabet: list[str], context: dict) -> tuple[str, ...]:
    states, glyphs = channel["states"], channel["glyph_alphabet"]
    if (channel.get("schema_version") != 1 or len(states) != 1 or not isinstance(states[0], str)
            or channel["initial"] != {states[0]: 1.} or glyphs != context["glyph_alphabet"]
            or channel["stop_probability"] != context["stop_probability"]
            or context["source_alphabet"] != alphabet):
        raise ValueError("Fixed deterministic channel/context mismatch")
    rows = {}
    for row in channel["rows"]:
        if row["state"] != states[0] or row["letter"] in rows or len(row["emissions"]) != 1:
            raise ValueError("Invalid deterministic channel row")
        emission = row["emissions"][0]
        unit = emission["glyphs"]
        if (emission["next_state"] != states[0] or emission["probability"] != 1.
                or not isinstance(unit, str) or not 1 <= len(unit) <= context["max_emission_length"]
                or set(unit) - set(glyphs)):
            raise ValueError("Invalid deterministic emission")
        rows[row["letter"]] = unit
    if set(rows) != set(alphabet):
        raise ValueError("Incomplete fixed channel")
    return tuple(rows[char] for char in alphabet)


def literal_model_bits(units: tuple[str, ...], context: dict) -> int:
    # Zero-bit singleton state/probability compositions, fixed bounded headers.
    def width(cardinality):
        return (cardinality - 1).bit_length()
    return (width(context.get("source_count", 1)) + width(context["max_states"])
            + len(units) * (width(context["max_alternatives"]) + width(context["max_emission_length"]))
            + sum(map(len, units)) * width(len(context["glyph_alphabet"])))


def _logsum(values: list[float]) -> float:
    if not values:
        return -math.inf
    high = max(values)
    return high + math.log(math.fsum(math.exp(value - high) for value in values)) if high != -math.inf else high


def infer_record(source: dict, units: tuple[str, ...], record: str, rho: float, *, max_nodes=3_000_000) -> dict:
    """Reachability followed by backward suffix sums/maxima; no production DP."""
    alphabet, order, rows = source["alphabet"], source["order"], source["probabilities"]
    if len(units) != len(alphabet) or any(not unit for unit in units) or not 0 < rho < 1:
        raise ValueError("Invalid inference inputs")
    matches = [[(char, offset + len(unit)) for char, unit in zip(alphabet, units, strict=True)
                if record.startswith(unit, offset)] for offset in range(len(record))]
    reachable = [set() for _ in range(len(record) + 1)]
    reachable[0].add("")
    count = 1
    for offset, edges in enumerate(matches):
        for context in reachable[offset]:
            for char, end in edges:
                if rows[context][char] == 0:
                    continue
                following = (context + char)[-order:] if order else ""
                if following not in reachable[end]:
                    reachable[end].add(following)
                    count += 1
                    if count > max_nodes:
                        raise RuntimeError("Independent exact lattice exceeded node cap")
    table = {(len(record), context): (math.log(rho), math.log(rho), None) for context in reachable[-1]}
    log_continue = math.log1p(-rho)
    for offset in range(len(record) - 1, -1, -1):
        for context in reachable[offset]:
            marginal_terms, best, choice = [], -math.inf, None
            for char, end in matches[offset]:
                probability = rows[context][char]
                following = (context + char)[-order:] if order else ""
                suffix = table.get((end, following))
                if probability == 0 or suffix is None or suffix[0] == -math.inf:
                    continue
                weight = math.log(probability) + log_continue
                marginal_terms.append(weight + suffix[0])
                if weight + suffix[1] > best:
                    best, choice = weight + suffix[1], (char, end, following)
            table[offset, context] = (_logsum(marginal_terms), best, choice)
    marginal, best, _ = table[0, ""]
    if marginal == -math.inf:
        return {"log_likelihood": None, "joint_log_probability": None, "plaintext": None, "nodes": count}
    offset, context, chars = 0, "", []
    while offset < len(record):
        char, offset, context = table[offset, context][2]
        chars.append(char)
    return {"log_likelihood": marginal, "joint_log_probability": best, "plaintext": "".join(chars), "nodes": count}


def reading_log_probability(source: dict, units: tuple[str, ...], text: str, record: str, rho: float) -> float:
    mapping = dict(zip(source["alphabet"], units, strict=True))
    if not isinstance(text, str) or set(text) - set(mapping) or "".join(mapping[char] for char in text) != record:
        raise ValueError("Archived reading does not reencode to its ciphertext")
    context, terms = "", [math.log(rho), len(text) * math.log1p(-rho)]
    for char in text:
        probability = source["probabilities"][context][char]
        if probability == 0:
            raise ValueError("Archived reading has zero posterior support")
        terms.append(math.log(probability))
        context = (context + char)[-source["order"]:] if source["order"] else ""
    return math.fsum(terms)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_distances(left: str, rights: list[str]) -> list[int]:
    """Ordinary integer DP vectorized over references and prefix columns.

    Horizontal insertion relaxation is cumulative-min(row[j]-j)+j. This is
    deliberately different from the production bit-vector distance routine.
    """
    if not rights:
        return []
    width = max(map(len, rights))
    letters = np.full((len(rights), width), -1, dtype=np.int32)
    lengths = np.asarray(list(map(len, rights)), dtype=np.intp)
    for row, text in enumerate(rights):
        letters[row, :len(text)] = [ord(char) for char in text]
    indices = np.arange(width + 1, dtype=np.int32)
    previous = np.broadcast_to(indices, (len(rights), width + 1)).copy()
    for i, char in enumerate(left, 1):
        current = np.empty_like(previous)
        current[:, 0] = i
        current[:, 1:] = np.minimum(previous[:, 1:] + 1, previous[:, :-1] + (letters != ord(char)))
        previous = np.minimum.accumulate(current - indices, axis=1) + indices
    return previous[np.arange(len(rights)), lengths].tolist()


def dictionary_edit_floor(alphabet: list[str], units: tuple[str, ...], record: str,
                          truth: str, *, max_cells=5_000_000) -> int | None:
    """Independent 0/1 shortest-path traversal of parse/edit product states."""
    if (len(record) + 1) * (len(truth) + 1) > max_cells:
        raise ValueError("Independent edit-product cell cap exceeded")
    matches = [[(char, offset + len(unit)) for char, unit in zip(alphabet, units, strict=True)
                if record.startswith(unit, offset)] for offset in range(len(record) + 1)]
    queue, distances = deque([(0, 0, 0)]), {(0, 0): 0}
    while queue:
        cost, offset, position = queue.popleft()
        if cost != distances[offset, position]:
            continue
        if offset == len(record) and position == len(truth):
            return cost
        edges = [(offset, position + 1, 1)] if position < len(truth) else []
        for char, end in matches[offset]:
            edges.append((end, position, 1))
            if position < len(truth):
                edges.append((end, position + 1, int(char != truth[position])))
        for end, following, weight in edges:
            if cost + weight < distances.get((end, following), math.inf):
                distances[end, following] = cost + weight
                item = (cost + weight, end, following)
                if weight:
                    queue.append(item)
                else:
                    queue.appendleft(item)
    return None


def bank_hash(samples: list[str]) -> str:
    return digest(json.dumps(samples, ensure_ascii=True, separators=(",", ":")).encode())


def audit_decision(row: dict, source: dict, units: tuple[str, ...], record: str, rho: float,
                   expected_seed: int, inference: dict) -> dict:
    if row["seed"] != expected_seed or row["candidate_draws"] != 32 or row["risk_draws"] != 256:
        raise ValueError("Decision sampling configuration mismatch")
    for label in ("candidate", "risk"):
        tag = "candidates" if label == "candidate" else "risk"
        expected = digest(f"unit-channel-decision/v1:{tag}:{expected_seed}".encode())
        if row[f"{label}_seed"] != expected:
            raise ValueError("Decision stream seed mismatch")
    if inference["plaintext"] is None:
        raise ValueError("Registered diagnostic unexpectedly has an impossible observation")
    if row["status"] != "ok" or row["tie_break"] != "map_first_then_candidate_first_occurrence":
        raise ValueError("Decision status/tie convention mismatch")
    delta = close(row["log_likelihood"], inference["log_likelihood"], "decision marginal")
    delta = max(delta, close(row["map_log_probability"], inference["joint_log_probability"], "decision MAP joint"))
    close(reading_log_probability(source, units, row["map_plaintext"], record, rho),
          inference["joint_log_probability"], "decision MAP literal support")
    for label, expected_length in (("candidate", 32), ("risk", 256)):
        samples = row[f"{label}_samples"]
        if len(samples) != expected_length or bank_hash(samples) != row[f"{label}_bank_sha256"]:
            raise ValueError("Decision bank length/hash mismatch")
        for text in set(samples):
            reading_log_probability(source, units, text, record, rho)
    candidates = list(dict.fromkeys([row["map_plaintext"], *row["candidate_samples"]]))
    risks = Counter(row["risk_samples"])
    refs, weights = list(risks), list(risks.values())
    totals = [sum(distance * weight for distance, weight in zip(edit_distances(text, refs), weights, strict=True))
              for text in candidates]
    if len(row["candidate_risks"]) != len(candidates):
        raise ValueError("Decision candidate inventory mismatch")
    for text, total, reported in zip(candidates, totals, row["candidate_risks"], strict=True):
        if reported["plaintext"] != text or reported["total_edit_distance"] != total:
            raise ValueError("Decision candidate order/total edit risk mismatch")
        close(reported["mean_edit_distance"], total / 256, "candidate mean edit risk", 1e-12)
    winner = min(range(len(candidates)), key=totals.__getitem__)
    if row["plaintext"] != candidates[winner]:
        raise ValueError("Decision winner/tie rule mismatch")
    close(row["estimated_risk"], totals[winner] / 256, "selected edit risk", 1e-12)
    close(row["map_estimated_risk"], totals[0] / 256, "MAP edit risk", 1e-12)
    return {"maximum_score_delta": delta, "candidate_reference_pairs": len(candidates) * 256,
            "distinct_distance_pairs_computed": len(candidates) * len(refs), "bank_draws_checked": 288,
            "numerically_resampled": False}


def _metrics(rows: list[dict], truth: list[str] | None) -> dict:
    texts = [row["plaintext"] for row in rows]
    edits = None if truth is None else [edit_distance(text or "", gold) for text, gold in zip(texts, truth, strict=True)]
    return {"supported_records": sum(text is not None for text in texts),
            "decoded_characters": sum(len(text or "") for text in texts),
            "gold_characters": None if truth is None else sum(map(len, truth)),
            "edits": None if edits is None else sum(edits), "record_edits": edits,
            "exact_records": None if truth is None else sum(text == gold for text, gold in zip(texts, truth, strict=True))}


def audit_payloads(manifest: dict, corpus: dict, selection: dict, predictions: dict, evaluation: dict,
                   payloads: dict, freezes: dict, *, expected_author_characters=50_000) -> dict:
    """Pure fixture-friendly replay. Byte/Git provenance is checked by the CLI."""
    source_audit = source_selection_audit(corpus, selection, payloads,
                                        expected_author_characters=expected_author_characters)
    old = payloads[manifest["source"]["path"]]["source_model"]
    if selection["old_source"] != manifest["source"]:
        raise ValueError("Old-source identity mismatch")
    sources = {"old": old, **payloads[selection["models"]["path"]]["models"]}
    for source in sources.values():
        validate_source(source)
        if source["alphabet"] != old["alphabet"]:
            raise ValueError("Source alphabet ordering mismatch")
    if set(predictions["cases"]) != set(CASE_NAMES) or set(evaluation["cases"]) != set(CASE_NAMES):
        raise ValueError("Registered case inventory mismatch")
    records_checked = literal_agreements = decisions_checked = draws = pairs = unique_pairs = floors_checked = 0
    max_delta = 0.
    for case_index, name in enumerate(CASE_NAMES):
        case, predicted, measured = manifest["cases"][name], predictions["cases"][name], evaluation["cases"][name]
        frozen = freezes[name]
        if (frozen["source_sha256"] != manifest["source"]["sha256"]
                or frozen["input_sha256"] != case["artifacts"]["fit"]["sha256"]):
            raise ValueError("Frozen learned dictionary identity mismatch")
        answer = payloads[case["artifacts"]["answer"]["path"]]
        channels = {"learned": frozen["channel"]}
        if case["positive"]:
            channels["oracle"] = answer["gold_channel"]
        if (set(predicted["arms"]) != set(channels) or set(measured["arms"]) != set(channels)
                or predicted["positive"] != case["positive"] or measured["positive"] != case["positive"]):
            raise ValueError("Fixed channel arm/positive inventory mismatch")
        fit = payloads[case["artifacts"]["fit"]["path"]]
        for arm, channel in channels.items():
            units = channel_units(channel, old["alphabet"], fit["context"])
            rho = channel["stop_probability"]
            # Legacy channel-versus-iid-family selector, unchanged from DEV003.
            bits = literal_model_bits(units, fit["context"]) + 1
            report, evaluated = predicted["arms"][arm], measured["arms"][arm]
            if (report["model_bits_with_selector"] != bits or evaluated["model_bits_with_selector"] != bits
                    or set(report["sources"]) != set(sources) or set(evaluated["sources"]) != set(sources)
                    or set(report["decisions"]) != {"fit", "transfer"}
                    or set(evaluated["decisions"]) != {"fit", "transfer"}):
                raise ValueError("Source panel/model-cost inventory mismatch")
            if case["positive"]:
                if set(evaluated.get("dictionary_edit_floor", {})) != {"fit", "transfer"}:
                    raise ValueError("Dictionary edit-floor split inventory mismatch")
            elif "dictionary_edit_floor" in evaluated:
                raise ValueError("Null case must not have a truth-based edit floor")
            for split_index, (split, count) in enumerate((("fit", 4), ("transfer", 2))):
                observed = payloads[case["artifacts"][split]["path"]]["records"]
                truth = answer["plaintext"][split] if case["positive"] else None
                if len(observed) != count or (truth is not None and len(truth) != count):
                    raise ValueError("Registered record inventory mismatch")
                if truth is not None:
                    floors = [dictionary_edit_floor(old["alphabet"], units, record, gold)
                              for record, gold in zip(observed, truth, strict=True)]
                    floor_report = {"record_edits": floors,
                                    "edits": None if any(value is None for value in floors) else sum(floors)}
                    if evaluated["dictionary_edit_floor"][split] != floor_report:
                        raise ValueError("Dictionary edit-floor mismatch")
                    floors_checked += len(floors)
                old_inference = []
                for source_name, source in sources.items():
                    if (set(report["sources"][source_name]) != {"fit", "transfer"}
                            or set(evaluated["sources"][source_name]) != {"fit", "transfer"}):
                        raise ValueError("Source prediction split mismatch")
                    rows = report["sources"][source_name][split]
                    if len(rows) != count:
                        raise ValueError("Prediction record count mismatch")
                    for record, row in zip(observed, rows, strict=True):
                        independent = infer_record(source, units, record, rho)
                        for field in ("log_likelihood", "joint_log_probability"):
                            max_delta = max(max_delta, close(row[field], independent[field], field))
                        if independent["plaintext"] is None:
                            if row["plaintext"] is not None:
                                raise ValueError("Impossible reading mismatch")
                        else:
                            max_delta = max(max_delta, close(reading_log_probability(source, units, row["plaintext"],
                                                                                    record, rho),
                                                             independent["joint_log_probability"], "MAP literal"))
                        literal_agreements += row["plaintext"] == independent["plaintext"]
                        records_checked += 1
                        if source_name == "old":
                            old_inference.append(independent)
                    expected = _metrics(rows, truth)
                    actual = evaluated["sources"][source_name][split]
                    for field, value in expected.items():
                        if actual[field] != value:
                            raise ValueError(f"Evaluation metric mismatch: {field}")
                    likelihood = math.fsum(row["log_likelihood"] for row in rows)
                    for field, value in (("log_likelihood", likelihood), ("data_bits", -likelihood / math.log(2)),
                                         ("total_bits", bits - likelihood / math.log(2))):
                        max_delta = max(max_delta, close(actual[field], value, f"evaluated {field}"))
                    surprisals = [(row["log_likelihood"] - row["joint_log_probability"]) / math.log(2) for row in rows]
                    if len(actual["map_surprisal_bits"]) != len(surprisals):
                        raise ValueError("Surprisal record inventory mismatch")
                    for actual_value, value in zip(actual["map_surprisal_bits"], surprisals, strict=True):
                        max_delta = max(max_delta, close(actual_value, value, "MAP surprisal"))
                decisions = report["decisions"][split]
                if len(decisions) != count:
                    raise ValueError("Decision record count mismatch")
                for record_index, (row, record, reference) in enumerate(zip(decisions, observed, old_inference, strict=True)):
                    if row["map_plaintext"] != report["sources"]["old"][split][record_index]["plaintext"]:
                        raise ValueError("Decision MAP differs from archived old-source MAP")
                    seed = 61101 + 1000 * case_index + 100 * split_index + record_index + (10000 if arm == "oracle" else 0)
                    replay = audit_decision(row, old, units, record, rho, seed, reference)
                    max_delta = max(max_delta, replay["maximum_score_delta"])
                    pairs += replay["candidate_reference_pairs"]
                    unique_pairs += replay["distinct_distance_pairs_computed"]
                    draws += replay["bank_draws_checked"]
                    decisions_checked += 1
                if evaluated["decisions"][split] != _metrics(decisions, truth):
                    raise ValueError("Decision evaluation metrics mismatch")
    return {"experiment": EXPERIMENT, "status": "pass", "source_selection": source_audit,
            "source_record_predictions_checked": records_checked, "literal_MAP_agreements": literal_agreements,
            "alternative_score_tied_MAPs": records_checked - literal_agreements,
            "decision_records_checked": decisions_checked, "bank_draws_checked": draws,
            "dictionary_edit_floor_records_checked": floors_checked,
            "candidate_reference_edit_pairs_checked": pairs, "distinct_edit_pairs_computed": unique_pairs,
            "maximum_inference_or_metric_delta": max_delta, "score_absolute_tolerance": TOLERANCE,
            "posterior_banks_numerically_resampled": False,
            "scope": "all source-selection candidates, refit tables, fixed-channel predictions, risks and metrics"}


def verified_json(root: Path, artifact: dict) -> dict:
    path = root / artifact["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes repository")
    raw = path.read_bytes()
    if digest(raw) != artifact["sha256"] or ("bytes" in artifact and len(raw) != artifact["bytes"]):
        raise ValueError("Artifact hash/size mismatch")
    decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return json.loads(decoded)


def frozen_json(root: Path, commit: str, path: str) -> tuple[dict, str]:
    raw = (root / path).read_bytes()
    archived = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, check=True, capture_output=True).stdout
    if raw != archived:
        raise ValueError(f"File differs from frozen revision: {path}")
    return json.loads(raw), digest(raw)


def run_audit(root: Path, freeze: str, output: str) -> dict:
    """Read empirical artifacts only when called explicitly by the gated CLI."""
    started, cpu_started = time.monotonic(), time.process_time()
    destination = root / output
    if destination.exists():
        raise FileExistsError("Refusing to overwrite prior independent audit")
    auditor_path = "scripts/audit_blind_channel_dev004.py"
    archived = subprocess.run(["git", "show", f"{freeze}:{auditor_path}"], cwd=root, check=True, capture_output=True).stdout
    if archived != (root / auditor_path).read_bytes():
        raise ValueError("Auditor is not frozen at requested revision")
    manifest, manifest_sha = frozen_json(root, freeze, MANIFEST)
    if manifest_sha != "8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea":
        raise ValueError("Registered development case manifest changed")
    corpus, corpus_sha = frozen_json(root, freeze, CORPUS)
    selection, selection_sha = frozen_json(root, freeze, SELECTION)
    prediction_manifest, prediction_sha = frozen_json(root, freeze, PREDICTIONS)
    evaluation = json.loads((root / EVALUATION).read_bytes())
    if (selection["corpus_manifest_sha256"] != corpus_sha
            or prediction_manifest["source_selection_sha256"] != selection_sha
            or evaluation["prediction_freeze"] != freeze
            or prediction_manifest["cases"] != list(CASE_NAMES)
            or prediction_manifest["plaintext_metrics_computed"] is not False
            or evaluation["predictions"] != prediction_manifest["predictions"]):
        raise ValueError("Stage provenance mismatch")
    predictions = verified_json(root, prediction_manifest["predictions"])
    if (predictions["source_selection_sha256"] != selection_sha
            or predictions["selection_freeze"] != prediction_manifest["selection_freeze"]):
        raise ValueError("Archived prediction provenance mismatch")
    payloads, artifacts = {}, []
    def load(artifact):
        if artifact["path"] not in payloads:
            payloads[artifact["path"]] = verified_json(root, artifact)
            artifacts.append(artifact)
    load(manifest["source"])
    load(selection["models"])
    for name in ("caesar", "virgil"):
        spec = corpus["sources"][name]
        load({"path": spec["derived_path"], "sha256": spec["derived_sha256"]})
    freezes = {}
    for name in CASE_NAMES:
        freezes[name], freeze_sha = frozen_json(root, freeze, f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json")
        if predictions["cases"][name]["freeze_sha256"] != freeze_sha:
            raise ValueError("Learned channel freeze hash mismatch")
        for split in ("fit", "transfer", "answer"):
            load(manifest["cases"][name]["artifacts"][split])
    result = audit_payloads(manifest, corpus, selection, predictions, evaluation, payloads, freezes)
    result.update(prediction_freeze=freeze, task_manifest_sha256=manifest_sha, corpus_manifest_sha256=corpus_sha,
                  source_selection_sha256=selection_sha, prediction_manifest_sha256=prediction_sha,
                  predictions=prediction_manifest["predictions"], evaluation_sha256=digest((root / EVALUATION).read_bytes()),
                  auditor_source_sha256=digest(archived), verified_artifacts=artifacts,
                  wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                  paid_spend_usd=0, final_authors_accessed=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--after-evaluation", action="store_true")
    parser.add_argument("--output", default=f"results/{EXPERIMENT}/independent_audit.json")
    args = parser.parse_args(argv)
    if not args.after_evaluation:
        parser.error("Empirical artifact reads require explicit --after-evaluation authorization")
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
    def timed_out(_signum, _frame):
        raise TimeoutError("Independent audit exceeded its 20-minute wall deadline")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(1200)
    result = run_audit(ROOT, args.freeze, args.output)
    print(json.dumps({key: result[key] for key in ("status", "source_record_predictions_checked",
                                                "decision_records_checked", "maximum_inference_or_metric_delta",
                                                "wall_seconds", "cpu_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
