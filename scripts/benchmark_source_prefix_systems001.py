"""Frozen cipher-only source-prefix systems benchmark; no empirical panel."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import signal
import time
from collections import defaultdict
from fractions import Fraction as F

import numpy as np

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import BASE_PATHS, INPUT_PATHS
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.joint_key_proposal import unit_pool
from voynich.source_prefix_inverse import search_source_prefix

EXP = "SOURCE-PREFIX-SYSTEMS-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
SOURCE = "results/LATIN-SOURCE-COMPACT-001/large.json"
PATHS = [*BASE_PATHS, *INPUT_PATHS, SOURCE, "src/voynich/compact_suffix_source.py",
         "src/voynich/compact_suffix_adapter.py", "src/voynich/dense_suffix_adapter.py",
         "src/voynich/source_prefix_inverse.py", "scripts/benchmark_source_prefix_systems001.py",
         "scripts/audit_source_prefix_systems001.py", "tests/test_source_prefix_inverse.py",
         "docs/experiments/SOURCE-PREFIX-SYSTEMS-001.md"]


def rational_row(prefix):
    return (F(1, 3), F(2, 3)) if not prefix else ((F(3, 4), F(1, 4)) if prefix[-1] == 0 else (F(1, 5), F(4, 5)))


def tiny_prior(text):
    value = F(1, 4)*F(3, 4)**len(text)
    for i, r in enumerate(text):
        value *= rational_row(text[:i])[r]
    return value


def full_key_reference(cipher):
    """Alternate algorithm: enumerate full dictionaries and source strings."""
    pool = [(g,) for g in range(2)]+list(itertools.product(range(2), repeat=2))
    candidates = [tuple(x for n in range(1, len(c)+1) for x in itertools.product(range(2), repeat=n)) for c in cipher]
    result = defaultdict(F)
    for key in itertools.product(pool, repeat=2):
        choices = [tuple(x for x in candidates[r] if tuple(g for s in x for g in key[s]) == c) for r, c in enumerate(cipher)]
        for texts in itertools.product(*choices):
            weight = F(1, 36)
            for text in texts:
                weight *= tiny_prior(text)
            result[texts] += weight
    return dict(result)


def verify_tiny(result, reference):
    low = math.exp(result["found_log_mass"])
    high = math.exp(result["evidence_log_upper"])
    total = float(sum(reference.values()))
    if low > total+1e-12 or high < total-1e-12:
        raise ValueError("Tiny evidence interval misses independent full-key sum")
    visited = {tuple(map(tuple, r["source_records"])): r["log_mass"] for r in result["readings"]}
    for reading, value in visited.items():
        if reading not in reference or math.exp(value) > float(reference[reading])+1e-12:
            raise ValueError("Tiny reading support/lower mass is invalid")
    if result["complete_search"]:
        if visited.keys() != reference.keys() or abs(low-total) > 1e-12:
            raise ValueError("Exhaustive tiny support/count differs")
        if any(abs(math.exp(v)-float(reference[x])) > 1e-12 for x, v in visited.items()):
            raise ValueError("Tiny reading marginal differs")
    if result["reading_bound_separated"] and reference:
        best = tuple(map(tuple, result["readings"][0]["source_records"]))
        if reference[best] != max(reference.values()):
            raise ValueError("Separated tiny reading is not truly best")


class DenseProvider:
    def __init__(self, source):
        self.source = source

    def __call__(self, prefix):
        history = "".join(self.source.alphabet[r] for r in prefix)
        return self.source.row(self.source.state(history))


def load_source():
    selected = json.loads((ROOT/SOURCE).read_text())
    if artifact(ROOT/selected["counts"]["path"]) != selected["counts"]:
        raise ValueError("Frozen trained source counts changed")
    source = DenseSuffixAdapter(CompactSuffixSource.load(ROOT/selected["counts"]["path"]), selected["selected"]["tau"])
    if len(source.alphabet) != 23:
        raise ValueError("Declared source alphabet differs")
    return source, selected


def array_identity(source):
    return {"probabilities": hashlib.sha256(source.probabilities).hexdigest(),
            "transitions": hashlib.sha256(source.transitions).hexdigest()}


def fixtures(source):
    result, pool = [], unit_pool(6)
    for index, (kind, length) in enumerate(itertools.product(("repetitive", "markov"), (64, 224))):
        rng = np.random.default_rng(73141+index)
        key = tuple(pool[int(i)] for i in rng.integers(42, size=23))
        if kind == "repetitive":
            common = int(np.argmax(source.row(0)))
            texts = ((common,)*length, (common,)*length)
        else:
            texts = []
            for _ in range(2):
                state, text = 0, []
                for _ in range(length):
                    row = int(rng.choice(23, p=source.row(state)))
                    text.append(row)
                    state = source.step(state, row)
                texts.append(tuple(text))
            texts = tuple(texts)
        cipher = tuple(tuple(g for r in text for g in key[r]) for text in texts)
        result.append({"kind": kind, "length": length, "seed": 73141+index,
                       "cipher": cipher, "generation_source": texts, "generation_key": key})
    return result


def verify_dense(result, cipher, source):
    """Alternate literal and scalar-state score replay, no generation answer."""
    scores = defaultdict(list)
    for t in result["terminals"]:
        texts, key = tuple(map(tuple, t["source_records"])), t["used_key"]
        if tuple(tuple(g for r in text for g in key[r]) for text in texts) != tuple(map(tuple, cipher)):
            raise ValueError("Completion violates literal shared-key observations")
        used = {r for text in texts for r in text}
        if any((k is not None) != (r in used) for r, k in enumerate(key)):
            raise ValueError("Unused rows must remain unassigned")
        terms = [-len(used)*math.log(42)]
        for text in texts:
            terms.append(math.log(1/225)+len(text)*math.log1p(-1/225))
            state = 0
            for row in text:
                terms.append(math.log(float(source.row(state)[row])))
                state = source.step(state, row)
        value = math.fsum(terms)
        if abs(value-t["log_mass"]) > 1e-9:
            raise ValueError("Literal candidate full source/prior score differs")
        scores[texts].append(value)
    visited = {tuple(map(tuple, r["source_records"])): r for r in result["readings"]}
    if visited.keys() != scores.keys():
        raise ValueError("Visited reading aggregation differs")
    for text, values in scores.items():
        high = max(values)
        total = high+math.log(math.fsum(math.exp(v-high) for v in values))
        if abs(total-visited[text]["log_mass"]) > 1e-9 or len(values) != visited[text]["visited_assignments"]:
            raise ValueError("Candidate key-marginal aggregation differs")


def run(freeze):
    require_frozen(freeze, PATHS)
    OUT.mkdir(parents=True, exist_ok=True)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    BULK.mkdir(parents=True, exist_ok=True)
    trace, completed = BULK/"workloads.jsonl", []
    try:
        source, selected = load_source()
        original = array_identity(source)
        generated = fixtures(source)
        inputs = save_new(BULK/"fixtures.json", {"fixtures": generated, "source": selected["counts"]})
        observed = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
        with trace.open("x") as stream:
            for i, cipher in enumerate(itertools.product(observed, repeat=2)):
                reference = full_key_reference(cipher)
                for bonus in (0., 4.):
                    for width, budget in ((10000, 10000), (1, 3), (3, 20), (3, 60)):
                        result = search_source_prefix(cipher, lambda p: tuple(map(float, rational_row(p))),
                            rows=2, glyphs=2, rho=.25, max_frontier=width, max_expanded=budget, progress_bonus=bonus)
                        verify_tiny(result, reference)
                        row = {"type": "tiny", "case": i, "bonus": bonus, "width": width, "budget": budget,
                               "cipher": cipher, "result": result}
                        stream.write(json.dumps(convert_finite(row), sort_keys=True, allow_nan=False)+"\n")
                        completed.append({"type": "tiny", "case": i})
            for i, fixture in enumerate(generated):
                for bonus in (0., 4.):
                    def observe(_):
                        if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                            raise MemoryError("2GiB sampled host guard")
                    result = search_source_prefix(fixture["cipher"], DenseProvider(source), max_expanded=5000,
                        max_frontier=4096, max_terminals=512, progress_bonus=bonus, observe=observe)
                    verify_dense(result, fixture["cipher"], source)
                    row = {"type": "dense", "case": i, "bonus": bonus, "result": result}
                    stream.write(json.dumps(convert_finite(row), sort_keys=True, allow_nan=False)+"\n")
                    stream.flush()
                    completed.append({"type": "dense", "case": i, "bonus": bonus,
                        **{k: result[k] for k in ("expanded", "generated", "source_calls", "pruned_states", "maximum_active_states", "stop_reason", "reading_bound_separated")},
                        "terminals": len(result["terminals"]), "readings": len(result["readings"]),
                        "resources": resource_report(wall, cpu)})
                    print(json.dumps(completed[-1]), flush=True)
        if array_identity(source) != original or len(completed) != 296:
            raise ValueError("Immutable source or fixed288tiny+8dense grid differs")
        save_new(OUT/"result.json", {"engineering_gate": "PASS_fixed_work_and_numerical_checks",
            "freeze": freeze, "source": selected["counts"], "array_identity": original,
            "dense_array_bytes": source.array_bytes, "fixtures": inputs, "trace": artifact(trace),
            "tiny_calls": 288, "dense_results": [r for r in completed if r["type"] == "dense"],
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "scope": "Artificial/forced-length workload, no empirical plaintext-recovery or interval certificate qualification."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"freeze": freeze, "error": repr(exc), "completed_workloads": len(completed),
            "trace": artifact(trace) if trace.exists() else None, "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


def convert_finite(value):
    """JSON preserves impossible masses explicitly, never emits Infinity/NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        if value == -math.inf:
            return None
        raise ValueError("Only negative-infinite unsupported mass is serializable")
    if isinstance(value, dict):
        return {k: convert_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_finite(v) for v in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
