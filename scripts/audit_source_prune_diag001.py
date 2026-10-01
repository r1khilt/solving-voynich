"""One full observer replay, plus alternate prefix and surrogate arithmetic."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import signal
import time

import numpy as np

from scripts.run_source_prune_diag001 import (
    ARMS, OUT, PARENT, NativePruneObserver, array_identity, artifact, config,
    finite, guard, inputs, limit_resources, load_bound, load_source, resource_report,
    save_new, summarize_trace, validate_trace,
)
from voynich.guided_source_particles import IidSuffixGuide


def alternate_states(fixture, p, goto, *, glyphs=6, rho=1/225, schedule="balanced"):
    """Independent scheduler and per-record rescoring at every gold prefix.

    Unlike incremental native/Python observers, recompute contexts and all
    probability factors from both original prefixes at each observation.
    """
    texts, key, cipher = fixture["generation_source"], fixture["generation_key"], fixture["cipher"]
    pool = [tuple(x) for n in (1, 2) for x in itertools.product(range(glyphs), repeat=n)]
    positions, offsets, states = [0]*len(texts), [0]*len(texts), []
    lengths = list(map(len, cipher))
    while True:
        partial, contexts, factors, used = [-1]*23, [], [], set()
        for r, text in enumerate(texts):
            context = 0
            for row in text[:positions[r]]:
                factors.extend([math.log1p(-rho), math.log(float(p[context, row]))])
                context = int(goto[context, row])
                used.add(row)
                partial[row] = key[row]
            if positions[r] == len(text):
                factors.append(math.log(rho))
                context = 0
            contexts.append(context)
        factors.extend([-math.log(len(pool))]*len(used))
        score = math.fsum(factors)
        unfinished = [r for r in range(len(texts)) if positions[r] < len(texts[r])]
        record = min(unfinished, key=lambda r: (offsets[r]/lengths[r], r)) if schedule == "balanced" and unfinished else (unfinished[0] if unfinished else None)
        table = offsets+[0]*(2-len(texts))+contexts+[0]*(2-len(texts))+partial
        table += [-1, -1, 1] if record is None else [texts[record][positions[record]], len(pool[key[texts[record][positions[record]]]]), 1]
        if len(table) != 30:
            raise ValueError("Alternate state layout changed")
        states.append((sum(offsets), table, score, list(positions)))
        if record is None:
            break
        row = texts[record][positions[record]]
        offsets[record] += len(pool[key[row]])
        positions[record] += 1
    return states


def alternate_priorities(fixture, p, goto, trace, gold_states, cfg):
    states = alternate_states(fixture, p, goto, glyphs=cfg["glyphs"], rho=cfg["rho"], schedule=cfg["schedule"])
    if len(states) != len(trace):
        raise ValueError("Independent gold state count differs")
    guide = IidSuffixGuide(fixture["cipher"], p[0], glyphs=cfg["glyphs"], rho=cfg["rho"], cache_entries=64, max_tables=10000)
    maximum, checked = 0., 0
    lengths = list(map(len, fixture["cipher"]))
    for (layer, target, score, _), d, saved in zip(states, trace, gold_states, strict=True):
        if target != saved or layer != d["target_glyph_layer"]:
            raise ValueError("Independent gold contexts/offsets/assignments differ")
        delta = abs(score-d["prefix_log_mass"])
        if not math.isfinite(delta) or delta > 1e-7:
            raise ValueError("Independent prefix mass differs")
        maximum, checked = max(maximum, delta), checked+1
        if d["ghost_pre_priority"] is None:
            continue
        # Sum geometric admissible length probabilities directly, rather than
        # the engine's closed-form expm1 identity.
        bound = 0.
        for r, length in enumerate(lengths):
            remaining = length-target[r]
            if remaining:
                probability = math.fsum(cfg["rho"]*(1-cfg["rho"])**n for n in range((remaining+1)//2, remaining+1))
                bound += math.log(probability)
        offset = np.array([target[:len(lengths)]], dtype=np.int64)
        key = np.array([target[4:4+p.shape[1]]], dtype=np.int16)
        surrogate = float(guide(key, offset, offset == np.array(lengths))[0]) if cfg["guidance"] == "iid" else bound
        for actual, expected in ((d["ghost_pre_priority"], score+bound), (d["ghost_guide_priority"], score+surrogate)):
            delta = abs(actual-expected)
            if not math.isfinite(delta) or delta > 1e-7:
                raise ValueError("Independent ghost priority differs")
            maximum, checked = max(maximum, delta), checked+1
    return checked, maximum


def audit():
    result = json.loads((OUT/"result.json").read_text())
    _, parent, cases, _, closed, build = inputs(result["freeze"])
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time()})
    limit_resources(1800, 1600)
    wall, cpu, rows, size, count, max_delta = time.monotonic(), time.process_time(), [], 0, 0, 0.
    try:
        source, selected = load_source()
        if selected["counts"] != result["source"] or array_identity(source) != result["source_arrays"]:
            raise ValueError("Source changed")
        if result["parent_result"] != artifact(PARENT/"result.json") or result["parent_audit"] != artifact(PARENT/"audit.json") or parent["fixtures"] != result["fixtures"]:
            raise ValueError("Parent changed")
        assert result["observer_build"] == artifact(OUT/"native-build.json")
        assert result["numpy_version"] == np.__version__
        assert result["paid_spend_usd"] == result["resources"]["paid_spend_usd"] == 0
        assert 0 <= result["resources"]["wall_seconds"] <= 1800
        assert 0 <= result["resources"]["cpu_seconds"] <= 1600
        assert 0 < result["resources"]["peak_rss_bytes"] <= 2*1024**3
        engine = NativePruneObserver(source.probabilities, source.transitions, build["build"])
        for fixture in cases:
            for arm in ARMS:
                name = f"case{fixture['case']}-{arm}"
                spec = result["workloads"][len(rows)]
                assert spec == artifact(OUT/f"{name}.json")
                row = load_bound(spec)
                assert (row["name"], row["case"], row["arm"], row["config"]) == (name, fixture["case"], arm, config(arm))
                assert row["original_cell"] == closed["workloads"][len(rows)]
                baseline_row = load_bound(row["original_cell"])
                baseline = load_bound(baseline_row["fitted"])
                stored = load_bound(row["diagnostic"])
                replay, trace = engine.observe(fixture["cipher"], fixture["generation_source"], fixture["generation_key"], **config(arm))
                assert finite(replay) == stored["control"] == baseline
                assert trace == stored["trace"]
                validate_trace(trace, fixture["cipher"], config(arm))
                assert row["ordinary_output_exact_match"] is True and row["summary"] == summarize_trace(trace)
                checked, delta = alternate_priorities(fixture, source.probabilities, source.transitions, trace, stored["gold_states"], config(arm))
                count, max_delta = count+checked, max(max_delta, delta)
                assert 0 <= row["wall_seconds"] <= result["resources"]["wall_seconds"]
                assert 0 <= row["cpu_seconds"] <= result["resources"]["cpu_seconds"]
                size += row["diagnostic"]["bytes"]
                guard(wall, cpu, size)
                rows.append(row)
                print(f"Audited {name}: {checked} alternate arithmetic checks", flush=True)
        assert len(rows) == len(result["workloads"]) == result["ordinary_output_exact_match_cells"] == 16
        assert size == result["bulk_output_bytes"] and array_identity(source) == result["source_arrays"]
        save_new(OUT/"audit.json", {"status": "PASS", "result": artifact(OUT/"result.json"), "workloads": 16,
            "alternate_prefix_priority_checks": count, "maximum_alternate_delta": max_delta,
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "same_author_and_search_implementation": True, "independent_agent_review": False,
            "independent_full_search": False, "ordinary_outputs_match_all_original_cells": True})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json", {"error": repr(exc), "completed_workloads": len(rows), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    audit()
