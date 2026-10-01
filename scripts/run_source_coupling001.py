"""Gold-assisted objective interventions; never an unassisted cipher solver."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import signal
import time

import numpy as np

from scripts.run_source_revise001 import (
    PATHS as REVISE_PATHS, POOL, ROOT, NativeMarginal, array_identity, artifact,
    inputs as parent_inputs, limit_resources, load_source, require_frozen,
    resource_report, save_new, score_key,
)
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter

EXP = "SOURCE-COUPLING-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
PARENT = ROOT/"results/SOURCE-REVISE-001"
PATHS = sorted(set([*REVISE_PATHS, "scripts/run_source_coupling001.py",
    "scripts/audit_source_coupling001.py", "tests/test_source_coupling001.py",
    "docs/experiments/SOURCE-COUPLING-001.md",
    "docs/research/source-key-coupling-2026-10-01.md",
    "results/SOURCE-REVISE-001/result.json", "results/SOURCE-REVISE-001/audit.json",
    *[f"results/SOURCE-REVISE-001/case{case}-{seed}-{guide}.json"
      for case, seed, guide in itertools.product(range(4), (75511, 75513), ("none", "iid"))]]))
MODELS = ("context", "iid")
TOLERANCE, JOINT_GAIN = .1, 1.


def iid_control(source):
    """Same smoothed empirical root counts, no context; new validated arrays."""
    root = {k: v.copy() for k, v in source.compact.levels[0].items()}
    compact = CompactSuffixSource(source.compact.alphabet, [root])
    iid = DenseSuffixAdapter(compact, source.tau, max_array_bytes=4096)
    if (not np.array_equal(iid.probabilities[0], source.row(source.state("")))
            or np.any(iid.transitions) or iid.order != 0):
        raise ValueError("IID control must preserve the exact source root distribution")
    return iid


def replace_rows(base, rows, target):
    out = list(base)
    for row in rows:
        out[row] = target[row]
    return tuple(out)


def finite_difference(values, rows):
    """Anchored discrete derivative, undefined if any subface has zero mass."""
    terms = []
    for size in range(len(rows)+1):
        for subset in itertools.combinations(rows, size):
            value = values[subset]
            if value is None:
                return None
            terms.append((-1 if (len(rows)-size) % 2 else 1)*value)
    return math.fsum(terms)


def decoy_targets(base, gold, rows, seed, pool=POOL):
    rng, target = random.Random(seed), list(base)
    for row in rows:
        options = tuple(u for u in pool if len(u) == len(gold[row]) and u not in (base[row], gold[row]))
        if not options:
            raise ValueError("Matched length-changing wrong replacement has no support")
        target[row] = rng.choice(options)
    return tuple(target)


def summarize(base_score, local, faces, endpoint):
    """Descriptive face witnesses; no barrier outside tested binary faces."""
    def gain(value):
        return None if value is None else value-base_score
    local_gains = [gain(r["value"]) for r in local]
    best_local = max((g for g in local_gains if g is not None), default=-math.inf)
    grouped = {}
    for size in (1, 2, 3):
        selected = [r for r in faces if len(r["rows"]) == size]
        finite = [r for r in selected if r["value"] is not None]
        joint = [r for r in selected if size >= 2 and r["joint_only_improvement"]]
        derivatives = [r["finite_difference"] for r in selected if r["finite_difference"] is not None]
        best = max(finite, key=lambda r: r["value"], default=None)
        grouped[str(size)] = {"faces": len(selected), "unsupported": len(selected)-len(finite),
            "best_gain_nats": None if best is None else gain(best["value"]),
            "best_bank_index": None if best is None else best["index"],
            "improving_faces": sum(r["value"]-base_score > JOINT_GAIN for r in finite),
            "joint_only_improving_faces": len(joint),
            "support_bridges_with_all_proper_nonempty_subfaces_unsupported": sum(r["support_bridge"] for r in selected),
            "finite_derivatives": len(derivatives),
            "derivative_min": min(derivatives, default=None),
            "derivative_max": max(derivatives, default=None),
            "derivative_median": None if not derivatives else float(np.median(derivatives))}
    best_joint = max((r for r in faces if len(r["rows"]) >= 2 and r["value"] is not None),
        key=lambda r: r["value"], default=None)
    return {"all_single_alternatives": len(local), "unsupported_single_alternatives": sum(g is None for g in local_gains),
        "best_single_gain_nats": None if best_local == -math.inf else best_local,
        "single_local_optimum_within_point1_nats": best_local <= TOLERANCE,
        "best_single_bank_index": None if best_local == -math.inf else max((r for r in local if r["value"] is not None), key=lambda r: r["value"])["index"],
        "by_size": grouped,
        "best_joint_bank_index": None if best_joint is None else best_joint["index"],
        "best_joint_gain_nats": None if best_joint is None else gain(best_joint["value"]),
        "endpoint_gain_nats": gain(endpoint),
        "joint_only_improving_faces": sum(r["joint_only_improvement"] for r in faces),
        "scope": "Anchored truth/decoy binary faces through size3; no global barrier or unassisted recovery claim."}


def probe(base, gold, used_rows, seed, score, observe=lambda _: None, pool=POOL):
    """Gold is deliberately used in interventions; no search/reading output."""
    base, gold, pool = tuple(base), tuple(gold), tuple(pool)
    used = tuple(sorted(used_rows))
    if (len(base) != len(gold) or not base or len(set(pool)) != len(pool)
            or any(u not in pool for u in (*base, *gold))
            or any(type(r) is not int or not 0 <= r < len(base) for r in used)
            or len(set(used)) != len(used)):
        raise ValueError("Invalid complete keys or used-row intervention set")
    mismatched = tuple(r for r in used if base[r] != gold[r])
    decoy = decoy_targets(base, gold, mismatched, seed, pool)
    bank, indices = [], {}
    def evaluate(key):
        if key not in indices:
            pair = score(key)
            if set(pair) != set(MODELS):
                raise ValueError("Both contextual and IID scores required")
            values = {m: pair[m]["log_key_mass"] for m in MODELS}
            if any(v is not None and not math.isfinite(v) for v in values.values()):
                raise ArithmeticError("Numerical failure is not zero support")
            if (values["context"] is None) != (values["iid"] is None):
                raise ValueError("Positive source controls must preserve deterministic channel support")
            indices[key] = len(bank)
            bank.append({"index": len(bank), "units": key, "models": pair})
            if len(bank) % 64 == 0:
                observe(len(bank))
        return indices[key]
    initial = evaluate(base)
    if any(bank[initial]["models"][m]["log_key_mass"] is None for m in MODELS):
        raise ValueError("Selected base key must have positive likelihood")
    local, families = [], {}
    for row in range(len(base)):
        for unit in pool:
            if unit != base[row]:
                target = list(base)
                target[row] = unit
                local.append({"row": row, "unit": unit, "index": evaluate(tuple(target))})
    for family, target in (("gold", gold), ("decoy", decoy)):
        faces = []
        for size in range(1, min(3, len(mismatched))+1):
            for rows in itertools.combinations(mismatched, size):
                faces.append({"rows": rows, "index": evaluate(replace_rows(base, rows, target))})
        families[family] = {"target": target, "faces": faces,
            "endpoint_index": evaluate(replace_rows(base, mismatched, target))}
    gold_index = evaluate(gold)
    maximum = 1+len(base)*(len(pool)-1)+2*sum(math.comb(len(mismatched), k) for k in range(1, min(3, len(mismatched))+1))+3
    if len(bank) > maximum:
        raise AssertionError("Registered candidate enumeration bound exceeded")
    summaries = {}
    for model in MODELS:
        base_score = bank[initial]["models"][model]["log_key_mass"]
        local_values = [{**r, "value": bank[r["index"]]["models"][model]["log_key_mass"]} for r in local]
        summaries[model] = {}
        for family, specs in families.items():
            values = {(): base_score, **{tuple(r["rows"]): bank[r["index"]]["models"][model]["log_key_mass"] for r in specs["faces"]}}
            faces = []
            for r in specs["faces"]:
                rows, value = tuple(r["rows"]), values[tuple(r["rows"])]
                proper = [values[s] for n in range(1, len(rows)) for s in itertools.combinations(rows, n)]
                improving = value is not None and value-base_score > JOINT_GAIN
                faces.append({**r, "value": value, "finite_difference": finite_difference(values, rows),
                    "joint_only_improvement": len(rows) >= 2 and improving and all(v is None or v-base_score <= TOLERANCE for v in proper),
                    "support_bridge": len(rows) >= 2 and value is not None and all(v is None for v in proper)})
            summaries[model][family] = summarize(base_score, local_values, faces,
                bank[specs["endpoint_index"]]["models"][model]["log_key_mass"])
        if bank[families["gold"]["endpoint_index"]]["models"][model]["log_key_mass"] is None:
            raise ValueError("Correct all used rows must admit the generating reading")
    return {"base": base, "gold": gold, "used_rows": used, "mismatched_used_rows": mismatched,
        "decoy_seed": seed, "decoy": decoy, "bank": bank, "local": local,
        "families": families, "gold_full_key_index": gold_index, "summaries": summaries,
        "maximum_unique_keys": maximum, "oracle_assisted": True,
        "unassisted_recovery_claimed": False, "global_barrier_claimed": False}


def signals(rows):
    own = [r["summaries"]["context"] for r in rows]
    missed = sum(r["gold"]["best_single_gain_nats"] is not None and r["gold"]["best_single_gain_nats"] > 1. for r in own)
    coupled = sum(r["gold"]["best_joint_gain_nats"] is not None
        and r["gold"]["best_joint_gain_nats"] > max(0., r["gold"]["best_single_gain_nats"] or 0.)+10.
        and r["gold"]["joint_only_improving_faces"] > r["decoy"]["joint_only_improving_faces"] for r in own)
    return {"missed_single_improvement_cells": missed, "gold_joint_advantage_cells": coupled,
        "missed_single_improvements_signal": "SUPPORTED" if missed >= 8 else "NOT_SUPPORTED",
        "gold_assisted_joint_advantage_signal": "SUPPORTED" if coupled >= 8 else "NOT_SUPPORTED",
        "not_recovery_qualification": True}


def inputs(freeze):
    require_frozen(freeze, PATHS)
    benchmark, _, cases = parent_inputs(freeze)
    parent = json.loads((PARENT/"result.json").read_text())
    audit = json.loads((PARENT/"audit.json").read_text())
    if (audit["result"] != artifact(PARENT/"result.json") or len(parent["workloads"]) != 16
            or audit["status"] != "PASS_full_score_rng_trace_and_reading_replay"):
        raise ValueError("Completed fully audited exposed parent required")
    return benchmark, parent, cases


def run(freeze):
    benchmark, parent, cases = inputs(freeze)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time()})
    limit_resources(1800, 1600)
    wall, cpu, rows, bulk_bytes = time.monotonic(), time.process_time(), [], 0
    try:
        source, selected = load_source()
        iid = iid_control(source)
        sources = {"context": source, "iid": iid}
        scorers = {m: NativeMarginal(s, benchmark["build"]) for m, s in sources.items()}
        identities = {m: array_identity(s) for m, s in sources.items()}
        for spec in parent["workloads"]:
            if artifact(ROOT/spec["path"]) != spec:
                raise ValueError("Parent compact result changed")
            old = json.loads((ROOT/spec["path"]).read_text())
            if artifact(ROOT/old["prediction"]["path"]) != old["prediction"]:
                raise ValueError("Parent prediction changed")
            base = json.loads((ROOT/old["prediction"]["path"]).read_text())["revised_key"]
            case, seed = old["case"], old["seed"]
            fixture = cases[case]
            cipher = tuple("".join("ABCDEF"[g] for g in c) for c in fixture["cipher"])
            gold = tuple(POOL[i] for i in fixture["generation_key"])
            used = {i for text in fixture["generation_source"] for i in text}
            cell_wall, cell_cpu = time.monotonic(), time.process_time()
            def score(key):
                return {m: score_key(scorer, cipher, key) for m, scorer in scorers.items()}
            def observe(_):
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered2GiB sampled host cap")
            payload = probe(base, gold, used, 76111+101*case+(seed-75511), score, observe)
            before = payload["bank"][0]["models"]["context"]["log_key_mass"]
            gold_score = payload["bank"][payload["gold_full_key_index"]]["models"]["context"]["log_key_mass"]
            if (abs(before-old["search_summary"]["best_score"]) > 1e-7
                    or abs(gold_score-old["diagnostic"]["gold_full_key_log_mass"]) > 1e-7):
                raise ValueError("Original objective/base/gold score changed")
            packed = save_new(BULK/f"{old['name']}.json.gz", payload, compressed=True)
            bulk_bytes += packed["bytes"]
            if bulk_bytes > 128*1024**2:
                raise MemoryError("Registered128MiB bulk cap")
            observe(0)
            row = {"name": old["name"], "case": case, "seed": seed, "guidance": old["guidance"],
                "parent": spec, "payload": packed, "summaries": payload["summaries"],
                "unique_keys": len(payload["bank"]), "maximum_unique_keys": payload["maximum_unique_keys"],
                "mismatched_used_rows": len(payload["mismatched_used_rows"]),
                "wall_seconds": time.monotonic()-cell_wall, "cpu_seconds": time.process_time()-cell_cpu}
            save_new(OUT/f"{old['name']}.json", row)
            rows.append(row)
            g = row["summaries"]["context"]["gold"]
            print(f"{old['name']}: single {g['best_single_gain_nats']}; joint {g['best_joint_gain_nats']}; gold joint-only {g['joint_only_improving_faces']}; {row['unique_keys']} keys", flush=True)
            del payload
        if len(rows) != 16 or identities != {m: array_identity(s) for m, s in sources.items()}:
            raise ValueError("Fixed16/source identity changed")
        save_new(OUT/"result.json", {"freeze": freeze, "parent": artifact(PARENT/"result.json"),
            "source": selected["counts"], "source_arrays": identities, "native_build": benchmark["build"],
            "workloads": [artifact(OUT/f"{r['name']}.json") for r in rows], "signals": signals(rows),
            "unique_key_evaluations": sum(r["unique_keys"] for r in rows), "bulk_output_bytes": bulk_bytes,
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "scope": "Gold-assisted fixed objective intervention and source-context ablation; not a solver or recovery qualification."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_workloads": len(rows), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
