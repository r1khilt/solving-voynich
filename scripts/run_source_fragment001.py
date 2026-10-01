"""Frozen coherent fragment proposals and matched random blocks on exposed cases."""
from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

from scripts.run_source_revise001 import (
    PATHS as PARENT_PATHS, POOL, ROOT, NativeMarginal, array_identity, artifact,
    inputs as old_inputs, limit_resources, load_source, read_key, require_frozen,
    resource_report, save_new, score_key,
)
from voynich.fragment_key_proposals import (
    FragmentMatcher, build_fragment_native, coherent_key, gram_table, window_offsets,
)
from voynich.unit_channel_decision import edit_distance

EXP = "SOURCE-FRAGMENT-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
PARENT = ROOT/"results/SOURCE-REVISE-001"
ARMS = ("fragment", "random_length_matched")
MAX_KEYS, ROUNDS = 8192, 3
PATHS = sorted(set([*PARENT_PATHS,
    "src/voynich/native_fragment_match.cpp", "src/voynich/fragment_key_proposals.py",
    "scripts/run_source_fragment001.py", "scripts/audit_source_fragment001.py",
    "tests/test_fragment_key_proposals.py", "tests/test_source_fragment001.py",
    "docs/research/fragment-proposals-2026-10-01.md", "docs/experiments/SOURCE-FRAGMENT-001.md",
    "results/SOURCE-FRAGMENT-001/native-build.json",
    "results/SOURCE-REVISE-001/result.json", "results/SOURCE-REVISE-001/audit.json",
    *[f"results/SOURCE-REVISE-001/case{case}-75511-iid.json" for case in range(4)]]))


def inputs(freeze):
    require_frozen(freeze, PATHS)
    benchmark, _, cases = old_inputs(freeze)
    parent = json.loads((PARENT/"result.json").read_text())
    audit = json.loads((PARENT/"audit.json").read_text())
    if audit["result"] != artifact(PARENT/"result.json") or len(parent["workloads"]) != 16:
        raise ValueError("Audited16-cell parent required")
    selected = []
    for case in range(4):
        spec = next(s for s in parent["workloads"] if s["path"] == f"results/SOURCE-REVISE-001/case{case}-75511-iid.json")
        if artifact(ROOT/spec["path"]) != spec:
            raise ValueError("Closed parent metadata changed")
        row = json.loads((ROOT/spec["path"]).read_text())
        if (row["case"], row["seed"], row["guidance"]) != (case, 75511, "iid") or artifact(ROOT/row["prediction"]["path"]) != row["prediction"]:
            raise ValueError("Chosen parent prediction changed")
        prediction = json.loads((ROOT/row["prediction"]["path"]).read_text())
        selected.append((spec, row, prediction))
    build = json.loads((OUT/"native-build.json").read_text())
    if artifact(ROOT/build["library"]["path"]) != build["library"]:
        raise ValueError("Closed fragment library changed")
    return benchmark, cases, selected, build


def random_blocks(windows, case):
    rng = random.Random(77111+case)
    result = []
    for window in windows:
        here = []
        for match in window["scan"]["matches"]:
            partial = match["partial_key"]
            random_partial = [(-1 if p == -1 else POOL.index(rng.choice(tuple(u for u in POOL if len(u) == len(POOL[p]))))) for p in partial]
            here.append({**match, "partial_key": random_partial})
        result.append({**window, "scan": {**window["scan"], "matches": here}})
    return result


def fit_one(records, base, windows, source, native, *, max_keys=MAX_KEYS, rounds=ROUNDS):
    """Only cipher, source library and cipher-derived warm key; no gold args."""
    if type(max_keys) is not int or max_keys < 1 or type(rounds) is not int or rounds < 1:
        raise ValueError("Positive finite fit budgets required")
    base = tuple(base)
    scores = [score_key(native, records, base)]
    if scores[0]["log_key_mass"] is None:
        raise ValueError("Supported cipher-only warm key required")
    seen, best, events, round_rows, stopped = {base: 0}, 0, [], [], False
    for iteration in range(rounds):
        current, start = tuple(scores[best]["units"]), best
        first_event = len(events)
        for w, window in enumerate(windows):
            for m, match in enumerate(window["scan"]["matches"]):
                key = coherent_key(current, match["partial_key"], POOL)
                if key not in seen:
                    if len(scores) >= max_keys:
                        stopped = True
                        break
                    seen[key] = len(scores)
                    scores.append(score_key(native, records, key))
                index = seen[key]
                events.append({"round": iteration, "window": w, "match": m, "index": index})
                value = scores[index]["log_key_mass"]
                if value is not None and value > scores[best]["log_key_mass"]:
                    best = index
            if stopped:
                break
        round_rows.append({"round": iteration, "base_index": start, "best_index": best,
                           "first_event": first_event, "events": len(events)-first_event})
        if stopped:
            break
    before = read_key(source, native, base, records, scores[0])
    after = read_key(source, native, scores[best]["units"], records, scores[best])
    return {"scores": scores, "events": events, "rounds": round_rows, "best_index": best,
            "warm_prediction": before, "final_prediction": after,
            "stop_reason": "key_cap" if stopped else "round_cap",
            "global_optimality_claimed": False, "posterior_sampling_claimed": False}


def diagnostic(fitted, fixture, source):
    truth = ["".join(source.alphabet[r] for r in record) for record in fixture["generation_source"]]
    gold = [POOL[r] for r in fixture["generation_key"]]
    used = sorted({r for record in fixture["generation_source"] for r in record})
    result = {}
    for stage, index, prediction in (("warm", 0, fitted["warm_prediction"]),
            ("final", fitted["best_index"], fitted["final_prediction"])):
        texts, key = prediction["source_records"], fitted["scores"][index]["units"]
        result[stage] = {"edit_errors": sum(edit_distance(a, b) for a, b in zip(texts, truth, strict=True)),
            "exact_records": sum(a == b for a, b in zip(texts, truth, strict=True)),
            "true_source_letters": sum(map(len, truth)), "used_rows": len(used),
            "matched_used_rows": sum(key[r] == gold[r] for r in used)}
    result["score_gain_nats"] = fitted["scores"][fitted["best_index"]]["log_key_mass"]-fitted["scores"][0]["log_key_mass"]
    result["gold_used_key_in_scored_bank"] = any(all(s["units"][r] == gold[r] for r in used) for s in fitted["scores"])
    return result


def coverage(windows, fixture, grams):
    """Post-prediction diagnosis of grammar/span support; never fit feedback."""
    library = {tuple(map(int, g)): i for i, g in enumerate(grams)}
    rows = []
    key = fixture["generation_key"]
    for window in windows:
        text = fixture["generation_source"][window["record"]]
        offsets = [0]
        for r in text:
            offsets.append(offsets[-1]+len(POOL[key[r]]))
        offset = window["offset"]
        boundary = offset in offsets[:-1]
        position = offsets.index(offset) if boundary else None
        fragment = () if position is None else tuple(text[position:position+12])
        gram = library.get(fragment)
        retained = False if gram is None else any(m["gram"] == gram and all(m["partial_key"][r] == key[r] for r in set(fragment)) for m in window["scan"]["matches"])
        rows.append({"record": window["record"], "offset": offset, "true_unit_boundary": boundary,
                     "true12gram_in_library": gram is not None, "true_fragment_binding_retained": retained})
    return rows


def clauses(rows):
    fragments = [r for r in rows if r["arm"] == "fragment"]
    controls = {r["case"]: r for r in rows if r["arm"] == "random_length_matched"}
    warm = sum(r["diagnostic"]["warm"]["edit_errors"] for r in fragments)
    final = sum(r["diagnostic"]["final"]["edit_errors"] for r in fragments)
    return {"all8_complete": len(rows) == 8,
            "fragment_edits_reduced_at_least10_percent": warm > 0 and final <= .9*warm,
            "fragment_beats_random_edits_at_least3_cases": sum(r["diagnostic"]["final"]["edit_errors"] < controls[r["case"]]["diagnostic"]["final"]["edit_errors"] for r in fragments) >= 3}


def run(freeze):
    benchmark, cases, selected, build = inputs(freeze)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "paid_spend_usd": 0})
    limit_resources(2400, 2200)
    wall, cpu, rows, bytes_out, windows_out = time.monotonic(), time.process_time(), [], 0, []
    try:
        source, selected_source = load_source()
        identity = array_identity(source)
        native = NativeMarginal(source, benchmark["build"])
        grams, counts = gram_table(source.compact)
        if len(grams) != 20202:
            raise ValueError("Frozen20202 retained source12grams required")
        matcher = FragmentMatcher(grams, counts, build["build"])
        for case, (parent_spec, _, prediction) in enumerate(selected):
            records = tuple("".join("ABCDEF"[g] for g in r) for r in cases[case]["cipher"])
            base = tuple(prediction["revised_key"])
            windows = []
            for record, offset in window_offsets(records):
                scan = matcher.match(tuple("ABCDEF".index(g) for g in records[record][offset:]))
                windows.append({"record": record, "offset": offset, "scan": scan})
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered2GiB sampled host cap")
            scans = save_new(BULK/f"case{case}-windows.json.gz", windows, compressed=True)
            windows_out.append(scans)
            bytes_out += scans["bytes"]
            for arm in ARMS:
                cell_wall, cell_cpu = time.monotonic(), time.process_time()
                proposals = windows if arm == "fragment" else random_blocks(windows, case)
                fitted = fit_one(records, base, proposals, source, native)
                if fitted["warm_prediction"] != prediction["revised"]:
                    raise ValueError("Same cipher-only baseline reader changed")
                # Close immutable full fit and predictions BEFORE any gold diagnostics.
                packed = save_new(BULK/f"case{case}-{arm}.json.gz", fitted, compressed=True)
                bytes_out += packed["bytes"]
                if bytes_out > 128*1024**2 or resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered bulk/host cap")
                row = {"case": case, "arm": arm, "parent": parent_spec, "windows": scans,
                    "fitted": packed, "scored_keys": len(fitted["scores"]), "events": len(fitted["events"]),
                    "stop_reason": fitted["stop_reason"], "diagnostic": diagnostic(fitted, cases[case], source),
                    "cell_wall_seconds": time.monotonic()-cell_wall, "cell_cpu_seconds": time.process_time()-cell_cpu}
                save_new(OUT/f"case{case}-{arm}.json", row)
                rows.append(row)
                print(f"case{case}-{arm}: {row['diagnostic']['warm']['edit_errors']} -> {row['diagnostic']['final']['edit_errors']} edits; {row['scored_keys']} keys", flush=True)
                del fitted
            # Gold boundary/library diagnostics only after BOTH arm predictions close.
            diagnostic_rows = coverage(windows, cases[case], grams)
            saved_coverage = save_new(BULK/f"case{case}-coverage.json", diagnostic_rows)
            bytes_out += saved_coverage["bytes"]
            windows_out[-1] = {"scans": scans, "coverage": saved_coverage,
                "windows": len(windows), "nodes": sum(w["scan"]["nodes"] for w in windows),
                "total_fragment_matches": sum(w["scan"]["total_matches"] for w in windows),
                "true_boundary_windows": sum(r["true_unit_boundary"] for r in diagnostic_rows),
                "true12gram_windows": sum(r["true12gram_in_library"] for r in diagnostic_rows),
                "retained_true_binding_windows": sum(r["true_fragment_binding_retained"] for r in diagnostic_rows)}
        if array_identity(source) != identity or bytes_out > 128*1024**2:
            raise ValueError("Original source arrays changed or final bulk cap exceeded")
        signal_clauses = clauses(rows)
        save_new(OUT/"result.json", {"freeze": freeze, "source": selected_source["counts"], "source_arrays": identity,
            "native_build": benchmark["build"], "fragment_build": artifact(OUT/"native-build.json"),
            "parent": artifact(PARENT/"result.json"), "library_fragments": len(grams), "windows": windows_out,
            "workloads": [artifact(OUT/f"case{r['case']}-{r['arm']}.json") for r in rows],
            "exploratory_signal_clauses": signal_clauses, "exploratory_signal": "SUPPORTED" if all(signal_clauses.values()) else "NOT_SUPPORTED",
            "scored_keys": sum(r["scored_keys"] for r in rows), "proposal_events": sum(r["events"] for r in rows),
            "resources": resource_report(wall, cpu), "bulk_output_bytes": bytes_out, "paid_spend_usd": 0,
            "scope": "Exposed four artificial keys; source-guided heuristic proposals, no posterior/recovery/decipherment qualification."})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"error": repr(exc), "completed_workloads": len(rows),
            "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


def prepare():
    build = build_fragment_native(BULK/"build")
    save_new(OUT/"native-build.json", {"build": build, "library": artifact(Path(build["library_path"])),
        "scope": "Compiler preparation only; no empirical corpus/template/cipher call."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.freeze:
        run(args.freeze)
    else:
        parser.error("run requires --freeze")
