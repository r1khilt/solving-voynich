"""Full same-algorithm replay plus alternate scalar/template checks."""
import argparse
import gzip
import json
import math
import signal
import time

import numpy as np

from scripts.run_source_fragment001 import (
    ARMS, OUT, PARENT, POOL, ROOT, NativeMarginal, array_identity, artifact, clauses,
    coverage, diagnostic, fit_one, inputs, limit_resources, load_source,
    random_blocks, resource_report, save_new,
)
from voynich.fragment_key_proposals import FragmentMatcher, gram_table, window_offsets
from voynich.known_plaintext_keys import solve_known_plaintext
from voynich.native_suffix_marginal import marginal_python


def scalar_template_check(matcher, observed, build):
    """Four deterministic individual source templates, every legal end offset."""
    indices = sorted({0, len(matcher.grams)//3, 2*len(matcher.grams)//3, len(matcher.grams)-1})
    checked = 0
    for index in indices:
        gram = tuple(map(int, matcher.grams[index]))
        one = FragmentMatcher(np.array([gram], dtype=np.int32), np.array([1]), build)
        found = one.match(observed, keep=4096)
        reference = []
        for end in range(12, min(24, len(observed))+1):
            result = solve_known_plaintext([gram], [observed[:end]], node_cap=100_000, solution_cap=8192)
            if not result["complete"]:
                raise ValueError("Scalar template check incomplete")
            for key in result["solutions"]:
                encoded = [(-1 if unit is None else one.pool.index(unit)) for unit in key]
                reference.append({"gram": 0, "consumed": end, "partial_key": encoded})
        reference.sort(key=lambda m: (m["gram"], m["partial_key"], m["consumed"]))
        if found["matches"] != reference or found["total_matches"] != len(reference):
            raise ValueError("Alternate word-equation/native template discrepancy")
        checked += 1
    return checked


def event_check(fitted, proposals):
    """Reconstruct coherent changes and earliest maximal selection directly."""
    scores, events, seen, best = fitted["scores"], fitted["events"], {tuple(fitted["scores"][0]["units"]): 0}, 0
    for round_row in fitted["rounds"]:
        assert round_row["base_index"] == best
        base = scores[best]["units"]
        here = events[round_row["first_event"]:round_row["first_event"]+round_row["events"]]
        expected = [(w, m) for w, window in enumerate(proposals) for m in range(len(window["scan"]["matches"]))]
        assert [(e["window"], e["match"]) for e in here] == expected[:len(here)]
        for event in here:
            partial = proposals[event["window"]]["scan"]["matches"][event["match"]]["partial_key"]
            key = tuple(base[r] if p == -1 else POOL[p] for r, p in enumerate(partial))
            index = event["index"]
            assert event["round"] == round_row["round"] and tuple(scores[index]["units"]) == key
            if key not in seen:
                assert index == len(seen)
                seen[key] = index
            assert seen[key] == index
            value = scores[index]["log_key_mass"]
            if value is not None and value > scores[best]["log_key_mass"]:
                best = index
        assert best == round_row["best_index"]
    assert len(seen) == len(scores) and best == fitted["best_index"]
    assert best == max(range(len(scores)), key=lambda i: -math.inf if scores[i]["log_key_mass"] is None else scores[i]["log_key_mass"])
    return len(events)


def audit():
    result = json.loads((OUT/"result.json").read_text())
    benchmark, cases, selected, build = inputs(result["freeze"])
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time()})
    limit_resources(2400, 2200)
    wall, cpu, rows, keys, events, bulk_bytes, checks, max_delta = time.monotonic(), time.process_time(), [], 0, 0, 0, 0, 0.
    try:
        source, source_spec = load_source()
        native = NativeMarginal(source, benchmark["build"])
        grams, counts = gram_table(source.compact)
        matcher = FragmentMatcher(grams, counts, build["build"])
        assert len(grams) == result["library_fragments"] == 20202
        assert source_spec["counts"] == result["source"] and array_identity(source) == result["source_arrays"]
        assert result["native_build"] == benchmark["build"] and result["fragment_build"] == artifact(OUT/"native-build.json")
        assert result["parent"] == artifact(PARENT/"result.json")
        expected_windows, workload_index = [], 0
        for case, (parent_spec, _, prediction) in enumerate(selected):
            records = tuple("".join("ABCDEF"[g] for g in r) for r in cases[case]["cipher"])
            base = tuple(prediction["revised_key"])
            window_spec = result["windows"][case]
            assert artifact(ROOT/window_spec["scans"]["path"]) == window_spec["scans"]
            bulk_bytes += window_spec["scans"]["bytes"]
            stored = json.loads(gzip.decompress((ROOT/window_spec["scans"]["path"]).read_bytes()))
            windows = []
            for record, offset in window_offsets(records):
                observed = tuple("ABCDEF".index(g) for g in records[record][offset:])
                scan = matcher.match(observed)
                windows.append({"record": record, "offset": offset, "scan": scan})
                checks += scalar_template_check(matcher, observed, build["build"])
            assert windows == stored
            for arm in ARMS:
                spec = result["workloads"][workload_index]
                workload_index += 1
                assert spec["path"] == f"results/SOURCE-FRAGMENT-001/case{case}-{arm}.json" and artifact(ROOT/spec["path"]) == spec
                row = json.loads((ROOT/spec["path"]).read_text())
                assert row["case"] == case and row["arm"] == arm and row["parent"] == parent_spec and row["windows"] == window_spec["scans"]
                assert artifact(ROOT/row["fitted"]["path"]) == row["fitted"]
                bulk_bytes += row["fitted"]["bytes"]
                fitted = json.loads(gzip.decompress((ROOT/row["fitted"]["path"]).read_bytes()))
                proposals = windows if arm == "fragment" else random_blocks(windows, case)
                replay = fit_one(records, base, proposals, source, native)
                assert json.loads(json.dumps(replay)) == fitted
                events += event_check(fitted, proposals)
                keys += len(fitted["scores"])
                assert row["scored_keys"] == len(fitted["scores"]) and row["events"] == len(fitted["events"])
                assert row["stop_reason"] == fitted["stop_reason"] and fitted["warm_prediction"] == prediction["revised"]
                assert diagnostic(fitted, cases[case], source) == row["diagnostic"]
                for index in sorted({0, fitted["best_index"]}):
                    score = fitted["scores"][index]
                    for r, observed in enumerate(records):
                        alt = marginal_python(source, score["units"], observed, 1/225, max_nodes=2_000_000, max_edges=8_000_000)
                        assert score["record_log_likelihoods"][r] is not None
                        delta = abs(alt.log_likelihood-score["record_log_likelihoods"][r])
                        assert delta <= 1e-7
                        max_delta = max(max_delta, delta)
                rows.append(row)
                if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered2GiB sampled audit cap")
                print(f"Audited case{case}-{arm}: {len(fitted['scores'])} keys", flush=True)
                del fitted, replay
            assert artifact(ROOT/window_spec["coverage"]["path"]) == window_spec["coverage"]
            bulk_bytes += window_spec["coverage"]["bytes"]
            covered = coverage(windows, cases[case], grams)
            assert covered == json.loads((ROOT/window_spec["coverage"]["path"]).read_text())
            expected_windows.append({"scans": window_spec["scans"], "coverage": window_spec["coverage"],
                "windows": len(windows), "nodes": sum(w["scan"]["nodes"] for w in windows),
                "total_fragment_matches": sum(w["scan"]["total_matches"] for w in windows),
                "true_boundary_windows": sum(r["true_unit_boundary"] for r in covered),
                "true12gram_windows": sum(r["true12gram_in_library"] for r in covered),
                "retained_true_binding_windows": sum(r["true_fragment_binding_retained"] for r in covered)})
        assert expected_windows == result["windows"] and len(rows) == 8
        assert keys == result["scored_keys"] and events == result["proposal_events"] and bulk_bytes == result["bulk_output_bytes"] <= 128*1024**2
        signal_clauses = clauses(rows)
        assert signal_clauses == result["exploratory_signal_clauses"] and result["exploratory_signal"] == ("SUPPORTED" if all(signal_clauses.values()) else "NOT_SUPPORTED")
        assert array_identity(source) == result["source_arrays"]
        save_new(OUT/"audit.json", {"status": "PASS_full_template_and_score_replay", "result": artifact(OUT/"result.json"),
            "checked_keys": keys, "checked_events": events, "alternate_scalar_templates": checks,
            "maximum_python_native_delta": max_delta, "resources": resource_report(wall, cpu),
            "independent_agent_review": False, "independent_full_search_algorithm": False,
            "paid_spend_usd": 0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json", {"error": repr(exc), "completed_workloads": len(rows),
            "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()
