"""Complete native score replay, RNG/kernel accounting and scalar reading checks."""
import argparse
import gzip
import json
import math
import random
import signal
import time
from dataclasses import asdict

from scripts.run_source_revise001 import (
    OUT, POOL, ROOT, array_identity, artifact, config_for, diagnostics,
    inputs, limit_resources, load_source, NativeMarginal, read_key, require_frozen,
    PATHS, resource_report, save_new, score_key, warm_key,
)
from voynich.native_suffix_marginal import marginal_python


def replay_rng_trace(search, cfg):
    """Independent mixture draws/acceptance/exchange reconstruction, no scorer."""
    bank, events = search["bank"], search["events"]
    assert len(bank) == search["scored_keys"] <= cfg.max_scored_keys
    assert len({tuple(b["units"]) for b in bank}) == len(bank)
    assert search["temperatures"] == list(cfg.temperatures)
    assert all(b["index"] == i for i, b in enumerate(bank))
    assert search["initial_replica_indices"] == [0]*len(cfg.temperatures)
    values = [-math.inf if b["score"] is None else b["score"] for b in bank]
    assert values[search["best_index"]] == max(values) == search["best_score"]
    assert bank[search["best_index"]]["units"] == search["best_units"]
    rng, replicas, seen = random.Random(cfg.seed), [0]*len(cfg.temperatures), {tuple(bank[0]["units"]): 0}
    counts = {kind: {"proposed": 0, "accepted": 0} for kind in ("replace", "swap", "block")}
    proposals, self_moves, exchanges, exchange_accepted = 0, 0, 0, 0
    expected, rounds = [], 0
    for iteration in range(cfg.iterations):
        expected.extend(("proposal", iteration, r) for r in range(len(replicas)))
        if (iteration+1) % cfg.swap_interval == 0:
            expected.extend(("exchange", iteration, left) for left in range(rounds % 2, len(replicas)-1, 2))
            rounds += 1
    assert [(e["event"], e["iteration"], e.get("replica", e.get("left"))) for e in events] == expected[:len(events)]
    for event in events:
        if event["event"] == "proposal":
            replica, previous = event["replica"], event["previous_index"]
            assert replicas[replica] == previous
            key, proposed = bank[previous]["units"], list(bank[previous]["units"])
            draw = rng.randrange(sum(cfg.proposal_weights))
            if draw < cfg.proposal_weights[0]:
                kind, rows = "replace", (rng.randrange(len(key)),)
                proposed[rows[0]] = rng.choice(tuple(u for u in POOL if u != key[rows[0]]))
            elif draw < sum(cfg.proposal_weights[:2]):
                kind, rows = "swap", tuple(sorted(rng.sample(range(len(key)), 2)))
                proposed[rows[0]], proposed[rows[1]] = key[rows[1]], key[rows[0]]
            else:
                size = rng.choice(tuple(n for n in cfg.block_sizes if n <= len(key)))
                kind, rows = "block", tuple(sorted(rng.sample(range(len(key)), size)))
                for row in rows:
                    proposed[row] = rng.choice(POOL)
            candidate = event["candidate_index"]
            assert kind == event["kind"] and list(rows) == event["rows"]
            assert [proposed[r] for r in rows] == event["new_units"] and proposed == bank[candidate]["units"]
            if tuple(proposed) not in seen:
                assert candidate == len(seen)
                seen[tuple(proposed)] = candidate
            assert seen[tuple(proposed)] == candidate
            threshold = -math.inf if values[candidate] == -math.inf else min(0., (values[candidate]-values[previous])/cfg.temperatures[replica])
            counts[kind]["proposed"] += 1
            counts[kind]["accepted"] += int(event["accepted"])
            proposals += 1
            self_moves += int(candidate == previous)
            if event["accepted"]:
                replicas[replica] = candidate
        else:
            a, b = event["left"], event["right"]
            assert b == a+1 and [replicas[a], replicas[b]] == event["previous_indices"]
            threshold = min(0., (1/cfg.temperatures[a]-1/cfg.temperatures[b])*(values[replicas[b]]-values[replicas[a]]))
            exchanges += 1
            exchange_accepted += int(event["accepted"])
            if event["accepted"]:
                replicas[a], replicas[b] = replicas[b], replicas[a]
        draw = rng.random()
        log_draw = math.log(draw) if draw else -math.inf
        assert (None if threshold == -math.inf else threshold) == event["log_acceptance"]
        assert (None if log_draw == -math.inf else log_draw) == event["log_uniform"]
        assert event["accepted"] == (log_draw < threshold)
    assert replicas == search["final_replica_indices"]
    assert counts == search["proposal_counts"] and proposals == search["proposals"] and self_moves == search["self_proposals"]
    assert search["exchange_counts"] == {"proposed": exchanges, "accepted": exchange_accepted}
    assert search["cache_hits"] == 1+proposals-len(bank) and len(seen) == len(bank)
    assert search["completed_iterations"] == proposals//len(replicas)
    assert not search["global_optimality_claimed"] and not search["stationary_distribution_claimed"]
    return {"proposals": proposals, "keys": len(bank), "exchanges": exchanges}


def audit():
    record = json.loads((OUT/"result.json").read_text())
    require_frozen(record["freeze"], PATHS)
    benchmark, parent, cases = inputs(record["freeze"])
    save_new(OUT/"audit-started.json", {"freeze": record["freeze"], "start_unix": time.time()})
    limit_resources(2400, 2200)
    wall, cpu, rows, checked_keys, checked_proposals = time.monotonic(), time.process_time(), [], 0, 0
    try:
        source, selected = load_source()
        native = NativeMarginal(source, benchmark["build"])
        if (array_identity(source) != record["source_arrays"] or selected["counts"] != record["source"]
                or len(record["workloads"]) != 16 or record["native_build"] != benchmark["build"]):
            raise ValueError("Immutable source/native/fixed16 inventory changed")
        max_delta, bulk_bytes = 0., 0
        for spec, parent_spec in zip(record["workloads"], parent["workloads"], strict=True):
            assert artifact(ROOT/spec["path"]) == spec
            row = json.loads((ROOT/spec["path"]).read_text())
            assert row["parent"] == parent_spec
            metadata = json.loads((ROOT/parent_spec["path"]).read_text())
            case, seed = row["case"], row["seed"]
            assert (row["name"], case, seed, row["guidance"]) == (metadata["name"], metadata["case"], metadata["seed"], metadata["guidance"])
            bank = json.loads(gzip.decompress((ROOT/metadata["prediction"]["path"]).read_bytes()))["bank"]
            warm, completion = warm_key(bank, case, seed)
            cfg = config_for(case, seed)
            assert row["config"] == json.loads(json.dumps(asdict(cfg)))
            for key in ("fitted", "prediction"):
                assert artifact(ROOT/row[key]["path"]) == row[key]
                bulk_bytes += row[key]["bytes"]
            fitted = json.loads(gzip.decompress((ROOT/row["fitted"]["path"]).read_bytes()))
            records = tuple("".join("ABCDEF"[g] for g in c) for c in cases[case]["cipher"])
            counts = replay_rng_trace(fitted["search"], cfg)
            checked_keys += counts["keys"]
            checked_proposals += counts["proposals"]
            assert fitted["scores"][0]["units"] == list(warm)
            assert len(fitted["scores"]) == fitted["search"]["scored_keys"]
            for i, stored in enumerate(fitted["scores"]):
                value = score_key(native, records, stored["units"])
                value["units"] = list(value["units"])
                assert value == stored and fitted["search"]["bank"][i]["score"] == stored["log_key_mass"]
                assert fitted["search"]["bank"][i]["units"] == stored["units"]
                if i % 64 == 0 and resource_report(wall, cpu)["peak_rss_bytes"] > 4*1024**3:
                    raise MemoryError("Registered4GiB sampled audit cap")
            # Full-record scalar traversal cross-check on each warm and selected key.
            for index in sorted({0, fitted["search"]["best_index"]}):
                score = fitted["scores"][index]
                for r, observed in enumerate(records):
                    reference = marginal_python(source, score["units"], observed, 1/225,
                        max_nodes=2_000_000, max_edges=8_000_000)
                    expected = score["record_log_likelihoods"][r]
                    assert expected is not None and reference.log_likelihood != -math.inf
                    delta = abs(expected-reference.log_likelihood)
                    max_delta = max(max_delta, delta)
                    assert delta <= 1e-7
            before = read_key(source, native, warm, records, fitted["scores"][0])
            after = read_key(source, native, fitted["search"]["best_units"], records,
                fitted["scores"][fitted["search"]["best_index"]])
            assert before == fitted["warm_prediction"] and after == fitted["final_prediction"]
            prediction = json.loads((ROOT/row["prediction"]["path"]).read_text())
            assert prediction == json.loads(json.dumps({"completion": completion, "warm": before, "revised": after,
                "warm_key": warm, "revised_key": fitted["search"]["best_units"]}))
            assert diagnostics(fitted, cases[case], source, native) == row["diagnostic"]
            assert {k: v for k, v in fitted["search"].items() if k not in ("bank", "events")} == row["search_summary"]
            rows.append(row)
            print(f"Audited {row['name']}: {counts['keys']} keys", flush=True)
            del fitted, bank
        totals = {stage: {metric: sum(r["diagnostic"][stage][metric] for r in rows) for metric in
            ("edit_errors", "exact_records", "true_source_letters", "matched_gold_used_rows", "gold_used_rows")}
            for stage in ("warm", "revised", "gold_key_reader")}
        assert totals == record["totals"] and bulk_bytes == record["bulk_output_bytes"]
        clauses = {"all16_complete": len(rows) == 16,
            "edits_reduced_at_least_25_percent": totals["warm"]["edit_errors"] > 0 and totals["revised"]["edit_errors"] <= .75*totals["warm"]["edit_errors"],
            "better_edits_in_at_least12_cells": sum(r["diagnostic"]["revised"]["edit_errors"] < r["diagnostic"]["warm"]["edit_errors"] for r in rows) >= 12}
        assert clauses == record["exploratory_signal_clauses"]
        assert record["exploratory_signal"] == ("SUPPORTED" if all(clauses.values()) else "NOT_SUPPORTED")
        assert record["parent"] == artifact(ROOT/"results/SOURCE-GUIDE-001/result.json")
        assert array_identity(source) == record["source_arrays"]
        save_new(OUT/"audit.json", {"status": "PASS_full_score_rng_trace_and_reading_replay", "result": artifact(OUT/"result.json"),
            "workloads": 16, "checked_keys": checked_keys, "checked_proposals": checked_proposals,
            "maximum_python_native_delta": max_delta, "resources": resource_report(wall, cpu),
            "independent_agent_review": False, "independent_full_search_algorithm": False,
            "native_source_replay_and_scalar_traversal": True, "paid_spend_usd": 0})
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
