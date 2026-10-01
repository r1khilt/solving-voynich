"""One full intervention/score replay and alternate scalar DP audit."""
import argparse
import gzip
import itertools
import json
import math
import signal
import time

from scripts.run_source_coupling001 import (
    MODELS, OUT, PATHS, POOL, ROOT, NativeMarginal, array_identity, artifact,
    iid_control, inputs, limit_resources, load_source, probe, replace_rows,
    require_frozen, resource_report, save_new, score_key, signals,
)
from voynich.native_suffix_marginal import marginal_python


def validate_inventory(payload, base, gold, used, pool=POOL):
    """Separate literal enumeration, independent of probe's scoring/cache loops."""
    bank = payload["bank"]
    assert all(r["index"] == i for i, r in enumerate(bank))
    keys = [tuple(r["units"]) for r in bank]
    assert len(set(keys)) == len(keys) <= payload["maximum_unique_keys"]
    assert keys[0] == tuple(base) and keys[payload["gold_full_key_index"]] == tuple(gold)
    expected_local = {(row, unit) for row, old in enumerate(base) for unit in pool if unit != old}
    assert {(r["row"], r["unit"]) for r in payload["local"]} == expected_local
    assert len(payload["local"]) == len(expected_local)
    for r in payload["local"]:
        key = list(base)
        key[r["row"]] = r["unit"]
        assert tuple(key) == keys[r["index"]]
    mismatch = sorted(r for r in used if base[r] != gold[r])
    assert payload["mismatched_used_rows"] == mismatch
    expect_subsets = {s for k in (1, 2, 3) for s in itertools.combinations(mismatch, k)}
    for family in ("gold", "decoy"):
        spec = payload["families"][family]
        target = gold if family == "gold" else payload["decoy"]
        assert spec["target"] == list(target)
        assert {tuple(r["rows"]) for r in spec["faces"]} == expect_subsets
        assert len(spec["faces"]) == len(expect_subsets)
        for r in spec["faces"]:
            assert keys[r["index"]] == replace_rows(base, r["rows"], target)
        assert keys[spec["endpoint_index"]] == replace_rows(base, mismatch, target)
    for row in mismatch:
        assert payload["decoy"][row] not in (base[row], gold[row])
        assert len(payload["decoy"][row]) == len(gold[row])
    assert payload["oracle_assisted"] and not payload["unassisted_recovery_claimed"] and not payload["global_barrier_claimed"]


def focus_indices(payload):
    """At most10 unique keys percell, at most40 scalar record/model calls."""
    indices = {0, payload["gold_full_key_index"], *(s["endpoint_index"] for s in payload["families"].values())}
    for model in MODELS:
        for family in ("gold", "decoy"):
            s = payload["summaries"][model][family]
            indices.update(i for i in (s["best_single_bank_index"], s["best_joint_bank_index"]) if i is not None)
    assert len(indices) <= 10
    return sorted(indices)


def audit():
    record = json.loads((OUT/"result.json").read_text())
    require_frozen(record["freeze"], PATHS)
    benchmark, parent, cases = inputs(record["freeze"])
    save_new(OUT/"audit-started.json", {"freeze": record["freeze"], "start_unix": time.time()})
    limit_resources(1800, 1600)
    wall, cpu, rows, checked, scalar_calls, delta, bulk = time.monotonic(), time.process_time(), [], 0, 0, 0., 0
    try:
        source, selected = load_source()
        sources = {"context": source, "iid": iid_control(source)}
        scorers = {m: NativeMarginal(s, benchmark["build"]) for m, s in sources.items()}
        assert record["source"] == selected["counts"] and record["native_build"] == benchmark["build"]
        assert record["source_arrays"] == {m: array_identity(s) for m, s in sources.items()}
        assert len(record["workloads"]) == len(parent["workloads"]) == 16
        for spec, old_spec in zip(record["workloads"], parent["workloads"], strict=True):
            assert artifact(ROOT/spec["path"]) == spec
            row = json.loads((ROOT/spec["path"]).read_text())
            assert row["parent"] == old_spec
            old = json.loads((ROOT/old_spec["path"]).read_text())
            assert (row["name"], row["case"], row["seed"], row["guidance"]) == (old["name"], old["case"], old["seed"], old["guidance"])
            assert artifact(ROOT/row["payload"]["path"]) == row["payload"]
            bulk += row["payload"]["bytes"]
            payload = json.loads(gzip.decompress((ROOT/row["payload"]["path"]).read_bytes()))
            fixture = cases[row["case"]]
            cipher = tuple("".join("ABCDEF"[g] for g in c) for c in fixture["cipher"])
            base = json.loads((ROOT/old["prediction"]["path"]).read_text())["revised_key"]
            gold = tuple(POOL[i] for i in fixture["generation_key"])
            used = {i for text in fixture["generation_source"] for i in text}
            validate_inventory(payload, base, gold, used)
            mapping = {}
            for index, stored in enumerate(payload["bank"]):
                key = tuple(stored["units"])
                values = {m: score_key(scorer, cipher, key) for m, scorer in scorers.items()}
                assert json.loads(json.dumps(values)) == stored["models"]
                mapping[key] = values
                checked += 1
                if index % 64 == 0 and resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                    raise MemoryError("Registered2GiB sampled audit cap")
            reconstructed = probe(base, gold, used, 76111+101*row["case"]+(row["seed"]-75511), mapping.__getitem__)
            assert json.loads(json.dumps(reconstructed)) == payload
            assert row["summaries"] == payload["summaries"] and row["unique_keys"] == len(payload["bank"])
            assert row["mismatched_used_rows"] == len(payload["mismatched_used_rows"])
            assert row["maximum_unique_keys"] == payload["maximum_unique_keys"]
            for index in focus_indices(payload):
                stored = payload["bank"][index]
                for model, s in sources.items():
                    for r, observed in enumerate(cipher):
                        value = marginal_python(s, stored["units"], observed, 1/225, max_nodes=2_000_000, max_edges=8_000_000)
                        expected = stored["models"][model]["record_log_likelihoods"][r]
                        assert (expected is None) == (value.log_likelihood == -math.inf)
                        difference = 0. if expected is None else abs(value.log_likelihood-expected)
                        assert difference <= 1e-7
                        delta = max(delta, difference)
                        scalar_calls += 1
            assert abs(payload["bank"][0]["models"]["context"]["log_key_mass"]-old["search_summary"]["best_score"]) <= 1e-7
            assert abs(payload["bank"][payload["gold_full_key_index"]]["models"]["context"]["log_key_mass"]-old["diagnostic"]["gold_full_key_log_mass"]) <= 1e-7
            rows.append(row)
            print(f"Audited {row['name']}: {row['unique_keys']} paired-source key scores", flush=True)
            del mapping, payload, reconstructed
        assert checked == record["unique_key_evaluations"] and bulk == record["bulk_output_bytes"]
        assert signals(rows) == record["signals"]
        assert record["parent"] == artifact(ROOT/"results/SOURCE-REVISE-001/result.json")
        assert record["source_arrays"] == {m: array_identity(s) for m, s in sources.items()}
        save_new(OUT/"audit.json", {"status": "PASS_all_key_interventions_native_and_scalar_replay", "result": artifact(OUT/"result.json"),
            "workloads": len(rows), "checked_unique_keys": checked, "checked_native_record_scores": 4*checked,
            "scalar_record_scores": scalar_calls, "maximum_scalar_native_delta": delta,
            "resources": resource_report(wall, cpu), "paid_spend_usd": 0,
            "independent_agent_review": False, "independent_intervention_inventory": True,
            "summary_replay_reuses_probe": True, "not_a_recovery_solver": True})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"audit-failure.json", {"error": repr(exc), "completed_workloads": len(rows), "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()
