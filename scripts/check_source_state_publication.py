"""Post-outcome static hashes/arithmetic/alternate edits; NO model/search calls."""
from __future__ import annotations

import gzip
import json
import math

import numpy as np

from scripts.run_joint_key_train001 import BASE_PATHS, INPUT_PATHS
from scripts.run_source_coupling001 import PATHS as COUPLING_PATHS
from scripts.run_source_fragment001 import PATHS as FRAGMENT_PATHS
from scripts.run_source_state_systems001 import (
    ARMS, OUT, PATHS, POOL, ROOT, artifact, config, require_frozen, save_new,
)


def edits(a, b):
    # Independent one-row dynamic program, no project edit_distance import.
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def total(logs):
    logs = [v for v in logs if v is not None]
    if not logs:
        return None
    best = max(logs)
    return best+math.log(math.fsum(math.exp(v-best) for v in logs))


def check():
    result = json.loads((OUT/"result.json").read_text())
    audit = json.loads((OUT/"audit.json").read_text())
    assert audit["status"] == "PASS_full_native_state_and_prediction_replay"
    assert audit["result"] == artifact(OUT/"result.json")
    assert artifact(ROOT/result["fixtures"]["path"]) == result["fixtures"]
    assert artifact(ROOT/result["source"]["path"]) == result["source"]
    require_frozen(result["freeze"], PATHS)
    require_frozen("9053abf", [*BASE_PATHS, *INPUT_PATHS])
    require_frozen("6e6eda61", FRAGMENT_PATHS)
    require_frozen("7d925af", COUPLING_PATHS)
    fixtures = json.loads((ROOT/result["fixtures"]["path"]).read_text())["fixtures"]
    with np.load(ROOT/result["source"]["path"], allow_pickle=False) as counts:
        alphabet = str(counts["alphabet"])
    checked, bulk, rows, max_delta = 0, 0, [], 0.
    for i, spec in enumerate(result["workloads"]):
        assert artifact(ROOT/spec["path"]) == spec
        row = json.loads((ROOT/spec["path"]).read_text())
        case, arm = i//4, ARMS[i%4]
        assert (row["case"], row["arm"], row["config"]) == (case, arm, config(arm))
        for field in ("fitted", "prediction"):
            assert artifact(ROOT/row[field]["path"]) == row[field]
            bulk += row[field]["bytes"]
        fitted = json.loads(gzip.decompress((ROOT/row["fitted"]["path"]).read_bytes()))
        assert {k: v for k, v in fitted.items() if k not in ("terminals", "trace")} == row["summary"]
        terms = fitted["terminals"]
        assert len(terms) == min(512, fitted["terminal_states"])
        assert len({tuple(t["used_key"]) for t in terms}) == len(terms)
        assert all(len(t["used_key"]) == 23 and all(-1 <= k < 42 for k in t["used_key"])
            and t["best_leaf_log_mass"] <= t["log_mass"]+1e-10 for t in terms)
        for recorded, calculated in ((fitted["returned_terminal_log_mass"], total(t["log_mass"] for t in terms)),
                                    (fitted["evidence_log_upper"], total([fitted["found_log_mass"], fitted["unresolved_log_mass_upper"]]))):
            if recorded is None or calculated is None:
                assert recorded is calculated
            else:
                delta = abs(recorded-calculated)
                assert delta <= 1e-9
                max_delta = max(max_delta, delta)
        assert fitted["interval_arithmetic_certificate"] is False and fitted["full_key_posterior_claimed"] is False
        prediction = json.loads((ROOT/row["prediction"]["path"]).read_text())
        if prediction["status"] == "read":
            truth = ["".join(alphabet[r] for r in text) for text in fixtures[case]["generation_source"]]
            text = prediction["prediction"]["source_records"]
            record_edits = [edits(a, b) for a, b in zip(text, truth, strict=True)]
            assert record_edits == row["diagnostic"]["record_edits"]
            assert sum(record_edits) == row["diagnostic"]["edit_errors"]
            assert sum(a == b for a, b in zip(text, truth, strict=True)) == row["diagnostic"]["exact_records"]
            cipher = ["".join("ABCDEF"[g] for g in record) for record in fixtures[case]["cipher"]]
            assert ["".join(prediction["key"][alphabet.index(c)] for c in record) for record in text] == cipher
            partial = terms[0]["used_key"]
            fill = list(map(int, np.random.default_rng(79101+case).integers(42, size=23)))
            assert fill == prediction["filler_indices"] and partial == prediction["partial_key"]
            assert [POOL[fill[r] if k < 0 else k] for r, k in enumerate(partial)] == prediction["key"]
            checked += 2
        else:
            assert not terms and row["diagnostic"]["edit_errors"] is None
        rows.append(row)
    assert bulk == result["bulk_output_bytes"] == 273963
    assert checked == audit["alternate_fixed_key_record_scores"] == 26
    assert rows[3]["diagnostic"]["edit_errors"] == 0 and rows[3]["diagnostic"]["matched_gold_used_rows"] == 19
    for arm, expected in (("guided", 899), ("wide_guided", 788)):
        assert sum(r["diagnostic"]["edit_errors"] for r in rows if r["arm"] == arm) == expected
    save_new(OUT/"publication-audit.json", {"status": "PASS_static_bindings_and_alternate_edits",
        "result": artifact(OUT/"result.json"), "audit": artifact(OUT/"audit.json"),
        "checked_cells": len(rows), "alternate_edit_and_literal_records": checked,
        "maximum_mass_arithmetic_delta": max_delta, "no_search_score_neural_calls": True,
        "frozen_state_paths": len(PATHS), "frozen_training_paths": len(BASE_PATHS)+len(INPUT_PATHS),
        "frozen_fragment_paths": len(FRAGMENT_PATHS), "frozen_coupling_paths": len(COUPLING_PATHS),
        "post_outcome_utility": True, "paid_spend_usd": 0})


if __name__ == "__main__":
    check()
