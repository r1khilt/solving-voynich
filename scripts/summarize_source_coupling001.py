"""Post-outcome closed-archive arithmetic only; no scoring or solver calls."""
import gzip
import json
import math

from scripts.run_blind_channel_dev004 import save_new
from scripts.run_latin_source_model001 import ROOT, artifact


def summarize():
    out = ROOT/"results/SOURCE-COUPLING-001"
    result = json.loads((out/"result.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    assert audit["result"] == artifact(out/"result.json")
    assert audit["status"] == "PASS_all_key_interventions_native_and_scalar_replay"
    rows = []
    for spec in result["workloads"]:
        assert artifact(ROOT/spec["path"]) == spec
        row = json.loads((ROOT/spec["path"]).read_text())
        assert artifact(ROOT/row["payload"]["path"]) == row["payload"]
        payload = json.loads(gzip.decompress((ROOT/row["payload"]["path"]).read_bytes()))
        summary = row["summaries"]["context"]["gold"]
        base, gold, used = payload["base"], payload["gold"], payload["used_rows"]
        single = payload["bank"][summary["best_single_bank_index"]]["units"]
        def value(index):
            return payload["bank"][index]["models"]["context"]["log_key_mass"]
        singles = [value(r["index"]) for r in payload["families"]["gold"]["faces"] if len(r["rows"]) == 1]
        prediction = None if None in singles else math.fsum(v-value(0) for v in singles)
        endpoint = summary["endpoint_gain_nats"]
        lower_gains = [value(r["index"])-value(0) for r in payload["families"]["gold"]["faces"] if value(r["index"]) is not None]
        rows.append({"name": row["name"],
            "best_single_gold_used_matches_before": sum(base[i] == gold[i] for i in used),
            "best_single_gold_used_matches_after": sum(single[i] == gold[i] for i in used),
            "known_wrong_single_corrections_unsupported": sum(v is None for v in singles),
            "additive_gold_single_prediction_nats": prediction,
            "gold_used_endpoint_gain_nats": endpoint,
            "endpoint_minus_additive_prediction_nats": None if prediction is None else endpoint-prediction,
            "endpoint_above_every_supported_size_le3_gold_correction_nats": endpoint-max([0., *lower_gains])})
    assert len(rows) == 16
    return {"status": "PASS_closed_archive_derived_arithmetic", "result": artifact(out/"result.json"),
        "audit": artifact(out/"audit.json"),
        "scope": "Post-outcome exploratory arithmetic only, not preregistered signal or new scoring/search", "rows": rows,
        "best_single_increases_gold_used_matches_cells": sum(r["best_single_gold_used_matches_after"] > r["best_single_gold_used_matches_before"] for r in rows),
        "best_single_decreases_gold_used_matches_cells": sum(r["best_single_gold_used_matches_after"] < r["best_single_gold_used_matches_before"] for r in rows),
        "positive_endpoint_with_unsupported_single_gold_correction_cells": sum(r["known_wrong_single_corrections_unsupported"] > 0 for r in rows),
        "additive_prediction_finite_cells": sum(r["additive_gold_single_prediction_nats"] is not None for r in rows),
        "no_new_source_or_neural_or_native_score_evaluations": True, "paid_spend_usd": 0}


if __name__ == "__main__":
    result = summarize()
    path = ROOT/"results/SOURCE-COUPLING-001/post-outcome-arithmetic.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
        print("PASS existing post-outcome arithmetic; no overwrite or new scoring")
    else:
        save_new(path, result)
        print("Saved post-outcome arithmetic; no new scoring")
