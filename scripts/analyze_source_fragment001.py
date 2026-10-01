"""Post-outcome exposed grammar coverage; no likelihood, fitting or neural calls."""
import argparse
import gzip
import json
import math

import numpy as np

from scripts.run_source_fragment001 import OUT, POOL, ROOT, artifact, save_new
from voynich.compact_suffix_source import CompactSuffixSource


def analyze():
    result = json.loads((OUT/"result.json").read_text())
    audit = json.loads((OUT/"audit.json").read_text())
    if audit["result"] != artifact(OUT/"result.json"):
        raise ValueError("Closed full audit required")
    source = CompactSuffixSource.load(ROOT/result["source"]["path"])
    assert artifact(ROOT/result["source"]["path"]) == result["source"]
    guide = json.loads((ROOT/"results/SOURCE-GUIDE-001/result.json").read_text())
    assert artifact(ROOT/guide["fixtures"]["path"]) == guide["fixtures"]
    cases = json.loads((ROOT/guide["fixtures"]["path"]).read_text())["fixtures"]
    depth_profile, histories, case_rows = [], [0]*13, []
    for depth in range(1, 13):
        keys = source.levels[depth]["contexts"]
        counts = []
        for case in cases:
            available = hits = 0
            for text in case["generation_source"]:
                for start in range(len(text)-depth+1):
                    code = 0
                    for row in text[start:start+depth]:
                        code = 23*code+row
                    index = int(np.searchsorted(keys, np.uint64(code)))
                    hits += index < len(keys) and int(keys[index]) == code
                    available += 1
            counts.append({"case": case["case"], "source_windows": available, "in_library": hits})
        depth_profile.append({"depth": depth, "library_size": len(keys), "cases": counts,
            "source_windows": sum(c["source_windows"] for c in counts), "in_library": sum(c["in_library"] for c in counts)})
    effective_context_mass = {depth: [] for depth in (1, 2, 4, 8, 12)}
    # Scalar history bookkeeping, not scoring targets or calling source.row().
    for case in cases:
        for text in case["generation_source"]:
            for position in range(len(text)):
                totals, max_depth = {}, 0
                for depth in range(1, min(12, position)+1):
                    code = 0
                    for row in text[position-depth:position]:
                        code = 23*code+row
                    level = source.levels[depth]
                    index = int(np.searchsorted(level["contexts"], np.uint64(code)))
                    n = int(level["totals"][index]) if index < len(level["contexts"]) and int(level["contexts"][index]) == code else 0
                    totals[depth] = n
                    if n:
                        max_depth = depth
                histories[max_depth] += 1
                for depth, values in effective_context_mass.items():
                    values.append(1-math.prod(64/(totals.get(d, 0)+64) for d in range(depth, 13)))
    # Independently decode every depth12 numeric code into a tuple set.
    tuple_library = set()
    for original in source.levels[12]["contexts"]:
        code, letters = int(original), []
        for _ in range(12):
            letters.append(code % 23)
            code //= 23
        tuple_library.add(tuple(reversed(letters)))
    for case, window_spec in zip(cases, result["windows"], strict=True):
        assert artifact(ROOT/window_spec["scans"]["path"]) == window_spec["scans"]
        windows = json.loads(gzip.decompress((ROOT/window_spec["scans"]["path"]).read_bytes()))
        gold = case["generation_key"]
        eligible, exact_binding, all_matches = 0, 0, 0
        for window in windows:
            text = case["generation_source"][window["record"]]
            offset, positions = 0, {}
            for i, row in enumerate(text):
                if i+12 <= len(text):
                    positions[offset] = i
                offset += len(POOL[gold[row]])
            eligible += window["offset"] in positions
            for match in window["scan"]["matches"]:
                assigned = [r for r, unit in enumerate(match["partial_key"]) if unit >= 0]
                assert assigned
                exact_binding += all(match["partial_key"][r] == gold[r] for r in assigned)
                all_matches += 1
        full_windows = [tuple(text[i:i+12]) for text in case["generation_source"] for i in range(len(text)-11)]
        tuple_hits = sum(window in tuple_library for window in full_windows)
        assert tuple_hits == depth_profile[-1]["cases"][case["case"]]["in_library"]
        arms = []
        for arm in ("fragment", "random_length_matched"):
            row = json.loads((OUT/f"case{case['case']}-{arm}.json").read_text())
            assert artifact(ROOT/row["fitted"]["path"]) == row["fitted"]
            fitted = json.loads(gzip.decompress((ROOT/row["fitted"]["path"]).read_bytes()))
            supported = [s for s in fitted["scores"] if s["log_key_mass"] is not None]
            arms.append({"arm": arm, "unique_keys": len(fitted["scores"]), "supported": len(supported),
                "unsupported": len(fitted["scores"])-len(supported), "best_index": fitted["best_index"],
                "score_gain": row["diagnostic"]["score_gain_nats"],
                "warm_edits": row["diagnostic"]["warm"]["edit_errors"], "final_edits": row["diagnostic"]["final"]["edit_errors"],
                "used_rows": row["diagnostic"]["final"]["used_rows"], "matched_used_rows": row["diagnostic"]["final"]["matched_used_rows"]})
        case_rows.append({"case": case["case"], "eligible_selected_true12_starts": eligible,
            "all_true12_source_windows": len(full_windows), "true12_in_library": tuple_hits,
            "retained_matches": all_matches, "retained_partial_bindings_entirely_gold_correct": exact_binding,
            "arms": arms})
    return {"result": artifact(OUT/"result.json"), "audit": artifact(OUT/"audit.json"),
        "fixtures": guide["fixtures"], "source": result["source"], "depth_profile": depth_profile,
        "history_max_retained_suffix_depth_counts": histories,
        "mean_source_mixture_weight_at_depth_or_above": {str(d): math.fsum(v)/len(v) for d, v in effective_context_mass.items()},
        "cases": case_rows, "all_source_letters": sum(histories),
        "scope": "Exploratory AFTER outcomes, four exposed cases. No new scores/search/neural calls or success-criterion revision. Mixture weights are coefficients, not target probability contributions."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    derived = analyze()
    destination = OUT/"post-outcome-coverage.json"
    if args.write:
        save_new(destination, derived)
    elif json.loads(destination.read_text()) != derived:
        raise ValueError("Closed post-outcome arithmetic changed")
    print(json.dumps({"cases": derived["cases"], "depth_profile": [{k:v for k,v in d.items() if k != 'cases'} for d in derived['depth_profile']],
                      "history_counts": derived["history_max_retained_suffix_depth_counts"],
                      "context_mixture_weights": derived["mean_source_mixture_weight_at_depth_or_above"]}))
