"""Post-outcome read-only closed-artifact publication checks; no model calls."""
from __future__ import annotations

import argparse
import json
import math

from scripts.run_source_prune_diag002 import (
    ARMS, OUT, PARENT, PATHS, artifact, config, load_bound, require_frozen, save_new,
)


def check():
    result = json.loads((OUT/"result.json").read_text())
    require_frozen(result["freeze"], PATHS)
    audit = json.loads((OUT/"audit.json").read_text())
    assert audit["status"] == "PASS" and audit["result"] == artifact(OUT/"result.json")
    assert audit["ordinary_outputs_match_all_original_cells"] is True
    assert audit["maximum_alternate_delta"] <= 1e-7
    assert result["parent_result"] == artifact(PARENT/"result.json")
    assert result["parent_audit"] == artifact(PARENT/"audit.json")
    parent = load_bound(result["parent_result"])
    count, total, losses, surviving = 0, 0, {}, []
    for case in range(4):
        for arm in ARMS:
            name = f"case{case}-{arm}"
            spec = result["workloads"][count]
            assert spec == artifact(OUT/f"{name}.json")
            row = load_bound(spec)
            assert row["name"] == name and row["config"] == config(arm)
            assert row["original_cell"] == parent["workloads"][count]
            original = load_bound(row["original_cell"])
            data = load_bound(row["diagnostic"])
            assert data["control"] == load_bound(original["fitted"])
            assert row["ordinary_output_exact_match"] is True
            trace, states = data["trace"], data["gold_states"]
            assert len(trace) == len(states) == row["summary"]["processed_gold_states"]
            death, alive, previous = None, True, -1
            for i, (d, state) in enumerate(zip(trace, states, strict=True)):
                assert d["processed"] and len(state) == 30 and state[29] == 1
                assert previous < d["layer"] == d["target_glyph_layer"] == sum(state[:2])
                assert math.isfinite(d["prefix_log_mass"])
                assert bool(d["alive_before"]) == alive
                if alive and not d["alive_after"]:
                    assert death is None
                    death = i
                if not alive:
                    assert not d["path_present"] and not d["path_expanded"] and not d["alive_after"]
                alive, previous = bool(d["alive_after"]), d["layer"]
            s, end = row["summary"], trace[-1]
            assert s["literal_path_survived"] == alive
            assert s["first_loss"] == (trace[death] if death is not None else None)
            assert s["terminal_used_mapping_best_rank"] == (end["terminal_used_mapping_best_rank"] or None)
            assert s["terminal_used_mapping_returned"] == bool(end["terminal_used_mapping_returned"])
            if alive:
                surviving.append(name)
            else:
                cause = "geometric_prebeam" if not trace[death]["pre_kept"] else "final_beam"
                assert s["first_loss_stage"] == cause
                losses[name] = {"cause": cause, "source_letters_consumed": death,
                    "total_source_letters": len(trace)-1, "glyph_layer": trace[death]["layer"],
                    "assigned_rows": sum(k >= 0 for k in states[death][4:27]),
                    "aggregate_pre_rank": trace[death]["pre_rank"],
                    "aggregate_final_rank": trace[death]["guide_rank"] or None,
                    "raw_ghost_pre_strict_better": trace[death]["ghost_pre_better"],
                    "raw_ghost_final_strict_better": trace[death]["ghost_guide_better"]}
            count += 1
            total += row["diagnostic"]["bytes"]
    assert count == len(result["workloads"]) == audit["workloads"] == result["ordinary_output_exact_match_cells"] == 16
    assert total == result["bulk_output_bytes"]
    for resource in (result["resources"], audit["resources"]):
        assert 0 <= resource["wall_seconds"] <= 1800 and 0 <= resource["cpu_seconds"] <= 1600
        assert 0 < resource["peak_rss_bytes"] <= 2*1024**3 and resource["paid_spend_usd"] == 0
    return {"status": "PASS_read_only_closed_publication_checks", "result": artifact(OUT/"result.json"),
        "audit": artifact(OUT/"audit.json"), "frozen_paths_checked": len(PATHS), "workloads": count,
        "literal_survivors": surviving, "first_losses": losses, "bulk_bytes": total,
        "no_new_source_search_scoring_training_or_neural_calls": True,
        "independent_full_search_audit": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(OUT/"publication-audit.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
