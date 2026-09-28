"""Read only completed model freezes and their hash-verified search traces.

No ciphertext, corpus, transfer, answers or evaluation report is opened. This
checks recorded search integrity and budget coverage, without replaying data
scores or assessing plaintext correctness. Missing freezes remain pending.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-001"
CASE_NAMES = tuple(f"{family}-key{key}{suffix}" for family in ("A", "B")
                   for key in (1, 2) for suffix in ("", "-shuffle"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _score(score, label):
    required = {"model_bits", "data_bits", "total_bits", "log_likelihood"}
    if not isinstance(score, dict) or required - set(score):
        raise ValueError(f"Incomplete score: {label}")
    if any(isinstance(score[key], bool) or not isinstance(score[key], (int, float))
           or not math.isfinite(score[key]) for key in required):
        raise ValueError(f"Invalid score numbers: {label}")
    if type(score["model_bits"]) is not int or score["model_bits"] < 0:
        raise ValueError(f"Invalid model-bit count: {label}")
    if abs(score["total_bits"] - (score["model_bits"] + score["data_bits"])) > 1e-9:
        raise ValueError(f"Score components do not add: {label}")


def validate_case(frozen: dict, full: dict) -> dict:
    """Validate one complete in-memory trace and return a compact summary."""
    for key in ("channel", "score", "config", "seconds", "stop_reason", "proposals", "completed_candidates"):
        if full[key] != frozen[key]:
            raise ValueError(f"Frozen/full mismatch: {key}")
    config, trace = full["config"], full["trace"]
    requested_states = config["state_counts"]
    requested_restarts = config["restarts_per_state"]
    proposals, completed, scored_records = 0, 0, set()
    attempted, successful = set(), set()
    initializations, kinds, accepted_kinds, rejected = Counter(), Counter(), Counter(), Counter()
    phases, accepted_initializations, accepted_proposals = Counter(), 0, 0
    details = defaultdict(lambda: {"initialization_attempts": 0, "initialization_methods": Counter(),
                                   "proposals": 0, "accepted_proposals": 0, "scored_stages": 0})
    best = None
    returned_seen = full["channel"] is None
    for index, event in enumerate(trace):
        state_count, restart = event["state_count"], event["restart"]
        key = restart, state_count
        if state_count not in requested_states or not 0 <= restart < requested_restarts:
            raise ValueError("Trace used an unrequested state count or restart")
        item = details[key]
        if event["event"] == "initialization":
            attempted.add(key)
            item["initialization_attempts"] += 1
            method = event["method"]
            initializations[method] += 1
            item["initialization_methods"][method] += 1
            if not 0 <= event["attempt"] < config["initialization_attempts"]:
                raise ValueError("Initialization attempt exceeds budget")
            if event["accepted"]:
                successful.add(key)
                accepted_initializations += 1
        elif event["event"] == "proposal":
            if key not in successful:
                raise ValueError("Proposal appeared without a supported initialization")
            if not 0 <= event["step"] < config["proposals_per_restart"]:
                raise ValueError("Proposal index exceeds budget")
            proposals += 1
            item["proposals"] += 1
            kind = event["move"]["kind"]
            kinds[kind] += 1
            if event["accepted"]:
                accepted_proposals += 1
                item["accepted_proposals"] += 1
                accepted_kinds[kind] += 1
        else:
            raise ValueError("Unknown search event")
        if not event["accepted"]:
            rejected[event["rejection_reason"]] += 1
        scored = []
        for stage in event["stages"]:
            if stage["status"] != "scored":
                if stage["score"] is not None or stage["status"] not in ("unsupported", "time_limit"):
                    raise ValueError("An incomplete stage has a score or unknown status")
                continue
            _score(stage["score"], f"event {index}")
            completed += 1
            item["scored_stages"] += 1
            phases[stage["phase"]] += 1
            scored_records.add(stage["records_scored"])
            scored.append(stage)
            value = stage["score"]["total_bits"]
            best = value if best is None else min(best, value)
            if stage["channel"] == full["channel"] and stage["score"] == full["score"]:
                returned_seen = True
        if scored:
            chosen = event["selected_score"]
            if chosen["total_bits"] != min(stage["score"]["total_bits"] for stage in scored):
                raise ValueError("Event selected a worse raw/refined score")
            if not any(stage["channel"] == event["selected_channel"] and stage["score"] == chosen for stage in scored):
                raise ValueError("Selected event model was not fully scored")
            if event["event"] == "proposal" and event["accepted"]:
                if event["parent_score"]["total_bits"] - chosen["total_bits"] <= config["improvement_tolerance_bits"]:
                    raise ValueError("Accepted proposal does not exceed the configured improvement threshold")
        elif event["accepted"]:
            raise ValueError("Accepted event has no completed score")
        if event["best_total_bits"] != best:
            raise ValueError("Trace best score is not the minimum completed stage score")
    if len(scored_records) > 1:
        raise ValueError("Completed stages scored different record counts")
    if proposals != full["proposals"] or completed != full["completed_candidates"]:
        raise ValueError("Recorded proposal/completed-candidate counts differ from trace")
    if any(item["proposals"] > config["proposals_per_restart"]
           or item["initialization_attempts"] > config["initialization_attempts"] for item in details.values()):
        raise ValueError("Trace event counts exceed a configured budget")
    if best is None:
        if full["channel"] is not None or full["score"] is not None:
            raise ValueError("Returned model has no fully scored candidate")
    elif (full["channel"] is None or full["score"]["total_bits"] != best or not returned_seen):
        raise ValueError("Returned model is not a recorded minimum-scored candidate")
    selected = None
    if full["channel"] is not None:
        channel = full["channel"]
        alternatives = [emission for row in channel["rows"] for emission in row["emissions"]]
        selected = {"states": len(channel["states"]), "rows": len(channel["rows"]),
                    "alternatives": len(alternatives),
                    "alternatives_per_row": dict(sorted(Counter(str(len(row["emissions"])) for row in channel["rows"]).items())),
                    "emission_length_counts": dict(sorted(Counter(str(len(item["glyphs"])) for item in alternatives).items())),
                    "unique_units": len({item["glyphs"] for item in alternatives}),
                    "distinct_destination_unit_pairs": len({(item["next_state"], item["glyphs"]) for item in alternatives})}
    restart_details = []
    for (restart, state_count), item in sorted(details.items()):
        restart_details.append({"restart": restart, "state_count": state_count,
                                **item, "initialization_methods": dict(item["initialization_methods"])})
    return {"status": "verified", "source_freeze": frozen["source_freeze"],
            "stop_reason": full["stop_reason"], "seed": config["seed"],
            "requested_state_counts": requested_states,
            "attempted_state_counts": sorted({state_count for _, state_count in attempted}),
            "requested_restarts_per_state": requested_restarts,
            "attempted_restart_state_pairs": len(attempted), "supported_restart_state_pairs": len(successful),
            "maximum_restart_state_pairs": len(requested_states) * requested_restarts,
            "requested_proposals_per_restart": config["proposals_per_restart"],
            "maximum_proposals": len(requested_states) * requested_restarts * config["proposals_per_restart"],
            "proposals": proposals, "completed_candidates": completed, "completed_stage_phases": dict(phases),
            "records_per_completed_stage": next(iter(scored_records)) if scored_records else None,
            "initialization_methods": dict(initializations), "accepted_initializations": accepted_initializations,
            "proposal_move_counts": dict(kinds), "accepted_move_counts": dict(accepted_kinds),
            "accepted_proposals": accepted_proposals, "rejection_reasons": dict(rejected),
            "restart_details": restart_details, "selected_model": selected, "score": full["score"],
            "minimum_recorded_total_bits": best, "returned_minimum_verified": True,
            "wall_seconds": frozen["seconds"], "cpu_seconds_before_final_write": frozen["cpu_seconds_before_final_write"],
            "peak_rss_bytes": frozen["peak_rss_bytes"], "configured_wall_seconds": config["max_seconds"],
            "unit_pool_size": len(full["unit_pool"])}


def summarize(root: Path, names=CASE_NAMES):
    result = {"experiment": EXPERIMENT, "scope": "recorded_search_integrity_and_budget_coverage_only",
              "score_replay_performed": False, "ciphertext_corpus_transfer_answers_or_evaluation_opened": False,
              "cases": {}, "pending": [], "errors": []}
    for name in names:
        path = root / f"results/{EXPERIMENT}/{name}_freeze.json"
        if not path.is_file():
            result["pending"].append(name)
            continue
        try:
            freeze_raw = path.read_bytes()
            frozen = json.loads(freeze_raw)
            if frozen["case_id"] != name:
                raise ValueError("Freeze case identity differs")
            artifact = frozen["full_output"]
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Full-output path is not repository-relative")
            compressed = (root / relative).read_bytes()
            if digest(compressed) != artifact["sha256"] or len(compressed) != artifact["bytes"]:
                raise ValueError("Full-output compressed hash/size differs")
            full = json.loads(gzip.decompress(compressed))
            case = validate_case(frozen, full)
            case.update(freeze_sha256=digest(freeze_raw), full_output_sha256=digest(compressed),
                        full_output_bytes=len(compressed))
            result["cases"][name] = case
        except (json.JSONDecodeError, EOFError, FileNotFoundError):
            # Do not retry a potentially incomplete write while a fit is active.
            result["pending"].append(name)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            result["errors"].append({"case_id": name, "error": str(exc)})
    result["status"] = "fail" if result["errors"] else ("pending" if result["pending"] else "pass")
    result["cases_verified"] = len(result["cases"])
    result["summarizer_source_sha256"] = digest(Path(__file__).read_bytes())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"results/{EXPERIMENT}/search_summary.json")
    arguments = parser.parse_args(argv)
    result = summarize(ROOT)
    destination = ROOT / arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cases_verified": result["cases_verified"],
                      "pending": result["pending"], "errors": result["errors"]}, indent=2))
    if result["errors"]:
        raise SystemExit("Recorded search trace integrity check failed")


if __name__ == "__main__":
    main()
