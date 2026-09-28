"""After-fitting trace accounting; never open ciphertext, sources or answers.

This independently checks trace structure, literal code lengths and arithmetic,
not the numerical likelihoods of empirical candidates. Compact-file hashes are
reported for the subsequent model freeze; archive hashes/sizes are checked
against those compact files. Git-freeze authentication is external to this tool.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-003"
CASE_NAMES = ("B-key1", "B-key1-shuffle", "B-key2", "B-key2-shuffle")
SOURCE_FREEZE = "830a53f79f318d90e933fe67b4e4c559bd875cdb"
SOURCE_SHA256 = "7de1c233b3ea69eb632247062f97db86040666daa13ba73bba28bbb51b3493b0"
MANIFEST_SHA256 = "8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea"
SOURCE_PATH = "results/BLIND-CHANNEL-DEV-001/source_selection.json"
MANIFEST_PATH = "data/manifests/blind_channel_dev001.json"
# Registered side information confirmed from the builder/protocol, not loaded
# from a ciphertext artifact or selected-source payload at audit time.
CONTEXT = {"source_alphabet": list("abcdefghiklmnopqrstuxyz"), "glyph_alphabet": list("ABCDEF"),
           "denominator": 32, "max_states": 2, "max_emission_length": 2,
           "max_alternatives": 3, "source_count": 1, "stop_probability": 1 / 225}
SOURCE_PATHS = {
    "scripts/run_blind_channel_dev003.py", "src/voynich/unit_channel_search.py",
    "src/voynich/batched_unit_channel.py", "src/voynich/finite_state_channel.py",
    "src/voynich/finite_state_channel_fit.py", "src/voynich/finite_state_channel_search.py",
    "scripts/run_blind_channel_dev001.py", "scripts/evaluate_blind_channel_dev001.py",
    "scripts/evaluate_naibbe001.py", "src/voynich/naibbe_key_search.py",
    "scripts/audit_blind_channel_dev001.py", "docs/experiments/BLIND-CHANNEL-DEV-003.md",
    "tests/test_batched_unit_channel.py", "tests/test_batched_unit_channel_independent.py",
    "tests/test_unit_channel_search.py", "tests/test_unit_channel_search_independent.py",
    "tests/test_run_blind_channel_dev003.py", MANIFEST_PATH, SOURCE_PATH,
}
PAYLOAD_FIELDS = {
    "schema_version", "candidate_family", "channel", "score", "units", "trace", "stop_reason",
    "evaluated_neighbors", "completed_sweeps", "local_optima", "best_is_certified_local_optimum",
    "seconds", "config", "source_index", "unit_pool", "kernel_fallbacks", "scored_initializations",
}
METADATA_FIELDS = {
    "experiment", "case_id", "source_freeze", "source_sha256", "input_sha256", "full_output",
    "cpu_seconds_before_final_write", "peak_rss_bytes", "environment", "source_hashes",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=0.):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"Unexpected fields: {label}")


def same(left, right):
    """JSON-value equality that does not identify booleans with integers."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def json_load(raw):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON field: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def registered_config(index):
    return {"seed": 52101 + 101 * index, "restarts": 16, "max_sweeps": 80,
            "max_seconds": 300., "batch_size": 256, "max_units": 256,
            "improvement_tolerance_bits": 1e-8}


def model_bits(units, context):
    def width(count):
        return (count - 1).bit_length()

    return (width(context["source_count"]) + width(context["max_states"])
            + len(units) * (width(context["max_alternatives"]) + width(context["max_emission_length"]))
            + width(len(context["glyph_alphabet"])) * sum(map(len, units)))


def ordered_neighbors(units, pool):
    for left, right in itertools.combinations(range(len(units)), 2):
        if units[left] != units[right]:
            changed = list(units)
            changed[left], changed[right] = changed[right], changed[left]
            yield {"kind": "swap", "rows": [left, right]}, changed
    for row in range(len(units)):
        for unit in pool:
            if unit != units[row]:
                yield {"kind": "replace", "row": row, "unit": unit}, [*units[:row], unit, *units[row + 1:]]


def check_score(value, units, context, label):
    if value is None:
        return
    fields(value, {"model_bits", "data_bits", "total_bits", "log_likelihood"}, label)
    require(integer(value["model_bits"]) and value["model_bits"] == model_bits(units, context),
            f"Literal model cost differs: {label}")
    require(finite(value["data_bits"]) and finite(value["total_bits"])
            and finite(value["log_likelihood"], -math.inf) and value["log_likelihood"] <= 1e-10,
            f"Invalid score: {label}")
    require(abs(value["data_bits"] + value["log_likelihood"] / math.log(2)) <= 1e-9,
            f"Log likelihood/data bits arithmetic differs: {label}")
    require(abs(value["total_bits"] - value["model_bits"] - value["data_bits"]) <= 1e-9,
            f"Score sum differs: {label}")


def check_diagnostics(value, candidates, context, record_count, source_order):
    expected = {"candidate_count": len(candidates), "record_count": record_count, "source_order": source_order,
                "source_alphabet_size": len(context["source_alphabet"]),
                "maximum_actual_unit_length": max(map(len, itertools.chain.from_iterable(candidates))),
                "candidate_record_pairs": len(candidates) * record_count}
    expected["rolling_buffer_bytes"] = ((expected["maximum_actual_unit_length"] + 1) * len(candidates)
                                        * (len(context["source_alphabet"]) + 1) * 8)
    fields(value, {*expected, "log_domain_fallbacks"}, "kernel diagnostics")
    require(all(type(value[key]) is int and value[key] == item for key, item in expected.items()),
            "Kernel dimensions or rolling-buffer accounting differs")
    fallback = value["log_domain_fallbacks"]
    require(integer(fallback) and fallback <= len(candidates) * record_count, "Invalid fallback count")
    return fallback


def validate_case(frozen: dict, full: dict, context: dict, *, record_count=4, source_order=1) -> dict:
    """Validate serialized accounting using supplied fixed side information.

    No source probabilities, observations, engine or production search helpers
    are used. Null/support labels and recorded likelihoods are not replayed.
    """
    fields(full, PAYLOAD_FIELDS, "full payload")
    fields(frozen, (PAYLOAD_FIELDS - {"trace"}) | METADATA_FIELDS, "compact payload")
    for key in PAYLOAD_FIELDS - {"trace"}:
        require(same(full[key], frozen[key]), f"Compact/full payload mismatch: {key}")
    require(full["schema_version"] == 1 and type(full["schema_version"]) is int
            and full["candidate_family"] == "one_state_deterministic_literal_units", "Unexpected family/schema")
    config = full["config"]
    fields(config, registered_config(0), "config")
    require(type(config["seed"]) is int and integer(config["restarts"], 1) and integer(config["max_sweeps"])
            and integer(config["batch_size"], 1) and integer(config["max_units"], 1)
            and finite(config["max_seconds"]) and config["max_seconds"] > 0
            and finite(config["improvement_tolerance_bits"]), "Invalid search config")
    require(integer(full["source_index"]) and full["source_index"] < context["source_count"], "Invalid source index")
    require(finite(full["seconds"]), "Invalid elapsed time")
    require(full["stop_reason"] in {"time_limit", "budget_exhausted", "no_supported_initialization"}, "Invalid stop reason")
    pool = ["".join(chars) for length in range(1, context["max_emission_length"] + 1)
            for chars in itertools.product(context["glyph_alphabet"], repeat=length)]
    require(full["unit_pool"] == pool and len(pool) <= config["max_units"], "Literal pool differs or exceeds cap")
    size = len(context["source_alphabet"])
    require(size >= len(context["glyph_alphabet"]), "Initializer coverage capacity differs")

    def valid_units(units):
        require(isinstance(units, list) and len(units) == size and all(unit in pool for unit in units), "Invalid units")

    best_units = best_score = current_units = current_score = None
    restart_index, next_sweep = -1, 0
    closed = True
    examined = completed = local = fallbacks = 0
    certified = set()
    methods, moves, accepted_moves = Counter(), Counter(), Counter()
    restart_details = []
    generator = random.Random(config["seed"])
    trace = full["trace"]
    require(isinstance(trace, list), "Trace must be a list")

    def consider(units, score):
        nonlocal best_units, best_score
        if score is not None and (best_score is None or score["total_bits"] < best_score["total_bits"]):
            best_units, best_score = units, score

    for index, event in enumerate(trace):
        require(isinstance(event, dict), "Trace event must be an object")
        if event.get("event") == "restart":
            fields(event, {"event", "restart", "seed", "method", "units", "score", "status", "kernel_diagnostics"}, "restart")
            require(closed or next_sweep == config["max_sweeps"], "Previous restart stopped before its allowed trajectory")
            restart_index += 1
            require(type(event["restart"]) is int and event["restart"] == restart_index < config["restarts"], "Restart sequence/budget differs")
            seed = generator.getrandbits(64)
            require(type(event["seed"]) is int and event["seed"] == seed, "Restart seed differs")
            units, score = event["units"], event["score"]
            valid_units(units)
            require(set(context["glyph_alphabet"]) <= set(units), "Initial singleton coverage differs")
            if restart_index == 0:
                method = "source_start_and_glyph_frequency_singletons"
                counts = Counter(units)
                require(all(len(unit) == 1 for unit in units) and max(counts.values()) - min(counts.values()) <= 1,
                        "Frequency initializer has invalid singleton multiplicities")
            else:
                method = "random_singleton_coverage_plus_uniform_pool"
                rng = random.Random(seed)
                rows, glyphs = list(range(size)), list(context["glyph_alphabet"])
                rng.shuffle(rows)
                rng.shuffle(glyphs)
                expected = [""] * size
                for row, glyph in zip(rows, glyphs):
                    expected[row] = glyph
                for row in rows[len(glyphs):]:
                    expected[row] = rng.choice(pool)
                require(units == expected, "Random initializer does not follow its recorded seed")
            require(event["method"] == method, "Initialization method differs")
            check_score(score, units, context, "initialization")
            require(event["status"] == ("scored" if score is not None else "unsupported"), "Initial support label differs")
            fallbacks += check_diagnostics(event["kernel_diagnostics"], [units], context, record_count, source_order)
            methods[method] += 1
            consider(units, score)
            current_units, current_score, next_sweep, closed = units, score, 0, score is None
            restart_details.append({"restart": restart_index, "seed": seed, "method": method, "supported": score is not None,
                                    "sweeps": 0, "completed_sweeps": 0, "neighbors": 0, "accepted_moves": 0})
            continue

        fields(event, {"event", "restart", "sweep", "parent_units", "parent_score", "neighbors", "batches",
                       "expected_neighbors", "evaluated_neighbors", "complete", "accepted", "selected_units",
                       "selected_score", "selected_move", "stop_reason"}, "sweep")
        require(event["event"] == "sweep" and not closed and current_score is not None, "Sweep has no active supported parent")
        require(type(event["restart"]) is int and event["restart"] == restart_index
                and type(event["sweep"]) is int and event["sweep"] == next_sweep < config["max_sweeps"], "Sweep sequence/budget differs")
        require(same(event["parent_units"], current_units) and same(event["parent_score"], current_score), "Parent trajectory differs")
        expected_count = size * (len(pool) - 1) + sum(a != b for a, b in itertools.combinations(current_units, 2))
        require(type(event["expected_neighbors"]) is int and event["expected_neighbors"] == expected_count, "Neighborhood cardinality differs")
        rows = event["neighbors"]
        require(isinstance(rows, list) and len(rows) <= expected_count, "Too many neighbors")
        selected_units, selected_score, selected_move = current_units, current_score, None
        for row, (move, units) in zip(rows, ordered_neighbors(current_units, pool), strict=False):
            fields(row, {"move", "units", "score", "status"}, "neighbor")
            require(same(row["move"], move) and same(row["units"], units), "Neighbor is not the exact ordered neighborhood prefix")
            check_score(row["score"], units, context, "neighbor")
            require(row["status"] == ("scored" if row["score"] is not None else "unsupported"), "Neighbor support label differs")
            moves[move["kind"]] += 1
            consider(units, row["score"])
            if row["score"] is not None and row["score"]["total_bits"] < selected_score["total_bits"]:
                selected_units, selected_score, selected_move = units, row["score"], move
        require(same(event["selected_units"], selected_units) and same(event["selected_score"], selected_score)
                and same(event["selected_move"], selected_move), "Sweep selected a nonminimum or wrong tie-breaking candidate")
        first = 0
        require(isinstance(event["batches"], list), "Batches must be a list")
        for batch in event["batches"]:
            fields(batch, {"first_neighbor", "count", "kernel_diagnostics"}, "batch")
            count = batch["count"]
            require(type(batch["first_neighbor"]) is int and batch["first_neighbor"] == first
                    and integer(count, 1) and count == min(config["batch_size"], expected_count - first)
                    and first + count <= len(rows), "Batch boundaries/count differ")
            fallbacks += check_diagnostics(batch["kernel_diagnostics"], [row["units"] for row in rows[first:first + count]],
                                           context, record_count, source_order)
            first += count
        require(first == len(rows) and type(event["evaluated_neighbors"]) is int
                and event["evaluated_neighbors"] == len(rows), "Completed neighbor/batch accounting differs")
        complete = len(rows) == expected_count
        accepted = current_score["total_bits"] - selected_score["total_bits"] > config["improvement_tolerance_bits"]
        require(type(event["complete"]) is bool and event["complete"] == complete
                and type(event["accepted"]) is bool and event["accepted"] == accepted, "Completion/acceptance label differs")
        reason = "time_limit" if not complete else ("improved" if accepted else "local_optimum")
        require(event["stop_reason"] == reason, "Sweep stopping reason differs")
        if not complete:
            require(index == len(trace) - 1 and full["stop_reason"] == "time_limit", "Partial sweep is not terminal")
        if complete and not accepted:
            certified.add(tuple(current_units))
            local += 1
        if accepted:
            accepted_moves[selected_move["kind"]] += 1
            current_units, current_score = selected_units, selected_score
        next_sweep += 1
        closed = not complete or not accepted
        examined += len(rows)
        completed += int(complete)
        details = restart_details[-1]
        details["sweeps"] += 1
        details["completed_sweeps"] += int(complete)
        details["neighbors"] += len(rows)
        details["accepted_moves"] += int(accepted)

    for key, expected in {"evaluated_neighbors": examined, "completed_sweeps": completed,
                          "local_optima": local, "kernel_fallbacks": fallbacks,
                          "scored_initializations": restart_index + 1}.items():
        require(type(full[key]) is int and full[key] == expected, f"Aggregate trace accounting differs: {key}")
    require(full["units"] == best_units, "Returned units are not the first best completed candidate")
    require(type(full["best_is_certified_local_optimum"]) is bool
            and full["best_is_certified_local_optimum"] == (best_units is not None and tuple(best_units) in certified),
            "Returned key does not own its local certificate")
    if best_units is None:
        require(full["channel"] is None and full["score"] is None, "Unsupported search returned a channel/score")
        require(full["stop_reason"] != "budget_exhausted", "Unsupported search has wrong final reason")
    else:
        require(full["stop_reason"] != "no_supported_initialization", "Supported search has wrong final reason")
        expected_channel = {"schema_version": 1, "states": ["s0"], "glyph_alphabet": list(context["glyph_alphabet"]),
                            "initial": {"s0": 1.}, "stop_probability": context["stop_probability"],
                            "max_emission_length": context["max_emission_length"],
                            "rows": [{"state": "s0", "letter": letter,
                                      "emissions": [{"next_state": "s0", "glyphs": best_units[context["source_alphabet"].index(letter)],
                                                     "probability": 1.}]} for letter in sorted(context["source_alphabet"])]}
        require(same(full["channel"], expected_channel), "Returned channel does not encode selected units")
        check_score(full["score"], best_units, context, "returned replay")
        require(full["score"] is not None, "Supported search omitted final score")
        slack = 1e-10 * max(1., abs(best_score["log_likelihood"]))
        require(abs(full["score"]["log_likelihood"] - best_score["log_likelihood"]) <= slack,
                "Final-engine score differs beyond its declared numerical slack")
    if full["stop_reason"] != "time_limit":
        require(restart_index + 1 == config["restarts"] and (closed or next_sweep == config["max_sweeps"]),
                "Count-budget run ended before its recorded budget")
    else:
        require(full["seconds"] >= config["max_seconds"], "Time-limit reason precedes the time budget")
    return {"status": "verified", "score": full["score"], "minimum_recorded_score": best_score,
            "returned_minimum_verified": True, "requested_restarts": config["restarts"],
            "scored_initializations": restart_index + 1, "supported_initializations": sum(row["supported"] for row in restart_details),
            "evaluated_neighbors": examined, "completed_sweeps": completed, "local_optima": local,
            "best_is_certified_local_optimum": full["best_is_certified_local_optimum"], "kernel_fallbacks": fallbacks,
            "initialization_methods": dict(methods), "move_counts": dict(moves), "accepted_move_counts": dict(accepted_moves),
            "restart_details": restart_details, "unit_pool_size": len(pool), "stop_reason": full["stop_reason"],
            "selected_unique_units": None if best_units is None else len(set(best_units)),
            "selected_length_counts": None if best_units is None else dict(sorted(Counter(map(len, best_units)).items())),
            "wall_seconds": full["seconds"], "configured_wall_seconds": config["max_seconds"]}


def _read(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Artifact resolves outside the supplied root")
    return path.read_bytes()


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is not None


def summarize(root: Path) -> dict:
    """Read a completed registered panel only; never write any file."""
    root = Path(root)
    result = {"experiment": EXPERIMENT, "scope": "archive_integrity_and_recorded_search_accounting_only",
              "empirical_likelihood_replay_performed": False, "ciphertext_source_transfer_answers_or_evaluation_opened": False,
              "first_initializer_frequency_ranking_replayed": False, "source_hashes_authenticated_here": False,
              "cases": {}, "pending": [], "errors": [], "status": "pending", "cases_verified": 0}
    campaign_path = f"results/{EXPERIMENT}/campaign.json"
    if not (root / campaign_path).is_file():
        result["pending"] = list(CASE_NAMES)
        return result
    try:
        campaign_raw = _read(root, campaign_path)
        campaign = json_load(campaign_raw)
        fields(campaign, {"experiment", "source_freeze", "status", "workers", "cases", "wall_seconds"}, "campaign")
        require(campaign["experiment"] == EXPERIMENT and campaign["source_freeze"] == SOURCE_FREEZE, "Campaign identity differs")
        if campaign["status"] != "complete":
            result["pending"] = list(CASE_NAMES)
            return result
        require(campaign["workers"] == 2 and set(campaign["cases"]) == set(CASE_NAMES)
                and finite(campaign["wall_seconds"]), "Campaign panel/resources differ")
        for name in CASE_NAMES:
            row = campaign["cases"][name]
            fields(row, {"returncode", "wall_seconds", "log"}, "campaign case")
            require(type(row["returncode"]) is int and row["returncode"] == 0 and finite(row["wall_seconds"])
                    and row["log"] == f"outputs/{EXPERIMENT}/{name}.log", "Campaign has an incomplete/failed case")
        result["campaign_sha256"] = digest(campaign_raw)
        for name in CASE_NAMES:
            if not all((root / path).is_file() for path in
                       (f"results/{EXPERIMENT}/{name}_freeze.json", f"outputs/{EXPERIMENT}/{name}.json.gz")):
                result["pending"].append(name)
        if result["pending"]:
            return result
        for index, name in enumerate(CASE_NAMES):
            raw = _read(root, f"results/{EXPERIMENT}/{name}_freeze.json")
            frozen = json_load(raw)
            require(frozen["experiment"] == EXPERIMENT and frozen["case_id"] == name
                    and frozen["source_freeze"] == SOURCE_FREEZE and frozen["source_sha256"] == SOURCE_SHA256
                    and _hex(frozen["input_sha256"], 64), "Compact case/source identity differs")
            require(frozen["config"] == registered_config(index) and frozen["source_index"] == 0, "Registered config/index differs")
            artifact = frozen["full_output"]
            fields(artifact, {"path", "bytes", "sha256"}, "archive descriptor")
            require(artifact["path"] == f"outputs/{EXPERIMENT}/{name}.json.gz"
                    and integer(artifact["bytes"], 1) and artifact["bytes"] <= 200 * 1024 * 1024
                    and _hex(artifact["sha256"], 64), "Archive descriptor differs")
            compressed = _read(root, artifact["path"])
            require(len(compressed) == artifact["bytes"] and digest(compressed) == artifact["sha256"], "Archive hash/size differs")
            full = json_load(gzip.decompress(compressed))
            case = validate_case(frozen, full, CONTEXT)
            require(finite(frozen["cpu_seconds_before_final_write"]) and integer(frozen["peak_rss_bytes"]), "Invalid CPU/RSS metadata")
            environment = frozen["environment"]
            fields(environment, {"python", "numpy", "platform", "numerical_thread_limits"}, "environment")
            require(all(isinstance(environment[key], str) and environment[key] for key in ("python", "numpy", "platform")), "Invalid environment labels")
            require(environment["numerical_thread_limits"] == {key: "1" for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "Numerical thread limits differ")
            hashes = frozen["source_hashes"]
            require(isinstance(hashes, dict) and set(hashes) == SOURCE_PATHS and all(_hex(value, 64) for value in hashes.values())
                    and hashes[SOURCE_PATH] == SOURCE_SHA256 and hashes[MANIFEST_PATH] == MANIFEST_SHA256, "Source-hash metadata differs")
            case.update(freeze_sha256=digest(raw), full_output_sha256=digest(compressed), full_output_bytes=len(compressed),
                        source_freeze=SOURCE_FREEZE, input_sha256=frozen["input_sha256"], source_sha256=SOURCE_SHA256,
                        cpu_seconds_before_final_write=frozen["cpu_seconds_before_final_write"], peak_rss_bytes=frozen["peak_rss_bytes"])
            result["cases"][name] = case
        result["status"] = "pass"
    except (KeyError, TypeError, ValueError, OSError, EOFError) as error:
        result["status"] = "fail"
        result["errors"].append(str(error))
    result["cases_verified"] = len(result["cases"])
    result["auditor_sha256"] = digest(Path(__file__).read_bytes())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-fitting", action="store_true", help="Confirm the registered fitting panel has finished")
    parser.add_argument("--output", help="Optional output JSON path; defaults to printing without writing")
    arguments = parser.parse_args(argv)
    if not arguments.after_fitting:
        parser.error("--after-fitting is required before reading selected-model archives")
    result = summarize(ROOT)
    if result["status"] != "pass":
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        raise SystemExit("Refusing an incomplete or invalid registered panel")
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        destination = Path(arguments.output)
        if not destination.is_absolute():
            destination = ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
