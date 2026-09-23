"""Independent CPU-only artifact audit for the frozen TEACH-0004 campaign.

Reconstructs the evaluation suite and scoring without importing the generator,
trainer, model, or any accelerator library. A partial campaign is INCOMPLETE.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess


REVISION = "e6e2aaab2d9413132b38b7da95661e857fd0558f"
SOURCE_PATHS = (
    "docs/experiments/TEACH-0004.md",
    "src/voynich/workspace/teacher4_tasks.py",
    "src/voynich/workspace/teacher4_models.py",
    "src/voynich/workspace/teacher4_train.py",
    "src/voynich/workspace/teacher2_train.py",
)
ARMS = ("two_read", "one_read", "dense_row")
REPLICATES = ("0", "1")
PARAMETERS = {"two_read": 222215, "one_read": 222215, "dense_row": 883591}
CONFIG = {
    "init_seeds": [64111, 64121], "train_seeds": [64211, 64221],
    "eval_seed": 64311, "steps_per_arm": 5000, "batch_size": 64,
    "eval_size": 256, "learning_rate": .0003, "max_seconds": 3600.0, "max_mps_bytes": 8 * 1024**3,
    "benchmark_steps": 24, "benchmark_warmup_steps": 4,
    "benchmark_max_seconds": 600.0,
}
NAMES, KEYS, OBJECTS = (tuple(range(a, b)) for a, b in ((9, 21), (21, 33), (33, 45)))
TASK_MARKERS = {4: "composed", 5: "direct", 6: "copy", 8: "first_hop"}
CELL_SPECS = {
    "composed_train_train": ("composed", "train", "train", 256),
    "composed_holdout_train": ("composed", "holdout", "train", 256),
    "composed_train_holdout": ("composed", "train", "holdout", 256),
    "composed_holdout_holdout": ("composed", "holdout", "holdout", 256),
    "first_hop_train": ("first_hop", "train", "train", 256),
    "first_hop_holdout": ("first_hop", "holdout", "train", 256),
    "first_hop_pairs_holdout": ("first_hop", "holdout", "train", 256),
    "direct_train": ("direct", "train", "train", 256),
    "direct_holdout": ("direct", "train", "holdout", 256),
    "direct_pairs_holdout": ("direct", "train", "holdout", 256),
    "copy": ("copy", "holdout", "holdout", 256),
    "factorial": ("composed", "holdout", "holdout", 512),
}


class AuditError(ValueError):
    """An artifact contradicts the registered study or its own records."""


def need(condition, message):
    if not condition:
        raise AuditError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def same(actual, expected, where):
    if isinstance(expected, dict):
        need(type(actual) is dict and set(actual) == set(expected), f"{where}: keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        need(type(actual) is list and len(actual) == len(expected), f"{where}: length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{where}[{index}]")
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             f"{where}: {actual!r} != {expected!r}")
    else:
        need(type(actual) is type(expected) and actual == expected,
             f"{where}: {actual!r} != {expected!r}")


def partition(kind, left, right):
    payload = b"TEACH-0004-v1/" + kind.encode() + b"/"
    payload += bytes([*sorted(left), 255, *sorted(right)])
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "holdout"


def parse_tokens(tokens):
    need(type(tokens) in (list, tuple) and len(tokens) == 14
         and all(type(token) is int for token in tokens), "Bad token sequence")
    need(tuple(tokens[:2]) == (1, 2) and tokens[6] == 3 and tokens[13] == 7,
         "Bad episode markers")
    names, assigned = (tokens[2], tokens[4]), (tokens[3], tokens[5])
    keys, objects = (tokens[7], tokens[9]), (tokens[8], tokens[10])
    need(len(set(names)) == len(set(keys)) == len(set(objects)) == 2
         and set(assigned) == set(keys) and all(x in NAMES for x in names)
         and all(x in KEYS for x in keys) and all(x in OBJECTS for x in objects),
         "Bad episode tables")
    task = TASK_MARKERS.get(tokens[11])
    need(task is not None, "Bad task marker")
    query = tokens[12]
    if task == "first_hop":
        need(query in names, "Bad first-hop query")
        f_row = names.index(query)
        answer, g_row = assigned[f_row], None
    elif task == "composed":
        need(query in names, "Bad composed query")
        f_row = names.index(query)
        g_row = keys.index(assigned[f_row])
        answer = objects[g_row]
    elif task == "direct":
        need(query in keys, "Bad direct query")
        f_row, g_row = None, keys.index(query)
        answer = objects[g_row]
    else:
        need(query in OBJECTS, "Bad copy query")
        f_row, g_row, answer = None, None, query
    f_family = (tuple(sorted(names)), tuple(sorted(keys)))
    g_family = (tuple(sorted(keys)), tuple(sorted(objects)))
    return (task, answer, f_family, g_family, partition("F", names, keys),
            partition("G", keys, objects), f_row, g_row)


def episode(names, assigned, keys, objects, task, query):
    marker = {value: key for key, value in TASK_MARKERS.items()}[task]
    tokens = [1, 2, names[0], assigned[0], names[1], assigned[1],
              3, keys[0], objects[0], keys[1], objects[1], marker, query, 7]
    return tokens, parse_tokens(tokens)[1]


def draw(rng, task, f, g):
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        objects = tuple(rng.sample(OBJECTS, 2))
        if (partition("F", names, keys), partition("G", keys, objects)) != (f, g):
            continue
        assigned = keys if rng.randrange(2) == 0 else keys[::-1]
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        query = (rng.choice(names) if task in ("first_hop", "composed") else
                 rng.choice(keys) if task == "direct" else rng.choice(OBJECTS))
        return episode(names, assigned, table_keys, objects, task, query)
    raise AuditError("Evaluation sampler did not find a family")


def quartet(rng):
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        objects = tuple(rng.sample(OBJECTS, 4))
        if partition("F", names, keys) != "holdout" or any(
                partition("G", keys, pair) != "holdout"
                for pair in (objects[:2], objects[2:])):
            continue
        query = rng.choice(names)
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        first = keys if rng.randrange(2) == 0 else keys[::-1]
        return [episode(names, assigned, table_keys, pair, "composed", query)
                for pair in (objects[:2], objects[2:])
                for assigned in (first, first[::-1])]
    raise AuditError("Evaluation sampler did not find a factorial quartet")


def expected_suite():
    rng = random.Random(64311)
    suite = {}
    for f, g in (("train", "train"), ("holdout", "train"),
                 ("train", "holdout"), ("holdout", "holdout")):
        suite[f"composed_{f}_{g}"] = [draw(rng, "composed", f, g) for _ in range(256)]
    for f in ("train", "holdout"):
        suite[f"first_hop_{f}"] = [draw(rng, "first_hop", f, "train") for _ in range(256)]
    pairs = []
    for _ in range(128):
        first = draw(rng, "first_hop", "holdout", "train")
        t = first[0]
        other = t[4] if t[12] == t[2] else t[2]
        pairs.extend((first, episode((t[2], t[4]), (t[3], t[5]), (t[7], t[9]),
                                     (t[8], t[10]), "first_hop", other)))
    suite["first_hop_pairs_holdout"] = pairs
    for g in ("train", "holdout"):
        suite[f"direct_{g}"] = [draw(rng, "direct", "train", g) for _ in range(256)]
    pairs = []
    for _ in range(128):
        first = draw(rng, "direct", "train", "holdout")
        t = first[0]
        other = t[9] if t[12] == t[7] else t[7]
        pairs.extend((first, episode((t[2], t[4]), (t[3], t[5]), (t[7], t[9]),
                                     (t[8], t[10]), "direct", other)))
    suite["direct_pairs_holdout"] = pairs
    suite["copy"] = [draw(rng, "copy", "holdout", "holdout") for _ in range(256)]
    suite["factorial"] = [item for _ in range(128) for item in quartet(rng)]
    return suite


def check_suite_structure(suite):
    need(set(suite) == set(CELL_SPECS), "Wrong evaluation cells")
    need(sum(len(items) for items in suite.values()) == 3328, "Wrong suite size")
    for name, examples in suite.items():
        task, f, g, count = CELL_SPECS[name]
        need(len(examples) == count, f"{name}: wrong cell size")
        for tokens, answer in examples:
            parsed = parse_tokens(tokens)
            need((parsed[0], parsed[1], parsed[4], parsed[5]) == (task, answer, f, g),
                 f"{name}: oracle or family mismatch")
    for name in ("first_hop_pairs_holdout", "direct_pairs_holdout"):
        for offset in range(0, len(suite[name]), 2):
            a, b = suite[name][offset:offset + 2]
            need(a[0][:12] == b[0][:12] and a[0][13] == b[0][13]
                 and a[0][12] != b[0][12] and a[1] != b[1],
                 f"{name}: invalid query reversal")
    for offset in range(0, 512, 4):
        group = suite["factorial"][offset:offset + 4]
        a, b, c, d = (item[0] for item in group)
        need(len({item[1] for item in group}) == 4
             and a[2:6] != b[2:6] and a[2:6] == c[2:6] and b[2:6] == d[2:6]
             and a[7:11] == b[7:11] and c[7:11] == d[7:11]
             and a[7:11] != c[7:11] and len({a[12], b[12], c[12], d[12]}) == 1,
             "Invalid factorial quartet")


def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [0.0 if success == 0 else max(0.0, center-radius),
            1.0 if success == total else min(1.0, center+radius)]


def score_cell(rows, name):
    total = len(rows)
    correct = sum(row["answer"] == row["prediction"] for row in rows)
    score = {"correct": correct, "total": total, "accuracy": correct/total,
             "wilson_95": wilson(correct, total),
             "candidate_member": sum(row["candidate_member"] for row in rows)}
    for side in ("f", "g"):
        applicable = [row for row in rows if row[f"{side}_read_answer"] is not None
                      and row[f"{side}_read_prediction"] is not None]
        if applicable:
            match = sum(row[f"{side}_read_answer"] == row[f"{side}_read_prediction"]
                        for row in applicable)
            score[f"{side}_read_correct"] = match
            score[f"{side}_read_total"] = len(applicable)
            score[f"{side}_read_accuracy"] = match/len(applicable)
    if name in ("first_hop_pairs_holdout", "direct_pairs_holdout", "factorial"):
        width = 4 if name == "factorial" else 2
        groups = [rows[i:i+width] for i in range(0, total, width)]
        exact = sum(all(row["answer"] == row["prediction"] for row in group)
                    for group in groups)
        score.update({"exact_groups": exact, "total_groups": len(groups),
                      "group_accuracy": exact/len(groups),
                      "group_wilson_95": wilson(exact, len(groups)),
                      "prediction_changes": sum(group[0]["prediction"] != group[1]["prediction"]
                                                for group in groups)})
        if width == 4:
            score["f_swap_prediction_changes"] = sum(
                group[a]["prediction"] != group[b]["prediction"]
                for group in groups for a, b in ((0, 1), (2, 3)))
            score["g_remap_prediction_changes"] = sum(
                group[a]["prediction"] != group[b]["prediction"]
                for group in groups for a, b in ((0, 2), (1, 3)))
    return score


def check_archive(path, suite):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            archived = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AuditError(f"Unreadable prediction archive: {path}") from exc
    need(type(archived) is dict and set(archived) == set(CELL_SPECS), "Wrong archive cells")
    scores = {}
    fields = {"index", "tokens", "task", "answer", "prediction", "f_family", "g_family",
              "candidate_member", "f_read_answer", "f_read_prediction", "g_read_answer",
              "g_read_prediction"}
    for name, examples in suite.items():
        rows = archived[name]
        need(type(rows) is list and len(rows) == len(examples), f"{name}: wrong archive size")
        for index, (row, (tokens, answer)) in enumerate(zip(rows, examples, strict=True)):
            parsed = parse_tokens(tokens)
            task, independent_answer, f_family, g_family, f_part, g_part, f_row, g_row = parsed
            need((task, independent_answer, f_part, g_part) ==
                 (CELL_SPECS[name][0], answer, CELL_SPECS[name][1], CELL_SPECS[name][2]),
                 f"{name}/{index}: independent parser disagrees")
            need(type(row) is dict and set(row) == fields, f"{name}/{index}: row fields")
            need(row["index"] == index and row["tokens"] == tokens
                 and row["answer"] == answer and row["task"] == task,
                 f"{name}/{index}: changed evaluation item")
            need(row["f_family"] == [list(x) for x in f_family]
                 and row["g_family"] == [list(x) for x in g_family],
                 f"{name}/{index}: family signature")
            need(row["f_read_answer"] == f_row and row["g_read_answer"] == g_row,
                 f"{name}/{index}: row oracle mismatch")
            need(all(value is None or (type(value) is int and value in (0, 1))
                     for value in (row["f_read_prediction"], row["g_read_prediction"])),
                 f"{name}/{index}: invalid row predictions")
            if "dense_row" in path.name:
                need(row["f_read_prediction"] is None
                     and row["g_read_prediction"] is None,
                     f"{name}/{index}: dense model reported memory attention")
            else:
                need(row["f_read_prediction"] is not None
                     and row["g_read_prediction"] is not None,
                     f"{name}/{index}: memory attention missing")
            legal = KEYS if task == "first_hop" else OBJECTS
            need(type(row["prediction"]) is int and row["prediction"] in legal,
                 f"{name}/{index}: illegal answer prediction")
            candidates = ((tokens[3], tokens[5]) if task == "first_hop"
                          else (tokens[8], tokens[10]))
            need(type(row["candidate_member"]) is bool
                 and row["candidate_member"] == (row["prediction"] in candidates),
                 f"{name}/{index}: candidate flag mismatch")
        scores[name] = score_cell(rows, name)
    return scores


def behavior(score):
    return {
        "first_hop_holdout_at_least_0.90": score["first_hop_holdout"]["accuracy"] >= .90,
        "first_hop_pair_at_least_0.80": score["first_hop_pairs_holdout"]["group_accuracy"] >= .80,
        "direct_holdout_at_least_0.90": score["direct_holdout"]["accuracy"] >= .90,
        "direct_pair_at_least_0.80": score["direct_pairs_holdout"]["group_accuracy"] >= .80,
        "both_held_composed_at_least_0.85": score["composed_holdout_holdout"]["accuracy"] >= .85,
        "crossed_cells_at_least_0.85": all(score[name]["accuracy"] >= .85 for name in
                                           ("composed_holdout_train", "composed_train_holdout")),
        "factorial_at_least_0.65": score["factorial"]["group_accuracy"] >= .65,
        "copy_at_least_0.95": score["copy"]["accuracy"] >= .95,
    }


def decision(scores):
    clauses = {}
    for rep in REPLICATES:
        memory = scores[rep]["two_read"]
        row = behavior(memory)
        for control in ("one_read", "dense_row"):
            row[f"composed_gain_over_{control}_at_least_0.15"] = (
                memory["composed_holdout_holdout"]["accuracy"]
                - scores[rep][control]["composed_holdout_holdout"]["accuracy"] >= .15)
            row[f"factorial_gain_over_{control}_at_least_0.15"] = (
                memory["factorial"]["group_accuracy"]
                - scores[rep][control]["factorial"]["group_accuracy"] >= .15)
        clauses[rep] = row
    qualified = all(all(row.values()) for row in clauses.values())
    return {"verdict": "qualified" if qualified else "not_qualified",
            "two_read_qualified": qualified, "clauses_by_seed": clauses}


def check_source(root, artifact):
    need(artifact.get("source_git_head") == REVISION, "Wrong frozen source revision")
    need(artifact.get("source_worktree_status") == [], "Dirty registered launch source")
    saved = artifact.get("source_sha256")
    need(type(saved) is dict and set(saved) == set(SOURCE_PATHS), "Source hash set differs")
    for path in SOURCE_PATHS:
        blob = subprocess.run(["git", "show", f"{REVISION}:{path}"], cwd=root,
                              capture_output=True, check=True, timeout=10).stdout
        need(saved[path] == digest(blob), f"Source hash mismatch: {path}")
    same(artifact.get("config"), CONFIG, "config")
    encoded = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    need(artifact.get("config_sha256") == digest(encoded), "Config hash mismatch")


def check_benchmark(root, report):
    path = root / "results/TEACH-0004/benchmark.json"
    need(path.is_file(), "Missing source-matched benchmark")
    benchmark = json.loads(path.read_text())
    need(report.get("benchmark_sha256") == file_digest(path), "Benchmark hash mismatch")
    check_source(root, benchmark)
    need(benchmark.get("experiment") == "TEACH-0004"
         and benchmark.get("mode") == "benchmark"
         and benchmark.get("status") == "pass", "Benchmark did not pass")
    measured = benchmark.get("measured")
    need(type(measured) is dict and set(measured) == set(ARMS), "Benchmark arms differ")
    for arm, row in measured.items():
        need(type(row) is dict and row.get("parameters") == PARAMETERS[arm]
             and row.get("timed_steps") == 20 and row.get("warmup_steps") == 4
             and type(row.get("timed_step_seconds")) is list
             and len(row["timed_step_seconds"]) == 20
             and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                     for x in row["timed_step_seconds"]), f"{arm}: invalid benchmark timing")
        same(row.get("median_timed_step_seconds"),
             float(statistics.median(row["timed_step_seconds"])), f"{arm}: median")
    projected = 2 * 5000 * sum(row["median_timed_step_seconds"] for row in measured.values())
    same(benchmark.get("projected_campaign_seconds"), projected, "benchmark projection")
    same(benchmark.get("conservative_projected_seconds"), 1.5*projected,
         "benchmark conservative projection")
    same(report.get("benchmark_conservative_projected_seconds"), 1.5*projected,
         "run benchmark projection")
    need(1.5*projected <= 3600, "Benchmark projection exceeded cap")
    need(type(benchmark.get("elapsed_seconds")) in (int, float)
         and 0 <= benchmark["elapsed_seconds"] <= 600
         and type(benchmark.get("peak_sampled_mps_allocated_bytes")) is int
         and 0 <= benchmark["peak_sampled_mps_allocated_bytes"] <= 8*1024**3,
         "Benchmark resource cap exceeded")
    numerical = benchmark.get("numerical_qualification", {})
    need(numerical.get("finite_loss_and_gradients") is True
         and type(numerical.get("max_recompute_logit_error")) in (int, float)
         and 0 <= numerical["max_recompute_logit_error"] <= .002,
         "Benchmark numerical gate failed")


def check_losses(path, record):
    need(path.is_file(), f"Missing loss archive: {path}")
    need(file_digest(path) == record.get("losses_sha256"), f"Loss hash mismatch: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            losses = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AuditError(f"Unreadable loss archive: {path}") from exc
    need(type(losses) is list and len(losses) == 5000
         and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                 for value in losses), f"Invalid loss series: {path}")
    history = record.get("history")
    need(type(history) is list and len(history) == 20, f"Invalid progress history: {path}")
    for row, step in zip(history, range(250, 5001, 250), strict=True):
        need(type(row) is dict and row.get("step") == step, f"Wrong progress step: {path}")
        same(row.get("last_100_loss"), sum(losses[step-100:step])/100,
             f"{path}: last-100 loss at {step}")
    return file_digest(path)


def audit(root):
    result_dir = root / "results/TEACH-0004"
    status_path = result_dir / "status.json"
    need(status_path.is_file(), "No scientific status file")
    status = json.loads(status_path.read_text())
    need(status.get("experiment") == "TEACH-0004", "Wrong experiment status")
    if status.get("status") != "complete":
        need(status.get("status") in ("running", "stopped"), "Unknown partial status")
        return {"experiment": "TEACH-0004", "audit": "incomplete",
                "recorded_status": status["status"],
                "completed_arm_count_reported": len(status.get("completed_arms", [])),
                "reason": status.get("reason", "Campaign has no final report"),
                "limitation": "Partial-arm scores are not a scientific verdict."}
    path = result_dir / "report.json"
    need(path.is_file(), "Complete status without final report")
    report = json.loads(path.read_text())
    need(report.get("experiment") == "TEACH-0004"
         and report.get("mode") == "scientific" and report.get("status") == "complete"
         and report.get("device") == "mps", "Final report metadata differs")
    check_source(root, report)
    check_benchmark(root, report)
    numerical = report.get("numerical_qualification", {})
    need(numerical.get("finite_loss_and_gradients") is True
         and type(numerical.get("max_recompute_logit_error")) in (int, float)
         and math.isfinite(numerical["max_recompute_logit_error"])
         and 0 <= numerical["max_recompute_logit_error"] <= .002,
         "Scientific numerical gate failed")
    elapsed = report.get("elapsed_seconds")
    peak = report.get("peak_sampled_mps_allocated_bytes")
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 3600,
         "Scientific time cap exceeded")
    need(type(peak) is int and 0 <= peak <= 8*1024**3, "Scientific MPS cap exceeded")
    need(status.get("report") == "report.json"
         and status.get("source_git_head") == REVISION
         and status.get("peak_sampled_mps_allocated_bytes") == peak,
         "Final status disagrees")
    suite = expected_suite()
    check_suite_structure(suite)
    payload = {name: [(tokens, answer) for tokens, answer in examples]
               for name, examples in suite.items()}
    suite_sha = digest(json.dumps(payload, sort_keys=True).encode())
    need(report.get("evaluation_suite_sha256") == suite_sha, "Evaluation suite hash mismatch")
    records, saved_scores = report.get("arms"), report.get("scores")
    need(type(records) is dict and type(saved_scores) is dict
         and set(records) == set(saved_scores) == set(REPLICATES), "Wrong replicate grid")
    recomputed = {}
    hashes = {}
    for rep in REPLICATES:
        need(set(records[rep]) == set(saved_scores[rep]) == set(ARMS),
             f"Replicate {rep}: wrong arm grid")
        recomputed[rep], hashes[rep] = {}, {}
        for arm in ARMS:
            record = records[rep][arm]
            need(record.get("steps") == 5000 and record.get("parameters") == PARAMETERS[arm],
                 f"{rep}/{arm}: steps or parameters differ")
            checkpoint = root / "outputs/TEACH-0004" / f"rep{rep}-{arm}.pt"
            losses = result_dir / f"losses-rep{rep}-{arm}.json.gz"
            prediction = result_dir / f"predictions-rep{rep}-{arm}.json.gz"
            need(checkpoint.is_file() and prediction.is_file(), f"{rep}/{arm}: missing artifact")
            checkpoint_sha = file_digest(checkpoint)
            need(checkpoint_sha == record.get("checkpoint_sha256"),
                 f"{rep}/{arm}: checkpoint hash mismatch")
            loss_sha = check_losses(losses, record)
            prediction_sha = file_digest(prediction)
            need(prediction_sha == record.get("predictions_sha256"),
                 f"{rep}/{arm}: prediction hash mismatch")
            recomputed[rep][arm] = check_archive(prediction, suite)
            same(saved_scores[rep][arm], recomputed[rep][arm], f"scores.{rep}.{arm}")
            hashes[rep][arm] = {"checkpoint_sha256": checkpoint_sha,
                                "losses_sha256": loss_sha,
                                "predictions_sha256": prediction_sha}
    expected_decision = decision(recomputed)
    same(report.get("decision"), expected_decision, "report.decision")
    same(status.get("decision"), expected_decision, "status.decision")
    same(report.get("completed_arms"), [[int(rep), arm] for rep in REPLICATES for arm in ARMS],
         "completed_arms")
    return {"experiment": "TEACH-0004", "audit": "pass",
            "source_git_head": REVISION, "source_hashes_verified": list(SOURCE_PATHS),
            "config_sha256": report["config_sha256"],
            "evaluation_suite_sha256": suite_sha,
            "evaluated_items_per_arm": 3328, "arm_artifact_hashes": hashes,
            "independently_computed_scores": recomputed, "decision": expected_decision,
            "elapsed_seconds": elapsed, "peak_sampled_mps_allocated_bytes": peak,
            "numerical_qualification": numerical,
            "limitations": [
                "File hashes do not prove every optimizer update ran as recorded.",
                "Checkpoint identity does not prove it generated the archived predictions.",
                "Holdouts are table families, not atomic token relations.",
                "The dense control has a different parameter count and compute cost.",
                "Synthetic qualification is neither a causal circuit nor Voynich claim.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    destination = args.output or root / "results/TEACH-0004/audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"audit": result["audit"],
                      "decision": result.get("decision", {}).get("verdict", "incomplete")},
                     indent=2))


if __name__ == "__main__":
    main()
