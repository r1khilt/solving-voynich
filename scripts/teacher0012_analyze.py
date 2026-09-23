#!/usr/bin/env python3
"""Independent CPU-only artifact audit for TEACH-0012.

This file intentionally does not import the experiment generator, model, or trainer.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE_REVISION = "92c570ebf17c46c29f362ef3c751944dc6331fa6"
SOURCE_PATHS = (
    "docs/experiments/TEACH-0012.md",
    "src/voynich/workspace/teacher12_tasks.py",
    "src/voynich/workspace/teacher12_models.py",
    "src/voynich/workspace/teacher12_train.py",
    "scripts/teacher0012_analyze.py",
    "tests/test_workspace_teacher12.py",
    "src/voynich/model.py",
)
ARMS = ("parsed_memory", "raw_shallow", "raw_looped", "raw_deep", "raw_null")
PARAMETERS = {
    "parsed_memory": 3_160_576,
    "raw_shallow": 14_693_376,
    "raw_looped": 14_694_912,
    "raw_deep": 41_965_568,
    "raw_null": 14_693_376,
}
CONFIG = {
    "init_seeds": [72121, 72131], "train_seeds": [72221, 72231],
    "eval_seed": 72311, "steps_per_arm": 8000, "batch_size": 64,
    "eval_groups": 128, "learning_rate": .0003, "max_seconds": 21600.0,
    "max_mps_bytes": 24 * 1024**3, "max_artifact_bytes": 4 * 1024**3,
    "benchmark_steps": 24, "benchmark_warmup_steps": 4,
    "benchmark_max_seconds": 1800.0,
}
PAD, BOS, EDGE, GAP, COMPOSE, DIRECT, COPY, FIRST_HOP, ANSWER = range(9)
SYMBOL_START, SYMBOL_COUNT = 16, 2048
SYMBOLS = tuple(range(SYMBOL_START, SYMBOL_START + SYMBOL_COUNT))
NAMESPACE = "TEACH-0012-v1"
TASK_MARKERS = {"composed": COMPOSE, "direct": DIRECT, "copy": COPY,
                "first_hop": FIRST_HOP}
GROUP_WIDTHS = {name: 4 for name in (
    "first_hop_query_groups", "direct_query_groups", "factorial",
    "order_groups", "distractor_groups", "boundary_groups")}


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def semantic_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def same(actual, expected, where):
    if isinstance(expected, dict):
        need(type(actual) is dict and set(actual) == set(expected), f"{where}: keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        need(type(actual) is list and len(actual) == len(expected), f"{where}: length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{where}[{index}]")
    elif isinstance(expected, float):
        need(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             f"{where}: {actual!r} != {expected!r}")
    else:
        need(type(actual) is type(expected) and actual == expected,
             f"{where}: {actual!r} != {expected!r}")


def partition(kind, left, right):
    value = [NAMESPACE, kind, sorted(left), sorted(right)]
    bucket = int(semantic_digest(value)[:16], 16) % 10
    return "train" if bucket < 8 else "confirm" if bucket == 9 else "development"


def partitions(f_rows, g_rows):
    return (partition("F", [a for a, _ in f_rows], [b for _, b in f_rows]),
            partition("G", [a for a, _ in g_rows], [b for _, b in g_rows]))


def oracle(rows, task, query):
    mapping = {}
    for left, right in rows:
        need(left not in mapping, "Duplicate row left side")
        mapping[left] = right
    if task == "copy":
        return query
    need(query in mapping, "Query absent from graph")
    first = mapping[query]
    if task in ("first_hop", "direct"):
        return first
    need(task == "composed" and first in mapping, "Malformed composed graph")
    return mapping[first]


def serialize(rows, task, query, rng, marker_dropout, max_gaps, styles):
    ordered = list(rows)
    rng.shuffle(ordered)
    tokens = [BOS]
    for index, (left, right) in enumerate(ordered):
        marker = rng.random() >= marker_dropout
        style = rng.choice(styles)
        if not marker:
            tokens.extend((left, right))
        elif style == "prefix":
            tokens.extend((EDGE, left, right))
        elif style == "infix":
            tokens.extend((left, EDGE, right))
        else:
            tokens.extend((left, right, EDGE))
        if index + 1 < len(ordered):
            tokens.extend([GAP] * rng.randint(0, max_gaps))
    tokens.extend((TASK_MARKERS[task], query, ANSWER))
    need(len(tokens) <= 128, "Sequence overflow")
    return tuple(tokens), tuple(ordered)


def make_episode(f_rows, g_rows, distractors, task, query, rng, spec):
    rows = tuple(f_rows) + tuple(g_rows) + tuple(distractors)
    need(len(f_rows) == len(g_rows) == 4 and len(distractors) % 2 == 0,
         "Bad row counts")
    answer = oracle(rows, task, query)
    f_part, g_part = partitions(f_rows, g_rows)
    logical = semantic_digest({"f": f_rows, "g": g_rows, "d": distractors,
                               "task": task, "query": query})
    tokens, ordered = serialize(rows, task, query, rng, *spec)
    render = semantic_digest({"logical": logical, "tokens": tokens})
    return {"tokens": tokens, "f_rows": tuple(f_rows), "g_rows": tuple(g_rows),
            "distractor_rows": tuple(distractors), "serialized_rows": ordered,
            "answer": answer, "task": task, "query": query,
            "f_partition": f_part, "g_partition": g_part,
            "logical_id": logical, "render_id": render,
            "distractor_chains": len(distractors) // 2,
            "marker_dropout": spec[0]}


def rerender(ep, rng, spec, task=None, query=None):
    return make_episode(ep["f_rows"], ep["g_rows"], ep["distractor_rows"],
                        task or ep["task"], ep["query"] if query is None else query,
                        rng, spec)


def draw_components(rng, chains):
    values = rng.sample(SYMBOLS, 12 + 3 * chains)
    names, keys, objects = tuple(values[:4]), tuple(values[4:8]), tuple(values[8:12])
    assigned, table_keys, object_order = list(keys), list(keys), list(objects)
    rng.shuffle(assigned)
    rng.shuffle(table_keys)
    rng.shuffle(object_order)
    f_rows = tuple(zip(names, assigned, strict=True))
    g_rows = tuple(zip(table_keys, object_order, strict=True))
    distractors = []
    offset = 12
    for _ in range(chains):
        name, key, obj = values[offset:offset + 3]
        offset += 3
        distractors.extend(((name, key), (key, obj)))
    return f_rows, g_rows, tuple(distractors)


def sample(rng, f_part, g_part, task, chains, spec):
    for _ in range(100_000):
        f_rows, g_rows, distractors = draw_components(rng, chains)
        if partitions(f_rows, g_rows) != (f_part, g_part):
            continue
        if task in ("first_hop", "composed"):
            query = rng.choice(tuple(a for a, _ in f_rows))
        elif task == "direct":
            query = rng.choice(tuple(a for a, _ in g_rows))
        else:
            query = rng.choice(tuple(b for _, b in g_rows))
        return make_episode(f_rows, g_rows, distractors, task, query, rng, spec)
    raise AuditError("Sampler exhausted")


def query_groups(rng, task, size):
    items, spec = [], (.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample(rng, "confirm", "confirm", task, 4, spec)
        queries = ([a for a, _ in base["f_rows"]] if task == "first_hop" else
                   [a for a, _ in base["g_rows"]])
        for query in queries:
            items.append(rerender(base, random.Random(rng.randrange(2**63)), spec,
                                  task, query))
    return items


def factorial(rng, size):
    items, spec = [], (.25, 2, ("prefix", "infix", "suffix"))
    while len(items) < 4 * size:
        base = sample(rng, "confirm", "confirm", "composed", 4, spec)
        query, alternate = base["f_rows"][0][0], base["f_rows"][1][0]
        assignments = dict(base["f_rows"])
        assignments[query], assignments[alternate] = assignments[alternate], assignments[query]
        f1 = tuple((left, assignments[left]) for left, _ in base["f_rows"])
        used = {value for row in base["f_rows"] + base["g_rows"] + base["distractor_rows"]
                for value in row}
        objects = tuple(rng.sample(sorted(set(SYMBOLS) - used), 4))
        g1 = tuple((left, obj) for (left, _), obj in zip(base["g_rows"], objects, strict=True))
        if partitions(f1, base["g_rows"]) != ("confirm", "confirm") or \
                partitions(base["f_rows"], g1) != ("confirm", "confirm"):
            continue
        quartet = [make_episode(
            f_rows, g_rows, base["distractor_rows"], "composed", query,
            random.Random(rng.randrange(2**63)), spec)
            for g_rows in (base["g_rows"], g1)
            for f_rows in (base["f_rows"], f1)]
        if len({ep["answer"] for ep in quartet}) == 4:
            items.extend(quartet)
    return items


def order_groups(rng, size):
    items, spec = [], (.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample(rng, "confirm", "confirm", "composed", 4, spec)
        items.extend(rerender(base, random.Random(rng.randrange(2**63)), spec)
                     for _ in range(4))
    return items


def boundary_groups(rng, size):
    items = []
    for _ in range(size):
        base = sample(rng, "confirm", "confirm", "composed", 4,
                      (0.0, 0, ("prefix",)))
        specs = ((0.0, 0, ("prefix",)),
                 (.25, 2, ("prefix", "infix", "suffix")),
                 (.50, 2, ("prefix", "infix", "suffix")),
                 (1.0, 2, ("prefix", "infix", "suffix")))
        items.extend(rerender(base, random.Random(rng.randrange(2**63)), spec)
                     for spec in specs)
    return items


def distractor_groups(rng, size):
    items, spec = [], (.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample(rng, "confirm", "confirm", "composed", 0, spec)
        used = {value for row in base["f_rows"] + base["g_rows"] for value in row}
        for chains in (0, 1, 2, 4):
            values = rng.sample(sorted(set(SYMBOLS) - used), 3 * chains)
            distractors = []
            for offset in range(0, len(values), 3):
                name, key, obj = values[offset:offset + 3]
                distractors.extend(((name, key), (key, obj)))
            items.append(make_episode(
                base["f_rows"], base["g_rows"], tuple(distractors), "composed",
                base["query"], random.Random(rng.randrange(2**63)), spec))
    return items


def expected_suite(seed=72311, size=128):
    rng = random.Random(seed)
    hard = (.25, 2, ("prefix", "infix", "suffix"))
    suite = {}
    for f_part, g_part in (("train", "train"), ("confirm", "train"),
                           ("train", "confirm"), ("confirm", "confirm")):
        suite[f"composed_{f_part}_{g_part}"] = [
            sample(rng, f_part, g_part, "composed", 4, hard) for _ in range(size)]
    for task in ("first_hop", "direct", "copy"):
        suite[f"{task}_confirm"] = [
            sample(rng, "confirm", "confirm", task, 4, hard) for _ in range(size)]
    suite["first_hop_query_groups"] = query_groups(rng, "first_hop", size)
    suite["direct_query_groups"] = query_groups(rng, "direct", size)
    suite["factorial"] = factorial(rng, size)
    suite["order_groups"] = order_groups(rng, size)
    suite["distractor_groups"] = distractor_groups(rng, size)
    suite["boundary_groups"] = boundary_groups(rng, size)
    suite["long_ood"] = [sample(rng, "confirm", "confirm", "composed", 6, hard)
                         for _ in range(size)]
    return suite


def wilson(correct, total):
    z, p = 1.959963984540054, correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [0.0 if correct == 0 else max(0.0, center-radius),
            1.0 if correct == total else min(1.0, center+radius)]


def score(rows, name):
    total = len(rows)
    correct = sum(row["prediction"] == row["answer"] for row in rows)
    result = {"correct": correct, "total": total, "accuracy": correct/total,
              "wilson_95": wilson(correct, total),
              "candidate_member": sum(row["candidate_member"] for row in rows),
              "mean_target_probability": sum(row["target_probability"] for row in rows)/total}
    if name in GROUP_WIDTHS:
        groups = [rows[offset:offset+4] for offset in range(0, total, 4)]
        exact = sum(all(row["prediction"] == row["answer"] for row in group)
                    for group in groups)
        consistent = sum(len({row["prediction"] for row in group}) == 1 for group in groups)
        result.update({"exact_groups": exact, "total_groups": len(groups),
                       "group_accuracy": exact/len(groups), "consistent_groups": consistent,
                       "consistency": consistent/len(groups),
                       "mean_distinct_predictions": sum(
                           len({row["prediction"] for row in group}) for group in groups)/len(groups)})
        if name == "boundary_groups":
            marker_free = [group[-1] for group in groups]
            count = sum(row["prediction"] == row["answer"] for row in marker_free)
            result.update({"marker_free_correct": count,
                           "marker_free_accuracy": count/len(marker_free)})
    return result


def decide(scores):
    clauses = {"primary": {}, "positive": {}, "null": {}}
    for rep in ("0", "1"):
        p = scores[rep]["raw_deep"]
        clauses["primary"][rep] = {
            "first_hop_at_least_0.95": p["first_hop_confirm"]["accuracy"] >= .95,
            "direct_at_least_0.95": p["direct_confirm"]["accuracy"] >= .95,
            "copy_at_least_0.98": p["copy_confirm"]["accuracy"] >= .98,
            "all_composed_cells_at_least_0.90": all(
                p[name]["accuracy"] >= .90 for name in p if name.startswith("composed_")),
            "f_query_groups_at_least_0.85": p["first_hop_query_groups"]["group_accuracy"] >= .85,
            "g_query_groups_at_least_0.85": p["direct_query_groups"]["group_accuracy"] >= .85,
            "factorial_at_least_0.75": p["factorial"]["group_accuracy"] >= .75,
            "order_groups_at_least_0.80": p["order_groups"]["group_accuracy"] >= .80,
            "distractor_groups_at_least_0.80": p["distractor_groups"]["group_accuracy"] >= .80,
            "boundary_groups_at_least_0.70": p["boundary_groups"]["group_accuracy"] >= .70,
            "marker_free_at_least_0.80": p["boundary_groups"]["marker_free_accuracy"] >= .80,
            "long_ood_at_least_0.80": p["long_ood"]["accuracy"] >= .80,
        }
        positive = scores[rep]["parsed_memory"]
        clauses["positive"][rep] = {
            "ungrouped_at_least_0.95": all(
                row["accuracy"] >= .95 for name, row in positive.items()
                if name not in GROUP_WIDTHS),
            "query_groups_at_least_0.90": all(
                positive[name]["group_accuracy"] >= .90 for name in
                ("first_hop_query_groups", "direct_query_groups")),
            "factorial_at_least_0.85": positive["factorial"]["group_accuracy"] >= .85,
        }
        null = scores[rep]["raw_null"]
        clauses["null"][rep] = {
            "composed_confirm_confirm_at_most_0.35":
                null["composed_confirm_confirm"]["accuracy"] <= .35,
            "factorial_at_most_0.10": null["factorial"]["group_accuracy"] <= .10,
        }
    primary_pass = all(all(row.values()) for row in clauses["primary"].values())
    positive_pass = all(all(row.values()) for row in clauses["positive"].values())
    null_pass = all(all(row.values()) for row in clauses["null"].values())
    depth, looping = {}, {}
    for rep in ("0", "1"):
        deep, shallow, looped = (scores[rep][arm] for arm in
                                  ("raw_deep", "raw_shallow", "raw_looped"))
        depth[rep] = {
            "confirm_confirm_gain_at_least_0.10":
                deep["composed_confirm_confirm"]["accuracy"]
                - shallow["composed_confirm_confirm"]["accuracy"] >= .10,
            "factorial_gain_at_least_0.10": deep["factorial"]["group_accuracy"]
            - shallow["factorial"]["group_accuracy"] >= .10,
        }
        gates = (
            looped["first_hop_confirm"]["accuracy"] >= .95,
            looped["direct_confirm"]["accuracy"] >= .95,
            looped["copy_confirm"]["accuracy"] >= .98,
            all(looped[name]["accuracy"] >= .90 for name in looped if name.startswith("composed_")),
            looped["first_hop_query_groups"]["group_accuracy"] >= .85,
            looped["direct_query_groups"]["group_accuracy"] >= .85,
            looped["factorial"]["group_accuracy"] >= .75,
            looped["order_groups"]["group_accuracy"] >= .80,
            looped["distractor_groups"]["group_accuracy"] >= .80,
            looped["boundary_groups"]["group_accuracy"] >= .70,
            looped["boundary_groups"]["marker_free_accuracy"] >= .80,
            looped["long_ood"]["accuracy"] >= .80,
        )
        looping[rep] = {
            "passes_primary_gates": all(gates),
            "confirm_confirm_within_0.05": deep["composed_confirm_confirm"]["accuracy"]
            - looped["composed_confirm_confirm"]["accuracy"] <= .05,
            "factorial_within_0.05": deep["factorial"]["group_accuracy"]
            - looped["factorial"]["group_accuracy"] <= .05,
        }
    verdict = "invalid" if not null_pass else "incomplete" if not positive_pass else \
        "qualified" if primary_pass else "not_qualified"
    return {"verdict": verdict,
            "raw_sequence_binding": "qualified" if primary_pass else "not_qualified",
            "positive_control": "pass" if positive_pass else "fail",
            "null_control": "pass" if null_pass else "fail",
            "depth_helpful": all(all(row.values()) for row in depth.values()),
            "looping_competitive": all(all(row.values()) for row in looping.values()),
            "clauses": clauses, "depth_clauses": depth, "looping_clauses": looping}


def git_blob(revision, path):
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                          capture_output=True, check=True, timeout=10).stdout


def verify_provenance(report, benchmark):
    head = report["source_git_head"]
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor",
                               FROZEN_SOURCE_REVISION, head], cwd=ROOT).returncode == 0
    need(ancestor, "Launch source does not descend from frozen registration")
    need(report["source_worktree_status"] == [], "Registered source was dirty at launch")
    need(set(report["source_sha256"]) == set(SOURCE_PATHS), "Wrong source path manifest")
    for path in SOURCE_PATHS:
        need(digest(git_blob(head, path)) == report["source_sha256"][path],
             f"Source blob hash mismatch: {path}")
    need(benchmark["source_git_head"] == head
         and benchmark["source_sha256"] == report["source_sha256"],
         "Benchmark/run source mismatch")


def normalized_episode(ep):
    return {key: list(value) if key == "tokens" else [list(row) for row in value]
            if key in ("f_rows", "g_rows", "distractor_rows", "serialized_rows") else value
            for key, value in ep.items()}


def audit(report_path, output_path):
    report = json.loads(report_path.read_text())
    result_dir = report_path.parent
    benchmark = json.loads((result_dir / "benchmark.json").read_text())
    need(report.get("experiment") == "TEACH-0012" and report.get("status") == "complete",
         "Campaign is not complete")
    same(report["config"], CONFIG, "config")
    need(benchmark.get("status") == "pass", "Benchmark did not pass")
    same(benchmark["config"], CONFIG, "benchmark.config")
    verify_provenance(report, benchmark)
    need(set(benchmark["measured"]) == set(ARMS), "Missing benchmark arms")
    for arm in ARMS:
        need(benchmark["measured"][arm]["parameters"] == PARAMETERS[arm]
             and benchmark["measured"][arm]["timed_steps"] == 20,
             f"Bad benchmark metadata for {arm}")
    suite = expected_suite()
    need(sum(map(len, suite.values())) == 4096, "Wrong suite size")
    suite_payload = {name: [normalized_episode(ep) for ep in episodes]
                     for name, episodes in suite.items()}
    suite_raw = json.dumps(suite_payload, sort_keys=True, separators=(",", ":")).encode()
    need(digest(suite_raw) == report["suite_sha256"], "Suite hash mismatch")
    rescored = {"0": {}, "1": {}}
    for rep in ("0", "1"):
        need(set(report["scores"][rep]) == set(ARMS), f"Missing score arms rep{rep}")
        for arm in ARMS:
            prediction_path = result_dir / f"predictions-rep{rep}-{arm}.json.gz"
            need(file_digest(prediction_path) == report["prediction_sha256"][rep][arm],
                 f"Prediction archive hash mismatch rep{rep} {arm}")
            rows_by_name = json.loads(gzip.decompress(prediction_path.read_bytes()))
            need(set(rows_by_name) == set(suite), f"Prediction cells differ rep{rep} {arm}")
            rescored[rep][arm] = {}
            for name, expected in suite.items():
                rows = rows_by_name[name]
                need(len(rows) == len(expected), f"{rep}/{arm}/{name}: row count")
                for index, (row, episode) in enumerate(zip(rows, expected, strict=True)):
                    normalized = normalized_episode(episode)
                    for field in ("tokens", "f_rows", "g_rows", "distractor_rows",
                                  "serialized_rows", "answer", "task", "query",
                                  "f_partition", "g_partition", "logical_id", "render_id",
                                  "distractor_chains", "marker_dropout"):
                        need(row[field] == normalized[field],
                             f"{rep}/{arm}/{name}/{index}: {field} mismatch")
                    need(row["index"] == index and row["length"] == len(episode["tokens"]),
                         f"{rep}/{arm}/{name}/{index}: index/length mismatch")
                    need(type(row["prediction"]) is int and row["prediction"] in SYMBOLS,
                         f"{rep}/{arm}/{name}/{index}: prediction outside symbols")
                    need(type(row["target_probability"]) in (int, float)
                         and math.isfinite(row["target_probability"])
                         and 0 <= row["target_probability"] <= 1,
                         f"{rep}/{arm}/{name}/{index}: bad probability")
                    candidates = ({episode["query"]} if episode["task"] == "copy" else
                                  {right for _, right in episode["f_rows"] + episode["g_rows"]
                                   + episode["distractor_rows"]})
                    need(row["candidate_member"] == (row["prediction"] in candidates),
                         f"{rep}/{arm}/{name}/{index}: candidate flag mismatch")
                rescored[rep][arm][name] = score(rows, name)
            same(report["scores"][rep][arm], rescored[rep][arm], f"scores.{rep}.{arm}")
            training = report["training"][rep][arm]
            need(training["parameters"] == PARAMETERS[arm]
                 and training["loss_count"] == 8000, f"Training metadata rep{rep} {arm}")
            loss_path = result_dir / f"losses-rep{rep}-{arm}.json.gz"
            need(file_digest(loss_path) == training["loss_archive_sha256"],
                 f"Loss archive hash mismatch rep{rep} {arm}")
            losses = json.loads(gzip.decompress(loss_path.read_bytes()))
            need(len(losses) == 8000 and all(type(x) in (int, float) and math.isfinite(x)
                                             for x in losses),
                 f"Bad losses rep{rep} {arm}")
            final = training["final_checkpoint"]
            checkpoint = ROOT / final["path"]
            need(checkpoint.is_file() and checkpoint.stat().st_size == final["bytes"]
                 and file_digest(checkpoint) == final["sha256"],
                 f"Final checkpoint mismatch rep{rep} {arm}")
            expected_milestones = {"250", "500", "1000", "2000", "4000"} \
                if arm == "raw_deep" else set()
            need(set(training["milestones"]) == expected_milestones,
                 f"Milestones mismatch rep{rep} {arm}")
            for milestone in training["milestones"].values():
                path = ROOT / milestone["path"]
                need(path.is_file() and file_digest(path) == milestone["sha256"],
                     f"Milestone checkpoint mismatch rep{rep} {arm}")
    expected_decision = decide(rescored)
    same(report["decision"], expected_decision, "decision")
    audit_report = {
        "audit": "pass", "experiment": "TEACH-0012",
        "frozen_source_revision": FROZEN_SOURCE_REVISION,
        "launch_source_revision": report["source_git_head"],
        "suite_sha256": report["suite_sha256"],
        "decision": expected_decision,
        "verified_prediction_archives": 10,
        "verified_loss_archives": 10,
        "verified_final_checkpoints": 10,
        "verified_milestone_checkpoints": 10,
    }
    output_path.write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n")
    return audit_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "results/TEACH-0012/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/TEACH-0012/audit.json")
    args = parser.parse_args()
    result = audit(args.report, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
