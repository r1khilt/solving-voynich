"""Independent, CPU-only artifact audit for the frozen TEACH-0002 study.

This module never imports the trainer, loads weights, or runs model inference.
It reconstructs the preregistered suite and scores from saved predictions.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess


REVISION = "a514a36a3e00fee1c0b8c99fe085f053f414a126"
SOURCE_PATHS = (
    "docs/experiments/TEACH-0002.md",
    "src/voynich/workspace/teacher2_tasks.py",
    "src/voynich/workspace/teacher2_train.py",
    "src/voynich/model.py",
)
ARMS = ("baseline", "curriculum", "null")
REPLICATES = ("0", "1")
NAMES = tuple(range(9, 21))
KEYS = tuple(range(21, 33))
OBJECTS = tuple(range(33, 45))
TYPES = {4: "composed", 5: "direct", 6: "copy", 8: "first_hop"}
CONFIG = {
    "batch_size": 128, "eval_seed": 62311, "eval_size": 256,
    "init_seeds": [62111, 62121], "learning_rate": 0.0003,
    "max_mps_bytes": 8 * 1024**3, "max_seconds": 1800.0,
    "steps_per_arm": 5000, "train_seeds": [62211, 62221],
}
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
    """A stored artifact contradicts the frozen registration or its own data."""


def need(condition: bool, explanation: str) -> None:
    if not condition:
        raise AuditError(explanation)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def partition(kind: str, first: tuple[int, int], second: tuple[int, int]) -> str:
    payload = b"TEACH-0002-v1/" + kind.encode() + b"/"
    payload += bytes((*sorted(first), 255, *sorted(second)))
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10
    return "train" if bucket <= 7 else "holdout"


def parse_tokens(tokens: list[int]) -> tuple[str, int, tuple, tuple, str, str]:
    need(len(tokens) == 14 and all(type(x) is int for x in tokens), "Bad token sequence")
    need(tokens[:2] == [1, 2] and tokens[6] == 3 and tokens[13] == 7,
         "Bad episode markers")
    names, assigned = (tokens[2], tokens[4]), (tokens[3], tokens[5])
    keys, objects = (tokens[7], tokens[9]), (tokens[8], tokens[10])
    need(len(set(names)) == len(set(assigned)) == len(set(keys)) == len(set(objects)) == 2,
         "Episode has duplicated table entries")
    need(all(x in NAMES for x in names) and all(x in KEYS for x in keys + assigned)
         and set(assigned) == set(keys) and all(x in OBJECTS for x in objects),
         "Episode has illegal table symbols")
    task = TYPES.get(tokens[11])
    need(task is not None, "Bad task marker")
    query = tokens[12]
    if task == "first_hop":
        need(query in names, "Bad first-hop query")
        answer = assigned[names.index(query)]
    elif task == "composed":
        need(query in names, "Bad composed query")
        answer = objects[keys.index(assigned[names.index(query)])]
    elif task == "direct":
        need(query in keys, "Bad direct query")
        answer = objects[keys.index(query)]
    else:
        need(query in OBJECTS, "Bad copy query")
        answer = query
    f_family = (tuple(sorted(names)), tuple(sorted(keys)))
    g_family = (tuple(sorted(keys)), tuple(sorted(objects)))
    return (task, answer, f_family, g_family, partition("F", names, keys),
            partition("G", keys, objects))


def episode(names: tuple[int, int], assigned: tuple[int, int],
            keys: tuple[int, int], objects: tuple[int, int], task: str,
            query: int) -> tuple[list[int], int]:
    marker = {"composed": 4, "direct": 5, "copy": 6, "first_hop": 8}[task]
    tokens = [1, 2, names[0], assigned[0], names[1], assigned[1],
              3, keys[0], objects[0], keys[1], objects[1], marker, query, 7]
    return tokens, parse_tokens(tokens)[1]


def draw(rng: random.Random, task: str, f: str, g: str) -> tuple[list[int], int]:
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        objects = tuple(rng.sample(OBJECTS, 2))
        if (partition("F", names, keys), partition("G", keys, objects)) != (f, g):
            continue
        assigned = keys if rng.randrange(2) == 0 else keys[::-1]
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        if task in ("first_hop", "composed"):
            query = rng.choice(names)
        elif task == "direct":
            query = rng.choice(keys)
        else:
            query = rng.choice(OBJECTS)
        return episode(names, assigned, table_keys, objects, task, query)
    raise AuditError("Evaluation sampler did not find a family")


def quartet(rng: random.Random) -> list[tuple[list[int], int]]:
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


def expected_suite() -> dict[str, list[tuple[list[int], int]]]:
    rng = random.Random(62311)
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
        pairs.extend((first, episode((t[2], t[4]), (t[3], t[5]),
                                     (t[7], t[9]), (t[8], t[10]), "first_hop", other)))
    suite["first_hop_pairs_holdout"] = pairs
    for g in ("train", "holdout"):
        suite[f"direct_{g}"] = [draw(rng, "direct", "train", g) for _ in range(256)]
    pairs = []
    for _ in range(128):
        first = draw(rng, "direct", "train", "holdout")
        t = first[0]
        other = t[9] if t[12] == t[7] else t[7]
        pairs.extend((first, episode((t[2], t[4]), (t[3], t[5]),
                                     (t[7], t[9]), (t[8], t[10]), "direct", other)))
    suite["direct_pairs_holdout"] = pairs
    suite["copy"] = [draw(rng, "copy", "holdout", "holdout") for _ in range(256)]
    suite["factorial"] = [item for _ in range(128) for item in quartet(rng)]
    return suite


def wilson(success: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def score_cell(rows: list[dict], name: str) -> dict:
    total = len(rows)
    correct = sum(row["prediction"] == row["answer"] for row in rows)
    score = {"correct": correct, "total": total, "accuracy": correct / total,
             "wilson_95": wilson(correct, total),
             "candidate_member": sum(row["candidate_member"] for row in rows)}
    if name in ("factorial", "first_hop_pairs_holdout", "direct_pairs_holdout"):
        width = 4 if name == "factorial" else 2
        groups = [rows[i:i + width] for i in range(0, total, width)]
        exact = sum(all(row["prediction"] == row["answer"] for row in group)
                    for group in groups)
        score.update({"exact_groups": exact, "total_groups": len(groups),
                      "group_accuracy": exact / len(groups),
                      "group_wilson_95": wilson(exact, len(groups)),
                      "prediction_reversals": sum(group[0]["prediction"] != group[1]["prediction"]
                                                  for group in groups)})
        if width == 4:
            score["f_swap_prediction_changes"] = sum(
                group[a]["prediction"] != group[b]["prediction"]
                for group in groups for a, b in ((0, 1), (2, 3)))
            score["g_remap_prediction_changes"] = sum(
                group[a]["prediction"] != group[b]["prediction"]
                for group in groups for a, b in ((0, 2), (1, 3)))
            score["f_swap_pair_exact"] = sum(
                all(group[i]["prediction"] == group[i]["answer"] for i in pair)
                for group in groups for pair in ((0, 1), (2, 3)))
            score["g_remap_pair_exact"] = sum(
                all(group[i]["prediction"] == group[i]["answer"] for i in pair)
                for group in groups for pair in ((0, 2), (1, 3)))
    return score


def paired_shortcut_diagnostics(rows: list[dict], name: str) -> dict:
    """Exploratory paired-query behavior, grouped by frozen F or G family."""
    need(name in ("first_hop_pairs_holdout", "direct_pairs_holdout")
         and len(rows) % 2 == 0, "Not a paired query cell")
    positions = {"first_candidate": 0, "second_candidate": 0, "noncandidate": 0}
    for row in rows:
        t = row["tokens"]
        first, second = ((t[3], t[5]) if name.startswith("first_hop")
                         else (t[8], t[10]))
        slot = ("first_candidate" if row["prediction"] == first
                else "second_candidate" if row["prediction"] == second
                else "noncandidate")
        positions[slot] += 1
    pair_count = len(rows) // 2
    changes = 0
    exact = 0
    families: dict[str, dict[str, bool]] = {}
    for i in range(0, len(rows), 2):
        first, second = rows[i:i + 2]
        family = first["f_family"] if name.startswith("first_hop") else first["g_family"]
        key = json.dumps(family, separators=(",", ":"))
        changed = first["prediction"] != second["prediction"]
        both = first["prediction"] == first["answer"] and second["prediction"] == second["answer"]
        changes += changed
        exact += both
        state = families.setdefault(key, {"any_change": False, "any_exact": False})
        state["any_change"] |= changed
        state["any_exact"] |= both
    family_changes = sum(state["any_change"] for state in families.values())
    family_exact = sum(state["any_exact"] for state in families.values())
    return {
        "candidate_positions": positions, "prediction_changes_on_query": changes,
        "pair_count": pair_count, "query_change_wilson_95": wilson(changes, pair_count),
        "pair_exact": exact, "pair_exact_wilson_95": wilson(exact, pair_count),
        "unique_families": len(families), "families_with_any_query_change": family_changes,
        "families_with_any_exact_pair": family_exact,
        "family_any_change_wilson_95": wilson(family_changes, len(families)),
        "family_any_exact_wilson_95": wilson(family_exact, len(families)),
    }


def same(actual, expected, where: str) -> None:
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"{where}: list differs")
        for i, value in enumerate(expected):
            same(actual[i], value, f"{where}[{i}]")
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
             f"{where}: {actual!r} != {expected!r}")
    else:
        need(actual == expected and type(actual) is type(expected),
             f"{where}: {actual!r} != {expected!r}")


def decision(scores: dict) -> dict:
    clauses = {}
    for replicate in REPLICATES:
        c, b, n = (scores[replicate][arm] for arm in ("curriculum", "baseline", "null"))
        compose = c["composed_holdout_holdout"]["accuracy"]
        clauses[replicate] = {
            "first_hop_holdout_at_least_0.90": c["first_hop_holdout"]["accuracy"] >= .90,
            "first_hop_paired_exact_at_least_0.80": c["first_hop_pairs_holdout"]["group_accuracy"] >= .80,
            "direct_train_and_holdout_at_least_0.90": all(
                c[f"direct_{part}"]["accuracy"] >= .90 for part in ("train", "holdout")),
            "direct_paired_exact_at_least_0.80": c["direct_pairs_holdout"]["group_accuracy"] >= .80,
            "both_holdout_composed_at_least_0.85": compose >= .85,
            "crossed_holdout_composed_each_at_least_0.85": all(
                c[cell]["accuracy"] >= .85
                for cell in ("composed_holdout_train", "composed_train_holdout")),
            "factorial_exact_at_least_0.65": c["factorial"]["group_accuracy"] >= .65,
            "copy_at_least_0.95": c["copy"]["accuracy"] >= .95,
            "composed_gain_vs_baseline_at_least_0.20": (
                compose - b["composed_holdout_holdout"]["accuracy"] >= .20),
            "factorial_gain_vs_baseline_at_least_0.20": (
                c["factorial"]["group_accuracy"] - b["factorial"]["group_accuracy"] >= .20),
            "composed_gain_vs_null_at_least_0.40": (
                compose - n["composed_holdout_holdout"]["accuracy"] >= .40),
            "null_factorial_at_most_0.10": n["factorial"]["group_accuracy"] <= .10,
        }
    return {"verdict": "qualified" if all(all(row.values()) for row in clauses.values())
            else "not_qualified", "clauses_by_replicate": clauses}


def check_source(root: Path, report: dict) -> None:
    need(report.get("source_git_head") == REVISION, "Wrong source revision")
    need(report.get("source_worktree_status") == [], "Dirty registered source at launch")
    saved = report.get("source_sha256")
    need(isinstance(saved, dict) and set(saved) == set(SOURCE_PATHS), "Source hash set differs")
    for path in SOURCE_PATHS:
        proc = subprocess.run(["git", "show", f"{REVISION}:{path}"], cwd=root,
                              capture_output=True, check=True, timeout=10)
        need(saved[path] == digest(proc.stdout), f"Source hash mismatch: {path}")
    same(report.get("config"), CONFIG, "config")
    encoded = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    need(report.get("config_sha256") == digest(encoded), "Config hash mismatch")


def check_suite_structure(suite: dict) -> None:
    for name, examples in suite.items():
        task, f, g, count = CELL_SPECS[name]
        need(len(examples) == count, f"{name}: wrong cell size")
        for tokens, answer in examples:
            parsed = parse_tokens(tokens)
            need(parsed[0] == task and parsed[1] == answer and parsed[4:] == (f, g),
                 f"{name}: wrong task, oracle answer or split")
    for name in ("first_hop_pairs_holdout", "direct_pairs_holdout"):
        for i in range(0, len(suite[name]), 2):
            first, second = suite[name][i:i + 2]
            a, b = first[0], second[0]
            need(a[:12] == b[:12] and a[13] == b[13] and a[12] != b[12]
                 and first[1] != second[1], f"{name}: invalid query reversal {i // 2}")
    for i in range(0, len(suite["factorial"]), 4):
        group = suite["factorial"][i:i + 4]
        a, b, c, d = (item[0] for item in group)
        need(len({item[1] for item in group}) == 4,
             f"factorial: repeated answer in quartet {i // 4}")
        need(a[2:6] != b[2:6] and a[2:6] == c[2:6] and b[2:6] == d[2:6]
             and a[7:11] == b[7:11] and c[7:11] == d[7:11]
             and a[7:11] != c[7:11] and len({a[12], b[12], c[12], d[12]}) == 1,
             f"factorial: invalid independent changes {i // 4}")


def check_archive(path: Path, expected: dict) -> tuple[dict, dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AuditError(f"Unreadable prediction archive: {path}") from exc
    need(isinstance(rows, dict) and set(rows) == set(CELL_SPECS),
         f"{path}: wrong cells")
    scores = {}
    for name, examples in expected.items():
        cell_rows = rows[name]
        need(isinstance(cell_rows, list) and len(cell_rows) == len(examples),
             f"{path}:{name}: wrong item count")
        for i, (row, (tokens, answer)) in enumerate(zip(cell_rows, examples, strict=True)):
            task, independent_answer, f_family, g_family, f_part, g_part = parse_tokens(tokens)
            spec = CELL_SPECS[name]
            need((task, f_part, g_part) == spec[:3] and answer == independent_answer,
                 f"{path}:{name}:{i}: independent oracle/split mismatch")
            need(type(row) is dict and set(row) == {
                "index", "tokens", "answer", "prediction", "task", "f_family",
                "g_family", "candidate_member"}, f"{path}:{name}:{i}: wrong row fields")
            need(row["index"] == i and row["tokens"] == tokens
                 and row["answer"] == answer and row["task"] == task,
                 f"{path}:{name}:{i}: changed evaluation example")
            need(row["f_family"] == [list(x) for x in f_family]
                 and row["g_family"] == [list(x) for x in g_family],
                 f"{path}:{name}:{i}: wrong family signature")
            allowed = KEYS if task == "first_hop" else OBJECTS
            need(type(row["prediction"]) is int and row["prediction"] in allowed,
                 f"{path}:{name}:{i}: illegal prediction")
            candidates = (tokens[3], tokens[5]) if task == "first_hop" else (tokens[8], tokens[10])
            need(type(row["candidate_member"]) is bool
                 and row["candidate_member"] == (row["prediction"] in candidates),
                 f"{path}:{name}:{i}: wrong candidate flag")
        scores[name] = score_cell(cell_rows, name)
    return scores, rows


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0002"
    report_path = result_dir / "report.json"
    need(report_path.exists(), "Final report.json is not present")
    report = json.loads(report_path.read_text())
    need(report.get("experiment") == "TEACH-0002" and report.get("status") == "complete",
         "Scientific run is not complete")
    check_source(root, report)
    need(report.get("device") == "mps", "Scientific run did not use registered MPS device")
    numerical = report.get("numerical_qualification", {})
    need(numerical.get("finite_loss") is True and numerical.get("finite_gradients") is True
         and type(numerical.get("max_cache_logit_error")) in (int, float)
         and math.isfinite(numerical["max_cache_logit_error"])
         and 0 <= numerical["max_cache_logit_error"] <= .002, "Numerical gate failed")
    elapsed = report.get("elapsed_seconds")
    peak = report.get("peak_sampled_mps_allocated_bytes")
    need(type(elapsed) in (float, int) and 0 <= elapsed <= 1800,
         "Global time cap was exceeded")
    need(type(peak) is int and 0 <= peak <= 8 * 1024**3,
         "Sampled MPS allocation cap was exceeded")
    status = json.loads((result_dir / "status.json").read_text())
    need(status.get("status") == "complete" and status.get("report") == "report.json"
         and status.get("source_git_head") == REVISION
         and status.get("peak_sampled_mps_allocated_bytes") == peak,
         "Final status file disagrees")
    suite = expected_suite()
    need(set(suite) == set(CELL_SPECS), "Independent suite cells differ")
    check_suite_structure(suite)
    suite_payload = {name: [(tokens, answer) for tokens, answer in examples]
                     for name, examples in suite.items()}
    suite_hash = digest(json.dumps(suite_payload, sort_keys=True).encode())
    need(report.get("evaluation_suite_sha256") == suite_hash, "Evaluation suite hash differs")
    arms = report.get("arms")
    saved_scores = report.get("scores")
    need(isinstance(arms, dict) and set(arms) == set(REPLICATES)
         and isinstance(saved_scores, dict) and set(saved_scores) == set(REPLICATES),
         "Wrong replicate set")
    computed = {}
    checkpoints = {}
    archives = {}
    shortcuts = {}
    for rep in REPLICATES:
        need(set(arms[rep]) == set(ARMS) and set(saved_scores[rep]) == set(ARMS),
             f"Replicate {rep}: wrong arms")
        computed[rep], checkpoints[rep], archives[rep], shortcuts[rep] = {}, {}, {}, {}
        for arm in ARMS:
            record = arms[rep][arm]
            need(record.get("steps") == 5000 and record.get("parameters") == 668032,
                 f"{rep}/{arm}: wrong steps or model size")
            history = record.get("history")
            need(isinstance(history, list) and len(history) == 20
                 and [row.get("step") for row in history] == list(range(250, 5001, 250))
                 and all(type(row.get("last_100_loss")) in (int, float)
                         and math.isfinite(row["last_100_loss"]) and row["last_100_loss"] >= 0
                         for row in history), f"{rep}/{arm}: invalid training history")
            checkpoint = root / "outputs/TEACH-0002" / f"rep{rep}-{arm}.pt"
            prediction = result_dir / f"predictions-rep{rep}-{arm}.json.gz"
            need(checkpoint.is_file() and prediction.is_file(), f"{rep}/{arm}: missing artifact")
            checkpoints[rep][arm] = file_digest(checkpoint)
            archives[rep][arm] = file_digest(prediction)
            need(checkpoints[rep][arm] == record.get("checkpoint_sha256"),
                 f"{rep}/{arm}: checkpoint hash mismatch")
            need(archives[rep][arm] == record.get("predictions_sha256"),
                 f"{rep}/{arm}: prediction hash mismatch")
            computed[rep][arm], rows = check_archive(prediction, suite)
            shortcuts[rep][arm] = {
                name: paired_shortcut_diagnostics(rows[name], name)
                for name in ("first_hop_pairs_holdout", "direct_pairs_holdout")
            }
            same(saved_scores[rep][arm], computed[rep][arm], f"scores.{rep}.{arm}")
    expected_decision = decision(computed)
    same(report.get("decision"), expected_decision, "report.decision")
    same(status.get("decision"), expected_decision, "status.decision")
    expected_completed = [[int(rep), arm] for rep in REPLICATES for arm in ARMS]
    same(report.get("completed_arms"), expected_completed, "completed_arms")
    return {
        "experiment": "TEACH-0002", "audit": "pass",
        "source_git_head": REVISION, "source_hashes_verified": list(SOURCE_PATHS),
        "config_sha256": report["config_sha256"], "evaluation_suite_sha256": suite_hash,
        "checkpoint_sha256": checkpoints, "prediction_sha256": archives,
        "evaluated_items_per_arm": sum(len(items) for items in suite.values()),
        "independently_computed_scores": computed, "decision": expected_decision,
        "exploratory_shortcut_diagnostics": shortcuts,
        "elapsed_seconds": elapsed, "peak_sampled_mps_allocated_bytes": peak,
        "numerical_qualification": numerical,
        "limitations": [
            "Artifact checks and metadata cannot prove that every optimizer update ran as recorded.",
            "A checkpoint hash proves file identity, not that its weights generated the archived predictions.",
            "Randomized test episodes can share atomic token relations despite held-out table families.",
            "This synthetic method test gives no Voynich decipherment or causal circuit evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    destination = args.output or root / "results/TEACH-0002/audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"audit": result["audit"], "decision": result["decision"]["verdict"],
                      "audited_items_per_arm": result["evaluated_items_per_arm"]}, indent=2))


if __name__ == "__main__":
    main()
