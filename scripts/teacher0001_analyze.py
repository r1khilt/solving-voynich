"""Independent, CPU-only integrity audit of the frozen TEACH-0001 report.

Replays the deterministic generator but does not load a model for inference.
It cannot independently rescore predictions because per-item predictions were
not retained in the compact report.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from voynich.workspace.teacher1_tasks import evaluation_suite, training_batch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def git_blob(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True,
        check=True, timeout=10,
    ).stdout


def family_sets(tokens: tuple[int, ...]) -> tuple[tuple[tuple[int, int], tuple[int, int]],
                                                   tuple[tuple[int, int], tuple[int, int]]]:
    require(len(tokens) == 14, "Episode length changed")
    f = (tuple(sorted((tokens[2], tokens[4]))), tuple(sorted((tokens[3], tokens[5]))))
    g = (tuple(sorted((tokens[7], tokens[9]))), tuple(sorted((tokens[8], tokens[10]))))
    return f, g


def partition(kind: str, family: tuple[tuple[int, int], tuple[int, int]]) -> str:
    left, right = family
    material = bytes([ord(kind), *left, 255, *right])
    bucket = int.from_bytes(hashlib.blake2s(material, digest_size=8).digest(), "big") % 10
    return "train" if bucket < 8 else "holdout"


def independent_answer(tokens: tuple[int, ...]) -> int:
    require(len(tokens) == 14 and tokens[0:2] == (1, 2) and tokens[6] == 3 and
            tokens[13] == 7, "Malformed episode markers")
    names = (tokens[2], tokens[4])
    first_values = (tokens[3], tokens[5])
    second_keys = (tokens[7], tokens[9])
    outputs = (tokens[8], tokens[10])
    require(set(names) <= set(range(8, 16)) and len(set(names)) == 2, "Malformed names")
    require(set(first_values) == set(second_keys) and
            set(second_keys) <= set(range(16, 24)), "Malformed keys")
    require(set(outputs) <= set(range(24, 32)) and len(set(outputs)) == 2,
            "Malformed outputs")
    query = tokens[12]
    if tokens[11] == 4:
        require(query in names, "Invalid composed query")
        key = first_values[names.index(query)]
        return outputs[second_keys.index(key)]
    if tokens[11] == 5:
        require(query in second_keys, "Invalid direct query")
        return outputs[second_keys.index(query)]
    if tokens[11] == 6:
        require(query in range(24, 32), "Invalid copy query")
        return query
    raise AssertionError("Unknown query marker")


def wilson(correct: int, total: int) -> list[float]:
    require(0 <= correct <= total and total > 0, "Invalid binomial count")
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [0.0 if correct == 0 else max(0.0, center - radius),
            1.0 if correct == total else min(1.0, center + radius)]


def independent_decision(scores: dict) -> dict:
    p, n = scores["primary"], scores["null"]

    def accuracy(group: dict) -> float:
        return group["correct"] / group["total"]

    both = accuracy(p["composed_holdout_holdout"])
    null_both = accuracy(n["composed_holdout_holdout"])
    clauses = {
        "both_heldout_composed_at_least_0.90": both >= 0.90,
        "crossed_heldout_composed_each_at_least_0.90": all(
            accuracy(p[name]) >= 0.90 for name in
            ("composed_holdout_train", "composed_train_holdout")),
        "factorial_exact_quartets_at_least_0.70": (
            p["factorial"]["exact_quartets"] / p["factorial"]["total_quartets"] >= 0.70),
        "direct_and_copy_each_at_least_0.95": all(accuracy(p[name]) >= 0.95
                                                   for name in ("direct", "copy")),
        "null_both_heldout_at_most_0.25": null_both <= 0.25,
        "primary_minus_null_at_least_0.60": both - null_both >= 0.60,
    }
    return {"verdict": "qualified" if all(clauses.values()) else "not_qualified", "clauses": clauses}


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0001"
    report = json.loads((result_dir / "report.json").read_text())
    status = json.loads((result_dir / "status.json").read_text())
    require(report["experiment"] == "TEACH-0001" and report["status"] == "complete",
            "Report not a complete TEACH-0001 run")
    require(report["source_worktree_status"] == [], "Registered source was dirty during run")
    require(status["status"] == "complete" and status["decision"] == report["decision"],
            "Status and report disagree")
    for key in ("source_git_head", "registration_sha256", "config_sha256",
                "peak_sampled_mps_allocated_bytes"):
        require(status[key] == report[key], f"Status mismatch: {key}")

    commit = report["source_git_head"]
    source_paths = {
        "tasks": "src/voynich/workspace/teacher1_tasks.py",
        "trainer": "src/voynich/workspace/teacher1_train.py",
        "model": "src/voynich/model.py",
    }
    for label, relative in source_paths.items():
        require(digest_bytes(git_blob(root, commit, relative)) == report["source_sha256"][label],
                f"Git source hash mismatch: {label}")
        require(digest_file(root / relative) == report["source_sha256"][label],
                f"Current source differs from frozen run: {label}")
    registration = "docs/experiments/TEACH-0001.md"
    require(digest_bytes(git_blob(root, commit, registration)) == report["registration_sha256"],
            "Git registration hash mismatch")
    require(digest_file(root / registration) == report["registration_sha256"],
            "Current registration differs from frozen run")
    config = report["config"]
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode()
    require(digest_bytes(config_bytes) == report["config_sha256"], "Config hash mismatch")
    require(config["steps_per_arm"] == 2500 and config["batch_size"] == 128 and
            config["eval_size"] == 256, "Registered size mismatch")

    checkpoint_summary = {}
    for arm, is_null in (("primary", False), ("null", True)):
        saved = report["arms"][arm]
        path = root / "outputs/TEACH-0001" / f"{arm}.pt"
        require(path.is_file(), f"Missing {arm} checkpoint")
        require(digest_file(path) == saved["checkpoint_sha256"],
                f"{arm} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["config"] == config and payload["null_labels"] is is_null,
                f"{arm} checkpoint metadata mismatch")
        params = sum(t.numel() for t in payload["model"].values())
        require(params == saved["parameter_count"] == 664704,
                f"{arm} parameter count mismatch")
        history = saved["history"]
        require(saved["steps"] == 2500 and [row["step"] for row in history] ==
                list(range(250, 2501, 250)), f"{arm} history steps mismatch")
        require(all(math.isfinite(row["last_100_loss"]) for row in history),
                f"{arm} nonfinite saved loss")
        checkpoint_summary[arm] = {"sha256": saved["checkpoint_sha256"],
                                   "parameters": params,
                                   "last_100_loss": history[-1]["last_100_loss"]}

    suite = evaluation_suite(config["eval_seed"], config["eval_size"])
    suite_bytes = json.dumps({k: [(ep.tokens, ep.answer) for ep in rows]
                              for k, rows in suite.items()}, sort_keys=True).encode()
    require(digest_bytes(suite_bytes) == report["eval_suite_sha256"],
            "Evaluation-suite digest mismatch")
    expected_names = {"composed_train_train", "composed_holdout_train",
                      "composed_train_holdout", "composed_holdout_holdout",
                      "direct", "copy", "factorial"}
    require(set(suite) == expected_names, "Evaluation cells changed")
    eval_family_f: dict[str, set] = {name: set() for name in suite}
    eval_family_g: dict[str, set] = {name: set() for name in suite}
    duplicates = {}
    for name, rows in suite.items():
        duplicates[name] = len(rows) - len({ep.tokens for ep in rows})
        for ep in rows:
            require(independent_answer(ep.tokens) == ep.answer, f"Wrong label: {name}")
            f, g = family_sets(ep.tokens)
            require(partition("F", f) == ep.f_partition and
                    partition("G", g) == ep.g_partition, f"Partition mismatch: {name}")
            eval_family_f[name].add(f)
            eval_family_g[name].add(g)
            if name.startswith("composed_"):
                _, wanted_f, wanted_g = name.split("_")
                require((ep.f_partition, ep.g_partition) == (wanted_f, wanted_g),
                        f"Cell partition mismatch: {name}")
            elif name in ("direct", "copy", "factorial"):
                require(ep.f_partition == ep.g_partition == "holdout",
                        f"Control partition mismatch: {name}")
    factorial = suite["factorial"]
    require(len(factorial) == 512, "Factorial size mismatch")
    for offset in range(0, len(factorial), 4):
        four = factorial[offset:offset + 4]
        require(len({ep.answer for ep in four}) == 4, "Factorial answers not distinct")
        require(four[0].tokens[6:] == four[1].tokens[6:] and
                four[2].tokens[6:] == four[3].tokens[6:] and
                four[0].tokens[:6] == four[2].tokens[:6] and
                four[1].tokens[:6] == four[3].tokens[:6],
                "Factorial F/G crossing malformed")

    # Replay the entire frozen CPU generator stream, never the model or optimizer.
    train_f, train_g = set(), set()
    train_task_counts = Counter()
    primary_targets = Counter()
    null_targets = Counter()
    for step in range(config["steps_per_arm"]):
        primary, labels = training_batch(config["train_seed"] + step, config["batch_size"])
        null, null_labels = training_batch(config["train_seed"] + step, config["batch_size"],
                                           null_labels=True)
        require(primary == null, "Primary/null input streams diverged")
        for ep, y, null_y in zip(primary, labels, null_labels, strict=True):
            require(independent_answer(ep.tokens) == ep.answer == y,
                    "Wrong primary training label")
            f, g = family_sets(ep.tokens)
            require(partition("F", f) == partition("G", g) == "train",
                    "Held-out family leaked into training")
            train_f.add(f)
            train_g.add(g)
            train_task_counts[ep.task] += 1
            primary_targets[y] += 1
            null_targets[null_y] += 1
            if ep.task != "composed":
                require(null_y == y, "Null altered direct/copy label")
    for name in suite:
        if "holdout" in name or name in ("direct", "copy", "factorial"):
            if name.startswith("composed_"):
                _, f_part, g_part = name.split("_")
            else:
                f_part = g_part = "holdout"
            if f_part == "holdout":
                require(train_f.isdisjoint(eval_family_f[name]), f"F-family leak in {name}")
            if g_part == "holdout":
                require(train_g.isdisjoint(eval_family_g[name]), f"G-family leak in {name}")

    scores = report["scores"]
    for arm in ("primary", "null"):
        require(set(scores[arm]) == expected_names, f"Score cells changed: {arm}")
        for name, row in scores[arm].items():
            require(row["total"] == len(suite[name]), f"Denominator mismatch: {arm}/{name}")
            require(0 <= row["correct"] <= row["total"], f"Invalid score: {arm}/{name}")
            require(math.isclose(row["accuracy"], row["correct"] / row["total"],
                                 abs_tol=1e-12), f"Accuracy mismatch: {arm}/{name}")
            require(all(math.isclose(a, b, abs_tol=1e-12)
                        for a, b in zip(row["wilson_95"], wilson(row["correct"], row["total"]),
                                        strict=True)), f"Wilson interval mismatch: {arm}/{name}")
            if name == "factorial":
                require(row["total_quartets"] == len(factorial) // 4,
                        f"Quartet denominator mismatch: {arm}")
                require(row["exact_quartets"] <= row["correct"] // 4,
                        f"Quartet score exceeds item score: {arm}")
                q = row["exact_quartets"]
                require(math.isclose(row["quartet_accuracy"], q / row["total_quartets"],
                                     abs_tol=1e-12), f"Quartet accuracy mismatch: {arm}")
                require(all(math.isclose(a, b, abs_tol=1e-12)
                            for a, b in zip(row["quartet_wilson_95"],
                                            wilson(q, row["total_quartets"]), strict=True)),
                        f"Quartet interval mismatch: {arm}")
    decision = independent_decision(scores)
    require(decision == report["decision"], "Registered decision mismatch")
    require(decision["verdict"] == "not_qualified", "Unexpected changed verdict")
    require(report["qualification"]["finite_gradients"] and
            report["qualification"]["finite_loss"] and
            report["qualification"]["max_cached_logit_error"] <= 0.002,
            "Numerical qualification failed")
    require(report["elapsed_seconds"] <= config["max_seconds"] and
            report["peak_sampled_mps_allocated_bytes"] <= config["max_mps_bytes"],
            "Resource cap exceeded")
    return {
        "audit": "pass", "experiment": "TEACH-0001", "source_git_head": commit,
        "decision": decision, "checkpoint_summary": checkpoint_summary,
        "training_episode_count": sum(train_task_counts.values()),
        "training_task_counts": dict(train_task_counts),
        "training_target_counts": {"primary": dict(primary_targets), "null": dict(null_targets)},
        "distinct_training_family_counts": {"F": len(train_f), "G": len(train_g)},
        "evaluation_duplicate_token_counts": duplicates,
        "theoretical_primary_two_choice_mixture_loss": 0.9 * math.log(2),
        "theoretical_null_uniform_composed_two_choice_direct_loss": (
            0.7 * math.log(8) + 0.2 * math.log(2)),
        "limitations": [
            "No per-example predictions or logits were saved, so checkpoint outputs cannot be independently rescored without model inference.",
            "Training stream is reconstructed from frozen source and seeds, not compared with per-step saved prompt hashes.",
            "Wilson intervals treat generated episodes as iid; repeated episodes/families make them descriptive.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    outcome = audit(root)
    path = root / "results/TEACH-0001/audit.json"
    path.write_text(json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"audit": outcome["audit"], "decision": outcome["decision"]["verdict"],
                      "episodes": outcome["training_episode_count"]}))


if __name__ == "__main__":
    main()
