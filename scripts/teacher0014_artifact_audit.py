"""Independent TEACH-0014 artifact/provenance audit; no model/trainer imports.

This validates files, source, training traces, resources and answer decisions.
It cannot prove that optimizer steps happened or that checkpoints emitted the
archived predictions; a separate numerical replay is required before a claim.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from scripts.teacher0014_behavior_audit import ARMS, audit_behavior
from scripts.teacher0014_suite_audit import audit_manifest


PARAMETERS = {
    arm: (24_757_760 if arm == "raw_dense_matched" else
          27_390_465 if arm in ("latent_rows_recurrent4", "latent_rows_diffuse")
          else 24_766_465) for arm in ARMS
}
SOURCE_PATHS = (
    "docs/experiments/TEACH-0014-design.md",
    "docs/experiments/TEACH-0014-benchmark-registration.md",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
    "src/voynich/workspace/teacher14_objectives.py",
    "src/voynich/workspace/teacher14_train.py",
    "scripts/teacher0014_suite_audit.py",
    "scripts/teacher0014_behavior_audit.py",
    "scripts/teacher0014_artifact_audit.py",
    "scripts/teacher0014_replay.py",
    "tests/test_workspace_teacher14_tasks.py",
    "tests/test_workspace_teacher14_models.py",
    "tests/test_workspace_teacher14_objectives.py",
    "tests/test_teacher0014_suite_audit.py",
    "tests/test_teacher0014_behavior_audit.py",
    "tests/test_workspace_teacher14_train.py",
    "tests/test_teacher0014_artifact_audit.py",
    "tests/test_teacher0014_replay.py",
)
EXPECTED_CONFIG = {
    "init_seeds": [84121, 84131], "train_seeds": [84221, 84231],
    "eval_seed": 84311, "steps_per_arm": 6000, "batch_size": 32,
    "causal_groups": 4, "eval_groups": 128, "learning_rate": 3e-4,
    "edge_weight": .2, "causal_weight": .2,
    "max_seconds": 28800.0, "max_mps_bytes": 12 * 1024**3,
    "max_artifact_bytes": 4 * 1024**3,
    "benchmark_steps": 24, "benchmark_warmup_steps": 4,
    "benchmark_max_seconds": 1800.0,
}
EXPECTED_SUITE_SHA256 = "6af176d376921caccc0d40641002b94a462b42670fc4794f5b354457d17fa827"


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _committed_sources(root: Path, head: str, hashes: dict) -> None:
    if not isinstance(head, str) or len(head) != 40 or any(
            char not in "0123456789abcdef" for char in head):
        raise ValueError("Malformed source commit")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise ValueError("Source file set differs from registration")
    for path in SOURCE_PATHS:
        raw = subprocess.run(["git", "show", f"{head}:{path}"], cwd=root,
                             capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != hashes[path]:
            raise ValueError(f"Committed source hash mismatch: {path}")


def _expected_losses(arm: str) -> set[str]:
    if arm == "latent_rows_edge_aux":
        return {"answer", "edge"}
    if arm in ("latent_rows_causal", "latent_rows_wrong_causal"):
        return {"answer", "causal"}
    if arm in ("latent_rows_recurrent4", "latent_rows_diffuse"):
        return {"answer", "edge_all_steps", "edge_denoise"}
    return {"answer"}


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def audit_trace(rows: list, arm: str, *, steps: int) -> dict:
    """Check every step and retain comparable input fingerprints."""
    if not isinstance(rows, list) or len(rows) != steps or arm not in ARMS:
        raise ValueError(f"{arm}: loss archive length or arm invalid")
    answers, labels, causal = [], [], []
    for step, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "step", "losses", "gradient_norm", "answer_input_sha256",
                "answer_label_sha256", "causal_input_sha256"} or row["step"] != step:
            raise ValueError(f"{arm}: malformed loss row at step {step}")
        losses = row["losses"]
        if not isinstance(losses, dict) or set(losses) != _expected_losses(arm):
            raise ValueError(f"{arm}: objective component drift at step {step}")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               or value < 0 for value in losses.values()) or (
                type(row["gradient_norm"]) not in (int, float) or
                not math.isfinite(row["gradient_norm"]) or
                row["gradient_norm"] < 0):
            raise ValueError(f"{arm}: nonfinite or negative loss/gradient")
        if not _valid_sha(row["answer_input_sha256"]) or not _valid_sha(
                row["answer_label_sha256"]):
            raise ValueError(f"{arm}: invalid answer input/label fingerprint")
        if arm in ("latent_rows_causal", "latent_rows_wrong_causal"):
            if not _valid_sha(row["causal_input_sha256"]):
                raise ValueError(f"{arm}: missing causal input fingerprint")
        elif row["causal_input_sha256"] is not None:
            raise ValueError(f"{arm}: unexpected causal input fingerprint")
        answers.append(row["answer_input_sha256"])
        labels.append(row["answer_label_sha256"])
        causal.append(row["causal_input_sha256"])
    return {"answer_inputs": answers, "answer_labels": labels,
            "causal_inputs": causal}


def _audit_benchmark(benchmark: dict, report: dict) -> None:
    if benchmark.get("status") != "pass" or benchmark.get("config") != EXPECTED_CONFIG:
        raise ValueError("Benchmark not passed under registered config")
    for key in ("source_git_head", "source_sha256", "config_sha256"):
        if benchmark.get(key) != report.get(key):
            raise ValueError(f"Benchmark/report {key} mismatch")
    measured = benchmark.get("measured")
    if not isinstance(measured, dict) or set(measured) != ARMS:
        raise ValueError("Benchmark arm set incomplete")
    for arm, row in measured.items():
        times = row.get("timed_step_seconds")
        if row.get("parameters") != PARAMETERS[arm] or (
                row.get("warmup_steps") != 4 or row.get("timed_steps") != 20 or
                not isinstance(times, list) or len(times) != 20 or
                any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0
                    for t in times)):
            raise ValueError(f"{arm}: malformed timing record")
        if not math.isclose(row.get("median_timed_step_seconds", -1),
                            statistics.median(times), rel_tol=1e-10,
                            abs_tol=1e-9):
            raise ValueError(f"{arm}: timing median mismatch")
    projection = sum(row["median_timed_step_seconds"] for row in measured.values())
    projection *= EXPECTED_CONFIG["steps_per_arm"] * 2
    conservative = projection * 1.5 + 1800.0
    if not math.isclose(benchmark.get("projected_training_seconds", -1),
                        projection, rel_tol=1e-10, abs_tol=1e-6) or not math.isclose(
            benchmark.get("conservative_projected_seconds", -1), conservative,
            rel_tol=1e-10, abs_tol=1e-6) or conservative > EXPECTED_CONFIG["max_seconds"]:
        raise ValueError("Benchmark resource projection invalid")
    if benchmark.get("peak_sampled_mps_allocated_bytes", math.inf) > (
            EXPECTED_CONFIG["max_mps_bytes"]):
        raise ValueError("Benchmark MPS allocation exceeded bound")
    _audit_mps_fraction(benchmark)


def _audit_mps_fraction(record: dict) -> None:
    recommended = record.get("mps_recommended_max_memory")
    fraction = record.get("mps_memory_fraction")
    if (type(recommended) is not int or recommended <= 0 or
            type(fraction) not in (int, float) or not math.isfinite(fraction) or
            not 0 < fraction <= .40 or
            fraction * recommended > EXPECTED_CONFIG["max_mps_bytes"] + 1):
        raise ValueError("Mac GPU memory fraction exceeds registered cap")


def _audit_replay_archive(archive: dict, manifest: dict,
                          predictions: dict) -> int:
    if archive.get("experiment") != "TEACH-0014" or (
            archive.get("manifest_sha256") != predictions.get("manifest_sha256")):
        raise ValueError("Replay archive suite identity mismatch")
    runs = archive.get("runs")
    if not isinstance(runs, dict) or set(runs) != ARMS:
        raise ValueError("Replay archive arm set incomplete")
    total = 0
    for arm, replicates in runs.items():
        if not isinstance(replicates, dict) or set(replicates) != {"0", "1"}:
            raise ValueError(f"{arm}: replay replicate set incomplete")
        for rep, panels in replicates.items():
            if not isinstance(panels, dict) or set(panels) != set(manifest["panels"]):
                raise ValueError(f"{arm}/{rep}: replay panel set incomplete")
            for name, rows in panels.items():
                episodes = manifest["panels"][name]
                indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
                if not isinstance(rows, list) or len(rows) != len(indices):
                    raise ValueError(f"{arm}/{rep}/{name}: replay sample count")
                for row, index in zip(rows, indices, strict=True):
                    if (not isinstance(row, dict) or
                            set(row) != {"index", "render_id", "logits"} or
                            type(row["index"]) is not int or
                            row["index"] != index or
                            row["render_id"] != episodes[index]["render_id"]):
                        raise ValueError(f"{arm}/{rep}/{name}: replay sample identity")
                    logits = row["logits"]
                    if (not isinstance(logits, list) or len(logits) != 2064 or
                            any(type(value) not in (int, float) or
                                not math.isfinite(value) for value in logits)):
                        raise ValueError(f"{arm}/{rep}/{name}: invalid replay logits")
                    prediction = 16 + max(range(2048),
                                          key=lambda offset: logits[16 + offset])
                    if prediction != predictions["runs"][arm][rep]["panels"][name][
                            index]["prediction"]:
                        raise ValueError(f"{arm}/{rep}/{name}: logit/prediction mismatch")
                    total += 1
    return total


def audit_artifacts(result_dir: Path, output_dir: Path,
                    root: Path) -> dict:
    report = json.loads((result_dir / "report.json").read_text())
    status = json.loads((result_dir / "status.json").read_text())
    benchmark_path = result_dir / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    if report.get("status") != "complete" or status.get("status") != "complete":
        raise ValueError("Campaign not complete")
    if report.get("experiment") != "TEACH-0014" or report.get("config") != (
            EXPECTED_CONFIG) or status.get("config") != EXPECTED_CONFIG:
        raise ValueError("Campaign config or experiment mismatch")
    if report.get("config_sha256") != _canonical_sha(EXPECTED_CONFIG) or (
            report.get("source_worktree_status") != []):
        raise ValueError("Campaign source/config not clean and frozen")
    _audit_mps_fraction(report)
    _committed_sources(root, report.get("source_git_head"),
                       report.get("source_sha256"))
    if report.get("benchmark_sha256") != _sha_file(benchmark_path):
        raise ValueError("Benchmark artifact hash mismatch")
    _audit_benchmark(benchmark, report)
    manifest_path = result_dir / "suite.json"
    manifest = json.loads(manifest_path.read_text())
    suite = audit_manifest(manifest)
    if (manifest.get("seed") != EXPECTED_CONFIG["eval_seed"] or
            manifest.get("group_count") != EXPECTED_CONFIG["eval_groups"] or
            suite["manifest_sha256"] != EXPECTED_SUITE_SHA256 or
            report.get("suite_sha256") != suite["manifest_sha256"] or
            report.get("manifest_file_sha256") != _sha_file(manifest_path)):
        raise ValueError("Suite identity or structure mismatch")
    pred_path = result_dir / "predictions.json.gz"
    if report.get("predictions_sha256") != _sha_file(pred_path):
        raise ValueError("Prediction artifact hash mismatch")
    predictions = json.loads(gzip.decompress(pred_path.read_bytes()))
    behavior = audit_behavior(manifest, predictions)
    replay_path = result_dir / "replay-logits.json.gz"
    if report.get("replay_logits_sha256") != _sha_file(replay_path):
        raise ValueError("Replay logits artifact hash mismatch")
    replay_archive = json.loads(gzip.decompress(replay_path.read_bytes()))
    replay_samples = _audit_replay_archive(replay_archive, manifest, predictions)
    saved_behavior_path = result_dir / "behavior-audit.json"
    if report.get("behavior_audit_sha256") != _sha_file(saved_behavior_path) or (
            json.loads(saved_behavior_path.read_text()) != behavior):
        raise ValueError("Saved answer decision differs from independent audit")
    training = report.get("training")
    if not isinstance(training, dict) or set(training) != ARMS:
        raise ValueError("Training arm set incomplete")
    fingerprints = {}
    for arm, replicates in training.items():
        if not isinstance(replicates, dict) or set(replicates) != {"0", "1"}:
            raise ValueError(f"{arm}: training replicate set incomplete")
        fingerprints[arm] = {}
        for rep, metadata in replicates.items():
            checkpoint = output_dir / (
                f"rep{rep}-{arm}-step{EXPECTED_CONFIG['steps_per_arm']}.pt")
            details = metadata.get("final_checkpoint", {})
            if details.get("path") != str(checkpoint) or not checkpoint.exists() or (
                    details.get("sha256") != _sha_file(checkpoint) or
                    details.get("bytes") != checkpoint.stat().st_size or
                    metadata.get("parameters") != PARAMETERS[arm] or
                    metadata.get("loss_count") != EXPECTED_CONFIG["steps_per_arm"]):
                raise ValueError(f"{arm}/{rep}: checkpoint or count invalid")
            loss_path = result_dir / f"losses-rep{rep}-{arm}.json.gz"
            if metadata.get("loss_archive_sha256") != _sha_file(loss_path):
                raise ValueError(f"{arm}/{rep}: loss hash mismatch")
            losses = json.loads(gzip.decompress(loss_path.read_bytes()))
            fingerprints[arm][rep] = audit_trace(
                losses, arm, steps=EXPECTED_CONFIG["steps_per_arm"])
    for rep in ("0", "1"):
        reference = fingerprints["oracle_rows_workspace"][rep]
        for arm in ARMS:
            row = fingerprints[arm][rep]
            if row["answer_inputs"] != reference["answer_inputs"]:
                raise ValueError(f"{arm}/{rep}: ordinary training input drift")
            if arm != "raw_null" and row["answer_labels"] != (
                    reference["answer_labels"]):
                raise ValueError(f"{arm}/{rep}: ordinary answer label drift")
        if (fingerprints["latent_rows_causal"][rep]["causal_inputs"] !=
                fingerprints["latent_rows_wrong_causal"][rep]["causal_inputs"]):
            raise ValueError(f"rep{rep}: causal objective input drift")
    if (not 0 <= report.get("elapsed_seconds", math.inf)
            <= EXPECTED_CONFIG["max_seconds"] or
            report.get("peak_sampled_mps_allocated_bytes", math.inf) >
            EXPECTED_CONFIG["max_mps_bytes"] or
            _artifact_bytes(result_dir) + _artifact_bytes(output_dir) >
            EXPECTED_CONFIG["max_artifact_bytes"]):
        raise ValueError("Campaign resource cap exceeded")
    return {"audit": "pass", "scope": "artifact_and_answer_behavior",
            "manifest_sha256": suite["manifest_sha256"],
            "source_git_head": report["source_git_head"],
            "training_traces": len(ARMS) * 2,
            "checked_steps": len(ARMS) * 2 * EXPECTED_CONFIG["steps_per_arm"],
            "replay_logit_samples": replay_samples,
            "answer_decisions": behavior["decisions"],
            "numerical_checkpoint_replay": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/TEACH-0014"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_artifacts(args.result_dir, args.output_dir, args.root)
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
