"""Independent no-model all-item audit of the TEACH-0020 development screen."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from scripts.teacher0014_artifact_audit import audit_trace
from scripts.teacher0014_behavior_audit import (
    _pass_extrapolation, _pass_twohop, _score_panel,
)
from scripts.teacher0014_suite_audit import audit_manifest
ARMS_DONE = (
    "oracle_rows_workspace", "latent_rows_answer", "latent_rows_edge_aux",
    "latent_rows_causal", "latent_rows_wrong_causal", "latent_rows_one_read",
)
EXPECTED_MANIFEST = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
MAX_SECONDS = 3600.0
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 1024**3
SOURCE_PATHS = (
    "docs/experiments/TEACH-0020-interrupted-development-screen.md",
    "scripts/teacher0020_dev_run.py",
    "scripts/teacher0020_dev_audit.py",
    "scripts/teacher0020_dev_replay.py",
    "tests/test_teacher0020_dev.py",
)
PRIMARY_SOURCES = (
    "docs/experiments/TEACH-0014-design.md",
    "docs/experiments/TEACH-0014-benchmark-registration.md",
    "docs/experiments/TEACH-0014-resource-amendment.md",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
    "src/voynich/workspace/teacher14_objectives.py",
    "src/voynich/workspace/teacher14_train.py",
    "scripts/teacher0014_suite_audit.py",
    "scripts/teacher0014_behavior_audit.py",
    "scripts/teacher0014_parser_audit.py",
    "scripts/teacher0014_artifact_audit.py",
    "scripts/teacher0014_replay.py",
    "tests/test_workspace_teacher14_tasks.py",
    "tests/test_workspace_teacher14_models.py",
    "tests/test_workspace_teacher14_objectives.py",
    "tests/test_teacher0014_suite_audit.py",
    "tests/test_teacher0014_behavior_audit.py",
    "tests/test_teacher0014_parser_audit.py",
    "tests/test_workspace_teacher14_train.py",
    "tests/test_teacher0014_artifact_audit.py",
    "tests/test_teacher0014_replay.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: Path, report: dict) -> None:
    for prefix, paths in (("screen", SOURCE_PATHS),
                          ("primary", PRIMARY_SOURCES)):
        head = report.get(f"{prefix}_source_git_head")
        hashes = report.get(f"{prefix}_source_sha256")
        if (not isinstance(head, str) or len(head) != 40 or
                not isinstance(hashes, dict) or set(hashes) != set(paths)):
            raise ValueError(f"{prefix} provenance incomplete")
        for name in paths:
            committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                       cwd=root, check=True,
                                       capture_output=True).stdout
            if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                    sha(root / name) != hashes[name]):
                raise ValueError(f"{prefix} source differs: {name}")


def _score(manifest: dict, payload: dict) -> dict:
    panels = payload.get("panels")
    samples = payload.get("samples")
    if (not isinstance(panels, dict) or not isinstance(samples, dict) or
            set(panels) != set(manifest["panels"]) or
            set(samples) != set(manifest["panels"])):
        raise ValueError("Development panel coverage differs")
    scores = {}
    for name, episodes in manifest["panels"].items():
        rows = panels[name]
        if not isinstance(rows, list) or len(rows) != len(episodes):
            raise ValueError("Development answer count differs")
        predicted = []
        for source, row in zip(episodes, rows, strict=True):
            if (not isinstance(row, dict) or set(row) != {
                    "render_id", "prediction"} or
                    row["render_id"] != source["render_id"] or
                    type(row["prediction"]) is not int or
                    not 16 <= row["prediction"] < 2064):
                raise ValueError("Development prediction identity differs")
            predicted.append(row["prediction"])
        scores[name] = _score_panel(
            name, [item["answer"] for item in episodes], predicted)
        indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
        chosen = samples[name]
        if not isinstance(chosen, list) or len(chosen) != len(indices):
            raise ValueError("Development sampled-logit count differs")
        for row, index in zip(chosen, indices, strict=True):
            logits = row.get("logits")
            if (set(row) != {"index", "render_id", "logits"} or
                    row["index"] != index or
                    row["render_id"] != episodes[index]["render_id"] or
                    not isinstance(logits, list) or len(logits) != 2064 or
                    any(type(value) not in (int, float) or
                        not math.isfinite(value) for value in logits)):
                raise ValueError("Development sampled-logit identity differs")
            argmax = 16 + max(range(2048), key=lambda j: logits[16 + j])
            if argmax != predicted[index]:
                raise ValueError("Development sampled-logit argmax differs")
    return {"panels": scores,
            "development_twohop_thresholds_met": _pass_twohop(
                {"panels": scores}),
            "development_extrapolation_thresholds_met": _pass_extrapolation(
                {"panels": scores})}


def audit(root: Path, primary_dir: Path, output_dir: Path,
          suite_path: Path, result_dir: Path) -> dict:
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    status = json.loads((result_dir / "status.json").read_text())
    if (report.get("experiment") != "TEACH-0020" or
            report.get("status") != "complete" or
            status.get("status") != "complete" or
            report.get("max_seconds") != MAX_SECONDS or
            report.get("max_mps_bytes") != MAX_MPS_BYTES or
            report.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES):
        raise ValueError("Development result/resource status invalid")
    _source(root, report)
    primary = json.loads((primary_dir / "status.json").read_text())
    if (primary.get("status") != "running" or
            (primary_dir / "report.json").exists() or
            report.get("primary_status_sha256") != sha(
                primary_dir / "status.json") or
            report.get("primary_benchmark_sha256") != sha(
                primary_dir / "benchmark.json") or
            report.get("primary_source_git_head") != primary.get(
                "source_git_head") or
            report.get("primary_source_sha256") != primary.get(
                "source_sha256")):
        raise ValueError("Interrupted primary identity differs")
    manifest = json.loads(suite_path.read_text())
    checked = audit_manifest(manifest)
    if (checked["manifest_sha256"] != EXPECTED_MANIFEST or
            report.get("manifest_sha256") != EXPECTED_MANIFEST or
            report.get("manifest_file_sha256") != sha(suite_path)):
        raise ValueError("Development suite identity differs")
    files = report.get("completed_checkpoint_files")
    archives = report.get("archives")
    if (not isinstance(files, dict) or set(files) != set(ARMS_DONE) or
            not isinstance(archives, dict) or set(archives) != set(ARMS_DONE)):
        raise ValueError("Development arm coverage differs")
    scores = {}
    for arm in ARMS_DONE:
        if set(files[arm]) != {"0", "1"} or set(archives[arm]) != {"0", "1"}:
            raise ValueError("Development seed coverage differs")
        scores[arm] = {}
        for rep in ("0", "1"):
            checkpoint = output_dir / f"rep{rep}-{arm}-step6000.pt"
            loss_path = primary_dir / f"losses-rep{rep}-{arm}.json.gz"
            detail = files[arm][rep]
            if (detail != {
                    "checkpoint_sha256": sha(checkpoint),
                    "loss_sha256": sha(loss_path),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "loss_bytes": loss_path.stat().st_size}):
                raise ValueError("Development checkpoint/loss hash differs")
            audit_trace(json.loads(gzip.decompress(loss_path.read_bytes())),
                        arm, steps=6000)
            path = result_dir / f"rows-{arm}-rep{rep}.json.gz"
            if archives[arm][rep] != {"sha256": sha(path),
                                     "bytes": path.stat().st_size}:
                raise ValueError("Development archive hash/size differs")
            payload = json.loads(gzip.decompress(path.read_bytes()))
            scores[arm][rep] = _score(manifest, payload)
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if artifact_bytes > MAX_ARTIFACT_BYTES:
        raise ValueError("Development artifact cap exceeded")
    return {"audit": "pass", "scope": "interrupted_development_only",
            "original_campaign_complete": False,
            "sampled_checkpoint_replay": "pending",
            "report_sha256": sha(report_path),
            "suite_sha256": EXPECTED_MANIFEST,
            "scores": scores}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root, root / "results/TEACH-0014-v3",
                   root / "outputs/TEACH-0014-v3",
                   root / "outputs/TEACH-0016/teach14-74111.json",
                   root / "results/TEACH-0020-interrupted-dev-v2")
    path = root / "results/TEACH-0020-interrupted-dev-v2/audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
