#!/usr/bin/env python3
"""Freeze the TEACH-0013 suite after an independently audited TEACH-0012 pass."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from voynich.workspace.teacher13_tasks import counterfactual_suite, semantic_layout


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "docs/experiments/TEACH-0013.md",
    "src/voynich/workspace/teacher13_tasks.py",
    "src/voynich/workspace/teacher13_intervene.py",
    "src/voynich/workspace/teacher13_score.py",
    "src/voynich/workspace/teacher13_discovery.py",
    "src/voynich/workspace/teacher13_geometry.py",
    "src/voynich/workspace/teacher13_confirm.py",
    "scripts/teacher0013_campaign.py",
    "scripts/teacher0013_confirm.py",
    "scripts/teacher0013_confirmation_audit.py",
    "scripts/teacher0013_audit.py",
    "scripts/teacher0013_discovery_audit.py",
    "scripts/teacher0013_prepare.py",
    "src/voynich/workspace/teacher12_tasks.py",
    "src/voynich/workspace/teacher12_models.py",
    "src/voynich/model.py",
    "docs/experiments/TEACH-0012-audit-hardening-amendment.md",
    "scripts/teacher0012_analyze.py",
    "tests/test_teacher0012_analyze.py",
    "pyproject.toml",
    "uv.lock",
    "tests/test_workspace_teacher13.py",
    "tests/test_workspace_teacher13_intervene.py",
    "tests/test_workspace_teacher13_score.py",
    "tests/test_workspace_teacher13_discovery.py",
    "tests/test_workspace_teacher13_geometry.py",
    "tests/test_workspace_teacher13_confirm.py",
    "tests/test_teacher0013_campaign.py",
    "tests/test_teacher0013_confirm.py",
    "tests/test_teacher0013_audit.py",
    "tests/test_teacher0013_discovery_audit.py",
    "tests/test_teacher0013_prepare.py",
)
TEACH12_LAUNCH_SOURCE_PATHS = (
    "docs/experiments/TEACH-0012.md",
    "docs/experiments/TEACH-0012-benchmark-gate-amendment.md",
    "src/voynich/workspace/teacher12_tasks.py",
    "src/voynich/workspace/teacher12_models.py",
    "src/voynich/workspace/teacher12_train.py",
    "scripts/teacher0012_analyze.py",
    "tests/test_workspace_teacher12.py",
    "src/voynich/model.py",
)
TEACH12_AUDITOR_PATHS = (
    "docs/experiments/TEACH-0012-audit-hardening-amendment.md",
    "scripts/teacher0012_analyze.py",
    "tests/test_teacher0012_analyze.py",
)
VARIANTS = (
    "base", "donor", "marker_free_base", "marker_free_donor",
    "reordered_base", "reordered_donor", "g_content_base", "g_content_donor",
    "binding_base", "binding_donor", "g_binding_base", "g_binding_donor",
    "format_donor", "distractor_donor", "first_hop_base", "first_hop_donor",
    "direct_base", "direct_donor", "direct_format_donor", "direct_order_donor",
    "direct_distractor_donor", "copy_control", "copy_format_donor",
    "copy_order_donor", "copy_distractor_donor",
)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _episodes(group, name):
    value = getattr(group, name)
    return value if isinstance(value, tuple) else (value,)


def suite_payload(seed=73111, discovery_groups=128, confirmation_groups=128):
    """Materialize logical groups together with exhaustive semantic position maps."""
    suite, generation_stats = counterfactual_suite(
        seed, discovery_groups=discovery_groups, confirmation_groups=confirmation_groups,
        return_stats=True)
    payload = {"experiment": "TEACH-0013", "namespace": "TEACH-0013-v1", "seed": seed,
               "generation_stats": generation_stats, "splits": {}}
    for split, groups in suite.items():
        payload["splits"][split] = []
        for group in groups:
            record = asdict(group)
            record["semantic_layouts"] = {
                name: [asdict(semantic_layout(episode)) for episode in _episodes(group, name)]
                for name in VARIANTS
            }
            payload["splits"][split].append(record)
    payload["semantic_label_order"] = sorted({
        label
        for group in payload["splits"]["discovery"]
        for layouts in group["semantic_layouts"].values()
        for layout in layouts
        for label in layout["labels"]
    })
    return payload


def source_provenance():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *SOURCE_PATHS], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise RuntimeError("Commit registered TEACH-0013 source before freezing the suite")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    hashes = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha_file(path)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True,
            capture_output=True).stdout
        if sha_bytes(committed) != hashes[relative]:
            raise RuntimeError(f"Registered source differs from HEAD: {relative}")
    return {"source_git_head": revision, "source_sha256": hashes}


def git_blob(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=ROOT, check=True,
        capture_output=True, timeout=30).stdout


def verify_teach12_provenance(report: dict, audit: dict) -> None:
    """Bind the entry decision to the exact committed trainer and auditor source."""
    auditor = audit.get("auditor", {})
    revision, hashes = auditor.get("revision"), auditor.get("sha256")
    if not isinstance(revision, str) or not isinstance(hashes, dict) \
            or set(hashes) != set(TEACH12_AUDITOR_PATHS):
        raise RuntimeError("Malformed or incomplete TEACH-0012 auditor provenance")
    for relative in TEACH12_AUDITOR_PATHS:
        if sha_bytes(git_blob(revision, relative)) != hashes[relative]:
            raise RuntimeError(f"TEACH-0012 auditor source mismatch: {relative}")
    launch = report.get("source_git_head")
    launch_hashes = report.get("source_sha256")
    if audit.get("launch_source_revision") != launch \
            or not isinstance(launch_hashes, dict) \
            or set(launch_hashes) != set(TEACH12_LAUNCH_SOURCE_PATHS):
        raise RuntimeError("TEACH-0012 launch provenance is inconsistent")
    for relative in TEACH12_LAUNCH_SOURCE_PATHS:
        if sha_bytes(git_blob(launch, relative)) != launch_hashes[relative]:
            raise RuntimeError(f"TEACH-0012 launch source mismatch: {relative}")
    if audit.get("suite_sha256") != report.get("suite_sha256"):
        raise RuntimeError("TEACH-0012 audit/report suite mismatch")


def independently_reaudit_teach12(report_path: Path, expected_audit: dict) -> None:
    """Recompute the TEACH-0012 audit so checkpoint metadata cannot be substituted."""
    with tempfile.TemporaryDirectory(prefix="teach0013-entry-") as directory:
        output = Path(directory) / "audit.json"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/teacher0012_analyze.py"),
             "--report", str(report_path), "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0 or not output.is_file():
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Independent TEACH-0012 re-audit failed: {message[-1000:]}")
        fresh = json.loads(output.read_text())
    for field in ("audit", "experiment", "launch_source_revision", "suite_sha256",
                  "decision", "verified_prediction_archives", "verified_loss_archives",
                  "verified_final_checkpoints", "verified_milestone_checkpoints"):
        if fresh.get(field) != expected_audit.get(field):
            raise RuntimeError(f"TEACH-0012 re-audit disagrees on {field}")


def verify_entry(report_path: Path, audit_path: Path):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if report.get("experiment") != "TEACH-0012" or report.get("status") != "complete":
        raise RuntimeError("TEACH-0012 report is absent or incomplete")
    if audit.get("experiment") != "TEACH-0012" or audit.get("audit") != "pass":
        raise RuntimeError("Passing independent TEACH-0012 audit required")
    if report.get("decision") != audit.get("decision"):
        raise RuntimeError("Trainer and independent auditor decisions disagree")
    verify_teach12_provenance(report, audit)
    independently_reaudit_teach12(report_path, audit)
    decision = audit["decision"]
    if decision.get("verdict") != "qualified" \
            or decision.get("raw_sequence_binding") != "qualified" \
            or decision.get("positive_control") != "pass" \
            or decision.get("null_control") != "pass":
        raise RuntimeError("TEACH-0013 entry condition was not met")
    checkpoints = {}
    for replicate in ("0", "1"):
        training = report["training"][replicate]["raw_deep"]
        rows = {"8000": training["final_checkpoint"], **training["milestones"]}
        checkpoints[replicate] = {}
        for step, metadata in sorted(rows.items(), key=lambda row: int(row[0])):
            path = ROOT / metadata["path"]
            if not path.is_file() or path.stat().st_size != metadata["bytes"] \
                    or sha_file(path) != metadata["sha256"]:
                raise RuntimeError(f"Raw-deep checkpoint mismatch: replicate {replicate} step {step}")
            checkpoints[replicate][step] = metadata
    return {"teach0012_report_sha256": sha_file(report_path),
            "teach0012_audit_sha256": sha_file(audit_path),
            "checkpoints": checkpoints}


def prepare(report_path: Path, audit_path: Path, output_dir: Path, result_dir: Path):
    provenance = source_provenance()
    entry = verify_entry(report_path, audit_path)
    suite_path = output_dir / "suite.json.gz"
    manifest_path = result_dir / "suite-manifest.json"
    if suite_path.exists() or manifest_path.exists():
        raise FileExistsError("No automatic TEACH-0013 suite overwrite")
    payload = suite_payload()
    raw = stable_json(payload)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    suite_path.write_bytes(compressed)
    manifest = {
        "experiment": "TEACH-0013", "status": "suite_frozen",
        "seed": payload["seed"],
        "generation_stats": payload["generation_stats"],
        "split_counts": {name: len(groups) for name, groups in payload["splits"].items()},
        "group_ids": {name: [group["group_id"] for group in groups]
                      for name, groups in payload["splits"].items()},
        "suite_path": str(suite_path.relative_to(ROOT)),
        "suite_uncompressed_sha256": sha_bytes(raw),
        "suite_gzip_sha256": sha_bytes(compressed),
        "suite_gzip_bytes": len(compressed),
        **entry, **provenance,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results/TEACH-0012/report.json")
    parser.add_argument("--audit", type=Path,
                        default=ROOT / "results/TEACH-0012/audit.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--result-dir", type=Path,
                        default=ROOT / "results/TEACH-0013")
    args = parser.parse_args()
    print(json.dumps(prepare(args.report, args.audit, args.output_dir, args.result_dir),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
