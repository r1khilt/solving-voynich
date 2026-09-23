#!/usr/bin/env python3
"""Freeze the TEACH-0013 suite after an independently audited TEACH-0012 pass."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from voynich.workspace.teacher13_tasks import counterfactual_suite, semantic_layout


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "docs/experiments/TEACH-0013.md",
    "src/voynich/workspace/teacher13_tasks.py",
    "src/voynich/workspace/teacher13_intervene.py",
    "scripts/teacher0013_prepare.py",
    "tests/test_workspace_teacher13.py",
    "tests/test_workspace_teacher13_intervene.py",
    "tests/test_teacher0013_prepare.py",
)
VARIANTS = (
    "base", "donor", "marker_free_base", "marker_free_donor",
    "reordered_base", "reordered_donor", "g_content_base", "g_content_donor",
    "format_donor", "distractor_donor", "first_hop_base", "first_hop_donor",
    "direct_base", "direct_donor", "copy_control",
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
    suite = counterfactual_suite(
        seed, discovery_groups=discovery_groups, confirmation_groups=confirmation_groups)
    payload = {"experiment": "TEACH-0013", "namespace": "TEACH-0013-v1", "seed": seed,
               "splits": {}}
    for split, groups in suite.items():
        payload["splits"][split] = []
        for group in groups:
            record = asdict(group)
            record["semantic_layouts"] = {
                name: [asdict(semantic_layout(episode)) for episode in _episodes(group, name)]
                for name in VARIANTS
            }
            payload["splits"][split].append(record)
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


def verify_entry(report_path: Path, audit_path: Path):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if report.get("experiment") != "TEACH-0012" or report.get("status") != "complete":
        raise RuntimeError("TEACH-0012 report is absent or incomplete")
    if audit.get("experiment") != "TEACH-0012" or audit.get("audit") != "pass":
        raise RuntimeError("Passing independent TEACH-0012 audit required")
    if report.get("decision") != audit.get("decision"):
        raise RuntimeError("Trainer and independent auditor decisions disagree")
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
