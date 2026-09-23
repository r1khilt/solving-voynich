#!/usr/bin/env python3
"""Benchmark and run TEACH-0013 discovery without confirmation access."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from voynich.workspace.teacher13_discovery import (
    fresh_panel_scores,
    load_suite,
    relation_endpoint_labels,
    residual_numerical_qualification,
    residual_screen,
    screen_conditions,
    two_site_identity_qualification,
    two_site_path_screen,
)
from voynich.workspace.teacher13_intervene import load_raw_checkpoint
from voynich.workspace.teacher13_score import (
    rank_secondary_residual_sites,
    select_residual_site,
    select_two_site_path,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "scripts/teacher0013_campaign.py"
TEST_PATH = "tests/test_teacher0013_campaign.py"
SECONDARY_FAMILIES = ("f_binding", "g_content", "g_binding", "order", "format", "distractor")
STAGE_B_SECONDS = 7200.0
CAMPAIGN_SECONDS = 14400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3


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


def verify_frozen_source(manifest: dict) -> dict:
    paths = tuple(manifest["source_sha256"])
    expected = set(paths) | {SCRIPT_PATH, TEST_PATH}
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted(expected)], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise RuntimeError("Commit all registered TEACH-0013 source before inference")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    hashes = {}
    for relative in sorted(expected):
        raw = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True,
            capture_output=True).stdout
        hashes[relative] = sha_bytes(raw)
        if relative in manifest["source_sha256"] \
                and hashes[relative] != manifest["source_sha256"][relative]:
            raise RuntimeError(f"Suite source changed after freeze: {relative}")
    return {"campaign_source_git_head": revision, "campaign_source_sha256": hashes}


def load_frozen(result_dir: Path, output_dir: Path):
    manifest_path = result_dir / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "TEACH-0013" or manifest.get("status") != "suite_frozen":
        raise RuntimeError("Frozen TEACH-0013 suite manifest required")
    suite_path = output_dir / "suite.json.gz"
    if sha_file(suite_path) != manifest["suite_gzip_sha256"]:
        raise RuntimeError("TEACH-0013 suite gzip hash mismatch")
    raw = gzip.decompress(suite_path.read_bytes())
    if sha_bytes(raw) != manifest["suite_uncompressed_sha256"]:
        raise RuntimeError("TEACH-0013 suite content hash mismatch")
    suite = load_suite(suite_path)
    if {name: len(groups) for name, groups in suite["splits"].items()} \
            != manifest["split_counts"]:
        raise RuntimeError("TEACH-0013 suite split counts changed")
    provenance = verify_frozen_source(manifest)
    return manifest, suite, provenance


def benchmark_projection(elapsed_seconds: float, *, benchmark_groups: int,
                         discovery_groups: int, benchmark_seeds: int = 1,
                         campaign_seeds: int = 2, materialized_bytes: int = 0,
                         family_item_factor: float = 4.5) -> dict:
    if min(elapsed_seconds, benchmark_groups, discovery_groups,
           benchmark_seeds, campaign_seeds, family_item_factor) <= 0:
        raise ValueError("Benchmark projection inputs must be positive")
    scale = (discovery_groups / benchmark_groups * campaign_seeds / benchmark_seeds
             * family_item_factor)
    stage_b = elapsed_seconds * scale * 1.5
    complete = stage_b * 2
    projected_bytes = int(materialized_bytes * scale * 1.5)
    return {"scale": scale, "conservative_stage_b_seconds": stage_b,
            "conservative_complete_seconds": complete,
            "projected_materialized_activation_bytes": projected_bytes,
            "family_item_factor": family_item_factor,
            "stage_b_pass": stage_b <= STAGE_B_SECONDS,
            "complete_pass": complete <= CAMPAIGN_SECONDS,
            "traffic_pass": projected_bytes <= MAX_TRAFFIC_BYTES}


def conditional_path_projection(elapsed_seconds: float, materialized_bytes: int, *,
                                benchmark_groups: int, discovery_groups: int,
                                residual_seconds: float, residual_bytes: int) -> dict:
    if min(elapsed_seconds, benchmark_groups, discovery_groups) <= 0 \
            or min(materialized_bytes, residual_seconds, residual_bytes) < 0:
        raise ValueError("Conditional path projection inputs are invalid")
    scale = discovery_groups / benchmark_groups * 2 * 1.5
    seconds = elapsed_seconds * scale
    traffic = int(materialized_bytes * scale)
    return {"scale": scale, "conservative_projected_path_seconds": seconds,
            "projected_path_materialized_bytes": traffic,
            "remaining_stage_b_seconds": STAGE_B_SECONDS - residual_seconds,
            "remaining_campaign_seconds": CAMPAIGN_SECONDS - residual_seconds,
            "stage_b_pass": seconds <= STAGE_B_SECONDS - residual_seconds,
            "campaign_pass": seconds <= CAMPAIGN_SECONDS - residual_seconds,
            "traffic_pass": traffic + residual_bytes <= MAX_TRAFFIC_BYTES}


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) \
        if path.exists() else 0


def checkpoint_path(manifest: dict, replicate: str, step="8000") -> Path:
    metadata = manifest["checkpoints"][replicate][step]
    path = ROOT / metadata["path"]
    if path.stat().st_size != metadata["bytes"] or sha_file(path) != metadata["sha256"]:
        raise RuntimeError(f"Checkpoint mismatch replicate {replicate} step {step}")
    return path


def validate_benchmark(report: dict, manifest: dict, provenance: dict) -> None:
    """Recompute the benchmark decision before allowing expensive discovery."""
    required = {"elapsed_seconds", "groups", "materialized_activation_bytes",
                "peak_sampled_current_allocated_bytes", "checkpoint_sha256",
                "suite_gzip_sha256", "campaign_source_sha256", "projection", "status"}
    if not required.issubset(report):
        raise RuntimeError("TEACH-0013 benchmark fields are incomplete")
    expected = benchmark_projection(
        report["elapsed_seconds"], benchmark_groups=report["groups"],
        discovery_groups=manifest["split_counts"]["discovery"],
        materialized_bytes=report["materialized_activation_bytes"])
    should_pass = (all(expected[name] for name in ("stage_b_pass", "complete_pass",
                                                   "traffic_pass"))
                   and report["peak_sampled_current_allocated_bytes"] <= MAX_CURRENT_BYTES)
    if report["projection"] != expected or report["status"] != ("pass" if should_pass else "stop"):
        raise RuntimeError("TEACH-0013 benchmark decision does not recompute")
    if report["groups"] != 32 \
            or report["checkpoint_sha256"] != manifest["checkpoints"]["0"]["8000"]["sha256"] \
            or report["suite_gzip_sha256"] != manifest["suite_gzip_sha256"] \
            or report["campaign_source_sha256"] != provenance["campaign_source_sha256"]:
        raise RuntimeError("TEACH-0013 benchmark provenance mismatch")


def benchmark(result_dir: Path, output_dir: Path, device: str):
    report_path = result_dir / "benchmark.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0013 benchmark overwrite")
    manifest, suite, provenance = load_frozen(result_dir, output_dir)
    groups = suite["splits"]["discovery"][:32]
    net, metadata = load_raw_checkpoint(checkpoint_path(manifest, "0"), device=device)
    if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
        raise RuntimeError("Benchmark requires the final raw-deep checkpoint")
    peak = {"current": 0, "driver": 0}
    traffic = {"bytes": 0}

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))

    def count_traffic(value):
        traffic["bytes"] += int(value)

    if device == "mps":
        torch.mps.synchronize()
        probe()
    start = time.monotonic()
    rows = residual_screen(
        net, groups, replicate=0, device=device,
        labels=suite["semantic_label_order"], resource_probe=probe,
        traffic_probe=count_traffic)
    if device == "mps":
        torch.mps.synchronize()
        probe()
    elapsed = time.monotonic() - start
    projection = benchmark_projection(
        elapsed, benchmark_groups=len(groups),
        discovery_groups=len(suite["splits"]["discovery"]),
        materialized_bytes=traffic["bytes"])
    report = {
        "experiment": "TEACH-0013", "mode": "benchmark",
        "status": "pass" if projection["stage_b_pass"] and projection["complete_pass"]
        and projection["traffic_pass"]
        and peak["current"] <= MAX_CURRENT_BYTES else "stop",
        "groups": len(groups), "rows_generated": len(rows),
        "elapsed_seconds": elapsed,
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "materialized_activation_bytes": traffic["bytes"],
        "checkpoint_sha256": manifest["checkpoints"]["0"]["8000"]["sha256"],
        "suite_gzip_sha256": manifest["suite_gzip_sha256"],
        "projection": projection, **provenance,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _write_rows(path: Path, rows: list[dict]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9,
                           mtime=0) as compressed:
            for row in rows:
                compressed.write(stable_json(row) + b"\n")
    return sha_file(path)


def _read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_path_prerequisites(result_dir: Path, manifest: dict, provenance: dict):
    decision_path = result_dir / "discovery-decision.json"
    audit_path = result_dir / "discovery-audit.json"
    decision, audit = json.loads(decision_path.read_text()), json.loads(audit_path.read_text())
    if decision.get("status") != "stage_b_residual_complete_path_pending" \
            or decision.get("residual_selection", {}).get("selection") is not None:
        raise RuntimeError("Two-site search is allowed only after no residual site qualifies")
    if audit.get("audit") != "pass" or audit.get("scope") != "residual_discovery" \
            or audit.get("discovery_decision_sha256") != sha_file(decision_path):
        raise RuntimeError("Passing matching independent residual-discovery audit required")
    if decision.get("suite_gzip_sha256") != manifest["suite_gzip_sha256"] \
            or decision.get("campaign_source_sha256") != provenance["campaign_source_sha256"]:
        raise RuntimeError("Residual decision provenance mismatch")
    return decision, decision_path, audit_path


def path_benchmark(result_dir: Path, output_dir: Path, device: str):
    report_path = result_dir / "path-benchmark.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0013 path benchmark overwrite")
    manifest, suite, provenance = load_frozen(result_dir, output_dir)
    residual, decision_path, audit_path = load_path_prerequisites(
        result_dir, manifest, provenance)
    groups = suite["splits"]["discovery"][:8]
    net, metadata = load_raw_checkpoint(checkpoint_path(manifest, "0"), device=device)
    if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
        raise RuntimeError("Path benchmark requires the final raw-deep checkpoint")
    peak, traffic = {"current": 0, "driver": 0}, {"bytes": 0}

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))

    def count_traffic(value):
        traffic["bytes"] += int(value)

    if device == "mps":
        torch.mps.synchronize()
        probe()
    start = time.monotonic()
    rows = two_site_path_screen(
        net, groups, replicate=0, device=device,
        source_labels=relation_endpoint_labels(suite["semantic_label_order"]),
        resource_probe=probe, traffic_probe=count_traffic)
    if device == "mps":
        torch.mps.synchronize()
        probe()
    elapsed = time.monotonic() - start
    projection = conditional_path_projection(
        elapsed, traffic["bytes"], benchmark_groups=len(groups),
        discovery_groups=len(suite["splits"]["discovery"]),
        residual_seconds=residual["elapsed_seconds"],
        residual_bytes=residual["materialized_activation_bytes"])
    passed = (projection["stage_b_pass"] and projection["campaign_pass"]
              and projection["traffic_pass"] and peak["current"] <= MAX_CURRENT_BYTES)
    report = {
        "experiment": "TEACH-0013", "mode": "path_benchmark",
        "status": "pass" if passed else "stop", "groups": len(groups),
        "rows_generated": len(rows), "elapsed_seconds": elapsed,
        "projection": projection,
        "measured_materialized_bytes": traffic["bytes"],
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "residual_decision_sha256": sha_file(decision_path),
        "residual_audit_sha256": sha_file(audit_path),
        "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def validate_path_benchmark(report: dict, residual: dict, manifest: dict,
                            provenance: dict) -> None:
    expected = conditional_path_projection(
        report["elapsed_seconds"], report["measured_materialized_bytes"],
        benchmark_groups=report["groups"],
        discovery_groups=manifest["split_counts"]["discovery"],
        residual_seconds=residual["elapsed_seconds"],
        residual_bytes=residual["materialized_activation_bytes"])
    passed = (expected["stage_b_pass"] and expected["campaign_pass"]
              and expected["traffic_pass"]
              and report["peak_sampled_current_allocated_bytes"] <= MAX_CURRENT_BYTES)
    if report.get("groups") != 8 or report.get("projection") != expected \
            or report.get("status") != ("pass" if passed else "stop") \
            or report.get("suite_gzip_sha256") != manifest["suite_gzip_sha256"] \
            or report.get("campaign_source_sha256") != provenance["campaign_source_sha256"]:
        raise RuntimeError("Conditional path benchmark does not recompute")


def path_discovery(result_dir: Path, output_dir: Path, device: str):
    report_path = result_dir / "path-decision.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0013 path discovery overwrite")
    manifest, suite, provenance = load_frozen(result_dir, output_dir)
    residual, decision_path, audit_path = load_path_prerequisites(
        result_dir, manifest, provenance)
    benchmark_path = result_dir / "path-benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    validate_path_benchmark(benchmark, residual, manifest, provenance)
    if benchmark.get("status") != "pass" \
            or benchmark.get("residual_decision_sha256") != sha_file(decision_path) \
            or benchmark.get("residual_audit_sha256") != sha_file(audit_path) \
            or benchmark.get("campaign_source_sha256") != provenance["campaign_source_sha256"]:
        raise RuntimeError("Passing matching path benchmark required")
    groups = suite["splits"]["discovery"]
    artifacts = {}
    start = time.monotonic()
    peak, traffic = {"current": 0, "driver": 0}, {"bytes": 0}
    remaining = STAGE_B_SECONDS - residual["elapsed_seconds"]

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        if time.monotonic() - start > remaining:
            raise RuntimeError("TEACH-0013 conditional path wall-time ceiling exceeded")
        if peak["current"] > MAX_CURRENT_BYTES:
            raise RuntimeError("TEACH-0013 path sampled MPS allocation ceiling exceeded")
        if traffic["bytes"] + residual["materialized_activation_bytes"] > MAX_TRAFFIC_BYTES:
            raise RuntimeError("TEACH-0013 cumulative Stage-B traffic ceiling exceeded")

    def count_traffic(value):
        traffic["bytes"] += int(value)

    for replicate in ("0", "1"):
        net, metadata = load_raw_checkpoint(checkpoint_path(manifest, replicate), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise RuntimeError("Path discovery requires final raw-deep checkpoints")
        rows = two_site_path_screen(
            net, groups, replicate=int(replicate), device=device,
            source_labels=relation_endpoint_labels(suite["semantic_label_order"]),
            resource_probe=probe, traffic_probe=count_traffic)
        path = result_dir / f"path-rows-rep{replicate}.jsonl.gz"
        artifacts[replicate] = {"path": str(path.relative_to(ROOT)),
                                "sha256": _write_rows(path, rows),
                                "rows": len(rows), "bytes": path.stat().st_size}
        probe()
        if tree_bytes(result_dir) > MAX_RESULT_BYTES:
            raise RuntimeError("TEACH-0013 path tracked-artifact ceiling exceeded")
        if device == "mps":
            torch.mps.synchronize()
            del net
            torch.mps.empty_cache()
    rows_all = [row for replicate in ("0", "1")
                for row in _read_rows(ROOT / artifacts[replicate]["path"])]
    selection = select_two_site_path(
        rows_all, tuple(suite["semantic_label_order"]),
        expected_groups_per_render=len(groups))
    del rows_all
    path_numerical = {}
    if selection["selection"]:
        selected = selection["selection"]
        raw_sites = tuple(["embed"] + [f"blocks.{index}.resid_post" for index in range(12)])
        for replicate in ("0", "1"):
            net, _ = load_raw_checkpoint(checkpoint_path(manifest, replicate), device=device)
            path_numerical[replicate] = two_site_identity_qualification(
                net, groups, early_site=raw_sites[selected["early_cut_index"]],
                source_label=selected["source_label"],
                late_site=raw_sites[selected["late_cut_index"]],
                destination_label=selected["destination_label"], device=device)
            count_traffic(path_numerical[replicate]["materialized_bytes"])
            probe()
            if device == "mps":
                torch.mps.synchronize()
                del net
                torch.mps.empty_cache()
    elapsed = time.monotonic() - start
    output_bytes, result_bytes = tree_bytes(output_dir), tree_bytes(result_dir)
    if output_bytes > MAX_OUTPUT_BYTES or result_bytes > MAX_RESULT_BYTES \
            or elapsed > remaining:
        raise RuntimeError("TEACH-0013 conditional path resource ceiling exceeded")
    numerical_pass = (not selection["selection"] or all(
        row["qualified"] for row in path_numerical.values()))
    status = ("stage_b_path_numerical_inconclusive" if selection["selection"]
              and not numerical_pass else "stage_b_path_complete" if selection["selection"]
              else "stage_b_no_static_or_two_site_mediator")
    report = {
        "experiment": "TEACH-0013", "mode": "path_discovery", "status": status,
        "suite_gzip_sha256": manifest["suite_gzip_sha256"],
        "residual_decision_sha256": sha_file(decision_path),
        "residual_audit_sha256": sha_file(audit_path),
        "path_benchmark_sha256": sha_file(benchmark_path),
        "path_selection": selection, "path_numerical_qualification": path_numerical,
        "artifacts": artifacts,
        "elapsed_seconds": elapsed,
        "cumulative_stage_b_seconds": residual["elapsed_seconds"] + elapsed,
        "materialized_activation_bytes": traffic["bytes"],
        "cumulative_stage_b_materialized_bytes": (
            residual["materialized_activation_bytes"] + traffic["bytes"]),
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "ignored_output_bytes": output_bytes, "tracked_result_bytes": result_bytes,
        **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def discovery(result_dir: Path, output_dir: Path, device: str):
    decision_path = result_dir / "discovery-decision.json"
    if decision_path.exists():
        raise FileExistsError("No automatic TEACH-0013 discovery rerun")
    manifest, suite, provenance = load_frozen(result_dir, output_dir)
    benchmark_path = result_dir / "benchmark.json"
    benchmark_report = json.loads(benchmark_path.read_text())
    validate_benchmark(benchmark_report, manifest, provenance)
    if benchmark_report.get("status") != "pass":
        raise RuntimeError("Passing matching TEACH-0013 benchmark required")
    groups = suite["splits"]["discovery"]
    clean, artifacts, numerical = {}, {}, {}
    start = time.monotonic()
    peak = {"current": 0, "driver": 0}
    traffic = {"bytes": 0}

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        if time.monotonic() - start > STAGE_B_SECONDS:
            raise RuntimeError("TEACH-0013 Stage-B wall-time ceiling exceeded")
        if peak["current"] > MAX_CURRENT_BYTES:
            raise RuntimeError("TEACH-0013 sampled MPS allocation ceiling exceeded")
        if traffic["bytes"] > MAX_TRAFFIC_BYTES:
            raise RuntimeError("TEACH-0013 materialized activation traffic ceiling exceeded")

    def count_traffic(value):
        traffic["bytes"] += int(value)
    for replicate in ("0", "1"):
        net, metadata = load_raw_checkpoint(checkpoint_path(manifest, replicate), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise RuntimeError("Discovery requires final raw-deep checkpoints")
        clean[replicate] = fresh_panel_scores(net, groups, device=device)
        numerical[replicate] = residual_numerical_qualification(
            net, groups, device=device)
        if not numerical[replicate]["qualified"]:
            raise RuntimeError(f"Replicate {replicate} failed residual numerical qualification")
        artifacts[replicate] = {}
        for family in ("f_content", *SECONDARY_FAMILIES):
            rows = residual_screen(
                net, groups, replicate=int(replicate), device=device,
                labels=suite["semantic_label_order"], resource_probe=probe,
                traffic_probe=count_traffic, family=family)
            path = result_dir / f"discovery-rows-rep{replicate}-{family}.jsonl.gz"
            artifacts[replicate][family] = {
                "path": str(path.relative_to(ROOT)), "sha256": _write_rows(path, rows),
                "rows": len(rows), "bytes": path.stat().st_size}
            probe()
        if device == "mps":
            torch.mps.synchronize()
            probe()
            del net
            torch.mps.empty_cache()
    primary_rows = [row for replicate in ("0", "1")
                    for row in _read_rows(ROOT / artifacts[replicate]["f_content"]["path"])]
    selection = select_residual_site(
        primary_rows, tuple(suite["semantic_label_order"]),
        expected_groups_per_render=len(groups))
    del primary_rows
    secondary_selection = {}
    for family in SECONDARY_FAMILIES:
        family_rows = [row for replicate in ("0", "1")
                       for row in _read_rows(ROOT / artifacts[replicate][family]["path"])]
        secondary_selection[family] = rank_secondary_residual_sites(
            family_rows, tuple(suite["semantic_label_order"]),
            expected_groups=len(groups) * len(screen_conditions(family)))
        del family_rows
    output_bytes, result_bytes = tree_bytes(output_dir), tree_bytes(result_dir)
    if output_bytes > MAX_OUTPUT_BYTES or result_bytes > MAX_RESULT_BYTES:
        raise RuntimeError("TEACH-0013 artifact ceiling exceeded")
    elapsed = time.monotonic() - start
    if elapsed > STAGE_B_SECONDS or traffic["bytes"] > MAX_TRAFFIC_BYTES:
        raise RuntimeError("TEACH-0013 Stage-B runtime or traffic ceiling exceeded")
    report = {
        "experiment": "TEACH-0013", "mode": "discovery",
        "status": ("stage_b_single_site_complete" if selection["selection"]
                   else "stage_b_residual_complete_path_pending"),
        "suite_gzip_sha256": manifest["suite_gzip_sha256"],
        "benchmark_sha256": sha_file(benchmark_path),
        "clean_discovery_scores": clean, "numerical_qualification": numerical,
        "residual_selection": selection,
        "secondary_residual_selections": secondary_selection,
        "next_stage": ("frozen_single_site_confirmation" if selection["selection"]
                       else "registered_two_site_path_screen_required"),
        "artifacts": artifacts, "elapsed_seconds": elapsed,
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"], **provenance,
        "materialized_activation_bytes": traffic["bytes"],
        "ignored_output_bytes": output_bytes, "tracked_result_bytes": result_bytes,
    }
    decision_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(
        "benchmark", "discovery", "path-benchmark", "path-discovery"))
    parser.add_argument("--result-dir", type=Path,
                        default=ROOT / "results/TEACH-0013")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    runners = {"benchmark": benchmark, "discovery": discovery,
               "path-benchmark": path_benchmark, "path-discovery": path_discovery}
    result = runners[args.mode](args.result_dir, args.output_dir, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
