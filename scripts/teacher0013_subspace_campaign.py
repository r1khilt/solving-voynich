#!/usr/bin/env python3
"""Fit and causally rank TEACH-0013 content/binding/order subspaces on discovery."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import random
import time

import torch

from voynich.workspace.teacher13_confirm import MediatorSpec
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_geometry import (
    linear_cka,
    orthogonal_procrustes,
    principal_angles,
    select_joint_causal_rank,
)
from voynich.workspace.teacher13_intervene import (
    load_raw_checkpoint,
    materialized_tensor_counter,
)
from voynich.workspace.teacher13_subspace import (
    factor_rank_screen,
    factor_state_panel,
    fit_registered_factor_geometries,
    summarize_factor_ranks,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_C_E_SECONDS = 5400.0
CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3
CANDIDATE_RANKS = (1, 2, 4, 8, 16, 32, 64)
MINIMUM_SPECTRAL_GAP = .05
FACTOR_THRESHOLDS = {
    "content": (.80, .60),
    "binding": (.75, .55),
    # Order rank is descriptive unless the separately registered shortcut target exists.
    "order": (.75, .55),
}


def _load_confirmation_runner():
    spec = importlib.util.spec_from_file_location(
        "teacher0013_confirm_dependency", ROOT / "scripts/teacher0013_confirm.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen confirmation dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prerequisite(result_dir: Path, stage_c) -> tuple[dict, dict, MediatorSpec]:
    report_path = result_dir / "confirmation-decision.json"
    audit_path = result_dir / "confirmation-audit.json"
    report, audit = (json.loads(path.read_text()) for path in (report_path, audit_path))
    if report.get("status") != "stage_c_complete" \
            or report.get("fresh_panel_decision", {}).get("label") != "pass":
        raise RuntimeError("Completed numerically qualified Stage C required")
    if audit.get("audit") != "pass" or audit.get("scope") != "stage_c_confirmation" \
            or audit.get("confirmation_decision_sha256") != stage_c.sha_file(report_path) \
            or audit.get("recipient_transfer_decision") \
            != report.get("recipient_transfer_decision"):
        raise RuntimeError("Passing matching independent Stage-C audit required")
    return report, audit, MediatorSpec(**report["mediator"])


def tensor_artifact(path: Path, payload: dict, stage_c) -> dict:
    torch.save(payload, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": stage_c.sha_file(path),
            "bytes": path.stat().st_size}


def geometry_payload(panels, fitted) -> dict:
    def contrast(value):
        return {"mean": value.mean, "basis": value.basis,
                "eigenvalues": value.eigenvalues,
                "factor_levels": value.factor_levels,
                "nuisance_cells": value.nuisance_cells,
                "participation_ratio": value.participation_ratio}

    return {
        "format": "TEACH-0013-factor-geometry-v2",
        "panels": {name: {"states": panel.states, "factor": panel.factor,
                           "nuisance": panel.nuisance, "blocks": panel.blocks,
                           "metadata": panel.metadata}
                   for name, panel in panels.items()},
        "geometry": {name: contrast(getattr(fitted, name))
                     for name in ("content", "binding", "order")},
        "orthogonal": {
            "raw": fitted.orthogonal.raw,
            "forward": fitted.orthogonal.forward,
            "reverse": fitted.orthogonal.reverse,
            "forward_eigenvalues": fitted.orthogonal.forward_eigenvalues,
            "reverse_eigenvalues": fitted.orthogonal.reverse_eigenvalues,
            "raw_principal_angles": fitted.orthogonal.raw_principal_angles,
        },
    }


def spectral_boundary_gap(spectrum: torch.Tensor, rank: int) -> float:
    """Relative residual-variance drop after a candidate prefix boundary."""
    spectrum = spectrum.double()
    if rank <= 0 or rank > len(spectrum):
        raise ValueError("Rank lies outside residualized factor spectrum")
    if rank == len(spectrum):
        return 1.0
    current, following = float(spectrum[rank - 1]), float(spectrum[rank])
    return max(0.0, min(1.0, (current - following) / max(current, 1e-12)))


def rank_measurements(summary: dict, spectrum: torch.Tensor | None = None) -> list[dict]:
    rows = [{"rank": int(rank), "item_accuracy": cell["item_accuracy"],
             "group_accuracy": cell["group_accuracy"],
             "mean_probability_gain": cell["mean_probability_gain"]}
            for rank, cell in summary.items() if rank != "full"]
    if spectrum is not None:
        for row in rows:
            row["spectral_boundary_gap"] = spectral_boundary_gap(spectrum, row["rank"])
    return rows


def joint_rank_decision(summaries: dict[str, dict], factor: str,
                        spectra: dict[str, torch.Tensor] | None = None) -> dict:
    full = {seed: rows["full"]["mean_probability_gain"]
            for seed, rows in summaries.items()}
    if any(not torch.isfinite(torch.tensor(value)).item() or value <= 0
           for value in full.values()):
        return {"selection": None, "reason": "nonpositive_full_probability_gain",
                "full_probability_gain_by_seed": full}
    item, group = FACTOR_THRESHOLDS[factor]
    return select_joint_causal_rank(
        {seed: rank_measurements(rows, None if spectra is None else spectra[seed])
         for seed, rows in summaries.items()},
        full_probability_gain_by_seed=full,
        minimum_item_accuracy=item, minimum_group_accuracy=group,
        minimum_spectral_gap=None if spectra is None else MINIMUM_SPECTRAL_GAP,
        candidate_ranks=CANDIDATE_RANKS)


def cross_seed_geometry(seed_payloads: dict) -> dict:
    left, right = seed_payloads["0"], seed_payloads["1"]
    result = {}
    for factor in ("content", "binding", "order"):
        left_panel, right_panel = left["panels"][factor], right["panels"][factor]
        if left_panel.metadata != right_panel.metadata:
            raise RuntimeError("Cross-seed factor panels are not item-aligned")
        mapping = orthogonal_procrustes(left_panel.states, right_panel.states)
        reverse = orthogonal_procrustes(right_panel.states, left_panel.states)
        left_basis = getattr(left["fitted"], factor).basis
        right_basis = getattr(right["fitted"], factor).basis
        result[factor] = {
            "linear_cka": linear_cka(left_panel.states, right_panel.states),
            "principal_angles": principal_angles(left_basis, right_basis),
            "aligned_principal_angles": principal_angles(
                mapping.rotation.T @ left_basis, right_basis),
            "seed0_to_seed1": asdict(mapping),
            "seed1_to_seed0": asdict(reverse),
        }

    factors = ("content", "binding", "order")

    def shared_map(right_block_map=None):
        left_rows, right_rows, means = [], [], {}
        for factor in factors:
            left_panel = seed_payloads["0"]["panels"][factor]
            right_panel = seed_payloads["1"]["panels"][factor]
            right_states = right_panel.states
            if right_block_map is not None:
                lookup = {(block, nuisance, level): index for index, (
                    block, nuisance, level) in enumerate(zip(
                        right_panel.blocks, right_panel.nuisance,
                        right_panel.factor, strict=True))}
                indices = [lookup[(right_block_map[block], nuisance, level)]
                           for block, nuisance, level in zip(
                               left_panel.blocks, left_panel.nuisance,
                               left_panel.factor, strict=True)]
                right_states = right_states[indices]
            left_mean, right_mean = left_panel.states.mean(0), right_states.mean(0)
            scale = len(left_panel.states) ** .5
            left_rows.append((left_panel.states - left_mean) / scale)
            right_rows.append((right_states - right_mean) / scale)
            means[factor] = {"seed0": left_mean, "seed1": right_mean}
        left, right = torch.cat(left_rows), torch.cat(right_rows)
        mapping = orthogonal_procrustes(left, right)
        return {"rotation": mapping.rotation, "normalized_residual": mapping.normalized_residual,
                "factor_means": means,
                "aligned_principal_angles": {factor: principal_angles(
                    mapping.rotation.T
                    @ seed_payloads["0"]["fitted"].orthogonal.forward[factor],
                    seed_payloads["1"]["fitted"].orthogonal.forward[factor])
                    for factor in factors}}

    blocks = sorted(set(seed_payloads["0"]["panels"]["content"].blocks))
    rng = random.Random(73341)
    donors = list(blocks)
    for right in range(len(donors) - 1, 0, -1):
        left = rng.randrange(right)
        donors[left], donors[right] = donors[right], donors[left]
    block_map = dict(zip(blocks, donors, strict=True))
    result["shared"] = shared_map()
    result["shared_deranged_control"] = {
        **shared_map(block_map), "seed": 73341,
        "donor_blocks": tuple(donors)}
    return result


def discovery(result_dir: Path, output_dir: Path, device: str) -> dict:
    report_path = result_dir / "subspace-discovery.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0013 subspace discovery overwrite")
    stage_c = _load_confirmation_runner()
    manifest = stage_c.load_manifest(result_dir)
    provenance = stage_c.verify_frozen_source(manifest)
    confirmation, confirmation_audit, spec = prerequisite(result_dir, stage_c)
    suite = stage_c.load_confirmation_suite(output_dir, manifest)
    groups = suite["splits"]["discovery"]
    if [group["group_id"] for group in groups] != manifest["group_ids"]["discovery"]:
        raise RuntimeError("Discovery group order or membership changed")

    start = time.monotonic()
    peak, traffic = {"current": 0, "driver": 0}, {"bytes": 0}
    artifacts, summaries, seed_payloads = {}, {}, {}

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        elapsed = time.monotonic() - start
        if elapsed + confirmation["elapsed_seconds"] > STAGE_C_E_SECONDS \
                or elapsed + confirmation["cumulative_campaign_seconds"] > CAMPAIGN_SECONDS:
            raise RuntimeError("TEACH-0013 Stage-D/campaign time ceiling exceeded")
        if peak["current"] > MAX_CURRENT_BYTES:
            raise RuntimeError("TEACH-0013 Stage-D sampled allocation ceiling exceeded")
        if traffic["bytes"] + confirmation["cumulative_materialized_bytes"] \
                > MAX_TRAFFIC_BYTES:
            raise RuntimeError("TEACH-0013 Stage-D cumulative traffic ceiling exceeded")

    def count_traffic(value):
        traffic["bytes"] += int(value)

    for replicate in ("0", "1"):
        net, metadata = load_raw_checkpoint(
            stage_c.checkpoint_path(manifest, replicate), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise RuntimeError("Stage D requires final raw-deep checkpoints")
        with materialized_tensor_counter(count_traffic):
            panels = {name: factor_state_panel(
                net, groups, spec, factor_name=name, episode_loader=episode_from_record,
                device=device) for name in ("content", "binding", "order")}
            fitted = fit_registered_factor_geometries(
                panels["content"], panels["binding"], panels["order"])
            rows_by_factor, captures = {}, {}
            for factor in ("content", "binding", "order"):
                basis = fitted.orthogonal.forward[factor]
                captures[factor] = stage_c.LogitCapture()
                rows_by_factor[factor] = ([] if basis.shape[1] == 0 else factor_rank_screen(
                    net, groups, spec, basis, factor_name=factor,
                    episode_loader=episode_from_record, candidate_ranks=CANDIDATE_RANKS,
                    device=device, logit_sink=captures[factor]))
        summaries[replicate] = {
            factor: ({} if not rows else summarize_factor_ranks(rows))
            for factor, rows in rows_by_factor.items()}
        geometry_path = output_dir / f"subspace-geometry-rep{replicate}.pt"
        if geometry_path.exists():
            raise FileExistsError("No automatic TEACH-0013 geometry artifact overwrite")
        artifacts[replicate] = {
            "geometry": tensor_artifact(
                geometry_path, geometry_payload(panels, fitted), stage_c),
            "rank_rows": {},
        }
        for factor, rows in rows_by_factor.items():
            if not rows:
                continue
            path = result_dir / f"subspace-ranks-rep{replicate}-{factor}.jsonl.gz"
            if path.exists():
                raise FileExistsError("No automatic TEACH-0013 rank-row overwrite")
            artifacts[replicate]["rank_rows"][factor] = {
                "path": str(path.relative_to(ROOT)), "sha256": stage_c.write_rows(path, rows),
                "bytes": path.stat().st_size, "rows": len(rows)}
            logits_path = output_dir / f"subspace-rank-logits-rep{replicate}-{factor}.pt"
            if logits_path.exists():
                raise FileExistsError("No automatic TEACH-0013 rank-logit overwrite")
            artifacts[replicate]["rank_rows"][factor]["exact_symbol_logits"] = \
                captures[factor].write(logits_path)
        seed_payloads[replicate] = {"panels": panels, "fitted": fitted}
        if device == "mps":
            torch.mps.synchronize()
            del net
            torch.mps.empty_cache()
        probe()

    selections = {}
    for factor in ("content", "binding", "order"):
        if any(not summaries[seed][factor] for seed in ("0", "1")):
            selections[factor] = {"selection": None, "reason": "empty_orthogonal_basis"}
        else:
            selections[factor] = joint_rank_decision(
                {seed: summaries[seed][factor] for seed in ("0", "1")}, factor,
                {seed: seed_payloads[seed]["fitted"].orthogonal.forward_eigenvalues[factor]
                 for seed in ("0", "1")})
    alignment = cross_seed_geometry(seed_payloads)
    alignment_path = output_dir / "subspace-cross-seed.pt"
    artifacts["cross_seed"] = tensor_artifact(
        alignment_path, {"format": "TEACH-0013-cross-seed-geometry-v2",
                         "alignment": alignment}, stage_c)
    elapsed = time.monotonic() - start
    output_bytes, result_bytes = stage_c.tree_bytes(output_dir), stage_c.tree_bytes(result_dir)
    if output_bytes > MAX_OUTPUT_BYTES or result_bytes > MAX_RESULT_BYTES:
        raise RuntimeError("TEACH-0013 Stage-D artifact ceiling exceeded")
    report = {
        "experiment": "TEACH-0013", "mode": "subspace_discovery",
        "status": "stage_d_discovery_complete", "mediator": asdict(spec),
        "confirmation_decision_sha256": stage_c.sha_file(
            result_dir / "confirmation-decision.json"),
        "confirmation_audit_sha256": stage_c.sha_file(
            result_dir / "confirmation-audit.json"),
        "confirmation_audit_status": confirmation_audit["audit"],
        "summaries": summaries, "rank_selections": selections,
        "artifacts": artifacts, "elapsed_seconds": elapsed,
        "cumulative_stage_c_e_seconds": confirmation["elapsed_seconds"] + elapsed,
        "cumulative_campaign_seconds": confirmation["cumulative_campaign_seconds"] + elapsed,
        "materialized_activation_bytes": traffic["bytes"],
        "cumulative_materialized_bytes": (
            confirmation["cumulative_materialized_bytes"] + traffic["bytes"]),
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "ignored_output_bytes": output_bytes, "tracked_result_bytes": result_bytes,
        "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path,
                        default=ROOT / "results/TEACH-0013")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    print(json.dumps(discovery(args.result_dir, args.output_dir, args.device),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
