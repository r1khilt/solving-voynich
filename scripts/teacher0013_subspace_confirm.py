#!/usr/bin/env python3
"""Sealed TEACH-0013 Stage-D causal-subspace confirmation campaign."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time

import torch

from voynich.workspace.teacher13_confirm import (
    MediatorSpec,
    cross_task_mediator_logits,
    diagnostic_rows,
    summarize_confirmation_rows,
)
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_geometry import (
    deranged_blocked_bases,
    haar_random_bases,
    orthogonal_factor_geometry,
    orthonormal_union,
)
from voynich.workspace.teacher13_intervene import (
    load_raw_checkpoint,
    materialized_tensor_counter,
    raw_forward,
)
from voynich.workspace.teacher13_subspace import (
    equal_norm_component_corruption_states,
    factor_equal_energy_rows,
    factor_transfer_rows,
    mean_ablation_states,
    mediator_endpoint_logits,
    mediator_state_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_COUNT = 32
CONTROL_SEEDS = {"content": 73411, "binding": 73421, "order": 73431}
CONTROL_SURFACES = {"content": ("marked",), "binding": ("marked",),
                    "order": ("f0_marked", "f1_marked")}
MAX_STAGE_C_E_SECONDS = 5400.0
MAX_CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3


def _load_stage_c():
    spec = importlib.util.spec_from_file_location(
        "teacher0013_confirm_dependency", ROOT / "scripts/teacher0013_confirm.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen Stage-C dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShardedLogitCapture:
    """Write each exact-logit record immediately to keep peak host memory bounded."""

    def __init__(self, directory: Path, stage_c):
        if directory.exists():
            raise FileExistsError("No automatic Stage-D exact-logit overwrite")
        directory.mkdir(parents=True)
        self.directory, self.stage_c, self.shards = directory, stage_c, []

    def __call__(self, condition, direction, metadata, logits, clean_logits):
        item_ids = tuple(row.get(
            "item_id", f"{condition}:{direction}:{row['logical_group_id']}:{row['recipient']}")
            for row in metadata)
        record = {"condition": condition, "direction": direction,
                  "logical_group_ids": tuple(row["logical_group_id"] for row in metadata),
                  "recipients": tuple(row["recipient"] for row in metadata),
                  "targets": tuple(row["target"] for row in metadata),
                  "item_ids": item_ids, "edited_symbol_logits": logits.contiguous(),
                  "clean_symbol_logits": clean_logits.contiguous()}
        path = self.directory / f"{len(self.shards):05d}.pt"
        torch.save({"format": "TEACH-0013-symbol-logit-shard-v1", "record": record}, path)
        self.shards.append({"path": str(path.relative_to(ROOT)),
                            "sha256": self.stage_c.sha_file(path),
                            "bytes": path.stat().st_size, "rows": len(item_ids)})

    def metadata(self):
        return {"format": "TEACH-0013-sharded-symbol-logits-v1",
                "shards": self.shards, "records": len(self.shards),
                "rows": sum(shard["rows"] for shard in self.shards)}


def prerequisite(result_dir: Path, stage_c):
    discovery_path = result_dir / "subspace-discovery.json"
    audit_path = result_dir / "subspace-discovery-audit.json"
    discovery, audit = (json.loads(path.read_text()) for path in (discovery_path, audit_path))
    if discovery.get("status") != "stage_d_discovery_complete" \
            or audit.get("audit") != "pass" or audit.get("scope") != "stage_d_discovery" \
            or audit.get("subspace_discovery_sha256") != stage_c.sha_file(discovery_path) \
            or audit.get("rank_selections") != discovery.get("rank_selections"):
        raise RuntimeError("Matching independently audited Stage-D discovery required")
    if discovery.get("rank_selections", {}).get("content", {}).get("selection") is None:
        raise RuntimeError("Stage-D confirmation requires a discovery-selected content rank")
    return discovery, audit, MediatorSpec(**discovery["mediator"])


def load_geometry(discovery, seed):
    metadata = discovery["artifacts"][seed]["geometry"]
    path = ROOT / metadata["path"]
    if not path.is_file() or path.stat().st_size != metadata["bytes"]:
        raise RuntimeError("Stage-D geometry artifact mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def selected_bases(discovery, geometry, seed):
    result = {}
    for factor in ("content", "binding", "order"):
        rank = discovery["rank_selections"][factor]["selection"]
        basis = geometry["orthogonal"]["forward"][factor]
        if rank is not None:
            if rank > basis.shape[1]:
                raise RuntimeError("Selected rank exceeds frozen factor basis")
            result[factor] = basis[:, :rank]
    return result


def deranged_controls(geometry, factor, rank):
    panel = geometry["panels"][factor]
    nulls = deranged_blocked_bases(
        panel["states"], panel["factor"], panel["nuisance"], panel["blocks"],
        count=CONTROL_COUNT, seed=CONTROL_SEEDS[factor])
    controls = []
    for null in nulls:
        raw = {name: geometry["geometry"][name]["basis"]
               for name in ("content", "binding", "order")}
        spectra = {name: geometry["geometry"][name]["eigenvalues"]
                   for name in ("content", "binding", "order")}
        raw[factor], spectra[factor] = null.basis, null.eigenvalues
        fitted = orthogonal_factor_geometry(
            raw["content"], raw["binding"], raw["order"], eigenvalues=spectra)
        basis = fitted.forward[factor]
        if basis.shape[1] < rank:
            raise RuntimeError("Deranged control rank fell below selected rank")
        controls.append({"basis": basis[:, :rank], "donor_blocks": null.donor_blocks})
    return tuple(controls)


def necessity_rows(net, groups, spec, basis, mean, capture, *, device, seed):
    episodes = tuple(episode_from_record(row) for group in groups
                     for name in ("base", "donor") for row in group[name])
    metadata = tuple({"logical_group_id": group["group_id"], "recipient": recipient,
                      "target": episode.answer, "fixed_donor_answer": episode.answer,
                      "base_render_id": episode.render_id,
                      "donor_render_id": episode.render_id,
                      "item_id": f"content:necessity:{assignment}:{group['group_id']}:{recipient}"}
                     for group in groups for assignment, name in (("f0", "base"),
                                                                  ("f1", "donor"))
                     for recipient, episode in enumerate(
                         tuple(episode_from_record(row) for row in group[name])))
    clean = raw_forward(net, episodes, device=device).logits
    pairs = mediator_state_pairs(net, episodes, episodes, spec, device=device)
    states = {
        "content_mean_ablation": mean_ablation_states(pairs.base, mean, basis),
        "content_equal_norm_corruption": equal_norm_component_corruption_states(
            pairs.base, mean, basis, seed=seed),
        "content_native_restoration": pairs.base,
    }
    rows = []
    for condition, replacement in states.items():
        logits = mediator_endpoint_logits(net, episodes, spec, replacement, device=device)
        tagged = tuple({**row, "item_id": f"{row['item_id']}:{condition}"}
                       for row in metadata)
        rows.extend(diagnostic_rows(
            logits, clean, episodes, metadata=tagged, condition=condition,
            direction="necessity", logit_sink=capture))
    return rows


def content_specificity_rows(net, groups, spec, basis, capture, *, device):
    rows = []
    for task, base_name, donor_name in (("first_hop", "first_hop_base", "first_hop_donor"),
                                        ("direct", "direct_base", "direct_donor")):
        bases = tuple(episode_from_record(group[base_name]) for group in groups)
        donors = tuple(episode_from_record(group[donor_name]) for group in groups)
        references = tuple(episode_from_record(group["donor"][0]) for group in groups)
        metadata = tuple({"logical_group_id": group["group_id"], "recipient": 0,
                          "target": donor.answer, "fixed_donor_answer": donor.answer,
                          "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                          "item_id": f"content:{task}:{group['group_id']}:0"}
                         for group, base, donor in zip(groups, bases, donors, strict=True))
        clean = raw_forward(net, bases, device=device).logits
        edited = cross_task_mediator_logits(
            net, bases, donors, references, spec, basis=basis, device=device)
        rows.extend(diagnostic_rows(
            edited, clean, bases, metadata=metadata, condition=f"content_{task}",
            direction="specificity", logit_sink=capture))
    bases = tuple(episode_from_record(group["copy_control"]) for group in groups)
    donors = tuple(episode_from_record(group["donor"][0]) for group in groups)
    metadata = tuple({"logical_group_id": group["group_id"], "recipient": 0,
                      "target": base.answer, "fixed_donor_answer": donor.answer,
                      "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                      "item_id": f"content:copy:{group['group_id']}:0"}
                     for group, base, donor in zip(groups, bases, donors, strict=True))
    clean = raw_forward(net, bases, device=device).logits
    edited = cross_task_mediator_logits(net, bases, donors, donors, spec, basis=basis, device=device)
    rows.extend(diagnostic_rows(
        edited, clean, bases, metadata=metadata, condition="content_copy",
        direction="specificity", logit_sink=capture))
    return rows


def score_rows(rows, factor):
    selected = {}
    prefix = f"{factor}_selected_"
    for row in rows:
        if row["condition"].startswith(prefix):
            selected.setdefault((row["condition"], row["direction"]), []).append(row)
    primary = {f"{condition}:{direction}": summarize_confirmation_rows(cell)
               for (condition, direction), cell in selected.items()}
    controls = {}
    for family in ("haar", "deranged", "equal_energy"):
        cells = {}
        for row in rows:
            if row["condition"].startswith(f"{factor}_{family}_"):
                cells.setdefault((row["condition"], row["direction"]), []).append(row)
        controls[family] = {f"{condition}:{direction}": summarize_confirmation_rows(cell)
                            for (condition, direction), cell in cells.items()}
    return {"primary": primary, "controls": controls}


def run_seed(net, groups, spec, discovery, geometry, seed, capture, *, device,
             probe=lambda: None):
    bases = selected_bases(discovery, geometry, seed)
    rows, controls = [], {}
    for factor, basis in bases.items():
        rows.extend(factor_transfer_rows(
            net, groups, spec, basis, factor_name=factor,
            condition_prefix=f"{factor}_selected", episode_loader=episode_from_record,
            device=device, logit_sink=capture))
        probe()
        rows.extend(factor_transfer_rows(
            net, groups, spec, basis, factor_name=factor,
            condition_prefix=f"{factor}_complement", component="complement",
            episode_loader=episode_from_record, device=device, logit_sink=capture))
        rank = basis.shape[1]
        haar = haar_random_bases(basis.shape[0], rank, CONTROL_COUNT,
                                 seed=CONTROL_SEEDS[factor])
        deranged = deranged_controls(geometry, factor, rank)
        controls[factor] = {"haar": haar, "deranged": deranged}
        for index, control in enumerate(haar):
            rows.extend(factor_transfer_rows(
                net, groups, spec, control, factor_name=factor,
                condition_prefix=f"{factor}_haar_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                device=device, logit_sink=capture))
            probe()
        for index, control in enumerate(deranged):
            rows.extend(factor_transfer_rows(
                net, groups, spec, control["basis"], factor_name=factor,
                condition_prefix=f"{factor}_deranged_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                device=device, logit_sink=capture))
            probe()
        for index in range(CONTROL_COUNT):
            rows.extend(factor_equal_energy_rows(
                net, groups, spec, basis, factor_name=factor,
                condition_prefix=f"{factor}_equal_energy_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                seed=CONTROL_SEEDS[factor] + 1000 + 20 * index,
                device=device, logit_sink=capture))
            probe()
    if "content" in bases and "binding" in bases:
        union = orthonormal_union(bases["content"], bases["binding"])
        for factor in ("content", "binding"):
            rows.extend(factor_transfer_rows(
                net, groups, spec, union, factor_name=factor,
                condition_prefix=f"content_binding_on_{factor}",
                episode_loader=episode_from_record, device=device, logit_sink=capture))
    if set(bases) == {"content", "binding", "order"}:
        union = orthonormal_union(bases["content"], bases["binding"], bases["order"])
        for factor in ("content", "binding", "order"):
            rows.extend(factor_transfer_rows(
                net, groups, spec, union, factor_name=factor,
                condition_prefix=f"all_factors_on_{factor}",
                episode_loader=episode_from_record, device=device, logit_sink=capture))
    content = bases["content"]
    rows.extend(necessity_rows(
        net, groups, spec, content, geometry["geometry"]["content"]["mean"], capture,
        device=device, seed=CONTROL_SEEDS["content"] + 5000))
    rows.extend(content_specificity_rows(net, groups, spec, content, capture, device=device))
    controls["selected"] = bases
    return rows, controls


def confirmation(result_dir: Path, output_dir: Path, device: str):
    report_path = result_dir / "subspace-confirmation.json"
    if report_path.exists():
        raise FileExistsError("No automatic Stage-D confirmation overwrite")
    stage_c = _load_stage_c()
    manifest = stage_c.load_manifest(result_dir)
    provenance = stage_c.verify_frozen_source(manifest)
    discovery, discovery_audit, spec = prerequisite(result_dir, stage_c)
    suite = stage_c.load_confirmation_suite(output_dir, manifest)
    groups = suite["splits"]["confirmation"]
    start, traffic = time.monotonic(), {"bytes": 0}
    peak = {"current": 0, "driver": 0}
    artifacts, scores = {}, {}

    def count(value):
        traffic["bytes"] += int(value)

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        elapsed = time.monotonic() - start
        if discovery["cumulative_stage_c_e_seconds"] + elapsed > MAX_STAGE_C_E_SECONDS \
                or discovery["cumulative_campaign_seconds"] + elapsed > MAX_CAMPAIGN_SECONDS \
                or peak["current"] > MAX_CURRENT_BYTES \
                or discovery["cumulative_materialized_bytes"] + traffic["bytes"] \
                > MAX_TRAFFIC_BYTES:
            raise RuntimeError("Stage-D confirmation resource ceiling exceeded")

    for seed in ("0", "1"):
        net, metadata = load_raw_checkpoint(stage_c.checkpoint_path(manifest, seed), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise RuntimeError("Stage-D confirmation requires final raw-deep checkpoints")
        geometry = load_geometry(discovery, seed)
        capture = ShardedLogitCapture(
            output_dir / f"subspace-confirmation-logits-rep{seed}", stage_c)
        with materialized_tensor_counter(count):
            rows, controls = run_seed(
                net, groups, spec, discovery, geometry, seed, capture,
                device=device, probe=probe)
        compact = result_dir / f"subspace-confirmation-rows-rep{seed}.jsonl.gz"
        controls_path = output_dir / f"subspace-confirmation-controls-rep{seed}.pt"
        if compact.exists() or controls_path.exists():
            raise FileExistsError("No automatic Stage-D compact/control overwrite")
        torch.save({"format": "TEACH-0013-subspace-controls-v1",
                    "controls": controls}, controls_path)
        scores[seed] = {factor: score_rows(rows, factor)
                        for factor in ("content", "binding", "order")
                        if discovery["rank_selections"][factor]["selection"] is not None}
        artifacts[seed] = {
            "rows": {"path": str(compact.relative_to(ROOT)),
                     "sha256": stage_c.write_rows(compact, rows),
                     "bytes": compact.stat().st_size, "rows": len(rows)},
            "exact_symbol_logits": capture.metadata(),
            "controls": {"path": str(controls_path.relative_to(ROOT)),
                         "sha256": stage_c.sha_file(controls_path),
                         "bytes": controls_path.stat().st_size},
        }
        if device == "mps":
            torch.mps.synchronize()
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
            del net
            torch.mps.empty_cache()
        probe()
    elapsed = time.monotonic() - start
    cumulative_time = discovery["cumulative_stage_c_e_seconds"] + elapsed
    cumulative_campaign = discovery["cumulative_campaign_seconds"] + elapsed
    cumulative_traffic = discovery["cumulative_materialized_bytes"] + traffic["bytes"]
    if cumulative_time > MAX_STAGE_C_E_SECONDS or cumulative_campaign > MAX_CAMPAIGN_SECONDS \
            or peak["current"] > MAX_CURRENT_BYTES or cumulative_traffic > MAX_TRAFFIC_BYTES:
        raise RuntimeError("Stage-D confirmation resource ceiling exceeded")
    if stage_c.tree_bytes(output_dir) > MAX_OUTPUT_BYTES \
            or stage_c.tree_bytes(result_dir) > MAX_RESULT_BYTES:
        raise RuntimeError("Stage-D confirmation artifact ceiling exceeded")
    report = {
        "experiment": "TEACH-0013", "mode": "subspace_confirmation",
        "status": "stage_d_confirmation_core_complete", "mediator": asdict(spec),
        "subspace_discovery_sha256": stage_c.sha_file(
            result_dir / "subspace-discovery.json"),
        "subspace_discovery_audit_sha256": stage_c.sha_file(
            result_dir / "subspace-discovery-audit.json"),
        "subspace_discovery_audit": discovery_audit["audit"],
        "rank_selections": discovery["rank_selections"], "scores": scores,
        "artifacts": artifacts, "elapsed_seconds": elapsed,
        "materialized_activation_bytes": traffic["bytes"],
        "cumulative_stage_c_e_seconds": cumulative_time,
        "cumulative_campaign_seconds": cumulative_campaign,
        "cumulative_materialized_bytes": cumulative_traffic,
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "ignored_output_bytes": stage_c.tree_bytes(output_dir),
        "tracked_result_bytes": stage_c.tree_bytes(result_dir),
        "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/TEACH-0013")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    print(json.dumps(confirmation(args.result_dir, args.output_dir, args.device),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
