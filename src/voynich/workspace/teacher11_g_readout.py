"""Frozen TEACH-0011 downstream QKV/G-edge decomposition; no run on import."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from .teacher2_train import write_json
from .teacher4_models import DenseRows
from .teacher5_intervene import build_groups, sha_file, stable_json
from .teacher7_dense_mechanism import _answers, _ids, continue_dense, score
from .teacher10_answer_readout import (
    cyclic_recipient,
    layer_parts_with_weights,
    prepare as prepare_answer_run,
    run_from_attention,
)


SOURCE_PATHS = (
    "docs/experiments/TEACH-0011.md",
    "src/voynich/workspace/teacher11_g_readout.py",
    "src/voynich/workspace/teacher10_answer_readout.py",
    "src/voynich/workspace/teacher7_dense_mechanism.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
)
QKV_CELLS = ("BBB", "DBB", "BDB", "BBD", "DDB", "DBD", "BDD", "DDD")
SLOT_NAMES = ("f0", "f1", "g0", "g1", "marker", "query")


@dataclass(frozen=True)
class Config:
    suite_seed: int = 71111
    random_seed: int = 71211
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0011 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0011 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def qkv_parts(layer, state, query_slot=5):
    normalized = layer.norm1(state)
    width, heads = normalized.shape[-1], layer.self_attn.num_heads
    head_width = width // heads
    projected = F.linear(normalized, layer.self_attn.in_proj_weight,
                         layer.self_attn.in_proj_bias)
    query, key, value = projected.chunk(3, dim=-1)
    batch, length, _ = query.shape
    query = query.view(batch, length, heads, head_width).transpose(1, 2)
    key = key.view(batch, length, heads, head_width).transpose(1, 2)
    value = value.view(batch, length, heads, head_width).transpose(1, 2)
    head_values, weights = torch._scaled_dot_product_attention_math(
        query, key, value, None, 0.0, False, None)
    attention = weights[:, :, query_slot]
    contributions = attention[..., None] * value
    head_query = head_values[:, :, query_slot]
    reference = layer_parts_with_weights(layer, state)["heads"][:, :, query_slot]
    return {"query": query[:, :, query_slot], "query_all": query,
            "key": key, "value": value, "attention": attention,
            "contributions": contributions, "head_query": head_query,
            "head_reconstruction_error": float((head_query - reference).abs().max())}


def hybrid_query(base, edited, cell):
    query = base["query_all"].clone()
    if cell[0] == "D":
        query[:, :, 5] = edited["query"]
    key = edited["key"] if cell[1] == "D" else base["key"]
    value = edited["value"] if cell[2] == "D" else base["value"]
    head_values, weights = torch._scaled_dot_product_attention_math(
        query, key, value, None, 0.0, False, None)
    attention = weights[:, :, 5]
    contributions = attention[..., None] * value
    return head_values[:, :, 5], attention, contributions


def project_all(layer, head_delta):
    return F.linear(head_delta.reshape(head_delta.shape[0], -1),
                    layer.self_attn.out_proj.weight, bias=None)


@torch.no_grad()
def prepare(net, groups, config):
    prepared = prepare_answer_run(net, groups, config)
    layer = net.transformer.layers[2]
    prepared["base_qkv"] = qkv_parts(layer, prepared["base_states"][2])
    prepared["edited_qkv"] = qkv_parts(layer, prepared["edited_cut2"])
    return prepared


def contribution_delta(prepared, slots):
    difference = (prepared["edited_qkv"]["contributions"]
                  - prepared["base_qkv"]["contributions"])
    return difference[:, :, slots].sum(dim=2)


def select_slots(screens):
    rows = []
    for mask in range(1, 64):
        key = str(mask)
        group_min = min(row[key]["donor_groups"]["accuracy"] for row in screens.values())
        item_min = min(row[key]["donor_items"]["accuracy"] for row in screens.values())
        rows.append((mask, mask.bit_count(), group_min, item_min))
    qualified = [row for row in rows if row[2] >= .60 and row[3] >= .75]
    if qualified:
        chosen = sorted(qualified, key=lambda row: (row[1], -row[2], -row[3], row[0]))[0]
        flag = True
    else:
        chosen = sorted(rows, key=lambda row: (-row[2], -row[3], row[1], row[0]))[0]
        flag = False
    return {"mask": chosen[0], "slots": [slot for slot in range(6)
                                            if chosen[0] & (1 << slot)],
            "slot_names": [name for slot, name in enumerate(SLOT_NAMES)
                           if chosen[0] & (1 << slot)],
            "size": chosen[1], "discovery_qualified": flag,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3]}


@torch.no_grad()
def source_screen(net, groups, config):
    prepared = prepare(net, groups, config)
    donor = _answers(groups, "donor_answers")
    base = _answers(groups, "base_answers")
    layer = net.transformer.layers[2]
    screen = {}
    for mask in range(1, 64):
        slots = [slot for slot in range(6) if mask & (1 << slot)]
        delta = project_all(layer, contribution_delta(prepared, slots))
        pred = run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + delta)
        screen[str(mask)] = score(pred, donor, base, len(groups), config.g_tables)
    return screen


def _g_rows(ids, groups, field):
    keys = torch.tensor([group[field] for group in groups]).repeat_interleave(3)
    return torch.where(ids[:, 7] == keys, 2, 3)


def _rowwise_delta(prepared, rows):
    difference = (prepared["edited_qkv"]["contributions"]
                  - prepared["base_qkv"]["contributions"])
    batch = torch.arange(len(rows))
    return difference[batch, :, rows]


def _aligned_attention(qkv, ids, groups):
    base_rows = _g_rows(ids, groups, "base_key")
    donor_rows = _g_rows(ids, groups, "donor_key")
    batch = torch.arange(len(ids))
    result = {}
    for head in range(4):
        weights = qkv["attention"][:, head]
        result[str(head)] = {
            "base_key_g": float(weights[batch, base_rows].mean()),
            "donor_key_g": float(weights[batch, donor_rows].mean()),
            "f_rows": float(weights[:, :2].sum(-1).mean()),
            "marker_query": float(weights[:, 4:6].sum(-1).mean()),
        }
    return result


def _sufficient(scores, name):
    row = scores[name]
    return (row["donor_items"]["accuracy"] >= .75
            and row["donor_groups"]["accuracy"] >= .60
            and row["cross_g_non_injection"]["accuracy"] >= .90)


@torch.no_grad()
def confirm_model(net, confirmation, selection, rep, config):
    prepared = prepare(net, confirmation, config)
    ids = prepared["base_ids"]
    donor_ids = _ids(confirmation, "donor")
    donor_targets = _answers(confirmation, "donor_answers")
    base_targets = _answers(confirmation, "base_answers")
    layer = net.transformer.layers[2]
    conditions = {}
    hybrid_outputs = {}
    for cell in QKV_CELLS:
        hybrid, _, _ = hybrid_query(prepared["base_qkv"], prepared["edited_qkv"], cell)
        delta = project_all(layer, hybrid - prepared["base_qkv"]["head_query"])
        conditions[f"qkv_{cell.lower()}"] = run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + delta)
        hybrid_outputs[cell] = hybrid

    selected_projected = project_all(layer, contribution_delta(prepared, selection["slots"]))
    conditions["selected_sources"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + selected_projected)
    for name, slots in {"all_sources": range(6), "both_g": (2, 3),
                        "non_g": (0, 1, 4, 5)}.items():
        delta = project_all(layer, contribution_delta(prepared, list(slots)))
        conditions[name] = run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + delta)
    base_rows = _g_rows(ids, confirmation, "base_key")
    donor_rows = _g_rows(ids, confirmation, "donor_key")
    base_g = project_all(layer, _rowwise_delta(prepared, base_rows))
    donor_g = project_all(layer, _rowwise_delta(prepared, donor_rows))
    conditions["base_key_g"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + base_g)
    conditions["donor_key_g"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + donor_g)

    generator = torch.Generator().manual_seed(config.random_seed + rep)
    random_delta = torch.randn(selected_projected.shape, generator=generator)
    random_delta /= random_delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_delta *= selected_projected.norm(dim=-1, keepdim=True)
    conditions["random"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + random_delta)
    mismatch = cyclic_recipient(selected_projected, len(confirmation), config.g_tables)
    conditions["cyclic_recipient"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + mismatch)
    conditions["clean_base"] = net(ids)[0].argmax(-1).cpu()
    conditions["clean_donor"] = net(donor_ids)[0].argmax(-1).cpu()
    conditions["upstream_key"] = continue_dense(
        net, ids, prepared["edited_cut2"], 2).argmax(-1).cpu()
    f_positive = prepared["base_states"][0].clone()
    donor_f = prepared["donor_states"][0][:, :2].repeat_interleave(config.g_tables, 0)
    f_positive[:, :2] = donor_f
    conditions["f_positive"] = continue_dense(net, ids, f_positive, 0).argmax(-1).cpu()

    scores = {name: score(pred, donor_targets, base_targets, len(confirmation),
                          config.g_tables) for name, pred in conditions.items()}
    base_sum = float((prepared["base_qkv"]["contributions"].sum(2)
                      - prepared["base_qkv"]["head_query"]).abs().max())
    edited_sum = float((prepared["edited_qkv"]["contributions"].sum(2)
                        - prepared["edited_qkv"]["head_query"]).abs().max())
    full_source = contribution_delta(prepared, list(range(6)))
    full_hybrid = hybrid_outputs["DDD"] - prepared["base_qkv"]["head_query"]
    numerical = {
        "base_head_reconstruction": prepared["base_qkv"]["head_reconstruction_error"],
        "edited_head_reconstruction": prepared["edited_qkv"]["head_reconstruction_error"],
        "base_source_sum": base_sum, "edited_source_sum": edited_sum,
        "full_source_delta": float((project_all(layer, full_source)
                                     - project_all(layer, full_hybrid)).abs().max()),
    }
    attention = {"base": _aligned_attention(prepared["base_qkv"], ids, confirmation),
                 "edited": _aligned_attention(prepared["edited_qkv"], ids, confirmation)}
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             **{f"{name}_prediction": int(pred[group * config.g_tables + g])
                for name, pred in conditions.items()}}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    return {"scores": scores, "numerical_errors": numerical,
            "attention_mass": attention}, rows


def decide(selection, metrics):
    clauses = {}
    for rep, row in metrics.items():
        scores = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "upstream_key_at_least_0.95": scores["upstream_key"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "ddd_sufficient": _sufficient(scores, "qkv_ddd"),
            "all_sources_sufficient": _sufficient(scores, "all_sources"),
            "selected_sources_sufficient": _sufficient(scores, "selected_sources"),
            "random_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40),
            "cyclic_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["cyclic_recipient"]["donor_items"]["accuracy"] >= .40),
            "numerical_errors_below_1e-6": max(row["numerical_errors"].values()) < 1e-6,
        }
    supported = selection["discovery_qualified"] and all(
        all(row.values()) for row in clauses.values())
    sufficient_cells = [cell for cell in QKV_CELLS if all(
        _sufficient(row["scores"], f"qkv_{cell.lower()}") for row in metrics.values())]
    singles = {"Q": "DBB", "K": "BDB", "V": "BBD"}
    single_pass = [name for name, cell in singles.items() if cell in sufficient_cells]
    if single_pass == ["Q"]:
        label = "q_only_sufficient"
    elif single_pass == ["K"]:
        label = "k_only_sufficient"
    elif single_pass == ["V"]:
        label = "v_only_sufficient"
    elif single_pass:
        label = "multiple_single_streams_sufficient"
    elif "DDD" in sufficient_cells:
        label = "interaction_required"
    else:
        label = "factorial_unresolved"
    return {"g_edge": "supported" if supported else "not_supported",
            "qkv_label": label, "sufficient_qkv_cells": sufficient_cells,
            "clauses_by_seed": clauses}


def run(config, checkpoint_dir, result_dir):
    config.validate()
    provenance = source_provenance(config)
    groups = build_groups(config.suite_seed,
                          config.discovery_groups + config.confirmation_groups)
    discovery = groups[:config.discovery_groups]
    confirmation = groups[config.discovery_groups:]
    suite_sha = hashlib.sha256(stable_json(groups).encode()).hexdigest()
    report4 = json.loads((result_dir.parent / "TEACH-0004" / "report.json").read_text())
    start = time.monotonic()
    models, screens, artifacts = {}, {}, {}
    for rep in range(2):
        checkpoint = checkpoint_dir / f"rep{rep}-dense_row.pt"
        expected = report4["arms"][str(rep)]["dense_row"]["checkpoint_sha256"]
        if sha_file(checkpoint) != expected:
            raise RuntimeError("TEACH-0011 checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net = DenseRows()
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        models[str(rep)] = net
        screens[str(rep)] = source_screen(net, discovery, config)
        artifacts[str(rep)] = {"checkpoint_sha256": expected}
    selection = select_slots(screens)
    metrics = {}
    for rep in range(2):
        row_metrics, rows = confirm_model(models[str(rep)], confirmation, selection, rep, config)
        metrics[str(rep)] = row_metrics
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        artifacts[str(rep)].update({"rows_sha256": sha_file(path), "rows": len(rows)})
    decision = decide(selection, metrics)
    elapsed = time.monotonic() - start
    if elapsed > config.max_seconds:
        raise RuntimeError("TEACH-0011 time cap exceeded")
    report = {"experiment": "TEACH-0011", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "selected_heads": [0, 1, 2, 3],
              "source_selection": selection, "discovery_screen": screens,
              "metrics": metrics, "artifacts": artifacts, "decision": decision,
              "elapsed_seconds": elapsed, "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0011"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"],
                      "source_selection": report["source_selection"],
                      "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
