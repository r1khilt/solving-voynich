"""Frozen TEACH-0009 QKV/source-edge decomposition; no run on import."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from .teacher2_train import write_json
from .teacher4_models import DenseRows
from .teacher5_intervene import build_groups, sha_file, stable_json
from .teacher7_dense_mechanism import _answers, _ids, continue_dense, manual_dense, score
from .teacher8_components import layer_parts


SOURCE_PATHS = (
    "docs/experiments/TEACH-0009.md",
    "src/voynich/workspace/teacher9_qkv.py",
    "src/voynich/workspace/teacher8_components.py",
    "src/voynich/workspace/teacher7_dense_mechanism.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
)
SELECTED_HEADS = (1, 3)
QKV_CELLS = ("BBB", "DBB", "BDB", "BBD", "DDB", "DBD", "BDD", "DDD")
SLOT_NAMES = ("f0", "f1", "g0", "g1", "marker", "query")


@dataclass(frozen=True)
class Config:
    suite_seed: int = 69111
    random_seed: int = 69211
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0009 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0009 sources before analysis: {status}")
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
    scores = (query[:, :, query_slot].unsqueeze(-2) @ key.transpose(-2, -1)).squeeze(-2)
    scores = scores / math.sqrt(head_width)
    attention = scores.softmax(dim=-1)
    contributions = attention[..., None] * value
    head_query = contributions.sum(dim=2)
    reference = layer_parts(layer, state)["heads"][:, :, query_slot]
    return {"query": query[:, :, query_slot], "key": key, "value": value,
            "attention": attention, "contributions": contributions,
            "head_query": head_query,
            "head_reconstruction_error": float((head_query - reference).abs().max())}


def hybrid_query(base, donor, cell):
    query = donor["query"] if cell[0] == "D" else base["query"]
    key = donor["key"] if cell[1] == "D" else base["key"]
    value = donor["value"] if cell[2] == "D" else base["value"]
    scores = (query.unsqueeze(-2) @ key.transpose(-2, -1)).squeeze(-2)
    scores = scores / math.sqrt(query.shape[-1])
    attention = scores.softmax(dim=-1)
    contributions = attention[..., None] * value
    return contributions.sum(dim=2), attention, contributions


def project_selected(layer, head_delta, selected=SELECTED_HEADS):
    used = torch.zeros_like(head_delta)
    used[:, selected] = head_delta[:, selected]
    return F.linear(used.reshape(used.shape[0], -1),
                    layer.self_attn.out_proj.weight, bias=None)


def predict_delta(net, ids, cut1, base_layer, projected_delta):
    attention = base_layer["attention"][:, 5] + projected_delta
    full_attention = base_layer["attention"].clone()
    full_attention[:, 5] = attention
    post_attention = cut1 + full_attention
    layer = net.transformer.layers[1]
    post_mlp = post_attention + layer._ff_block(layer.norm2(post_attention))
    return continue_dense(net, ids, post_mlp, 2).argmax(-1).cpu()


@torch.no_grad()
def prepare(net, groups, config):
    base_ids = _ids(groups, "base")
    donor_ids = _ids(groups, "donor", first_only=True)
    _, base_states = manual_dense(net, base_ids)
    _, donor_states = manual_dense(net, donor_ids)
    base_layer = layer_parts(net.transformer.layers[1], base_states[1])
    donor_layer = layer_parts(net.transformer.layers[1], donor_states[1])
    base_qkv = qkv_parts(net.transformer.layers[1], base_states[1])
    donor_single = qkv_parts(net.transformer.layers[1], donor_states[1])
    donor_qkv = {key: (value.repeat_interleave(config.g_tables, dim=0)
                       if isinstance(value, torch.Tensor) else value)
                 for key, value in donor_single.items()}
    return {"base_ids": base_ids, "base_states": base_states,
            "donor_states": donor_states, "base_layer": base_layer,
            "donor_layer": donor_layer, "base_qkv": base_qkv,
            "donor_qkv": donor_qkv, "donor_single_qkv": donor_single}


def contribution_delta(prepared, slots):
    difference = (prepared["donor_qkv"]["contributions"]
                  - prepared["base_qkv"]["contributions"])
    return difference[:, :, slots].sum(dim=2)


@torch.no_grad()
def source_screen(net, groups, config):
    prepared = prepare(net, groups, config)
    donor = _answers(groups, "donor_answers")
    base = _answers(groups, "base_answers")
    layer = net.transformer.layers[1]
    screen = {}
    for mask in range(1, 64):
        slots = [slot for slot in range(6) if mask & (1 << slot)]
        projected = project_selected(layer, contribution_delta(prepared, slots))
        predictions = predict_delta(net, prepared["base_ids"], prepared["base_states"][1],
                                    prepared["base_layer"], projected)
        screen[str(mask)] = score(predictions, donor, base, len(groups), config.g_tables)
    return screen


def select_slots(screens):
    rows = []
    for mask in range(1, 64):
        key = str(mask)
        group_min = min(row[key]["donor_groups"]["accuracy"] for row in screens.values())
        item_min = min(row[key]["donor_items"]["accuracy"] for row in screens.values())
        cross_min = min(row[key]["cross_g_non_injection"]["accuracy"]
                        for row in screens.values())
        rows.append((mask, mask.bit_count(), group_min, item_min, cross_min))
    qualified = [row for row in rows if row[2] >= .60 and row[4] >= .90]
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
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
            "minimum_cross_g_non_injection": chosen[4]}


def _aligned_attention(qkv, ids):
    relevant = torch.where(ids[:, 2] == ids[:, 12], 0, 1)
    batch = torch.arange(len(ids))
    result = {}
    for head in SELECTED_HEADS:
        weights = qkv["attention"][:, head]
        queried = weights[batch, relevant]
        other = weights[batch, 1 - relevant]
        result[str(head)] = {
            "queried_f": float(queried.mean()), "other_f": float(other.mean()),
            "g_rows": float(weights[:, 2:4].sum(-1).mean()),
            "marker": float(weights[:, 4].mean()), "query": float(weights[:, 5].mean()),
        }
    return result


def _rowwise_f_delta(prepared, ids, other=False):
    difference = (prepared["donor_qkv"]["contributions"]
                  - prepared["base_qkv"]["contributions"])
    relevant = torch.where(ids[:, 2] == ids[:, 12], 0, 1)
    if other:
        relevant = 1 - relevant
    batch = torch.arange(len(ids))
    return difference[batch, :, relevant]


@torch.no_grad()
def confirm_model(net, discovery, confirmation, selection, rep, config):
    prepared = prepare(net, confirmation, config)
    ids = prepared["base_ids"]
    layer = net.transformer.layers[1]
    donor_targets = _answers(confirmation, "donor_answers")
    base_targets = _answers(confirmation, "base_answers")
    conditions = {}
    qkv_predictions = {}
    qkv_outputs = {}
    for cell in QKV_CELLS:
        hybrid, _, _ = hybrid_query(prepared["base_qkv"], prepared["donor_qkv"], cell)
        projected = project_selected(layer, hybrid - prepared["base_qkv"]["head_query"])
        prediction = predict_delta(net, ids, prepared["base_states"][1],
                                   prepared["base_layer"], projected)
        qkv_predictions[cell] = prediction
        qkv_outputs[cell] = hybrid
        conditions[f"qkv_{cell.lower()}"] = prediction

    selected_head = contribution_delta(prepared, selection["slots"])
    selected_projected = project_selected(layer, selected_head)
    conditions["selected_sources"] = predict_delta(
        net, ids, prepared["base_states"][1], prepared["base_layer"], selected_projected)
    slot_sets = {"all_sources": range(6), "both_f": (0, 1), "non_f": (2, 3, 4, 5)}
    for name, slots in slot_sets.items():
        projected = project_selected(layer, contribution_delta(prepared, list(slots)))
        conditions[name] = predict_delta(net, ids, prepared["base_states"][1],
                                         prepared["base_layer"], projected)
    relevant_delta = project_selected(layer, _rowwise_f_delta(prepared, ids))
    other_delta = project_selected(layer, _rowwise_f_delta(prepared, ids, other=True))
    conditions["queried_f"] = predict_delta(
        net, ids, prepared["base_states"][1], prepared["base_layer"], relevant_delta)
    conditions["other_f"] = predict_delta(
        net, ids, prepared["base_states"][1], prepared["base_layer"], other_delta)

    generator = torch.Generator().manual_seed(config.random_seed + rep)
    random_delta = torch.randn(selected_projected.shape, generator=generator)
    random_delta /= random_delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_delta *= selected_projected.norm(dim=-1, keepdim=True)
    conditions["random"] = predict_delta(
        net, ids, prepared["base_states"][1], prepared["base_layer"], random_delta)
    shaped = selected_projected.view(len(confirmation), config.g_tables, -1)
    mismatch = torch.roll(shaped, shifts=-1, dims=0).reshape_as(selected_projected)
    mismatch *= (selected_projected.norm(dim=-1, keepdim=True)
                 / mismatch.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    conditions["mismatch"] = predict_delta(
        net, ids, prepared["base_states"][1], prepared["base_layer"], mismatch)

    donor_ids = _ids(confirmation, "donor")
    conditions["clean_base"] = net(ids)[0].argmax(-1).cpu()
    conditions["clean_donor"] = net(donor_ids)[0].argmax(-1).cpu()
    f_positive_state = prepared["base_states"][0].clone()
    donor_f = prepared["donor_states"][0][:, :2].repeat_interleave(config.g_tables, dim=0)
    f_positive_state[:, :2] = donor_f
    conditions["f_positive"] = continue_dense(net, ids, f_positive_state, 0).argmax(-1).cpu()

    base_sum_error = float((prepared["base_qkv"]["contributions"].sum(2)
                            - prepared["base_qkv"]["head_query"]).abs().max())
    donor_sum_error = float((prepared["donor_single_qkv"]["contributions"].sum(2)
                             - prepared["donor_single_qkv"]["head_query"]).abs().max())
    all_source = contribution_delta(prepared, list(range(6)))
    full_hybrid_delta = qkv_outputs["DDD"] - prepared["base_qkv"]["head_query"]
    delta_error = float((project_selected(layer, all_source)
                         - project_selected(layer, full_hybrid_delta)).abs().max())
    numerical = {
        "base_head_reconstruction": prepared["base_qkv"]["head_reconstruction_error"],
        "donor_head_reconstruction": prepared["donor_single_qkv"][
            "head_reconstruction_error"],
        "base_source_sum": base_sum_error, "donor_source_sum": donor_sum_error,
        "full_source_delta": delta_error,
    }
    scores = {name: score(pred, donor_targets, base_targets, len(confirmation),
                          config.g_tables) for name, pred in conditions.items()}
    attention = {"base": _aligned_attention(prepared["base_qkv"], ids),
                 "donor_g0_repeated": _aligned_attention(prepared["donor_qkv"], ids)}
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             **{f"{name}_prediction": int(pred[group * config.g_tables + g])
                for name, pred in conditions.items()}}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    return {"scores": scores, "numerical_errors": numerical,
            "attention_mass": attention}, rows


def _sufficient(scores, name):
    row = scores[name]
    return (row["donor_items"]["accuracy"] >= .75
            and row["donor_groups"]["accuracy"] >= .60
            and row["cross_g_non_injection"]["accuracy"] >= .90)


def decide(selection, metrics):
    clauses = {}
    for rep, row in metrics.items():
        scores = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "ddd_sufficient": _sufficient(scores, "qkv_ddd"),
            "all_sources_sufficient": _sufficient(scores, "all_sources"),
            "selected_sources_sufficient": _sufficient(scores, "selected_sources"),
            "random_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40),
            "mismatch_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["mismatch"]["donor_items"]["accuracy"] >= .40),
            "numerical_errors_below_1e-6": max(row["numerical_errors"].values()) < 1e-6,
        }
    supported = selection["discovery_qualified"] and all(
        all(row.values()) for row in clauses.values())
    qk = all(_sufficient(row["scores"], "qkv_ddb") for row in metrics.values())
    value = all(_sufficient(row["scores"], "qkv_bbd") for row in metrics.values())
    full = all(_sufficient(row["scores"], "qkv_ddd") for row in metrics.values())
    if qk and value:
        qkv_label = "qk_and_v_each_sufficient"
    elif qk:
        qkv_label = "qk_only_sufficient"
    elif value:
        qkv_label = "v_only_sufficient"
    elif full:
        qkv_label = "qk_v_conjunctive"
    else:
        qkv_label = "unresolved"
    return {"source_edge": "supported" if supported else "not_supported",
            "qkv_label": qkv_label, "clauses_by_seed": clauses}


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
            raise RuntimeError("TEACH-0009 checkpoint hash mismatch")
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
        row_metrics, rows = confirm_model(models[str(rep)], discovery, confirmation,
                                          selection, rep, config)
        metrics[str(rep)] = row_metrics
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        artifacts[str(rep)].update({"rows_sha256": sha_file(path), "rows": len(rows)})
    decision = decide(selection, metrics)
    elapsed = time.monotonic() - start
    if elapsed > config.max_seconds:
        raise RuntimeError("TEACH-0009 time cap exceeded")
    report = {"experiment": "TEACH-0009", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "selected_heads": list(SELECTED_HEADS),
              "source_selection": selection, "discovery_screen": screens,
              "metrics": metrics, "artifacts": artifacts, "decision": decision,
              "elapsed_seconds": elapsed, "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0009"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"],
                      "source_selection": report["source_selection"],
                      "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
