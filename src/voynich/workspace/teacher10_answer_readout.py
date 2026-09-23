"""Frozen TEACH-0010 downstream answer-write decomposition; no run on import."""

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
from .teacher7_dense_mechanism import _answers, _ids, continue_dense, manual_dense, score
from .teacher8_components import selected_head_delta


SOURCE_PATHS = (
    "docs/experiments/TEACH-0010.md",
    "docs/experiments/TEACH-0010-numerical-amendment.md",
    "src/voynich/workspace/teacher10_answer_readout.py",
    "src/voynich/workspace/teacher8_components.py",
    "src/voynich/workspace/teacher7_dense_mechanism.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
)


@dataclass(frozen=True)
class Config:
    suite_seed: int = 70111
    random_seed: int = 70211
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0010 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0010 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def layer_parts_with_weights(layer, state):
    """Expose heads using the weights returned by native MultiheadAttention."""
    normalized = layer.norm1(state)
    attention_write, weights = layer.self_attn(
        normalized, normalized, normalized, need_weights=True,
        average_attn_weights=False)
    width, heads = normalized.shape[-1], layer.self_attn.num_heads
    head_width = width // heads
    projected = F.linear(normalized, layer.self_attn.in_proj_weight,
                         layer.self_attn.in_proj_bias)
    value = projected.chunk(3, dim=-1)[2]
    batch, length, _ = value.shape
    value = value.view(batch, length, heads, head_width).transpose(1, 2)
    head_values = weights @ value
    concatenated = head_values.transpose(1, 2).reshape(batch, length, width)
    reconstructed = F.linear(concatenated, layer.self_attn.out_proj.weight,
                             layer.self_attn.out_proj.bias)
    post_attention = state + layer.dropout1(attention_write)
    mlp_write = layer._ff_block(layer.norm2(post_attention))
    post_mlp = post_attention + mlp_write
    native = layer(state)
    errors = {"attention": float((reconstructed - attention_write).abs().max()),
              "layer": float((post_mlp - native).abs().max())}
    return {"attention": attention_write, "post_attention": post_attention,
            "mlp": mlp_write, "post_mlp": post_mlp, "heads": head_values,
            "weights": weights, "errors": errors}


@torch.no_grad()
def prepare(net, groups, config):
    base_ids = _ids(groups, "base")
    donor_ref_ids = _ids(groups, "donor", first_only=True)
    _, base_states = manual_dense(net, base_ids)
    _, donor_states = manual_dense(net, donor_ref_ids)
    edited_cut2 = base_states[2].clone()
    edited_cut2[:, 5] = donor_states[2][:, 5].repeat_interleave(config.g_tables, dim=0)
    layer = net.transformer.layers[2]
    return {"base_ids": base_ids, "base_states": base_states, "donor_states": donor_states,
            "edited_cut2": edited_cut2,
            "base_parts": layer_parts_with_weights(layer, base_states[2]),
            "edited_parts": layer_parts_with_weights(layer, edited_cut2)}


def run_from_attention(net, prepared, query_attention):
    attention = prepared["base_parts"]["attention"].clone()
    attention[:, 5] = query_attention
    post_attention = prepared["base_states"][2] + attention
    layer = net.transformer.layers[2]
    post_mlp = post_attention + layer._ff_block(layer.norm2(post_attention))
    return continue_dense(net, prepared["base_ids"], post_mlp, 3).argmax(-1).cpu()


def run_from_post_attention(net, prepared, query_state):
    post_attention = prepared["base_parts"]["post_attention"].clone()
    post_attention[:, 5] = query_state
    layer = net.transformer.layers[2]
    post_mlp = post_attention + layer._ff_block(layer.norm2(post_attention))
    return continue_dense(net, prepared["base_ids"], post_mlp, 3).argmax(-1).cpu()


def run_from_mlp(net, prepared, query_mlp):
    mlp = prepared["base_parts"]["mlp"].clone()
    mlp[:, 5] = query_mlp
    state = prepared["base_parts"]["post_attention"] + mlp
    return continue_dense(net, prepared["base_ids"], state, 3).argmax(-1).cpu()


def select_heads(screens):
    rows = []
    for mask in range(1, 16):
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
    return {"mask": chosen[0], "heads": [head for head in range(4)
                                          if chosen[0] & (1 << head)],
            "size": chosen[1], "discovery_qualified": flag,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3]}


def cyclic_recipient(delta, groups, g_tables=3):
    shaped = delta.view(groups, g_tables, -1)
    shifted = torch.roll(shaped, shifts=-1, dims=1).reshape_as(delta)
    return shifted * (delta.norm(dim=-1, keepdim=True)
                      / shifted.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def repeat_g0(delta, groups, g_tables=3):
    shaped = delta.view(groups, g_tables, -1)
    return shaped[:, :1].expand(-1, g_tables, -1).reshape_as(delta)


@torch.no_grad()
def head_screen(net, groups, config):
    prepared = prepare(net, groups, config)
    donor = _answers(groups, "donor_answers")
    base = _answers(groups, "base_answers")
    layer = net.transformer.layers[2]
    screen = {}
    for mask in range(1, 16):
        selected = [head for head in range(4) if mask & (1 << head)]
        delta = selected_head_delta(
            layer, prepared["base_parts"]["heads"], prepared["edited_parts"]["heads"],
            selected)
        predictions = run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + delta)
        screen[str(mask)] = score(predictions, donor, base, len(groups), config.g_tables)
    return screen


@torch.no_grad()
def confirm_model(net, confirmation, selection, rep, config):
    prepared = prepare(net, confirmation, config)
    ids = prepared["base_ids"]
    donor_ids = _ids(confirmation, "donor")
    donor_targets = _answers(confirmation, "donor_answers")
    base_targets = _answers(confirmation, "base_answers")
    layer = net.transformer.layers[2]
    selected = selection["heads"]
    complement = [head for head in range(4) if head not in selected]
    selected_delta = selected_head_delta(
        layer, prepared["base_parts"]["heads"], prepared["edited_parts"]["heads"], selected)
    all_delta = selected_head_delta(
        layer, prepared["base_parts"]["heads"], prepared["edited_parts"]["heads"],
        list(range(4)))
    complement_delta = (selected_head_delta(
        layer, prepared["base_parts"]["heads"], prepared["edited_parts"]["heads"],
        complement) if complement else torch.zeros_like(selected_delta))
    conditions = {
        "clean_base": net(ids)[0].argmax(-1).cpu(),
        "clean_donor": net(donor_ids)[0].argmax(-1).cpu(),
        "upstream_key": continue_dense(net, ids, prepared["edited_cut2"], 2).argmax(-1).cpu(),
        "selected_heads": run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + selected_delta),
        "all_heads": run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + all_delta),
        "complement_heads": run_from_attention(
            net, prepared, prepared["base_parts"]["attention"][:, 5] + complement_delta),
        "mlp_only": run_from_mlp(net, prepared, prepared["edited_parts"]["mlp"][:, 5]),
        "post_attention": run_from_post_attention(
            net, prepared, prepared["edited_parts"]["post_attention"][:, 5]),
    }
    final_state = prepared["base_parts"]["post_mlp"].clone()
    final_state[:, 5] = prepared["edited_parts"]["post_mlp"][:, 5]
    conditions["final_state"] = continue_dense(net, ids, final_state, 3).argmax(-1).cpu()
    generator = torch.Generator().manual_seed(config.random_seed + rep)
    random_delta = torch.randn(selected_delta.shape, generator=generator)
    random_delta /= random_delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_delta *= selected_delta.norm(dim=-1, keepdim=True)
    conditions["random"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + random_delta)
    mismatch = cyclic_recipient(selected_delta, len(confirmation), config.g_tables)
    conditions["cyclic_recipient"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + mismatch)
    fixed_g0 = repeat_g0(selected_delta, len(confirmation), config.g_tables)
    conditions["fixed_g0"] = run_from_attention(
        net, prepared, prepared["base_parts"]["attention"][:, 5] + fixed_g0)
    f_positive_state = prepared["base_states"][0].clone()
    donor_f = prepared["donor_states"][0][:, :2].repeat_interleave(config.g_tables, dim=0)
    f_positive_state[:, :2] = donor_f
    conditions["f_positive"] = continue_dense(net, ids, f_positive_state, 0).argmax(-1).cpu()
    scores = {name: score(pred, donor_targets, base_targets, len(confirmation), config.g_tables)
              for name, pred in conditions.items()}
    numerical = {
        "base_attention": prepared["base_parts"]["errors"]["attention"],
        "base_layer": prepared["base_parts"]["errors"]["layer"],
        "edited_attention": prepared["edited_parts"]["errors"]["attention"],
        "edited_layer": prepared["edited_parts"]["errors"]["layer"],
    }
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             **{f"{name}_prediction": int(pred[group * config.g_tables + g])
                for name, pred in conditions.items()}}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    return {"scores": scores, "numerical_errors": numerical}, rows


def decide(selection, metrics):
    clauses = {}
    for rep, row in metrics.items():
        scores = row["scores"]
        selected = scores["selected_heads"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "upstream_key_at_least_0.95": scores["upstream_key"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "selected_items_at_least_0.75": selected["donor_items"]["accuracy"] >= .75,
            "selected_groups_at_least_0.60": selected["donor_groups"]["accuracy"] >= .60,
            "selected_non_injection_at_least_0.90": selected[
                "cross_g_non_injection"]["accuracy"] >= .90,
            "random_advantage_at_least_0.40": (
                selected["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40),
            "cyclic_advantage_at_least_0.40": (
                selected["donor_items"]["accuracy"]
                - scores["cyclic_recipient"]["donor_items"]["accuracy"] >= .40),
            "numerical_errors_below_1e-6": max(row["numerical_errors"].values()) < 1e-6,
        }
    supported = selection["discovery_qualified"] and all(
        all(row.values()) for row in clauses.values())
    return {"answer_write": "supported" if supported else "not_supported",
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
            raise RuntimeError("TEACH-0010 checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net = DenseRows()
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        models[str(rep)] = net
        screens[str(rep)] = head_screen(net, discovery, config)
        artifacts[str(rep)] = {"checkpoint_sha256": expected}
    selection = select_heads(screens)
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
        raise RuntimeError("TEACH-0010 time cap exceeded")
    report = {"experiment": "TEACH-0010", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "selection": selection,
              "discovery_screen": screens, "metrics": metrics, "artifacts": artifacts,
              "decision": decision, "elapsed_seconds": elapsed,
              "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0010"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"], "selection": report["selection"],
                      "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
