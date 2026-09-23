"""Frozen TEACH-0008 dense-layer component decomposition; no run on import."""

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
from .teacher6_geometry import key_centroid_basis
from .teacher7_dense_mechanism import _answers, _ids, continue_dense, manual_dense, score


SOURCE_PATHS = (
    "docs/experiments/TEACH-0008.md",
    "docs/experiments/TEACH-0008-numerical-amendment.md",
    "docs/experiments/TEACH-0008-native-residual-amendment.md",
    "src/voynich/workspace/teacher8_components.py",
    "src/voynich/workspace/teacher7_dense_mechanism.py",
    "src/voynich/workspace/teacher6_geometry.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
)


@dataclass(frozen=True)
class Config:
    suite_seed: int = 68111
    random_seed: int = 68211
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0008 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0008 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def layer_parts(layer, state):
    """Reproduce norm-first encoder layer and expose pre-out-projection head values."""
    normalized = layer.norm1(state)
    width, heads = normalized.shape[-1], layer.self_attn.num_heads
    head_width = width // heads
    qkv = F.linear(normalized, layer.self_attn.in_proj_weight,
                   layer.self_attn.in_proj_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    batch, length, _ = query.shape
    query = query.view(batch, length, heads, head_width).transpose(1, 2)
    key = key.view(batch, length, heads, head_width).transpose(1, 2)
    value = value.view(batch, length, heads, head_width).transpose(1, 2)
    head_values = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False,
        scale=1 / math.sqrt(head_width))
    concatenated = head_values.transpose(1, 2).reshape(batch, length, width)
    head_attention = F.linear(concatenated, layer.self_attn.out_proj.weight,
                              layer.self_attn.out_proj.bias)
    # Keep the native attention write as the residual-stream baseline.  The
    # exposed head values reconstruct it within the registered tolerance, but
    # using the native baseline avoids amplifying sub-ULP kernel-order noise in
    # the subsequent residual and MLP operations.
    attention_write = layer._sa_block(normalized, None, None)
    post_attention = state + layer.dropout1(attention_write)
    mlp_write = layer._ff_block(layer.norm2(post_attention))
    post_mlp = post_attention + mlp_write
    native = layer(state)
    errors = {"attention": float((head_attention - attention_write).abs().max()),
              "layer": float((post_mlp - native).abs().max())}
    return {"attention": attention_write, "post_attention": post_attention,
            "mlp": mlp_write, "post_mlp": post_mlp, "heads": head_values,
            "head_attention": head_attention,
            "errors": errors}


def selected_head_delta(layer, base_heads, donor_heads, selected, query_slot=5):
    difference = torch.zeros_like(base_heads)
    difference[:, selected, query_slot] = (
        donor_heads[:, selected, query_slot] - base_heads[:, selected, query_slot])
    batch, heads, length, head_width = difference.shape
    concatenated = difference.transpose(1, 2).reshape(batch, length, heads * head_width)
    return F.linear(concatenated, layer.self_attn.out_proj.weight, bias=None)[:, query_slot]


def run_from_attention(net, ids, cut1_state, base_parts, query_attention):
    attention = base_parts["attention"].clone()
    attention[:, 5] = query_attention
    post_attention = cut1_state + attention
    layer = net.transformer.layers[1]
    post_mlp = post_attention + layer._ff_block(layer.norm2(post_attention))
    return continue_dense(net, ids, post_mlp, 2).argmax(-1).cpu(), post_attention


def run_from_post_attention(net, ids, base_parts, query_state):
    post_attention = base_parts["post_attention"].clone()
    post_attention[:, 5] = query_state
    layer = net.transformer.layers[1]
    post_mlp = post_attention + layer._ff_block(layer.norm2(post_attention))
    return continue_dense(net, ids, post_mlp, 2).argmax(-1).cpu()


def run_from_mlp(net, ids, base_parts, query_mlp):
    mlp = base_parts["mlp"].clone()
    mlp[:, 5] = query_mlp
    return continue_dense(net, ids, base_parts["post_attention"] + mlp, 2).argmax(-1).cpu()


@torch.no_grad()
def decompose_inputs(net, base_ids, donor_ref_ids, config):
    _, base_states = manual_dense(net, base_ids)
    _, donor_states = manual_dense(net, donor_ref_ids)
    base_cut1 = base_states[1]
    donor_cut1 = donor_states[1].repeat_interleave(config.g_tables, dim=0)
    base_parts = layer_parts(net.transformer.layers[1], base_cut1)
    donor_parts_single = layer_parts(net.transformer.layers[1], donor_states[1])
    donor_parts = {key: (value.repeat_interleave(config.g_tables, dim=0)
                         if isinstance(value, torch.Tensor) else value)
                   for key, value in donor_parts_single.items()}
    return base_states, donor_states, base_cut1, donor_cut1, base_parts, donor_parts


@torch.no_grad()
def head_screen(net, groups, config):
    base_ids = _ids(groups, "base")
    donor_ref_ids = _ids(groups, "donor", first_only=True)
    _, _, _, _, base_parts, donor_parts = decompose_inputs(
        net, base_ids, donor_ref_ids, config)
    donor = _answers(groups, "donor_answers")
    base = _answers(groups, "base_answers")
    layer = net.transformer.layers[1]
    screen = {}
    for mask in range(1, 16):
        selected = [head for head in range(4) if mask & (1 << head)]
        delta = selected_head_delta(layer, base_parts["heads"], donor_parts["heads"], selected)
        predictions, _ = run_from_attention(
            net, base_ids, base_parts["post_attention"] - base_parts["attention"],
            base_parts, base_parts["attention"][:, 5] + delta)
        screen[str(mask)] = score(predictions, donor, base, len(groups), config.g_tables)
    return screen


def select_heads(screens):
    rows = []
    for mask in range(1, 16):
        key = str(mask)
        group_min = min(screen[key]["donor_groups"]["accuracy"] for screen in screens.values())
        item_min = min(screen[key]["donor_items"]["accuracy"] for screen in screens.values())
        cross_min = min(screen[key]["cross_g_non_injection"]["accuracy"]
                        for screen in screens.values())
        rows.append((mask, mask.bit_count(), group_min, item_min, cross_min))
    qualified = [row for row in rows if row[2] >= .60 and row[4] >= .90]
    if qualified:
        chosen = sorted(qualified, key=lambda row: (row[1], -row[2], -row[3], row[0]))[0]
        flag = True
    else:
        chosen = sorted(rows, key=lambda row: (-row[2], -row[3], row[1], row[0]))[0]
        flag = False
    return {"mask": chosen[0], "heads": [head for head in range(4)
                                          if chosen[0] & (1 << head)],
            "size": chosen[1], "discovery_qualified": flag,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
            "minimum_cross_g_non_injection": chosen[4]}


@torch.no_grad()
def confirm_model(net, discovery, confirmation, selection, rep, config):
    base_ids = _ids(confirmation, "base")
    donor_ids = _ids(confirmation, "donor")
    donor_ref_ids = _ids(confirmation, "donor", first_only=True)
    base_states, donor_states, base_cut1, _, base_parts, donor_parts = decompose_inputs(
        net, base_ids, donor_ref_ids, config)
    donor_targets = _answers(confirmation, "donor_answers")
    base_targets = _answers(confirmation, "base_answers")
    clean_base = net(base_ids)[0].argmax(-1).cpu()
    clean_donor = net(donor_ids)[0].argmax(-1).cpu()
    layer = net.transformer.layers[1]
    selected = selection["heads"]
    complement = [head for head in range(4) if head not in selected]
    selected_delta = selected_head_delta(
        layer, base_parts["heads"], donor_parts["heads"], selected)
    complement_delta = selected_head_delta(
        layer, base_parts["heads"], donor_parts["heads"], complement) if complement else \
        torch.zeros_like(selected_delta)
    all_delta = selected_head_delta(layer, base_parts["heads"], donor_parts["heads"],
                                    list(range(4)))
    selected_predictions, selected_post_attention = run_from_attention(
        net, base_ids, base_cut1, base_parts, base_parts["attention"][:, 5] + selected_delta)
    all_heads, _ = run_from_attention(
        net, base_ids, base_cut1, base_parts, base_parts["attention"][:, 5] + all_delta)
    complement_predictions, _ = run_from_attention(
        net, base_ids, base_cut1, base_parts, base_parts["attention"][:, 5] + complement_delta)
    mlp_predictions = run_from_mlp(net, base_ids, base_parts, donor_parts["mlp"][:, 5])
    post_attention_predictions = run_from_post_attention(
        net, base_ids, base_parts, donor_parts["post_attention"][:, 5])
    final_state = base_parts["post_mlp"].clone()
    final_state[:, 5] = donor_parts["post_mlp"][:, 5]
    final_predictions = continue_dense(net, base_ids, final_state, 2).argmax(-1).cpu()
    generator = torch.Generator().manual_seed(config.random_seed + rep)
    random_delta = torch.randn(selected_delta.shape, generator=generator)
    random_delta = random_delta / random_delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_delta *= selected_delta.norm(dim=-1, keepdim=True)
    random_predictions, _ = run_from_attention(
        net, base_ids, base_cut1, base_parts, base_parts["attention"][:, 5] + random_delta)
    f_positive_state = base_states[0].clone()
    donor_f = donor_states[0][:, :2].repeat_interleave(config.g_tables, dim=0)
    f_positive_state[:, :2] = donor_f
    f_positive = continue_dense(net, base_ids, f_positive_state, 0).argmax(-1).cpu()
    discovery_base_ids = _ids(discovery, "base", first_only=True)
    discovery_donor_ids = _ids(discovery, "donor", first_only=True)
    _, db_states = manual_dense(net, discovery_base_ids)
    _, dd_states = manual_dense(net, discovery_donor_ids)
    db_parts = layer_parts(layer, db_states[1])
    dd_parts = layer_parts(layer, dd_states[1])
    discovery_delta = selected_head_delta(layer, db_parts["heads"], dd_parts["heads"], selected)
    discovery_representation = torch.cat((db_parts["post_attention"][:, 5],
                                          db_parts["post_attention"][:, 5]
                                          + discovery_delta))
    discovery_labels = torch.cat((torch.tensor([g["base_key"] for g in discovery]),
                                  torch.tensor([g["donor_key"] for g in discovery])))
    geometry = key_centroid_basis(discovery_representation, discovery_labels)
    base_representation = base_parts["post_attention"][:, 5]
    selected_change = selected_post_attention[:, 5] - base_representation
    dimension_scores = {}
    for dimension in sorted({1, 2, 4, 8, geometry["rank"]}):
        used = geometry["basis"][:, :min(dimension, geometry["rank"])]
        projected = base_representation + selected_change @ used @ used.T
        pred = run_from_post_attention(net, base_ids, base_parts, projected)
        dimension_scores[str(min(dimension, geometry["rank"]))] = score(
            pred, donor_targets, base_targets, len(confirmation), config.g_tables)
    conditions = {"clean_base": clean_base, "clean_donor": clean_donor,
                  "selected_heads": selected_predictions, "all_heads": all_heads,
                  "complement_heads": complement_predictions, "mlp_only": mlp_predictions,
                  "post_attention": post_attention_predictions, "final_state": final_predictions,
                  "random": random_predictions, "f_positive": f_positive}
    scores = {name: score(pred, donor_targets, base_targets, len(confirmation), config.g_tables)
              for name, pred in conditions.items()}
    errors = {"base_attention": base_parts["errors"]["attention"],
              "base_layer": base_parts["errors"]["layer"],
              "donor_attention": donor_parts["errors"]["attention"],
              "donor_layer": donor_parts["errors"]["layer"]}
    metrics = {"scores": scores, "numerical_errors": errors, "rank": geometry["rank"],
               "singular_values": geometry["singular_values"].tolist(),
               "dimension_scores": dimension_scores}
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             **{f"{name}_prediction": int(pred[group * config.g_tables + g])
                for name, pred in conditions.items()}}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    return metrics, rows


def decide(selection, metrics):
    clauses = {}
    for rep, row in metrics.items():
        scores = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "selected_items_at_least_0.75": scores["selected_heads"]["donor_items"][
                "accuracy"] >= .75,
            "selected_groups_at_least_0.60": scores["selected_heads"]["donor_groups"][
                "accuracy"] >= .60,
            "selected_non_injection_at_least_0.90": scores["selected_heads"][
                "cross_g_non_injection"]["accuracy"] >= .90,
            "random_advantage_at_least_0.40": scores["selected_heads"]["donor_items"][
                "accuracy"] - scores["random"]["donor_items"]["accuracy"] >= .40,
            "numerical_errors_below_1e-6": max(row["numerical_errors"].values()) < 1e-6,
        }
    supported = selection["discovery_qualified"] and all(
        all(row.values()) for row in clauses.values())
    return {"head_write": "supported" if supported else "not_supported",
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
            raise RuntimeError("TEACH-0008 checkpoint hash mismatch")
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
        row_metrics, rows = confirm_model(models[str(rep)], discovery, confirmation,
                                          selection, rep, config)
        metrics[str(rep)] = row_metrics
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        artifacts[str(rep)].update({"rows_sha256": sha_file(path), "rows": len(rows)})
    verdict = decide(selection, metrics)
    if time.monotonic() - start > config.max_seconds:
        raise RuntimeError("TEACH-0008 time cap exceeded")
    report = {"experiment": "TEACH-0008", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "selection": selection,
              "discovery_screen": screens, "metrics": metrics, "artifacts": artifacts,
              "decision": verdict, "elapsed_seconds": time.monotonic() - start,
              "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0008"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"], "selection": report["selection"],
                      "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
