"""Frozen TEACH-0006 causal geometry analysis; no checkpoint access on import."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import torch
import torch.nn.functional as F

from .teacher2_train import wilson_95, write_json
from .teacher4_models import LearnedMemory
from .teacher5_intervene import _states, build_groups, sha_file, stable_json


SOURCE_PATHS = (
    "docs/experiments/TEACH-0006.md",
    "docs/experiments/TEACH-0006-rank-amendment.md",
    "src/voynich/workspace/teacher6_geometry.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
    "src/voynich/workspace/teacher4_tasks.py",
)


@dataclass(frozen=True)
class Config:
    suite_seed: int = 66111
    random_subspace_seed: int = 66211
    alignment_shuffle_seed: int = 66311
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    random_subspaces: int = 32
    rotations: int = 16
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0006 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0006 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def _episode_ids(groups, field, *, first_g_only=False):
    if first_g_only:
        return torch.tensor([group[field][0] for group in groups], dtype=torch.long)
    return torch.tensor([tokens for group in groups for tokens in group[field]], dtype=torch.long)


def _labels(groups, field):
    return torch.tensor([value for group in groups for value in group[field]], dtype=torch.long)


def _single_labels(groups, field):
    return torch.tensor([group[field] for group in groups], dtype=torch.long)


def key_centroid_basis(states, labels):
    """Return centered key centroids and their nonzero orthonormal span."""
    keys = sorted(set(labels.tolist()))
    mean = states.mean(0)
    centroids = torch.stack([states[labels == key].mean(0) for key in keys])
    # Center and estimate rank in float64. With K centered centroids the exact
    # rank is at most K-1; float32 centering left a spurious twelfth singular
    # value in the invalid first execution documented by the amendment.
    centered = centroids.double() - centroids.double().mean(0, keepdim=True)
    _, singular64, vh = torch.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * torch.finfo(centered.dtype).eps * singular64[0]
    rank = min(int((singular64 > tolerance).sum()), len(keys) - 1)
    basis = vh[:rank].T.to(states.dtype).contiguous()
    singular = singular64.to(states.dtype)
    return {"keys": keys, "mean": mean, "centroids": centroids, "basis": basis,
            "singular_values": singular, "rank": rank}


def nearest_centroid_accuracy(states, labels, centroids, keys):
    distances = torch.cdist(states, centroids)
    predictions = torch.tensor(keys, dtype=torch.long)[distances.argmin(-1)]
    return float((predictions == labels).float().mean()), predictions


def _rate(correct, total):
    return {"correct": int(correct), "total": int(total), "accuracy": correct / total,
            "wilson_95": wilson_95(int(correct), int(total))}


def score_predictions(predictions, donor_targets, base_targets, groups, g_tables=3):
    donor_correct = int((predictions == donor_targets).sum())
    base_correct = int((predictions == base_targets).sum())
    exact = sum(bool((predictions[i:i + g_tables] == donor_targets[i:i + g_tables]).all())
                for i in range(0, len(predictions), g_tables))
    return {"donor_items": _rate(donor_correct, len(predictions)),
            "base_items": _rate(base_correct, len(predictions)),
            "donor_groups": _rate(exact, groups)}


@torch.no_grad()
def predictions_with_first(net, ids, first):
    return _states(net, ids, first_override=first)["logits"].argmax(-1).cpu()


def _haar_basis(width, rank, generator):
    matrix = torch.randn(width, rank, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def _haar_rotation(width, generator):
    matrix = torch.randn(width, width, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r)).clamp_min(0) * 2 - 1
    return q * signs


def _percentile95(values):
    ordered = sorted(values)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def _pearson(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(), y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denom) if float(denom) else 0.0


def _ranks(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(x), dtype=torch.float64)
    return ranks


def _angles(first, second):
    singular = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    return [math.degrees(math.acos(float(value))) for value in singular]


def _procrustes(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    left, _, right_h = torch.linalg.svd(
        (source - source_mean).T @ (target - target_mean), full_matrices=False)
    rotation = left @ right_h
    fitted = (source - source_mean) @ rotation + target_mean
    residual = float((fitted - target).norm() / target.sub(target_mean).norm().clamp_min(1e-12))
    return source_mean, target_mean, rotation, residual


def _map_states(states, source_mean, target_mean, rotation):
    return (states - source_mean) @ rotation + target_mean


def within_model(net, discovery, confirmation, rep, config):
    discovery_base_ids = _episode_ids(discovery, "base", first_g_only=True)
    discovery_donor_ids = _episode_ids(discovery, "donor", first_g_only=True)
    confirmation_base_one = _episode_ids(confirmation, "base", first_g_only=True)
    confirmation_donor_one = _episode_ids(confirmation, "donor", first_g_only=True)
    confirmation_base_ids = _episode_ids(confirmation, "base")
    with torch.no_grad():
        discovery_base = _states(net, discovery_base_ids)["first"]
        discovery_donor = _states(net, discovery_donor_ids)["first"]
        confirmation_base = _states(net, confirmation_base_one)["first"]
        confirmation_donor = _states(net, confirmation_donor_one)["first"]
        clean = _states(net, confirmation_base_ids)
    discovery_states = torch.cat((discovery_base, discovery_donor))
    discovery_labels = torch.cat((_single_labels(discovery, "base_key"),
                                  _single_labels(discovery, "donor_key")))
    confirmation_states = torch.cat((confirmation_base, confirmation_donor))
    confirmation_labels = torch.cat((_single_labels(confirmation, "base_key"),
                                     _single_labels(confirmation, "donor_key")))
    geometry = key_centroid_basis(discovery_states, discovery_labels)
    centroid_accuracy, _ = nearest_centroid_accuracy(
        confirmation_states, confirmation_labels, geometry["centroids"], geometry["keys"])
    basis = geometry["basis"]
    rank = geometry["rank"]
    base_first = confirmation_base.repeat_interleave(config.g_tables, dim=0)
    donor_first = confirmation_donor.repeat_interleave(config.g_tables, dim=0)
    delta = donor_first - base_first
    donor_targets = _labels(confirmation, "donor_answers")
    base_targets = _labels(confirmation, "base_answers")
    clean_predictions = clean["logits"].argmax(-1).cpu()
    full_predictions = predictions_with_first(net, confirmation_base_ids, donor_first)
    projection = delta @ basis @ basis.T
    key_predictions = predictions_with_first(net, confirmation_base_ids, base_first + projection)
    complement_predictions = predictions_with_first(
        net, confirmation_base_ids, base_first + delta - projection)
    scores = {
        "clean": score_predictions(clean_predictions, donor_targets, base_targets,
                                   len(confirmation), config.g_tables),
        "full": score_predictions(full_predictions, donor_targets, base_targets,
                                  len(confirmation), config.g_tables),
        "key_span": score_predictions(key_predictions, donor_targets, base_targets,
                                      len(confirmation), config.g_tables),
        "complement": score_predictions(complement_predictions, donor_targets, base_targets,
                                        len(confirmation), config.g_tables),
    }
    dimension_scores = {}
    for dimension in sorted({1, 2, 4, 8, rank}):
        use = basis[:, :min(dimension, rank)]
        pred = predictions_with_first(net, confirmation_base_ids,
                                      base_first + delta @ use @ use.T)
        dimension_scores[str(min(dimension, rank))] = score_predictions(
            pred, donor_targets, base_targets, len(confirmation), config.g_tables)
    generator = torch.Generator().manual_seed(config.random_subspace_seed + rep)
    random_accuracies = []
    for _ in range(config.random_subspaces):
        random_basis = _haar_basis(delta.shape[1], rank, generator)
        pred = predictions_with_first(net, confirmation_base_ids,
                                      base_first + delta @ random_basis @ random_basis.T)
        random_accuracies.append(float((pred == donor_targets).float().mean()))
    leverage = basis.square().sum(-1)
    native_order = torch.argsort(leverage, descending=True)
    native_scores = {}
    for size in sorted({rank, min(2 * rank, 128), 32, 128}):
        mask = torch.zeros(128)
        mask[native_order[:size]] = 1
        pred = predictions_with_first(net, confirmation_base_ids, base_first + delta * mask)
        native_scores[str(size)] = score_predictions(pred, donor_targets, base_targets,
                                                     len(confirmation), config.g_tables)
    rotated_accuracies = []
    for _ in range(config.rotations):
        rotation = _haar_rotation(128, generator)
        rotated_basis = rotation.T @ basis
        order = torch.argsort(rotated_basis.square().sum(-1), descending=True)
        mask = torch.zeros(128)
        mask[order[:rank]] = 1
        rotated_delta = delta @ rotation
        patched = base_first + (rotated_delta * mask) @ rotation.T
        pred = predictions_with_first(net, confirmation_base_ids, patched)
        rotated_accuracies.append(float((pred == donor_targets).float().mean()))
    local_count = min(128, len(confirmation_base_ids))
    ids_local = confirmation_base_ids[:local_count]
    base_local = base_first[:local_count].detach().requires_grad_(True)
    donor_local = donor_first[:local_count]
    base_label_local = base_targets[:local_count]
    donor_label_local = donor_targets[:local_count]
    logits_local = _states(net, ids_local, first_override=base_local)["logits"]
    indices = torch.arange(local_count)
    margin = logits_local[indices, donor_label_local] - logits_local[indices, base_label_local]
    gradient = torch.autograd.grad(margin.sum(), base_local)[0]
    predicted_change = (gradient * (donor_local - base_local.detach())).sum(-1).detach()
    with torch.no_grad():
        donor_logits = _states(net, ids_local, first_override=donor_local)["logits"]
        donor_margin = (donor_logits[indices, donor_label_local]
                        - donor_logits[indices, base_label_local])
    actual_change = donor_margin - margin.detach()
    locality = {
        "items": local_count,
        "pearson": _pearson(predicted_change, actual_change),
        "spearman": _pearson(_ranks(predicted_change), _ranks(actual_change)),
        "sign_agreement": float((torch.sign(predicted_change) == torch.sign(actual_change)).float().mean()),
        "median_absolute_relative_error": float(torch.median(
            (predicted_change - actual_change).abs() / actual_change.abs().clamp_min(1e-8))),
    }
    trajectory = {}
    for fraction in (0.0, .25, .5, .75, 1.0):
        with torch.no_grad():
            logits = _states(net, confirmation_base_ids,
                             first_override=base_first + fraction * delta)["logits"]
            probs = F.softmax(logits, dim=-1)
            idx = torch.arange(len(logits))
        trajectory[str(fraction)] = {
            "donor_accuracy": float((logits.argmax(-1).cpu() == donor_targets).float().mean()),
            "base_accuracy": float((logits.argmax(-1).cpu() == base_targets).float().mean()),
            "mean_donor_probability": float(probs[idx, donor_targets].mean()),
            "mean_base_probability": float(probs[idx, base_targets].mean()),
        }
    within = []
    between = []
    normalized = F.normalize(confirmation_states - geometry["mean"], dim=-1)
    similarities = normalized @ normalized.T
    for i in range(len(confirmation_states)):
        for j in range(i + 1, len(confirmation_states)):
            (within if confirmation_labels[i] == confirmation_labels[j] else between).append(
                float(similarities[i, j]))
    metrics = {
        "rank": rank,
        "singular_values": geometry["singular_values"].tolist(),
        "centroid_energy_fraction": [float(value.square() / geometry["singular_values"].square().sum())
                                     for value in geometry["singular_values"]],
        "key_counts_discovery": {str(key): int((discovery_labels == key).sum())
                                 for key in geometry["keys"]},
        "key_counts_confirmation": {str(key): int((confirmation_labels == key).sum())
                                    for key in geometry["keys"]},
        "nearest_centroid_confirmation_accuracy": centroid_accuracy,
        "scores": scores,
        "dimension_scores": dimension_scores,
        "random_subspace_donor_accuracies": random_accuracies,
        "random_subspace_p95": _percentile95(random_accuracies),
        "native_neuron_scores": native_scores,
        "native_top_coordinates": native_order[:32].tolist(),
        "leverage": leverage.tolist(),
        "leverage_participation_ratio": float(leverage.sum().square() / leverage.square().sum()),
        "rotated_top_rank_donor_accuracies": rotated_accuracies,
        "rotated_top_rank_p95": _percentile95(rotated_accuracies),
        "cosine_within_mean": sum(within) / len(within),
        "cosine_between_mean": sum(between) / len(between),
        "locality": locality,
        "trajectory": trajectory,
    }
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             "clean_prediction": int(clean_predictions[group * config.g_tables + g]),
             "full_prediction": int(full_predictions[group * config.g_tables + g]),
             "key_span_prediction": int(key_predictions[group * config.g_tables + g]),
             "complement_prediction": int(complement_predictions[group * config.g_tables + g])}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    state = {"discovery": discovery_states, "confirmation_base": confirmation_base,
             "confirmation_donor": confirmation_donor, "basis": basis,
             "discovery_labels": discovery_labels, "confirmation_labels": confirmation_labels,
             "confirmation_base_ids": confirmation_base_ids, "base_first": base_first,
             "donor_targets": donor_targets, "base_targets": base_targets}
    return metrics, rows, state


def cross_seed(models, states, confirmation, config):
    first, second = states["0"], states["1"]
    mean0, mean1, rotation01, residual01 = _procrustes(first["discovery"], second["discovery"])
    mean1r, mean0r, rotation10, residual10 = _procrustes(second["discovery"], first["discovery"])
    rng = random.Random(config.alignment_shuffle_seed)
    permutation = list(range(len(first["discovery"])))
    rng.shuffle(permutation)
    shuffled = torch.tensor(permutation)
    sm0, sm1, shuffled01, shuffled_residual01 = _procrustes(
        first["discovery"], second["discovery"][shuffled])
    sm1r, sm0r, shuffled10, shuffled_residual10 = _procrustes(
        second["discovery"], first["discovery"][shuffled])
    results = {}
    mappings = {
        "0_to_1": (first, second, models["1"], mean0, mean1, rotation01,
                   sm0, sm1, shuffled01, residual01, shuffled_residual01),
        "1_to_0": (second, first, models["0"], mean1r, mean0r, rotation10,
                   sm1r, sm0r, shuffled10, residual10, shuffled_residual10),
    }
    for name, (source, target, net, source_mean, target_mean, rotation,
               shuffle_source_mean, shuffle_target_mean, shuffle_rotation,
               residual, shuffle_residual) in mappings.items():
        mapped = _map_states(source["confirmation_donor"], source_mean,
                             target_mean, rotation).repeat_interleave(config.g_tables, dim=0)
        shuffled_mapped = _map_states(source["confirmation_donor"], shuffle_source_mean,
                                      shuffle_target_mean, shuffle_rotation).repeat_interleave(
                                          config.g_tables, dim=0)
        predictions = predictions_with_first(net, target["confirmation_base_ids"], mapped)
        shuffled_predictions = predictions_with_first(
            net, target["confirmation_base_ids"], shuffled_mapped)
        donor_accuracy = float((predictions == target["donor_targets"]).float().mean())
        shuffled_accuracy = float(
            (shuffled_predictions == target["donor_targets"]).float().mean())
        results[name] = {"donor_accuracy": donor_accuracy,
                         "shuffled_donor_accuracy": shuffled_accuracy,
                         "advantage": donor_accuracy - shuffled_accuracy,
                         "discovery_relative_residual": residual,
                         "shuffled_discovery_relative_residual": shuffle_residual}
    aligned_basis0 = rotation01.T @ first["basis"]
    results["principal_angles_degrees"] = {
        "unaligned": _angles(first["basis"], second["basis"]),
        "aligned": _angles(aligned_basis0, second["basis"]),
    }
    return results


def decide(metrics, cross):
    clauses = {}
    axis = {}
    for rep, row in metrics.items():
        score = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": score["clean"]["base_items"]["accuracy"] >= .95,
            "full_patch_at_least_0.95": score["full"]["donor_items"]["accuracy"] >= .95,
            "rank_at_most_11": row["rank"] <= 11,
            "key_span_items_at_least_0.80": score["key_span"]["donor_items"]["accuracy"] >= .80,
            "key_span_groups_at_least_0.65": score["key_span"]["donor_groups"]["accuracy"] >= .65,
            "complement_preserves_base_at_least_0.90": score["complement"]["base_items"][
                "accuracy"] >= .90,
            "complement_donor_at_most_0.10": score["complement"]["donor_items"][
                "accuracy"] <= .10,
            "random_p95_advantage_at_least_0.40": score["key_span"]["donor_items"]["accuracy"]
                - row["random_subspace_p95"] >= .40,
            "nearest_centroid_at_least_0.90": row[
                "nearest_centroid_confirmation_accuracy"] >= .90,
        }
        native = row["native_neuron_scores"][str(row["rank"])]["donor_items"]["accuracy"]
        axis[rep] = {"native_top_rank_at_least_0.80": native >= .80,
                     "rotation_p95_advantage_at_least_0.30":
                         native - row["rotated_top_rank_p95"] >= .30}
    subspace = "causal_subspace_supported" if all(all(row.values()) for row in clauses.values()) \
        else "not_supported"
    alignment_clauses = {}
    for name in ("0_to_1", "1_to_0"):
        alignment_clauses[name] = {
            "donor_accuracy_at_least_0.80": cross[name]["donor_accuracy"] >= .80,
            "shuffled_advantage_at_least_0.40": cross[name]["advantage"] >= .40,
        }
    alignment = "cross_seed_causal_alignment" if all(
        all(row.values()) for row in alignment_clauses.values()) else "not_supported"
    axis_verdict = "native_axis_concentrated" if all(all(row.values()) for row in axis.values()) \
        else "not_supported"
    return {"causal_subspace": subspace, "causal_subspace_clauses": clauses,
            "cross_seed_alignment": alignment, "cross_seed_clauses": alignment_clauses,
            "native_axis": axis_verdict, "native_axis_clauses": axis}


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
    metrics, rows_by_rep, states, models, artifacts = {}, {}, {}, {}, {}
    for rep in range(2):
        checkpoint = checkpoint_dir / f"rep{rep}-two_read.pt"
        expected = report4["arms"][str(rep)]["two_read"]["checkpoint_sha256"]
        if sha_file(checkpoint) != expected:
            raise RuntimeError("TEACH-0006 checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net = LearnedMemory(True)
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        row_metrics, rows, state = within_model(net, discovery, confirmation, rep, config)
        metrics[str(rep)], rows_by_rep[str(rep)], states[str(rep)], models[str(rep)] = (
            row_metrics, rows, state, net)
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        artifacts[str(rep)] = {"checkpoint_sha256": expected, "rows_sha256": sha_file(path),
                               "rows": len(rows)}
    cross = cross_seed(models, states, confirmation, config)
    verdict = decide(metrics, cross)
    if time.monotonic() - start > config.max_seconds:
        raise RuntimeError("TEACH-0006 time cap exceeded")
    report = {"experiment": "TEACH-0006", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "artifacts": artifacts,
              "metrics": metrics, "cross_seed": cross, "decision": verdict,
              "elapsed_seconds": time.monotonic() - start, "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0006"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
