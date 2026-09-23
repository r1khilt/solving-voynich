"""Frozen TEACH-0007 dense-row causal-site discovery; no run on import."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from .teacher2_train import wilson_95, write_json
from .teacher4_models import DenseRows
from .teacher4_tasks import NAMES, make_episode, table_partitions
from .teacher5_intervene import build_groups, sha_file, stable_json
from .teacher6_geometry import _haar_basis, _haar_rotation, _percentile95
from .teacher6_geometry import key_centroid_basis, nearest_centroid_accuracy


SOURCE_PATHS = (
    "docs/experiments/TEACH-0007.md",
    "src/voynich/workspace/teacher7_dense_mechanism.py",
    "src/voynich/workspace/teacher6_geometry.py",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
    "src/voynich/workspace/teacher4_tasks.py",
)


@dataclass(frozen=True)
class Config:
    suite_seed: int = 67111
    random_state_seed: int = 67211
    random_subspace_seed: int = 67311
    discovery_groups: int = 128
    confirmation_groups: int = 128
    g_tables: int = 3
    random_subspaces: int = 32
    rotations: int = 16
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0007 configuration changed")


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0007 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def _ids(groups, field, first_only=False):
    if first_only:
        return torch.tensor([group[field][0] for group in groups], dtype=torch.long)
    return torch.tensor([tokens for group in groups for tokens in group[field]], dtype=torch.long)


def _answers(groups, field):
    return torch.tensor([answer for group in groups for answer in group[field]], dtype=torch.long)


def _single(groups, field):
    return torch.tensor([group[field] for group in groups], dtype=torch.long)


def manual_dense(net, ids):
    f_rows, g_rows, marker, query = net.interface(ids)
    state = torch.cat((f_rows, g_rows, marker[:, None], query[:, None]), dim=1)
    states = [state]
    for layer in net.transformer.layers:
        state = layer(state)
        states.append(state)
    answer = net.final_norm(state[:, -1])
    logits = net.heads(ids, answer, answer, answer)
    return logits, states


def continue_dense(net, ids, state, cut):
    for layer in net.transformer.layers[cut:]:
        state = layer(state)
    answer = net.final_norm(state[:, -1])
    return net.heads(ids, answer, answer, answer)


def _rate(correct, total):
    return {"correct": int(correct), "total": int(total), "accuracy": correct / total,
            "wilson_95": wilson_95(int(correct), int(total))}


def score(predictions, donor, base, groups, g_tables=3):
    donor_correct = int((predictions == donor).sum())
    base_correct = int((predictions == base).sum())
    exact = sum(bool((predictions[i:i + g_tables] == donor[i:i + g_tables]).all())
                for i in range(0, len(predictions), g_tables))
    cross = [i for i in range(len(predictions)) if i % g_tables]
    fixed_avoid = sum(int(predictions[i]) != int(donor[(i // g_tables) * g_tables])
                      for i in cross)
    return {"donor_items": _rate(donor_correct, len(predictions)),
            "base_items": _rate(base_correct, len(predictions)),
            "donor_groups": _rate(exact, groups),
            "cross_g_non_injection": _rate(fixed_avoid, len(cross))}


def _patch_slot(net, ids, base_state, cut, slot, patch):
    edited = base_state.clone()
    edited[:, slot] = patch
    return continue_dense(net, ids, edited, cut).argmax(-1).cpu()


def same_key_episodes(groups):
    episodes = []
    for group in groups:
        tokens = group["base"][0]
        names = [tokens[2], tokens[4]]
        assigned = [tokens[3], tokens[5]]
        query = tokens[12]
        other_index = 1 if names[0] == query else 0
        replacement = None
        fallback = None
        for candidate in NAMES:
            if candidate in names:
                continue
            if fallback is None:
                fallback = candidate
            candidate_names = names.copy()
            candidate_names[other_index] = candidate
            if table_partitions(tuple(candidate_names), (tokens[7], tokens[9]),
                                (tokens[8], tokens[10]))[0] == "holdout":
                replacement = candidate
                break
        if replacement is None:
            replacement = fallback
        if replacement is None:
            raise RuntimeError("No same-key distractor replacement")
        names[other_index] = replacement
        episode = make_episode(tuple(names), tuple(assigned), (tokens[7], tokens[9]),
                               (tokens[8], tokens[10]), "composed", query)
        if episode.answer != group["base_answers"][0]:
            raise AssertionError("Same-key episode changed target")
        episodes.append(list(episode.tokens))
    return torch.tensor(episodes, dtype=torch.long)


@torch.no_grad()
def screen_model(net, groups, config):
    base_ids = _ids(groups, "base")
    donor_ref_ids = _ids(groups, "donor", first_only=True)
    _, base_states = manual_dense(net, base_ids)
    _, donor_states = manual_dense(net, donor_ref_ids)
    donor = _answers(groups, "donor_answers")
    base = _answers(groups, "base_answers")
    screen = {}
    for cut in range(5):
        for slot in range(6):
            patch = donor_states[cut][:, slot].repeat_interleave(config.g_tables, dim=0)
            predictions = _patch_slot(net, base_ids, base_states[cut], cut, slot, patch)
            screen[f"{cut}:{slot}"] = score(predictions, donor, base, len(groups),
                                             config.g_tables)
    return screen


def select_site(screens):
    candidates = []
    for cut in range(5):
        for slot in range(6):
            key = f"{cut}:{slot}"
            group_min = min(screen[key]["donor_groups"]["accuracy"] for screen in screens.values())
            item_min = min(screen[key]["donor_items"]["accuracy"] for screen in screens.values())
            cross_min = min(screen[key]["cross_g_non_injection"]["accuracy"]
                            for screen in screens.values())
            candidates.append((cut, slot, group_min, item_min, cross_min))
    qualified = [row for row in candidates if row[2] >= .65 and row[4] >= .90]
    if qualified:
        earliest = min(row[0] for row in qualified)
        pool = [row for row in qualified if row[0] == earliest]
        chosen = sorted(pool, key=lambda row: (-row[2], -row[3], row[1]))[0]
        return {"cut": chosen[0], "slot": chosen[1], "discovery_qualified": True,
                "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
                "minimum_cross_g_non_injection": chosen[4]}
    chosen = sorted(candidates, key=lambda row: (-row[2], -row[3], row[0], row[1]))[0]
    return {"cut": chosen[0], "slot": chosen[1], "discovery_qualified": False,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
            "minimum_cross_g_non_injection": chosen[4]}


@torch.no_grad()
def confirm_model(net, discovery, confirmation, site, rep, config):
    cut, slot = site["cut"], site["slot"]
    base_ids = _ids(confirmation, "base")
    donor_ids = _ids(confirmation, "donor")
    donor_ref_ids = _ids(confirmation, "donor", first_only=True)
    base_one_ids = _ids(confirmation, "base", first_only=True)
    discovery_base_ids = _ids(discovery, "base", first_only=True)
    discovery_donor_ids = _ids(discovery, "donor", first_only=True)
    same_ids = same_key_episodes(confirmation)
    native_logits = net(base_ids)[0]
    manual_logits, base_states = manual_dense(net, base_ids)
    donor_logits, _ = manual_dense(net, donor_ids)
    _, donor_ref_states = manual_dense(net, donor_ref_ids)
    _, base_one_states = manual_dense(net, base_one_ids)
    _, same_states = manual_dense(net, same_ids)
    _, discovery_base_states = manual_dense(net, discovery_base_ids)
    _, discovery_donor_states = manual_dense(net, discovery_donor_ids)
    numerical_error = float((native_logits - manual_logits).abs().max())
    donor_targets = _answers(confirmation, "donor_answers")
    base_targets = _answers(confirmation, "base_answers")
    clean_base = manual_logits.argmax(-1).cpu()
    clean_donor = donor_logits.argmax(-1).cpu()
    donor_patch = donor_ref_states[cut][:, slot].repeat_interleave(config.g_tables, dim=0)
    selected = _patch_slot(net, base_ids, base_states[cut], cut, slot, donor_patch)
    generator = torch.Generator().manual_seed(config.random_state_seed + rep)
    random_patch = torch.randn(donor_patch.shape, generator=generator)
    random_patch = random_patch / random_patch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_patch *= donor_patch.norm(dim=-1, keepdim=True)
    random_predictions = _patch_slot(net, base_ids, base_states[cut], cut, slot, random_patch)
    same_patch = same_states[cut][:, slot].repeat_interleave(config.g_tables, dim=0)
    same_predictions = _patch_slot(net, base_ids, base_states[cut], cut, slot, same_patch)
    neighbor = slot - 1 if slot else 1
    neighbor_patch = donor_ref_states[cut][:, neighbor].repeat_interleave(
        config.g_tables, dim=0)
    neighbor_predictions = _patch_slot(
        net, base_ids, base_states[cut], cut, neighbor, neighbor_patch)
    f_positive_state = base_states[0].clone()
    donor_f = donor_ref_states[0][:, :2].repeat_interleave(config.g_tables, dim=0)
    f_positive_state[:, :2] = donor_f
    f_positive = continue_dense(net, base_ids, f_positive_state, 0).argmax(-1).cpu()
    all_patch_state = base_states[cut].clone()
    all_patch_state[:] = donor_ref_states[cut].repeat_interleave(config.g_tables, dim=0)
    all_patch = continue_dense(net, base_ids, all_patch_state, cut).argmax(-1).cpu()
    discovery_states = torch.cat((discovery_base_states[cut][:, slot],
                                  discovery_donor_states[cut][:, slot]))
    discovery_labels = torch.cat((_single(discovery, "base_key"),
                                  _single(discovery, "donor_key")))
    geometry = key_centroid_basis(discovery_states, discovery_labels)
    confirmation_base = base_one_states[cut][:, slot]
    confirmation_donor = donor_ref_states[cut][:, slot]
    confirmation_states = torch.cat((confirmation_base, confirmation_donor))
    confirmation_labels = torch.cat((_single(confirmation, "base_key"),
                                     _single(confirmation, "donor_key")))
    centroid_accuracy, _ = nearest_centroid_accuracy(
        confirmation_states, confirmation_labels, geometry["centroids"], geometry["keys"])
    base_slot = confirmation_base.repeat_interleave(config.g_tables, dim=0)
    donor_slot = confirmation_donor.repeat_interleave(config.g_tables, dim=0)
    delta = donor_slot - base_slot
    basis = geometry["basis"]
    projection = delta @ basis @ basis.T
    key_span = _patch_slot(net, base_ids, base_states[cut], cut, slot, base_slot + projection)
    complement = _patch_slot(net, base_ids, base_states[cut], cut, slot,
                             base_slot + delta - projection)
    dimension_scores = {}
    for dimension in sorted({1, 2, 4, 8, geometry["rank"]}):
        used = basis[:, :min(dimension, geometry["rank"])]
        pred = _patch_slot(net, base_ids, base_states[cut], cut, slot,
                           base_slot + delta @ used @ used.T)
        dimension_scores[str(min(dimension, geometry["rank"]))] = score(
            pred, donor_targets, base_targets, len(confirmation), config.g_tables)
    subspace_generator = torch.Generator().manual_seed(config.random_subspace_seed + rep)
    random_subspaces = []
    for _ in range(config.random_subspaces):
        random_basis = _haar_basis(128, geometry["rank"], subspace_generator)
        pred = _patch_slot(net, base_ids, base_states[cut], cut, slot,
                           base_slot + delta @ random_basis @ random_basis.T)
        random_subspaces.append(float((pred == donor_targets).float().mean()))
    leverage = basis.square().sum(-1)
    order = torch.argsort(leverage, descending=True)
    native_scores = {}
    for size in sorted({geometry["rank"], min(2 * geometry["rank"], 128), 32, 128}):
        mask = torch.zeros(128)
        mask[order[:size]] = 1
        pred = _patch_slot(net, base_ids, base_states[cut], cut, slot, base_slot + delta * mask)
        native_scores[str(size)] = score(pred, donor_targets, base_targets,
                                         len(confirmation), config.g_tables)
    rotated = []
    for _ in range(config.rotations):
        rotation = _haar_rotation(128, subspace_generator)
        rotated_basis = rotation.T @ basis
        rotated_order = torch.argsort(rotated_basis.square().sum(-1), descending=True)
        mask = torch.zeros(128)
        mask[rotated_order[:geometry["rank"]]] = 1
        patch = base_slot + (delta @ rotation * mask) @ rotation.T
        pred = _patch_slot(net, base_ids, base_states[cut], cut, slot, patch)
        rotated.append(float((pred == donor_targets).float().mean()))
    conditions = {"clean_base": clean_base, "clean_donor": clean_donor,
                  "selected": selected, "random": random_predictions,
                  "same_key": same_predictions, "neighbor": neighbor_predictions,
                  "f_positive": f_positive, "all_slots": all_patch,
                  "key_span": key_span, "complement": complement}
    scores = {name: score(pred, donor_targets, base_targets, len(confirmation), config.g_tables)
              for name, pred in conditions.items()}
    metrics = {"numerical_error": numerical_error, "neighbor_slot": neighbor,
               "rank": geometry["rank"], "singular_values": geometry["singular_values"].tolist(),
               "nearest_centroid_confirmation_accuracy": centroid_accuracy,
               "scores": scores, "dimension_scores": dimension_scores,
               "random_subspace_accuracies": random_subspaces,
               "random_subspace_p95": _percentile95(random_subspaces),
               "native_neuron_scores": native_scores,
               "leverage_participation_ratio": float(leverage.sum().square()
                                                     / leverage.square().sum()),
               "rotated_top_rank_accuracies": rotated,
               "rotated_top_rank_p95": _percentile95(rotated),
               "top_coordinates": order[:32].tolist()}
    rows = [{"group": group, "g_index": g,
             "base_answer": int(base_targets[group * config.g_tables + g]),
             "donor_answer": int(donor_targets[group * config.g_tables + g]),
             **{f"{name}_prediction": int(pred[group * config.g_tables + g])
                for name, pred in conditions.items()}}
            for group in range(len(confirmation)) for g in range(config.g_tables)]
    return metrics, rows


def decide(site, metrics):
    site_clauses, subspace_clauses = {}, {}
    for rep, row in metrics.items():
        scores = row["scores"]
        site_clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "selected_items_at_least_0.80": scores["selected"]["donor_items"]["accuracy"] >= .80,
            "selected_groups_at_least_0.65": scores["selected"]["donor_groups"]["accuracy"] >= .65,
            "selected_non_injection_at_least_0.90": scores["selected"][
                "cross_g_non_injection"]["accuracy"] >= .90,
            "random_advantage_at_least_0.40": scores["selected"]["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40,
            "same_key_preserves_base_at_least_0.90": scores["same_key"]["base_items"][
                "accuracy"] >= .90,
            "numerical_error_below_1e-6": row["numerical_error"] < 1e-6,
        }
        subspace_clauses[rep] = {
            "rank_at_most_11": row["rank"] <= 11,
            "key_span_items_at_least_0.75": scores["key_span"]["donor_items"]["accuracy"] >= .75,
            "key_span_groups_at_least_0.60": scores["key_span"]["donor_groups"]["accuracy"] >= .60,
            "complement_preserves_base_at_least_0.90": scores["complement"]["base_items"][
                "accuracy"] >= .90,
            "complement_donor_at_most_0.10": scores["complement"]["donor_items"][
                "accuracy"] <= .10,
            "random_p95_advantage_at_least_0.40": scores["key_span"]["donor_items"]["accuracy"]
                - row["random_subspace_p95"] >= .40,
            "nearest_centroid_at_least_0.90": row[
                "nearest_centroid_confirmation_accuracy"] >= .90,
        }
    site_pass = site["discovery_qualified"] and all(
        all(row.values()) for row in site_clauses.values())
    subspace_pass = site_pass and all(all(row.values()) for row in subspace_clauses.values())
    return {"emergent_key_site": "supported" if site_pass else "not_supported",
            "emergent_key_site_clauses": site_clauses,
            "dense_causal_subspace": "supported" if subspace_pass else "not_supported",
            "dense_causal_subspace_clauses": subspace_clauses}


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
            raise RuntimeError("TEACH-0007 checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net = DenseRows()
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        models[str(rep)] = net
        screens[str(rep)] = screen_model(net, discovery, config)
        artifacts[str(rep)] = {"checkpoint_sha256": expected}
    site = select_site(screens)
    metrics = {}
    for rep in range(2):
        row_metrics, rows = confirm_model(models[str(rep)], discovery, confirmation,
                                          site, rep, config)
        metrics[str(rep)] = row_metrics
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        artifacts[str(rep)].update({"rows_sha256": sha_file(path), "rows": len(rows)})
    verdict = decide(site, metrics)
    if time.monotonic() - start > config.max_seconds:
        raise RuntimeError("TEACH-0007 time cap exceeded")
    report = {"experiment": "TEACH-0007", "status": "complete", "config": asdict(config),
              **provenance, "suite_sha256": suite_sha, "site": site,
              "discovery_screen": screens, "metrics": metrics, "artifacts": artifacts,
              "decision": verdict, "elapsed_seconds": time.monotonic() - start,
              "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0007"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"], "site": report["site"],
                      "decision": report["decision"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
