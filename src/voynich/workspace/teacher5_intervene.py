"""Frozen TEACH-0005 cross-G interchange interventions; no run on import."""

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
from .teacher4_tasks import KEYS, NAMES, OBJECTS, make_episode, symbolic_oracle
from .teacher4_tasks import table_partitions


SOURCE_PATHS = (
    "docs/experiments/TEACH-0005.md",
    "src/voynich/workspace/teacher5_intervene.py",
    "src/voynich/workspace/teacher4_models.py",
    "src/voynich/workspace/teacher4_tasks.py",
)


@dataclass(frozen=True)
class Config:
    suite_seed: int = 65111
    random_seed: int = 65211
    groups: int = 128
    g_tables: int = 3
    max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0005 configuration changed")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ordered_pair(first, second, reverse):
    return (second, first) if reverse else (first, second)


def build_groups(seed=65111, count=128):
    """Construct fresh holdout F groups with three independently remapped holdout Gs."""
    rng = random.Random(seed)
    groups, signatures = [], set()
    attempts = 0
    while len(groups) < count and attempts < 100_000:
        attempts += 1
        n0, n1, n2 = rng.sample(NAMES, 3)
        k0, k1, k2 = rng.sample(KEYS, 3)
        base_names = _ordered_pair(n0, n1, bool(rng.randrange(2)))
        base_map = {n0: k0, n1: k1}
        donor_map = {n0: k1, n1: k0}
        base_assigned = tuple(base_map[name] for name in base_names)
        donor_assigned = tuple(donor_map[name] for name in base_names)
        if table_partitions(base_names, (k0, k1), OBJECTS[:2])[0] != "holdout":
            continue
        same_names = _ordered_pair(n0, n2, bool(rng.randrange(2)))
        same_map = {n0: k0, n2: k2}
        same_assigned = tuple(same_map[name] for name in same_names)
        if table_partitions(same_names, (k0, k2), OBJECTS[:2])[0] != "holdout":
            continue
        available = list(OBJECTS)
        g_rows = []
        for _ in range(3):
            for _ in range(1_000):
                pair = tuple(rng.sample(available, 2))
                table_keys = _ordered_pair(k0, k1, bool(rng.randrange(2)))
                output_map = {k0: pair[0], k1: pair[1]}
                table_outputs = tuple(output_map[key] for key in table_keys)
                if table_partitions(base_names, table_keys, table_outputs)[1] == "holdout":
                    g_rows.append((table_keys, table_outputs))
                    available.remove(pair[0])
                    available.remove(pair[1])
                    break
            else:
                break
        if len(g_rows) != 3:
            continue
        same_keys = same_outputs = None
        for _ in range(1_000):
            pair = tuple(rng.sample(available, 2))
            candidate_keys = _ordered_pair(k0, k2, bool(rng.randrange(2)))
            output_map = {k0: pair[0], k2: pair[1]}
            candidate_outputs = tuple(output_map[key] for key in candidate_keys)
            if table_partitions(same_names, candidate_keys, candidate_outputs)[1] == "holdout":
                same_keys, same_outputs = candidate_keys, candidate_outputs
                break
        if same_keys is None:
            continue
        six_outputs = tuple(output for _, table_outputs in g_rows for output in table_outputs)
        signature = (tuple(sorted(base_names)), tuple(sorted((k0, k1))), six_outputs)
        if signature in signatures:
            continue
        signatures.add(signature)
        base, donor, direct, copy = [], [], [], []
        for table_keys, table_outputs in g_rows:
            base.append(make_episode(base_names, base_assigned, table_keys, table_outputs,
                                     "composed", n0))
            donor.append(make_episode(base_names, donor_assigned, table_keys, table_outputs,
                                      "composed", n0))
            direct.append(make_episode(base_names, base_assigned, table_keys, table_outputs,
                                       "direct", k0))
            copy.append(make_episode(base_names, base_assigned, table_keys, table_outputs,
                                     "copy", table_outputs[0]))
        same_key = make_episode(same_names, same_assigned, same_keys, same_outputs,
                                "first_hop", n0)
        group = {
            "base": [list(ep.tokens) for ep in base],
            "donor": [list(ep.tokens) for ep in donor],
            "direct": [list(ep.tokens) for ep in direct],
            "copy": [list(ep.tokens) for ep in copy],
            "same_key": list(same_key.tokens),
            "base_answers": [ep.answer for ep in base],
            "donor_answers": [ep.answer for ep in donor],
            "direct_answers": [ep.answer for ep in direct],
            "copy_answers": [ep.answer for ep in copy],
            "base_key": k0,
            "donor_key": k1,
            "query": n0,
        }
        if any(symbolic_oracle(tuple(tokens)) != answer for field, answers in (
                ("base", group["base_answers"]), ("donor", group["donor_answers"]),
                ("direct", group["direct_answers"]), ("copy", group["copy_answers"]))
                for tokens, answer in zip(group[field], answers, strict=True)):
            raise AssertionError("Generated oracle mismatch")
        if symbolic_oracle(tuple(group["same_key"])) != k0:
            raise AssertionError("Same-key donor changed selected key")
        if len(set(group["base_answers"] + group["donor_answers"])) != 6:
            raise AssertionError("Cross-G outputs must all be distinct")
        groups.append(group)
    if len(groups) != count:
        raise RuntimeError(f"Generated only {len(groups)} of {count} groups")
    return groups


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0005 sources before analysis: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status,
            "config_sha256": hashlib.sha256(stable_json(asdict(config)).encode()).hexdigest()}


def _states(net, ids, *, first_override=None, attention_override=None,
            second_override=None):
    f_rows, g_rows, marker, query = net.interface(ids)
    q = query + marker
    f_values = net.f_value(f_rows)
    f_scores = (net.f_key(f_rows) * net.f_query(q)[:, None]).sum(-1) / math.sqrt(q.shape[-1])
    attention = F.softmax(f_scores, dim=-1)
    used_attention = attention if attention_override is None else attention_override
    first = (used_attention[:, :, None] * f_values).sum(1)
    if first_override is not None:
        first = first_override
    g_query = net.g_query(torch.cat((q, first), dim=-1))
    g_scores = (net.g_key(g_rows) * g_query[:, None]).sum(-1) / math.sqrt(q.shape[-1])
    g_attention = F.softmax(g_scores, dim=-1)
    second = (g_attention[:, :, None] * net.g_value(g_rows)).sum(1)
    if second_override is not None:
        second = second_override
    logits = net.heads(ids, first, second, net.copy_state(q))
    return {"logits": logits, "first": first, "second": second,
            "f_attention": attention, "g_attention": g_attention}


def _predictions(logits):
    return logits.argmax(-1).tolist()


def _target_probabilities(logits, targets):
    probs = F.softmax(logits, dim=-1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return probs[rows, torch.tensor(targets, device=logits.device)].tolist()


def _condition(logits, targets):
    return {"prediction": _predictions(logits),
            "target_probability": _target_probabilities(logits, targets)}


def _flatten(groups, field):
    return [tokens for group in groups for tokens in group[field]]


def _flatten_answers(groups, field):
    return [answer for group in groups for answer in group[field]]


def _rate(correct, total):
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "wilson_95": wilson_95(correct, total)}


def summarize(rows):
    eligible = [row for row in rows if row["eligible_group"]]
    eligible_groups = sorted({row["group"] for row in eligible})
    total_groups = len({row["group"] for row in rows})
    clean_groups = len(eligible_groups)
    summary = {"clean_groups": _rate(clean_groups, total_groups)}
    if not eligible:
        return summary
    conditions = {
        "donor_first": ("donor_answer",), "random_first": ("donor_answer",),
        "zero_first": ("base_answer",), "same_key_first": ("base_answer",),
        "attention_only": ("base_answer",), "rescue_base_first": ("base_answer",),
        "full_second": ("donor_reference_answer",),
        "clean_base": ("base_answer",), "clean_donor": ("donor_answer",),
        "clean_direct": ("direct_answer",), "patched_direct": ("direct_answer",),
        "clean_copy": ("copy_answer",), "patched_copy": ("copy_answer",),
    }
    for condition, (target,) in conditions.items():
        correct = sum(row[condition]["prediction"] == row[target] for row in eligible)
        summary[condition] = _rate(correct, len(eligible))
    exact = 0
    for group in eligible_groups:
        group_rows = [row for row in eligible if row["group"] == group]
        exact += all(row["donor_first"]["prediction"] == row["donor_answer"]
                     for row in group_rows)
    summary["donor_first_groups"] = _rate(exact, len(eligible_groups))
    cross = [row for row in eligible if row["g_index"] > 0]
    non_injected = sum(row["donor_first"]["prediction"] != row["donor_reference_answer"]
                       for row in cross)
    summary["donor_first_cross_g_non_injection"] = _rate(non_injected, len(cross))
    deltas = [row["donor_first"]["donor_target_probability"]
              - row["clean_base"]["donor_target_probability"] for row in eligible]
    summary["donor_target_probability_delta"] = {
        "mean": sum(deltas) / len(deltas), "minimum": min(deltas), "maximum": max(deltas)}
    summary["zero_first_base_accuracy_drop"] = (
        summary["clean_base"]["accuracy"] - summary["zero_first"]["accuracy"])
    summary["donor_first_random_advantage"] = (
        summary["donor_first"]["accuracy"] - summary["random_first"]["accuracy"])
    summary["direct_accuracy_drop"] = (
        summary["clean_direct"]["accuracy"] - summary["patched_direct"]["accuracy"])
    summary["copy_accuracy_drop"] = (
        summary["clean_copy"]["accuracy"] - summary["patched_copy"]["accuracy"])
    return summary


def decision(summaries, numerical_max):
    clauses = {}
    for rep, summary in summaries.items():
        positive = summary.get("full_second", {}).get("accuracy", 0) >= .80
        competent = summary["clean_groups"]["accuracy"] >= .95
        row = {
            "clean_groups_at_least_0.95": competent,
            "donor_first_groups_at_least_0.80": summary.get("donor_first_groups", {}).get(
                "accuracy", 0) >= .80,
            "donor_first_items_at_least_0.90": summary.get("donor_first", {}).get(
                "accuracy", 0) >= .90,
            "advantage_over_random_at_least_0.40": summary.get(
                "donor_first_random_advantage", -1) >= .40,
            "probability_delta_at_least_0.40": summary.get(
                "donor_target_probability_delta", {}).get("mean", -1) >= .40,
            "cross_g_non_injection_at_least_0.90": summary.get(
                "donor_first_cross_g_non_injection", {}).get("accuracy", 0) >= .90,
            "full_second_positive_at_least_0.80": positive,
            "zero_necessity_drop_at_least_0.30": summary.get(
                "zero_first_base_accuracy_drop", -1) >= .30,
            "rescue_at_least_0.95": summary.get("rescue_base_first", {}).get(
                "accuracy", 0) >= .95,
            "same_key_at_least_0.90": summary.get("same_key_first", {}).get(
                "accuracy", 0) >= .90,
            "attention_only_at_least_0.90": summary.get("attention_only", {}).get(
                "accuracy", 0) >= .90,
            "direct_drop_at_most_0.05": summary.get("direct_accuracy_drop", 1) <= .05,
            "copy_drop_at_most_0.05": summary.get("copy_accuracy_drop", 1) <= .05,
            "numerical_error_below_1e-6": numerical_max[rep] < 1e-6,
        }
        clauses[rep] = row
    if not all(clauses[rep]["clean_groups_at_least_0.95"] and
               clauses[rep]["full_second_positive_at_least_0.80"] for rep in clauses):
        verdict = "inconclusive"
    else:
        verdict = "causal_key_supported" if all(all(row.values()) for row in clauses.values()) \
            else "not_supported"
    return {"verdict": verdict, "clauses_by_seed": clauses}


@torch.no_grad()
def analyze_checkpoint(net, groups, rep, config):
    device = next(net.parameters()).device
    base_ids = torch.tensor(_flatten(groups, "base"), dtype=torch.long, device=device)
    donor_ids = torch.tensor(_flatten(groups, "donor"), dtype=torch.long, device=device)
    direct_ids = torch.tensor(_flatten(groups, "direct"), dtype=torch.long, device=device)
    copy_ids = torch.tensor(_flatten(groups, "copy"), dtype=torch.long, device=device)
    same_ids = torch.tensor([group["same_key"] for group in groups], dtype=torch.long,
                            device=device)
    base_targets = _flatten_answers(groups, "base_answers")
    donor_targets = _flatten_answers(groups, "donor_answers")
    direct_targets = _flatten_answers(groups, "direct_answers")
    copy_targets = _flatten_answers(groups, "copy_answers")
    base = _states(net, base_ids)
    donor = _states(net, donor_ids)
    direct = _states(net, direct_ids)
    copy = _states(net, copy_ids)
    same = _states(net, same_ids)
    donor_reference_indices = torch.arange(0, len(groups) * config.g_tables,
                                           config.g_tables, device=device)
    donor_first = donor["first"][donor_reference_indices].repeat_interleave(
        config.g_tables, dim=0)
    donor_second = donor["second"][donor_reference_indices].repeat_interleave(
        config.g_tables, dim=0)
    same_first = same["first"].repeat_interleave(config.g_tables, dim=0)
    generator = torch.Generator(device="cpu").manual_seed(config.random_seed + rep)
    random_first = torch.randn(donor_first.shape, generator=generator).to(device)
    random_first = random_first / random_first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_first = random_first * donor_first.norm(dim=-1, keepdim=True)
    donor_attention = donor["f_attention"][donor_reference_indices].repeat_interleave(
        config.g_tables, dim=0)
    conditions = {
        "clean_base": base,
        "clean_donor": donor,
        "donor_first": _states(net, base_ids, first_override=donor_first),
        "random_first": _states(net, base_ids, first_override=random_first),
        "zero_first": _states(net, base_ids, first_override=torch.zeros_like(donor_first)),
        "same_key_first": _states(net, base_ids, first_override=same_first),
        "attention_only": _states(net, base_ids, attention_override=donor_attention),
        "rescue_base_first": _states(net, base_ids, first_override=base["first"]),
        "full_second": _states(net, base_ids, second_override=donor_second),
        "clean_direct": direct,
        "patched_direct": _states(net, direct_ids, first_override=donor_first),
        "clean_copy": copy,
        "patched_copy": _states(net, copy_ids, first_override=donor_first),
    }
    if not all(bool(torch.isfinite(value["logits"]).all()) for value in conditions.values()):
        raise RuntimeError("Nonfinite TEACH-0005 logits")
    numerical = float((conditions["rescue_base_first"]["logits"] - base["logits"]).abs().max())
    prediction = {name: _predictions(value["logits"]) for name, value in conditions.items()}
    probability = {}
    for name, value in conditions.items():
        targets = (donor_targets if name in ("clean_donor", "donor_first", "random_first")
                   else direct_targets if name in ("clean_direct", "patched_direct")
                   else copy_targets if name in ("clean_copy", "patched_copy")
                   else [group["donor_answers"][0] for group in groups
                         for _ in range(config.g_tables)] if name == "full_second"
                   else base_targets)
        probability[name] = _target_probabilities(value["logits"], targets)
    clean_base_donor_prob = _target_probabilities(base["logits"], donor_targets)
    rows = []
    for flat_index in range(len(base_targets)):
        group_index, g_index = divmod(flat_index, config.g_tables)
        group = groups[group_index]
        eligible = all(
            prediction[condition][group_index * config.g_tables + j] == target[j]
            for j in range(config.g_tables)
            for condition, target in (("clean_base", group["base_answers"]),
                                      ("clean_donor", group["donor_answers"])))
        row = {
            "group": group_index, "g_index": g_index, "eligible_group": eligible,
            "base_tokens": group["base"][g_index], "donor_tokens": group["donor"][g_index],
            "direct_tokens": group["direct"][g_index], "copy_tokens": group["copy"][g_index],
            "same_key_tokens": group["same_key"],
            "base_answer": group["base_answers"][g_index],
            "donor_answer": group["donor_answers"][g_index],
            "donor_reference_answer": group["donor_answers"][0],
            "direct_answer": group["direct_answers"][g_index],
            "copy_answer": group["copy_answers"][g_index],
        }
        for name in conditions:
            target_key = ("donor_answer" if name in ("clean_donor", "donor_first", "random_first")
                          else "direct_answer" if name in ("clean_direct", "patched_direct")
                          else "copy_answer" if name in ("clean_copy", "patched_copy")
                          else "donor_reference_answer" if name == "full_second"
                          else "base_answer")
            row[name] = {"prediction": prediction[name][flat_index],
                         "target_probability": probability[name][flat_index],
                         "target": row[target_key]}
        row["clean_base"]["donor_target_probability"] = clean_base_donor_prob[flat_index]
        row["donor_first"]["donor_target_probability"] = probability["donor_first"][flat_index]
        rows.append(row)
    return rows, numerical


def run(config, checkpoint_dir, result_dir):
    config.validate()
    provenance = source_provenance(config)
    report4 = json.loads((result_dir.parent / "TEACH-0004" / "report.json").read_text())
    groups = build_groups(config.suite_seed, config.groups)
    suite_sha = hashlib.sha256(stable_json(groups).encode()).hexdigest()
    start = time.monotonic()
    summaries, numerical, artifacts = {}, {}, {}
    for rep in range(2):
        checkpoint = checkpoint_dir / f"rep{rep}-two_read.pt"
        expected = report4["arms"][str(rep)]["two_read"]["checkpoint_sha256"]
        if sha_file(checkpoint) != expected:
            raise RuntimeError(f"TEACH-0004 checkpoint hash mismatch for seed {rep}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("replicate") != rep or payload.get("arm") != "two_read":
            raise RuntimeError("Checkpoint metadata mismatch")
        net = LearnedMemory(True)
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        rows, numerical[str(rep)] = analyze_checkpoint(net, groups, rep, config)
        path = result_dir / f"rows-rep{rep}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(stable_json(rows).encode(), compresslevel=9, mtime=0))
        summaries[str(rep)] = summarize(rows)
        artifacts[str(rep)] = {"checkpoint_sha256": expected, "rows_sha256": sha_file(path),
                               "rows": len(rows)}
        if time.monotonic() - start > config.max_seconds:
            raise RuntimeError("TEACH-0005 time cap exceeded")
    verdict = decision(summaries, numerical)
    report = {"experiment": "TEACH-0005", "status": "complete", "config": asdict(config),
              **provenance, "teaching_checkpoint_source": report4["source_git_head"],
              "suite_sha256": suite_sha, "artifacts": artifacts, "summaries": summaries,
              "numerical_max_recompute_error": numerical, "decision": verdict,
              "elapsed_seconds": time.monotonic() - start,
              "torch_version": torch.__version__}
    write_json(result_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0005"))
    args = parser.parse_args()
    report = run(Config(), args.checkpoint_dir, args.result_dir)
    print(json.dumps({"status": report["status"],
                      "decision": report["decision"]["verdict"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
