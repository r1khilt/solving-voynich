"""Frozen, bounded TEACH-0002 curriculum comparison; no run on import."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer
from voynich.workspace.teacher2_tasks import (
    KEYS, OBJECTS, SEQUENCE_LENGTH, VOCAB_SIZE, Episode, evaluation_suite,
    family_signatures, symbolic_oracle, training_batch,
)


@dataclass(frozen=True)
class Config:
    init_seeds: tuple[int, int] = (62111, 62121)
    train_seeds: tuple[int, int] = (62211, 62221)
    eval_seed: int = 62311
    steps_per_arm: int = 5000
    batch_size: int = 128
    eval_size: int = 256
    learning_rate: float = 3e-4
    max_seconds: float = 1800.0
    max_mps_bytes: int = 8 * 1024**3

    def validate(self) -> None:
        if self.init_seeds != (62111, 62121) or self.train_seeds != (62211, 62221) or \
                self.eval_seed != 62311:
            raise ValueError("Frozen TEACH-0002 seeds changed")
        if self.steps_per_arm != 5000 or self.batch_size != 128 or self.eval_size != 256:
            raise ValueError("Frozen TEACH-0002 update/batch/evaluation sizes changed")
        if self.learning_rate != 3e-4 or not 0 < self.max_seconds <= 1800:
            raise ValueError("Frozen learning rate or time ceiling changed")
        if not 0 < self.max_mps_bytes <= 8 * 1024**3:
            raise ValueError("Memory ceiling exceeded")


class BudgetExceeded(RuntimeError):
    pass


def model() -> VoynichTransformer:
    return VoynichTransformer(ModelConfig(
        vocab_size=VOCAB_SIZE, pad_id=0, d_model=128, n_layers=4, n_heads=4,
        d_ff=256, context_length=SEQUENCE_LENGTH, dropout=0.0,
    ))


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def wilson_95(correct: int, total: int) -> list[float]:
    if not 0 <= correct <= total or total <= 0:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [0.0 if correct == 0 else max(0.0, center - radius),
            1.0 if correct == total else min(1.0, center + radius)]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance(config: Config) -> dict:
    root = Path(__file__).resolve().parents[3]
    paths = ["docs/experiments/TEACH-0002.md",
             "src/voynich/workspace/teacher2_tasks.py",
             "src/voynich/workspace/teacher2_train.py", "src/voynich/model.py"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=root,
                            capture_output=True, text=True, check=True, timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0002 sources before training: {status}")
    cfg_bytes = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
    return {"source_git_head": head, "source_sha256": {path: sha_file(root / path) for path in paths},
            "config_sha256": sha_bytes(cfg_bytes), "source_worktree_status": status}


def check_budget(config: Config, start: float, device: str, resource: dict) -> None:
    if device == "mps":
        torch.mps.synchronize()
        used = torch.mps.current_allocated_memory()
        resource["peak_sampled_mps_allocated_bytes"] = max(
            resource["peak_sampled_mps_allocated_bytes"], used)
        if used > config.max_mps_bytes:
            raise BudgetExceeded("Sampled MPS allocation exceeded 8 GiB")
    if time.monotonic() - start > config.max_seconds:
        raise BudgetExceeded("TEACH-0002 1800-second ceiling reached")


def task_loss(logits: torch.Tensor, episodes: list[Episode], targets: list[int],
              device: str) -> torch.Tensor:
    """Query marker fixes output type; compare only the 12 legal symbols of that type."""
    key_mask = torch.tensor([ep.task == "first_hop" for ep in episodes],
                            dtype=torch.bool, device=device)
    all_targets = torch.tensor(targets, dtype=torch.long, device=device)
    total = logits.new_zeros(())
    if bool(key_mask.any().item()):
        total = total + F.cross_entropy(
            logits[key_mask, KEYS[0]:KEYS[-1] + 1],
            all_targets[key_mask] - KEYS[0], reduction="sum")
    if bool((~key_mask).any().item()):
        total = total + F.cross_entropy(
            logits[~key_mask, OBJECTS[0]:OBJECTS[-1] + 1],
            all_targets[~key_mask] - OBJECTS[0], reduction="sum")
    return total / len(episodes)


def numerical_qualification(device: str) -> dict:
    torch.manual_seed(62999)
    net = model().to(device)
    episodes = []
    labels = []
    for step in (0, 1800, 3800):
        batch, targets = training_batch(62990 + step, 16, arm="curriculum", step=step)
        episodes.extend(batch)
        labels.extend(targets)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    net.eval()
    with torch.no_grad():
        plain = net(ids).logits[:, -1]
        hooked = net(ids, cache_names=["blocks.1.resid_post"]).logits[:, -1]
        error = float((plain - hooked).abs().max().item())
    net.train()
    loss = task_loss(net(ids).logits[:, -1], episodes, labels, device)
    loss.backward()
    finite_grad = all(p.grad is None or bool(torch.isfinite(p.grad).all().item())
                      for p in net.parameters())
    if error > .002 or not math.isfinite(float(loss.item())) or not finite_grad:
        raise RuntimeError("TEACH-0002 numerical qualification failed")
    del net
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return {"max_cache_logit_error": error, "finite_loss": True, "finite_gradients": True}


def train_arm(config: Config, replicate: int, arm: str, device: str, start: float,
              output_dir: Path, status: dict, status_path: Path, resource: dict
              ) -> tuple[VoynichTransformer, dict]:
    torch.manual_seed(config.init_seeds[replicate])
    net = model().to(device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=.01)
    history = []
    losses = []
    for step in range(config.steps_per_arm):
        check_budget(config, start, device, resource)
        episodes, targets = training_batch(config.train_seeds[replicate] + step,
                                           config.batch_size, arm=arm, step=step)
        ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(net(ids).logits[:, -1], episodes, targets, device)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Nonfinite {arm} loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"Nonfinite {arm} gradient at step {step}")
        optimizer.step()
        check_budget(config, start, device, resource)
        losses.append(float(loss.item()))
        if (step + 1) % 250 == 0:
            last100 = sum(losses[-100:]) / 100
            history.append({"step": step + 1, "last_100_loss": last100})
            status["progress"] = {
                "replicate": replicate, "arm": arm, "step": step + 1,
                "elapsed_seconds": time.monotonic() - start,
                "latest_loss": losses[-1], "last_100_loss": last100,
                **resource,
            }
            status.update(resource)
            write_json(status_path, status)
    checkpoint = output_dir / f"rep{replicate}-{arm}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "config": asdict(config), "replicate": replicate, "arm": arm}, temporary)
    temporary.replace(checkpoint)
    return net, {"steps": config.steps_per_arm, "history": history,
                 "checkpoint_sha256": sha_file(checkpoint), "parameters": net.parameter_count}


@torch.no_grad()
def evaluate(net: VoynichTransformer, suite: dict[str, list[Episode]], device: str,
             replicate: int, arm: str, result_dir: Path) -> tuple[dict, str]:
    net.eval()
    predictions = {}
    scores = {}
    for name, episodes in suite.items():
        rows = []
        for offset in range(0, len(episodes), 256):
            chunk = episodes[offset:offset + 256]
            ids = torch.tensor([ep.tokens for ep in chunk], dtype=torch.long, device=device)
            logits = net(ids).logits[:, -1]
            for i, ep in enumerate(chunk):
                choices = KEYS if ep.task == "first_hop" else OBJECTS
                prediction = choices[int(logits[i, choices[0]:choices[-1] + 1].argmax().item())]
                f_family, g_family = family_signatures(ep)
                candidates = (ep.tokens[3], ep.tokens[5]) if ep.task == "first_hop" else (
                    ep.tokens[8], ep.tokens[10])
                rows.append({"index": offset + i, "tokens": ep.tokens, "answer": ep.answer,
                             "prediction": prediction, "task": ep.task,
                             "f_family": f_family, "g_family": g_family,
                             "candidate_member": prediction in candidates})
        correct = sum(row["prediction"] == row["answer"] for row in rows)
        summary = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                   "wilson_95": wilson_95(correct, len(rows)),
                   "candidate_member": sum(row["candidate_member"] for row in rows)}
        if name in ("factorial", "first_hop_pairs_holdout", "direct_pairs_holdout"):
            width = 4 if name == "factorial" else 2
            groups = [rows[i:i + width] for i in range(0, len(rows), width)]
            exact = sum(all(row["prediction"] == row["answer"] for row in group)
                        for group in groups)
            summary.update({"exact_groups": exact, "total_groups": len(groups),
                            "group_accuracy": exact / len(groups),
                            "group_wilson_95": wilson_95(exact, len(groups)),
                            "prediction_reversals": sum(
                                group[0]["prediction"] != group[1]["prediction"]
                                for group in groups)})
            if width == 4:
                summary["f_swap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 1), (2, 3)))
                summary["g_remap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 2), (1, 3)))
                summary["f_swap_pair_exact"] = sum(
                    all(group[i]["prediction"] == group[i]["answer"] for i in pair)
                    for group in groups for pair in ((0, 1), (2, 3)))
                summary["g_remap_pair_exact"] = sum(
                    all(group[i]["prediction"] == group[i]["answer"] for i in pair)
                    for group in groups for pair in ((0, 2), (1, 3)))
        predictions[name] = rows
        scores[name] = summary
    raw = json.dumps(predictions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    archive = gzip.compress(raw, compresslevel=9, mtime=0)
    path = result_dir / f"predictions-rep{replicate}-{arm}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(path)
    return scores, sha_file(path)


def decide(scores: dict) -> dict:
    clauses = {}
    for replicate in ("0", "1"):
        data = scores[replicate]
        c, b, n = data["curriculum"], data["baseline"], data["null"]
        both = c["composed_holdout_holdout"]["accuracy"]
        clauses[replicate] = {
            "first_hop_holdout_at_least_0.90": c["first_hop_holdout"]["accuracy"] >= .90,
            "first_hop_paired_exact_at_least_0.80": (
                c["first_hop_pairs_holdout"]["group_accuracy"] >= .80),
            "direct_train_and_holdout_at_least_0.90": all(
                c[f"direct_{part}"]["accuracy"] >= .90 for part in ("train", "holdout")),
            "direct_paired_exact_at_least_0.80": c["direct_pairs_holdout"]["group_accuracy"] >= .80,
            "both_holdout_composed_at_least_0.85": both >= .85,
            "crossed_holdout_composed_each_at_least_0.85": all(
                c[name]["accuracy"] >= .85 for name in
                ("composed_holdout_train", "composed_train_holdout")),
            "factorial_exact_at_least_0.65": c["factorial"]["group_accuracy"] >= .65,
            "copy_at_least_0.95": c["copy"]["accuracy"] >= .95,
            "composed_gain_vs_baseline_at_least_0.20": (
                both - b["composed_holdout_holdout"]["accuracy"] >= .20),
            "factorial_gain_vs_baseline_at_least_0.20": (
                c["factorial"]["group_accuracy"] - b["factorial"]["group_accuracy"] >= .20),
            "composed_gain_vs_null_at_least_0.40": (
                both - n["composed_holdout_holdout"]["accuracy"] >= .40),
            "null_factorial_at_most_0.10": n["factorial"]["group_accuracy"] <= .10,
        }
    qualified = all(all(row.values()) for row in clauses.values())
    return {"verdict": "qualified" if qualified else "not_qualified", "clauses_by_replicate": clauses}


def run(config: Config, output_dir: Path, result_dir: Path, device: str) -> dict:
    config.validate()
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("Local CPU or available MPS required")
    provenance = source_provenance(config)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(.16)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0002", "status": "running", "device": device,
              "torch_version": torch.__version__, "platform": platform.platform(),
              "machine": platform.machine(), "config": asdict(config),
              **provenance, **resource}
    status_path = result_dir / "status.json"
    write_json(status_path, status)
    try:
        status["numerical_qualification"] = numerical_qualification(device)
        check_budget(config, start, device, resource)
        suite = evaluation_suite(config.eval_seed, config.eval_size)
        if any(symbolic_oracle(ep.tokens) != ep.answer for rows in suite.values() for ep in rows):
            raise AssertionError("Evaluation oracle mismatch")
        suite_bytes = json.dumps({k: [(ep.tokens, ep.answer) for ep in rows]
                                  for k, rows in suite.items()}, sort_keys=True).encode()
        status["evaluation_suite_sha256"] = sha_bytes(suite_bytes)
        summaries = {}
        arm_records = {}
        for replicate in range(2):
            key = str(replicate)
            summaries[key], arm_records[key] = {}, {}
            for arm in ("baseline", "curriculum", "null"):
                net, record = train_arm(config, replicate, arm, device, start,
                                        output_dir, status, status_path, resource)
                check_budget(config, start, device, resource)
                summary, prediction_sha = evaluate(net, suite, device, replicate, arm, result_dir)
                check_budget(config, start, device, resource)
                record["predictions_sha256"] = prediction_sha
                arm_records[key][arm] = record
                summaries[key][arm] = summary
                del net
                if device == "mps":
                    torch.mps.empty_cache()
                status["completed_arms"] = [(int(rep), name) for rep, row in arm_records.items()
                                            for name in row]
                status.update(resource)
                write_json(status_path, status)
        decision = decide(summaries)
        status.update({"status": "complete", "scores": summaries, "arms": arm_records,
                       "decision": decision, "elapsed_seconds": time.monotonic() - start,
                       **resource})
        write_json(result_dir / "report.json", status)
        write_json(status_path, {"experiment": "TEACH-0002", "status": "complete",
                                 "decision": decision, "report": "report.json",
                                 "source_git_head": provenance["source_git_head"], **resource})
        return status
    except (BudgetExceeded, RuntimeError, AssertionError) as exc:
        status.update({"status": "stopped", "reason": type(exc).__name__ + ": " + str(exc),
                       "decision": {"verdict": "incomplete"},
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/TEACH-0002"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0002"))
    args = parser.parse_args()
    result = run(Config(), args.output_dir, args.result_dir, args.device)
    print(json.dumps({"status": result["status"], "decision": result["decision"]["verdict"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
