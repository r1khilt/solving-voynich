"""Bounded TEACH-0012 raw serialized variable-binding campaign."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

from .teacher12_models import ARMS, collate_rows, model_for_arm
from .teacher12_tasks import (
    SYMBOL_START, evaluation_suite, pad_batch, training_batch,
)
from .teacher2_train import wilson_95, write_json


EXPECTED_PARAMETERS = {
    "parsed_memory": 3_160_576,
    "raw_shallow": 14_693_376,
    "raw_looped": 14_694_912,
    "raw_deep": 41_965_568,
    "raw_null": 14_693_376,
}
GROUP_WIDTHS = {
    "first_hop_query_groups": 4,
    "direct_query_groups": 4,
    "factorial": 4,
    "order_groups": 4,
    "distractor_groups": 4,
    "boundary_groups": 4,
}
MILESTONES = (250, 500, 1000, 2000, 4000)
SOURCE_PATHS = (
    "docs/experiments/TEACH-0012.md",
    "docs/experiments/TEACH-0012-benchmark-gate-amendment.md",
    "src/voynich/workspace/teacher12_tasks.py",
    "src/voynich/workspace/teacher12_models.py",
    "src/voynich/workspace/teacher12_train.py",
    "scripts/teacher0012_analyze.py",
    "tests/test_workspace_teacher12.py",
    "src/voynich/model.py",
)


@dataclass(frozen=True)
class Config:
    init_seeds: tuple[int, int] = (72121, 72131)
    train_seeds: tuple[int, int] = (72221, 72231)
    eval_seed: int = 72311
    steps_per_arm: int = 8000
    batch_size: int = 64
    eval_groups: int = 128
    learning_rate: float = 3e-4
    max_seconds: float = 6 * 3600.0
    max_mps_bytes: int = 24 * 1024**3
    max_artifact_bytes: int = 4 * 1024**3
    benchmark_steps: int = 24
    benchmark_warmup_steps: int = 4
    benchmark_max_seconds: float = 1800.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0012 configuration changed")


class ResourceStop(RuntimeError):
    pass


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0012 sources before accelerator use: {status}")
    cfg = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "source_worktree_status": status, "config_sha256": sha_bytes(cfg)}


def artifact_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def resource_check(config, start, resource, output_dir, *, benchmark):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        used = torch.mps.current_allocated_memory()
        resource["peak_sampled_mps_allocated_bytes"] = max(
            resource["peak_sampled_mps_allocated_bytes"], used)
        if used > config.max_mps_bytes:
            raise ResourceStop("TEACH-0012 sampled MPS allocation exceeded 24 GiB")
    ceiling = config.benchmark_max_seconds if benchmark else config.max_seconds
    if time.monotonic() - start > ceiling:
        raise ResourceStop("TEACH-0012 wall-time ceiling reached")
    if Path(output_dir).exists() and artifact_size(output_dir) > config.max_artifact_bytes:
        raise ResourceStop("TEACH-0012 ignored artifact ceiling exceeded 4 GiB")


def new_model(config, arm, replicate, device):
    torch.manual_seed(config.init_seeds[replicate])
    net = model_for_arm(arm).to(device)
    if net.parameter_count != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"Parameter count drift for {arm}: {net.parameter_count}")
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=.01)
    return net, optimizer


def batch_tensors(episodes, device):
    padded, _ = pad_batch(episodes)
    return torch.tensor(padded, dtype=torch.long, device=device)


def model_output(net, arm, episodes, device):
    ids = batch_tensors(episodes, device)
    if arm == "parsed_memory":
        return net(ids, *collate_rows(episodes, device))
    return net(ids)


def answer_loss(logits, answers):
    targets = torch.tensor(answers, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits[:, SYMBOL_START:], targets - SYMBOL_START)


def one_update(net, optimizer, config, replicate, arm, step, device):
    episodes, answers = training_batch(
        config.train_seeds[replicate] + step, config.batch_size, step=step,
        null_composed=arm == "raw_null")
    optimizer.zero_grad(set_to_none=True)
    output = model_output(net, arm, episodes, device)
    loss = answer_loss(output.logits, answers)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"Nonfinite {arm} loss at step {step}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError(f"Nonfinite {arm} gradient at step {step}")
    optimizer.step()
    return float(loss.item())


def numerical_check(config, arm, device):
    net, optimizer = new_model(config, arm, 0, device)
    episodes, _ = training_batch(config.train_seeds[0], 4, step=4100,
                                 null_composed=arm == "raw_null")
    net.eval()
    with torch.no_grad():
        first = model_output(net, arm, episodes, device).logits
        second = model_output(net, arm, episodes, device).logits
        error = float((first - second).abs().max().item())
    net.train()
    loss = one_update(net, optimizer, config, 0, arm, 0, device)
    finite = math.isfinite(loss) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in net.parameters())
    del net, optimizer
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    if error >= 1e-6 or not finite:
        raise RuntimeError(f"Numerical qualification failed for {arm}")
    return {"max_recompute_logit_error": error, "finite_loss_and_gradients": finite}


def benchmark(config, result_dir, output_dir):
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for the TEACH-0012 benchmark")
    provenance = source_provenance(config)
    path = result_dir / "benchmark.json"
    if path.exists():
        raise FileExistsError("No automatic TEACH-0012 benchmark rerun")
    torch.mps.set_per_process_memory_fraction(.38)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    report = {"experiment": "TEACH-0012", "mode": "benchmark", "status": "running",
              "config": asdict(config), "device": "mps", "torch_version": torch.__version__,
              "platform": platform.platform(), "machine": platform.machine(),
              **provenance, **resource}
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    try:
        report["numerical_qualification"] = {
            arm: numerical_check(config, arm, "mps") for arm in ARMS}
        measured = {}
        for arm in ARMS:
            net, optimizer = new_model(config, arm, 0, "mps")
            timings = []
            for step in range(config.benchmark_steps):
                before = time.monotonic()
                one_update(net, optimizer, config, 0, arm, step, "mps")
                torch.mps.synchronize()
                elapsed = time.monotonic() - before
                if step >= config.benchmark_warmup_steps:
                    timings.append(elapsed)
                resource_check(config, start, resource, output_dir, benchmark=True)
            measured[arm] = {
                "parameters": net.parameter_count,
                "warmup_steps": config.benchmark_warmup_steps,
                "timed_steps": len(timings),
                "timed_step_seconds": timings,
                "median_timed_step_seconds": statistics.median(timings),
            }
            del net, optimizer
            torch.mps.synchronize()
            torch.mps.empty_cache()
        projected = sum(row["median_timed_step_seconds"] for row in measured.values()) \
            * config.steps_per_arm * len(config.init_seeds)
        conservative = projected * 1.5
        report.update({"status": "pass" if conservative <= config.max_seconds else "stop",
                       "measured": measured, "projected_campaign_seconds": projected,
                       "conservative_projected_seconds": conservative,
                       "elapsed_seconds": time.monotonic() - start, **resource})
        if report["status"] != "pass":
            report["reason"] = "Conservative campaign projection exceeded six hours"
    except Exception as exc:
        report.update({"status": "stop", "reason": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(path, report)
        raise
    write_json(path, report)
    return report


def check_benchmark(config, result_dir, provenance):
    path = result_dir / "benchmark.json"
    if not path.exists():
        raise RuntimeError("Passing TEACH-0012 benchmark required before run")
    report = json.loads(path.read_text())
    serialized_config = json.loads(json.dumps(asdict(config)))
    valid = (report.get("status") == "pass" and report.get("config") == serialized_config
             and report.get("source_git_head") == provenance["source_git_head"]
             and report.get("source_sha256") == provenance["source_sha256"]
             and report.get("config_sha256") == provenance["config_sha256"]
             and report.get("conservative_projected_seconds", math.inf) <= config.max_seconds
             and report.get("peak_sampled_mps_allocated_bytes", math.inf)
             <= config.max_mps_bytes)
    if valid:
        valid = set(report.get("measured", {})) == set(ARMS) and all(
            report["measured"][arm].get("parameters") == EXPECTED_PARAMETERS[arm]
            and report["measured"][arm].get("timed_steps") == 20 for arm in ARMS)
    if not valid:
        raise RuntimeError("TEACH-0012 benchmark did not qualify or source/config changed")
    return report


def save_checkpoint(net, config, replicate, arm, step, output_dir):
    path = output_dir / f"rep{replicate}-{arm}-step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "config": asdict(config), "replicate": replicate, "arm": arm,
                "step": step, "parameters": net.parameter_count}, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def train_arm(config, replicate, arm, device, start, output_dir, result_dir, resource,
              status, status_path):
    net, optimizer = new_model(config, arm, replicate, device)
    net.train()
    losses, history, milestones = [], [], {}
    for step in range(config.steps_per_arm):
        loss = one_update(net, optimizer, config, replicate, arm, step, device)
        losses.append(loss)
        completed = step + 1
        if arm == "raw_deep" and completed in MILESTONES:
            milestones[str(completed)] = save_checkpoint(
                net, config, replicate, arm, completed, output_dir)
        if completed % 250 == 0:
            history.append({"step": completed, "last_100_loss": sum(losses[-100:]) / 100,
                            "loss": loss})
            status["progress"] = {"replicate": replicate, "arm": arm,
                                  "step": completed, "elapsed_seconds": time.monotonic() - start,
                                  "latest_loss": loss, "last_100_loss": history[-1]["last_100_loss"],
                                  **resource}
            status.update(resource)
            write_json(status_path, status)
            resource_check(config, start, resource, output_dir, benchmark=False)
    final = save_checkpoint(net, config, replicate, arm, config.steps_per_arm, output_dir)
    loss_path = result_dir / f"losses-rep{replicate}-{arm}.json.gz"
    raw = json.dumps(losses, separators=(",", ":"), allow_nan=False).encode()
    loss_path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return net, {"history": history, "final_checkpoint": final, "milestones": milestones,
                 "loss_archive_sha256": sha_file(loss_path), "loss_count": len(losses),
                 "parameters": net.parameter_count}


def _candidate_set(episode):
    if episode.task == "copy":
        return {episode.query}
    return {right for _, right in episode.rows}


def score_rows(rows, name):
    total = len(rows)
    correct = sum(row["prediction"] == row["answer"] for row in rows)
    score = {"correct": correct, "total": total, "accuracy": correct / total,
             "wilson_95": wilson_95(correct, total),
             "candidate_member": sum(row["candidate_member"] for row in rows),
             "mean_target_probability": sum(row["target_probability"] for row in rows) / total}
    width = GROUP_WIDTHS.get(name)
    if width:
        groups = [rows[offset:offset + width] for offset in range(0, total, width)]
        exact = sum(all(row["prediction"] == row["answer"] for row in group)
                    for group in groups)
        consistent = sum(len({row["prediction"] for row in group}) == 1 for group in groups)
        score.update({"exact_groups": exact, "total_groups": len(groups),
                      "group_accuracy": exact / len(groups),
                      "consistent_groups": consistent,
                      "consistency": consistent / len(groups),
                      "mean_distinct_predictions": sum(len({row["prediction"] for row in group})
                                                       for group in groups) / len(groups)})
        if name == "boundary_groups":
            marker_free = [group[-1] for group in groups]
            score["marker_free_correct"] = sum(
                row["prediction"] == row["answer"] for row in marker_free)
            score["marker_free_accuracy"] = score["marker_free_correct"] / len(marker_free)
    return score


@torch.no_grad()
def evaluate(net, arm, suite, device, replicate, result_dir):
    net.eval()
    scores, all_rows = {}, {}
    for name, episodes in suite.items():
        rows = []
        for offset in range(0, len(episodes), 64):
            chunk = episodes[offset:offset + 64]
            logits = model_output(net, arm, chunk, device).logits[:, SYMBOL_START:]
            probabilities = logits.softmax(-1)
            prediction = logits.argmax(-1) + SYMBOL_START
            for index, episode in enumerate(chunk):
                target_index = episode.answer - SYMBOL_START
                rows.append({
                    "index": offset + index,
                    "tokens": episode.tokens,
                    "f_rows": episode.f_rows,
                    "g_rows": episode.g_rows,
                    "distractor_rows": episode.distractor_rows,
                    "serialized_rows": episode.serialized_rows,
                    "answer": episode.answer,
                    "prediction": int(prediction[index].item()),
                    "target_probability": float(probabilities[index, target_index].item()),
                    "task": episode.task,
                    "query": episode.query,
                    "f_partition": episode.f_partition,
                    "g_partition": episode.g_partition,
                    "logical_id": episode.logical_id,
                    "render_id": episode.render_id,
                    "distractor_chains": episode.distractor_chains,
                    "marker_dropout": episode.marker_dropout,
                    "length": len(episode.tokens),
                    "candidate_member": int(prediction[index].item()) in _candidate_set(episode),
                })
        scores[name] = score_rows(rows, name)
        all_rows[name] = rows
    raw = json.dumps(all_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path = result_dir / f"predictions-rep{replicate}-{arm}.json.gz"
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return scores, sha_file(path)


def decision(scores):
    clauses = {"primary": {}, "positive": {}, "null": {}}
    for replicate in ("0", "1"):
        primary = scores[replicate]["raw_deep"]
        clauses["primary"][replicate] = {
            "first_hop_at_least_0.95": primary["first_hop_confirm"]["accuracy"] >= .95,
            "direct_at_least_0.95": primary["direct_confirm"]["accuracy"] >= .95,
            "copy_at_least_0.98": primary["copy_confirm"]["accuracy"] >= .98,
            "all_composed_cells_at_least_0.90": all(
                primary[name]["accuracy"] >= .90 for name in primary if name.startswith("composed_")),
            "f_query_groups_at_least_0.85":
                primary["first_hop_query_groups"]["group_accuracy"] >= .85,
            "g_query_groups_at_least_0.85":
                primary["direct_query_groups"]["group_accuracy"] >= .85,
            "factorial_at_least_0.75": primary["factorial"]["group_accuracy"] >= .75,
            "order_groups_at_least_0.80": primary["order_groups"]["group_accuracy"] >= .80,
            "distractor_groups_at_least_0.80":
                primary["distractor_groups"]["group_accuracy"] >= .80,
            "boundary_groups_at_least_0.70":
                primary["boundary_groups"]["group_accuracy"] >= .70,
            "marker_free_at_least_0.80":
                primary["boundary_groups"]["marker_free_accuracy"] >= .80,
            "long_ood_at_least_0.80": primary["long_ood"]["accuracy"] >= .80,
        }
        positive = scores[replicate]["parsed_memory"]
        clauses["positive"][replicate] = {
            "ungrouped_at_least_0.95": all(
                row["accuracy"] >= .95 for name, row in positive.items()
                if name not in GROUP_WIDTHS),
            "query_groups_at_least_0.90": all(
                positive[name]["group_accuracy"] >= .90 for name in
                ("first_hop_query_groups", "direct_query_groups")),
            "factorial_at_least_0.85": positive["factorial"]["group_accuracy"] >= .85,
        }
        null = scores[replicate]["raw_null"]
        clauses["null"][replicate] = {
            "composed_confirm_confirm_at_most_0.35":
                null["composed_confirm_confirm"]["accuracy"] <= .35,
            "factorial_at_most_0.10": null["factorial"]["group_accuracy"] <= .10,
        }
    primary_pass = all(all(row.values()) for row in clauses["primary"].values())
    positive_pass = all(all(row.values()) for row in clauses["positive"].values())
    null_pass = all(all(row.values()) for row in clauses["null"].values())
    depth = {}
    looping = {}
    for replicate in ("0", "1"):
        deep = scores[replicate]["raw_deep"]
        shallow = scores[replicate]["raw_shallow"]
        looped = scores[replicate]["raw_looped"]
        depth[replicate] = {
            "confirm_confirm_gain_at_least_0.10": deep["composed_confirm_confirm"]["accuracy"]
            - shallow["composed_confirm_confirm"]["accuracy"] >= .10,
            "factorial_gain_at_least_0.10": deep["factorial"]["group_accuracy"]
            - shallow["factorial"]["group_accuracy"] >= .10,
        }
        looped_primary = {
            "first": looped["first_hop_confirm"]["accuracy"] >= .95,
            "direct": looped["direct_confirm"]["accuracy"] >= .95,
            "copy": looped["copy_confirm"]["accuracy"] >= .98,
            "composed": all(looped[name]["accuracy"] >= .90
                            for name in looped if name.startswith("composed_")),
            "f_query": looped["first_hop_query_groups"]["group_accuracy"] >= .85,
            "g_query": looped["direct_query_groups"]["group_accuracy"] >= .85,
            "factorial": looped["factorial"]["group_accuracy"] >= .75,
            "order": looped["order_groups"]["group_accuracy"] >= .80,
            "distractor": looped["distractor_groups"]["group_accuracy"] >= .80,
            "boundary": looped["boundary_groups"]["group_accuracy"] >= .70,
            "marker_free": looped["boundary_groups"]["marker_free_accuracy"] >= .80,
            "long": looped["long_ood"]["accuracy"] >= .80,
        }
        looping[replicate] = {
            "passes_primary_gates": all(looped_primary.values()),
            "confirm_confirm_within_0.05": deep["composed_confirm_confirm"]["accuracy"]
            - looped["composed_confirm_confirm"]["accuracy"] <= .05,
            "factorial_within_0.05": deep["factorial"]["group_accuracy"]
            - looped["factorial"]["group_accuracy"] <= .05,
        }
    if not null_pass:
        verdict = "invalid"
    elif not positive_pass:
        verdict = "incomplete"
    else:
        verdict = "qualified" if primary_pass else "not_qualified"
    return {"verdict": verdict,
            "raw_sequence_binding": "qualified" if primary_pass else "not_qualified",
            "positive_control": "pass" if positive_pass else "fail",
            "null_control": "pass" if null_pass else "fail",
            "depth_helpful": all(all(row.values()) for row in depth.values()),
            "looping_competitive": all(all(row.values()) for row in looping.values()),
            "clauses": clauses, "depth_clauses": depth, "looping_clauses": looping}


def run(config, result_dir, output_dir):
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for the bounded TEACH-0012 campaign")
    provenance = source_provenance(config)
    benchmark_report = check_benchmark(config, result_dir, provenance)
    report_path = result_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("No automatic TEACH-0012 scientific rerun")
    torch.mps.set_per_process_memory_fraction(.38)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    suite = evaluation_suite(config.eval_seed, config.eval_groups)
    suite_raw = json.dumps({name: [asdict(ep) for ep in episodes]
                            for name, episodes in suite.items()},
                           sort_keys=True, separators=(",", ":")).encode()
    status = {"experiment": "TEACH-0012", "status": "running", "config": asdict(config),
              "device": "mps", "torch_version": torch.__version__,
              "platform": platform.platform(), "machine": platform.machine(),
              "suite_sha256": sha_bytes(suite_raw),
              "suite_items": sum(len(items) for items in suite.values()),
              "benchmark_sha256": sha_file(result_dir / "benchmark.json"),
              "benchmark_projection_seconds": benchmark_report["conservative_projected_seconds"],
              **provenance, **resource}
    result_dir.mkdir(parents=True, exist_ok=True)
    status_path = result_dir / "status.json"
    write_json(status_path, status)
    scores, training, prediction_hashes = {}, {}, {}
    try:
        for replicate in range(2):
            scores[str(replicate)] = {}
            training[str(replicate)] = {}
            prediction_hashes[str(replicate)] = {}
            for arm in ARMS:
                net, train_meta = train_arm(
                    config, replicate, arm, "mps", start, output_dir, result_dir,
                    resource, status, status_path)
                arm_scores, prediction_hash = evaluate(
                    net, arm, suite, "mps", replicate, result_dir)
                scores[str(replicate)][arm] = arm_scores
                training[str(replicate)][arm] = train_meta
                prediction_hashes[str(replicate)][arm] = prediction_hash
                del net
                torch.mps.synchronize()
                torch.mps.empty_cache()
                resource_check(config, start, resource, output_dir, benchmark=False)
        result = {**status, "status": "complete", "scores": scores, "training": training,
                  "prediction_sha256": prediction_hashes, "decision": decision(scores),
                  "elapsed_seconds": time.monotonic() - start,
                  "artifact_bytes": artifact_size(output_dir), **resource}
        write_json(report_path, result)
        status.update({"status": "complete", "elapsed_seconds": result["elapsed_seconds"],
                       "decision": result["decision"], **resource})
        write_json(status_path, status)
        return result
    except Exception as exc:
        status.update({"status": "stopped", "reason": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0012"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/TEACH-0012"))
    args = parser.parse_args()
    if args.mode == "benchmark":
        report = benchmark(Config(), args.result_dir, args.output_dir)
    else:
        report = run(Config(), args.result_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
