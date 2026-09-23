"""Bounded TEACH-0004 same-record dense versus learned-memory campaign."""

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

from .teacher2_train import task_loss, wilson_95, write_json
from .teacher4_models import model_for_arm
from .teacher4_tasks import KEYS, OBJECTS, evaluation_suite, family_signatures
from .teacher4_tasks import symbolic_oracle, training_batch


ARMS = ("two_read", "one_read", "dense_row")
EXPECTED_PARAMETERS = {"two_read": 222215, "one_read": 222215,
                       "dense_row": 883591}
SOURCE_PATHS = (
    "docs/experiments/TEACH-0004.md",
    "src/voynich/workspace/teacher4_tasks.py",
    "src/voynich/workspace/teacher4_models.py",
    "src/voynich/workspace/teacher4_train.py",
    "src/voynich/workspace/teacher2_train.py",
)


@dataclass(frozen=True)
class Config:
    init_seeds: tuple[int, int] = (64111, 64121)
    train_seeds: tuple[int, int] = (64211, 64221)
    eval_seed: int = 64311
    steps_per_arm: int = 5000
    batch_size: int = 64
    eval_size: int = 256
    learning_rate: float = 3e-4
    max_seconds: float = 3600.0
    max_mps_bytes: int = 8 * 1024**3
    benchmark_steps: int = 24
    benchmark_warmup_steps: int = 4
    benchmark_max_seconds: float = 600.0

    def validate(self):
        if self != Config():
            raise ValueError("Frozen TEACH-0004 config changed")


class ResourceStop(RuntimeError):
    pass


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_provenance(config):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0004 sources before MPS use: {status}")
    config_sha = hashlib.sha256(json.dumps(asdict(config), sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root/path) for path in SOURCE_PATHS},
            "source_worktree_status": status, "config_sha256": config_sha}


def resource_check(config, start, resource, *, benchmark):
    torch.mps.synchronize()
    used = torch.mps.current_allocated_memory()
    resource["peak_sampled_mps_allocated_bytes"] = max(
        resource["peak_sampled_mps_allocated_bytes"], used)
    if used > config.max_mps_bytes:
        raise ResourceStop("TEACH-0004 sampled MPS allocation exceeded 8 GiB")
    ceiling = config.benchmark_max_seconds if benchmark else config.max_seconds
    if time.monotonic()-start > ceiling:
        raise ResourceStop("TEACH-0004 wall-time cap reached")


def new_model(config, arm, rep, device):
    torch.manual_seed(config.init_seeds[rep])
    net = model_for_arm(arm).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=.01)
    return net, optimizer


def one_update(net, optimizer, config, rep, step, device):
    episodes, answers = training_batch(config.train_seeds[rep]+step, config.batch_size,
                                       arm="curriculum", step=step)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = net(ids)
    loss = task_loss(logits, episodes, answers, device)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"Nonfinite loss at step {step}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError(f"Nonfinite gradient at step {step}")
    optimizer.step()
    return float(loss.item())


def numerical_check(config):
    net, optimizer = new_model(config, "two_read", 0, "mps")
    episodes, _ = training_batch(config.train_seeds[0], 8, arm="curriculum", step=0)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device="mps")
    net.eval()
    with torch.no_grad():
        first = net(ids)[0]
        second = net(ids)[0]
        error = float((first-second).abs().max().item())
    net.train()
    loss = one_update(net, optimizer, config, 0, 0, "mps")
    finite = math.isfinite(loss) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in net.parameters())
    del net, optimizer
    torch.mps.synchronize()
    torch.mps.empty_cache()
    if error >= .002 or not finite:
        raise RuntimeError("TEACH-0004 numerical qualification failed")
    return {"max_recompute_logit_error": error, "finite_loss_and_gradients": finite}


def benchmark(config, result_dir):
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0004 benchmark")
    provenance = source_provenance(config)
    path = result_dir/"benchmark.json"
    if path.exists():
        raise FileExistsError("No automatic TEACH-0004 benchmark rerun")
    torch.mps.set_per_process_memory_fraction(.20)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0004", "mode": "benchmark", "status": "running",
              "config": asdict(config), **provenance, **resource}
    write_json(path, status)
    try:
        numerical = numerical_check(config)
        resource_check(config, start, resource, benchmark=True)
        measured = {}
        for arm in ARMS:
            net, optimizer = new_model(config, arm, 0, "mps")
            net.train()
            timings = []
            for step in range(config.benchmark_steps):
                resource_check(config, start, resource, benchmark=True)
                tick = time.monotonic()
                one_update(net, optimizer, config, 0, step, "mps")
                resource_check(config, start, resource, benchmark=True)
                if step >= config.benchmark_warmup_steps:
                    timings.append(time.monotonic()-tick)
            measured[arm] = {"parameters": sum(p.numel() for p in net.parameters()),
                             "median_timed_step_seconds": statistics.median(timings),
                             "timed_step_seconds": timings, "timed_steps": len(timings),
                             "warmup_steps": config.benchmark_warmup_steps}
            if measured[arm]["parameters"] != EXPECTED_PARAMETERS[arm]:
                raise RuntimeError(f"TEACH-0004 {arm} parameter count changed")
            del net, optimizer
            torch.mps.empty_cache()
            status.update({"measured": measured, "elapsed_seconds": time.monotonic()-start,
                           **resource})
            write_json(path, status)
        if measured["two_read"]["parameters"] != measured["one_read"]["parameters"]:
            raise RuntimeError("Memory arms must have exactly matched parameters")
        projection = 2*config.steps_per_arm*sum(
            row["median_timed_step_seconds"] for row in measured.values())
        conservative = 1.5*projection
        resource_check(config, start, resource, benchmark=True)
        status.update({"status": "pass" if conservative <= config.max_seconds else "fail",
                       "numerical_qualification": numerical,
                       "projected_campaign_seconds": projection,
                       "conservative_projected_seconds": conservative,
                       "elapsed_seconds": time.monotonic()-start, **resource})
        write_json(path, status)
        return status
    except (ResourceStop, RuntimeError) as exc:
        status.update({"status": "fail", "reason": type(exc).__name__+": "+str(exc),
                       "elapsed_seconds": time.monotonic()-start, **resource})
        write_json(path, status)
        raise


def check_benchmark(config, result_dir, provenance):
    path = result_dir/"benchmark.json"
    if not path.is_file():
        raise RuntimeError("Source-matched TEACH-0004 benchmark required")
    report = json.loads(path.read_text())
    measured = report.get("measured", {})
    valid = (type(measured) is dict and set(measured) == set(ARMS)
             and all(type(row) is dict for row in measured.values()))
    if valid:
        valid = all(row.get("timed_steps") == 20 and row.get("warmup_steps") == 4
                    and row.get("parameters") == EXPECTED_PARAMETERS[arm]
                    and type(row.get("timed_step_seconds")) is list
                    and len(row["timed_step_seconds"]) == 20
                    and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                            for value in row["timed_step_seconds"])
                    and type(row.get("median_timed_step_seconds")) in (int, float)
                    and math.isclose(row["median_timed_step_seconds"],
                                     statistics.median(row["timed_step_seconds"]),
                                     rel_tol=1e-12, abs_tol=1e-12)
                    for arm, row in measured.items())
    if valid:
        valid = measured["two_read"]["parameters"] == measured["one_read"]["parameters"]
    projection = (2*config.steps_per_arm*sum(row["median_timed_step_seconds"]
                  for row in measured.values())) if valid else float("inf")
    numerical = report.get("numerical_qualification", {})
    valid = (valid and report.get("experiment") == "TEACH-0004"
             and report.get("mode") == "benchmark" and report.get("status") == "pass"
             and report.get("config") == json.loads(json.dumps(asdict(config)))
             and all(report.get(key) == value for key, value in provenance.items())
             and numerical.get("finite_loss_and_gradients") is True
             and type(numerical.get("max_recompute_logit_error")) in (int, float)
             and 0 <= numerical["max_recompute_logit_error"] < .002
             and type(report.get("projected_campaign_seconds")) in (int, float)
             and math.isclose(report["projected_campaign_seconds"], projection,
                              rel_tol=1e-12, abs_tol=1e-9)
             and type(report.get("conservative_projected_seconds")) in (int, float)
             and math.isclose(report["conservative_projected_seconds"], 1.5*projection,
                              rel_tol=1e-12, abs_tol=1e-9)
             and 1.5*projection <= config.max_seconds
             and type(report.get("peak_sampled_mps_allocated_bytes")) is int
             and 0 <= report["peak_sampled_mps_allocated_bytes"] <= config.max_mps_bytes
             and type(report.get("elapsed_seconds")) in (int, float)
             and 0 <= report["elapsed_seconds"] <= config.benchmark_max_seconds)
    if not valid:
        raise RuntimeError("TEACH-0004 benchmark did not qualify this frozen source")
    return report


def pointer_answers(episode):
    t = episode.tokens
    f = (0 if t[12] == t[2] else 1) if episode.task in ("first_hop", "composed") else None
    key = (t[12] if episode.task == "direct" else
           t[3 if f == 0 else 5] if episode.task == "composed" else None)
    g = (0 if key == t[7] else 1) if key is not None else None
    return f, g


@torch.no_grad()
def evaluate(net, suite, device, rep, arm, result_dir):
    net.eval()
    scores, archived = {}, {}
    for name, episodes in suite.items():
        rows = []
        for offset in range(0, len(episodes), 128):
            batch = episodes[offset:offset+128]
            ids = torch.tensor([ep.tokens for ep in batch], dtype=torch.long, device=device)
            logits, first_attn, second_attn = net(ids)
            for index, ep in enumerate(batch):
                options = KEYS if ep.task == "first_hop" else OBJECTS
                prediction = options[int(logits[index, options[0]:options[-1]+1].argmax().item())]
                f_true, g_true = pointer_answers(ep)
                f_family, g_family = family_signatures(ep)
                t = ep.tokens
                candidates = (t[3], t[5]) if ep.task == "first_hop" else (t[8], t[10])
                rows.append({"index": offset+index, "tokens": t, "task": ep.task,
                             "answer": ep.answer, "prediction": prediction,
                             "f_family": f_family, "g_family": g_family,
                             "candidate_member": prediction in candidates,
                             "f_read_answer": f_true, "g_read_answer": g_true,
                             "f_read_prediction": (int(first_attn[index].argmax().item())
                                                   if first_attn is not None else None),
                             "g_read_prediction": (int(second_attn[index].argmax().item())
                                                   if second_attn is not None else None)})
        correct = sum(row["answer"] == row["prediction"] for row in rows)
        score = {"correct": correct, "total": len(rows), "accuracy": correct/len(rows),
                 "wilson_95": wilson_95(correct, len(rows)),
                 "candidate_member": sum(row["candidate_member"] for row in rows)}
        for side in ("f", "g"):
            applicable = [row for row in rows if row[f"{side}_read_answer"] is not None
                          and row[f"{side}_read_prediction"] is not None]
            if applicable:
                match = sum(row[f"{side}_read_answer"] == row[f"{side}_read_prediction"]
                            for row in applicable)
                score[f"{side}_read_correct"] = match
                score[f"{side}_read_total"] = len(applicable)
                score[f"{side}_read_accuracy"] = match/len(applicable)
        if name in ("first_hop_pairs_holdout", "direct_pairs_holdout", "factorial"):
            width = 4 if name == "factorial" else 2
            groups = [rows[i:i+width] for i in range(0, len(rows), width)]
            exact = sum(all(row["answer"] == row["prediction"] for row in group)
                        for group in groups)
            score.update({"exact_groups": exact, "total_groups": len(groups),
                          "group_accuracy": exact/len(groups),
                          "group_wilson_95": wilson_95(exact, len(groups)),
                          "prediction_changes": sum(group[0]["prediction"] != group[1]["prediction"]
                                                    for group in groups)})
            if width == 4:
                score["f_swap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 1), (2, 3)))
                score["g_remap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 2), (1, 3)))
        scores[name], archived[name] = score, rows
    payload = json.dumps(archived, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    path = result_dir/f"predictions-rep{rep}-{arm}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    temporary.replace(path)
    return scores, sha_file(path)


def decision(scores):
    clauses = {}
    for rep in ("0", "1"):
        memory = scores[rep]["two_read"]
        row = {
            "first_hop_holdout_at_least_0.90": memory["first_hop_holdout"]["accuracy"] >= .90,
            "first_hop_pair_at_least_0.80": memory["first_hop_pairs_holdout"]["group_accuracy"] >= .80,
            "direct_holdout_at_least_0.90": memory["direct_holdout"]["accuracy"] >= .90,
            "direct_pair_at_least_0.80": memory["direct_pairs_holdout"]["group_accuracy"] >= .80,
            "both_held_composed_at_least_0.85": memory["composed_holdout_holdout"]["accuracy"] >= .85,
            "crossed_cells_at_least_0.85": all(memory[name]["accuracy"] >= .85 for name in
                                               ("composed_holdout_train", "composed_train_holdout")),
            "factorial_at_least_0.65": memory["factorial"]["group_accuracy"] >= .65,
            "copy_at_least_0.95": memory["copy"]["accuracy"] >= .95,
        }
        for control in ("one_read", "dense_row"):
            row[f"composed_gain_over_{control}_at_least_0.15"] = (
                memory["composed_holdout_holdout"]["accuracy"]
                - scores[rep][control]["composed_holdout_holdout"]["accuracy"] >= .15)
            row[f"factorial_gain_over_{control}_at_least_0.15"] = (
                memory["factorial"]["group_accuracy"]
                - scores[rep][control]["factorial"]["group_accuracy"] >= .15)
        clauses[rep] = row
    qualified = all(all(row.values()) for row in clauses.values())
    return {"verdict": "qualified" if qualified else "not_qualified",
            "two_read_qualified": qualified, "clauses_by_seed": clauses}


def run(config, output_dir, result_dir):
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0004 run")
    provenance = source_provenance(config)
    benchmark_report = check_benchmark(config, result_dir, provenance)
    status_path = result_dir/"status.json"
    if status_path.exists():
        raise FileExistsError("No automatic TEACH-0004 scientific rerun")
    torch.mps.set_per_process_memory_fraction(.20)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0004", "mode": "scientific", "status": "running",
              "device": "mps", "torch_version": torch.__version__,
              "platform": platform.platform(), "machine": platform.machine(),
              "config": asdict(config), "benchmark_sha256": sha_file(result_dir/"benchmark.json"),
              "benchmark_conservative_projected_seconds": benchmark_report[
                  "conservative_projected_seconds"], **provenance, **resource}
    write_json(status_path, status)
    try:
        status["numerical_qualification"] = numerical_check(config)
        resource_check(config, start, resource, benchmark=False)
        suite = evaluation_suite(config.eval_seed, config.eval_size)
        if sum(len(rows) for rows in suite.values()) != 3328 or any(
                symbolic_oracle(ep.tokens) != ep.answer for rows in suite.values() for ep in rows):
            raise AssertionError("Evaluation suite changed or oracle failed")
        suite_payload = {name: [(ep.tokens, ep.answer) for ep in rows]
                         for name, rows in suite.items()}
        status["evaluation_suite_sha256"] = hashlib.sha256(
            json.dumps(suite_payload, sort_keys=True).encode()).hexdigest()
        records, scores = {}, {}
        for rep in range(2):
            key = str(rep)
            records[key], scores[key] = {}, {}
            for arm in ARMS:
                net, optimizer = new_model(config, arm, rep, "mps")
                net.train()
                losses, history = [], []
                for step in range(config.steps_per_arm):
                    resource_check(config, start, resource, benchmark=False)
                    loss = one_update(net, optimizer, config, rep, step, "mps")
                    resource_check(config, start, resource, benchmark=False)
                    losses.append(loss)
                    if (step+1) % 250 == 0:
                        recent = sum(losses[-100:])/100
                        history.append({"step": step+1, "last_100_loss": recent})
                        status.update({"progress": {"replicate": rep, "arm": arm,
                                                    "step": step+1, "latest_loss": loss,
                                                    "last_100_loss": recent,
                                                    "elapsed_seconds": time.monotonic()-start,
                                                    **resource}, **resource})
                        write_json(status_path, status)
                checkpoint = output_dir/f"rep{rep}-{arm}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({"model": {name: value.detach().cpu()
                                      for name, value in net.state_dict().items()},
                            "config": asdict(config), "replicate": rep, "arm": arm}, temporary)
                temporary.replace(checkpoint)
                loss_path = result_dir/f"losses-rep{rep}-{arm}.json.gz"
                loss_path.parent.mkdir(parents=True, exist_ok=True)
                loss_tmp = loss_path.with_suffix(loss_path.suffix+".tmp")
                loss_tmp.write_bytes(gzip.compress(
                    json.dumps(losses, separators=(",", ":"), allow_nan=False).encode(),
                    compresslevel=9, mtime=0))
                loss_tmp.replace(loss_path)
                resource_check(config, start, resource, benchmark=False)
                score, prediction_sha = evaluate(net, suite, "mps", rep, arm, result_dir)
                resource_check(config, start, resource, benchmark=False)
                records[key][arm] = {"steps": config.steps_per_arm,
                                     "parameters": sum(p.numel() for p in net.parameters()),
                                     "history": history,
                                     "checkpoint_sha256": sha_file(checkpoint),
                                     "losses_sha256": sha_file(loss_path),
                                     "predictions_sha256": prediction_sha}
                scores[key][arm] = score
                del net, optimizer
                torch.mps.empty_cache()
                status["completed_arms"] = [[int(r), name] for r in ("0", "1")
                                            for name in records.get(r, {})]
                status.update(resource)
                write_json(status_path, status)
        verdict = decision(scores)
        if sum(len(row) for row in records.values()) != 6:
            raise AssertionError("Six-arm campaign incomplete")
        status.update({"status": "complete", "decision": verdict, "scores": scores,
                       "arms": records, "elapsed_seconds": time.monotonic()-start,
                       **resource})
        write_json(result_dir/"report.json", status)
        write_json(status_path, {"experiment": "TEACH-0004", "status": "complete",
                                 "decision": verdict, "report": "report.json",
                                 "source_git_head": provenance["source_git_head"], **resource})
        return status
    except (ResourceStop, RuntimeError, AssertionError) as exc:
        status.update({"status": "stopped", "decision": {"verdict": "incomplete"},
                       "reason": type(exc).__name__+": "+str(exc),
                       "elapsed_seconds": time.monotonic()-start, **resource})
        write_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/TEACH-0004"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0004"))
    args = parser.parse_args()
    config = Config()
    result = (benchmark(config, args.result_dir) if args.mode == "benchmark"
              else run(config, args.output_dir, args.result_dir))
    print(json.dumps({"status": result["status"],
                      "decision": result.get("decision", {}).get("verdict"),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
