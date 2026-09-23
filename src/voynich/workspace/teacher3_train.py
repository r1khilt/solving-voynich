"""Frozen TEACH-0003 scale/alignment campaign; benchmark required before run."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch
from torch import nn
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer
from voynich.workspace.teacher2_train import task_loss, wilson_95, write_json
from voynich.workspace.teacher3_tasks import (
    KEYS, OBJECTS, Episode, evaluation_suite, family_signatures,
    symbolic_oracle, training_batch,
)


ARMS = ("dense_small", "dense_medium", "dense_large", "align_true", "align_random")
EXPECTED_CAMPAIGN_ARMS = 10
EXPECTED_EVALUATION_ITEMS_PER_ARM = 3328
SHAPES = {"dense_small": (4, 128, 4, 256),
          "dense_medium": (6, 384, 6, 1536),
          "dense_large": (8, 768, 12, 3072),
          "align_true": (8, 768, 12, 3072),
          "align_random": (8, 768, 12, 3072)}
EXPECTED_PARAMETERS = {"dense_small": 668544, "dense_medium": 14196864,
                       "dense_large": 75582720, "align_true": 75582720,
                       "align_random": 75582720}
SOURCE_PATHS = (
    "docs/experiments/TEACH-0003.md",
    "src/voynich/workspace/teacher3_tasks.py",
    "src/voynich/workspace/teacher3_train.py",
    "src/voynich/workspace/teacher2_train.py",
    "src/voynich/model.py",
)


@dataclass(frozen=True)
class Config:
    init_seeds: tuple[int, int] = (63111, 63121)
    train_seeds: tuple[int, int] = (63211, 63221)
    eval_seed: int = 63311
    steps_per_arm: int = 5000
    batch_size: int = 64
    eval_size: int = 256
    learning_rate: float = 3e-4
    aux_weight: float = 0.5
    max_seconds: float = 21600.0
    max_mps_bytes: int = 24 * 1024**3
    benchmark_steps: int = 24
    benchmark_warmup_steps: int = 4
    benchmark_max_seconds: float = 1800.0

    def validate(self) -> None:
        if self != Config():
            raise ValueError("Frozen TEACH-0003 configuration changed")


class ResourceStop(RuntimeError):
    pass


class RowHeadTeacher(nn.Module):
    """Causal backbone plus learned heads; no row-index operator in inference."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in SHAPES:
            raise ValueError("Unknown arm")
        layers, width, heads, ff = SHAPES[arm]
        self.backbone = VoynichTransformer(ModelConfig(
            vocab_size=45, pad_id=0, d_model=width, n_layers=layers,
            n_heads=heads, d_ff=ff, context_length=14, dropout=0.0,
        ))
        self.f_row = nn.Linear(width, 2, bias=False)
        self.g_row = nn.Linear(width, 2, bias=False)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured = []

        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            captured.append(output[:, -1])

        hook = self.backbone.final_norm.register_forward_hook(capture)
        try:
            output = self.backbone(ids)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Answer-position hidden state not captured")
        hidden = captured[0]
        return output.logits[:, -1], self.f_row(hidden), self.g_row(hidden)


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def config_hash(config: Config) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_provenance(config: Config) -> dict:
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                            cwd=root, capture_output=True, text=True, check=True,
                            timeout=5).stdout.splitlines()
    if status:
        raise RuntimeError(f"Commit frozen TEACH-0003 sources first: {status}")
    return {"source_git_head": head,
            "source_sha256": {path: sha_file(root / path) for path in SOURCE_PATHS},
            "config_sha256": config_hash(config), "source_worktree_status": status}


def row_targets(episodes: list[Episode], arm: str, seed: int
                ) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return selected item indices and two-row targets for auxiliary heads."""
    f_indices, f_targets, g_indices, g_targets = [], [], [], []
    random_labels = random.Random(seed ^ 0x71A3C003)
    for i, ep in enumerate(episodes):
        t = ep.tokens
        if ep.task in ("first_hop", "composed"):
            f_index = 0 if t[12] == t[2] else 1
            f_indices.append(i)
            f_targets.append(random_labels.randrange(2) if arm == "align_random" else f_index)
        if ep.task in ("direct", "composed"):
            key = t[12] if ep.task == "direct" else t[3 if t[12] == t[2] else 5]
            g_index = 0 if key == t[7] else 1
            g_indices.append(i)
            g_targets.append(random_labels.randrange(2) if arm == "align_random" else g_index)
    return f_indices, f_targets, g_indices, g_targets


def total_loss(answer_logits: torch.Tensor, f_logits: torch.Tensor,
               g_logits: torch.Tensor, episodes: list[Episode], answers: list[int],
               arm: str, seed: int, device: str) -> torch.Tensor:
    primary = task_loss(answer_logits, episodes, answers, device)
    if arm not in ("align_true", "align_random"):
        return primary
    f_indices, f_targets, g_indices, g_targets = row_targets(episodes, arm, seed)
    pieces = []
    if f_indices:
        pieces.append(F.cross_entropy(f_logits[f_indices],
                                      torch.tensor(f_targets, device=device), reduction="sum"))
    if g_indices:
        pieces.append(F.cross_entropy(g_logits[g_indices],
                                      torch.tensor(g_targets, device=device), reduction="sum"))
    count = len(f_indices) + len(g_indices)
    if not count:
        return primary
    return primary + Config().aux_weight * sum(pieces) / count


def resource_check(config: Config, start: float, resource: dict, *, benchmark: bool) -> None:
    torch.mps.synchronize()
    allocation = torch.mps.current_allocated_memory()
    resource["peak_sampled_mps_allocated_bytes"] = max(
        resource["peak_sampled_mps_allocated_bytes"], allocation)
    if allocation > config.max_mps_bytes:
        raise ResourceStop("TEACH-0003 sampled MPS allocation exceeded 24 GiB")
    ceiling = config.benchmark_max_seconds if benchmark else config.max_seconds
    if time.monotonic() - start > ceiling:
        raise ResourceStop("TEACH-0003 wall-time ceiling reached")


def one_update(net: RowHeadTeacher, optimizer: torch.optim.Optimizer,
               config: Config, arm: str, replicate: int, step: int,
               device: str) -> float:
    seed = config.train_seeds[replicate] + step
    episodes, answers = training_batch(seed, config.batch_size,
                                       arm="curriculum", step=step)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    answer_logits, f_logits, g_logits = net(ids)
    loss = total_loss(answer_logits, f_logits, g_logits,
                      episodes, answers, arm, seed, device)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"Nonfinite {arm} loss at step {step}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError(f"Nonfinite {arm} gradient at step {step}")
    optimizer.step()
    return float(loss.item())


def new_model(config: Config, arm: str, replicate: int, device: str
              ) -> tuple[RowHeadTeacher, torch.optim.Optimizer]:
    torch.manual_seed(config.init_seeds[replicate])
    net = RowHeadTeacher(arm).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=.01)
    return net, optimizer


def numerical_check(config: Config, device: str) -> dict:
    net, optimizer = new_model(config, "dense_large", 0, device)
    episodes, answers = training_batch(config.train_seeds[0], 8,
                                       arm="curriculum", step=0)
    ids = torch.tensor([ep.tokens for ep in episodes], dtype=torch.long, device=device)
    net.eval()
    with torch.no_grad():
        native = net.backbone(ids).logits[:, -1]
        wrapped = net(ids)[0]
        error = float((native - wrapped).abs().max().item())
    net.train()
    loss = one_update(net, optimizer, config, "align_true", 0, 0, device)
    finite = math.isfinite(loss) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item())
        for p in net.parameters())
    del net, optimizer
    torch.mps.synchronize()
    torch.mps.empty_cache()
    if error > .002 or not finite:
        raise RuntimeError("TEACH-0003 numerical qualification failed")
    return {"max_wrapped_native_logit_error": error, "finite_loss_and_gradients": finite}


def benchmark(config: Config, result_dir: Path) -> dict:
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0003 benchmark")
    provenance = source_provenance(config)
    torch.mps.set_per_process_memory_fraction(.40)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0003", "mode": "benchmark", "status": "running",
              "config": asdict(config), **provenance, **resource}
    path = result_dir / "benchmark.json"
    write_json(path, status)
    try:
        numerical = numerical_check(config, "mps")
        resource_check(config, start, resource, benchmark=True)
        measured = {}
        for arm in ARMS:
            net, optimizer = new_model(config, arm, 0, "mps")
            net.train()
            timings = []
            for step in range(config.benchmark_steps):
                resource_check(config, start, resource, benchmark=True)
                tick = time.monotonic()
                one_update(net, optimizer, config, arm, 0, step, "mps")
                resource_check(config, start, resource, benchmark=True)
                if step >= config.benchmark_warmup_steps:
                    timings.append(time.monotonic() - tick)
            measured[arm] = {"parameters": net.parameter_count,
                             "median_timed_step_seconds": statistics.median(timings),
                             "timed_step_seconds": timings,
                             "timed_steps": len(timings),
                             "warmup_steps": config.benchmark_warmup_steps}
            if measured[arm]["parameters"] != EXPECTED_PARAMETERS[arm]:
                raise RuntimeError(f"TEACH-0003 {arm} parameter count changed")
            del net, optimizer
            torch.mps.empty_cache()
            status.update({"measured": measured, **resource,
                           "elapsed_seconds": time.monotonic() - start})
            write_json(path, status)
        raw_projection = 2 * config.steps_per_arm * sum(
            row["median_timed_step_seconds"] for row in measured.values())
        conservative = 1.5 * raw_projection
        resource_check(config, start, resource, benchmark=True)
        pass_gate = conservative <= config.max_seconds
        status.update({"status": "pass" if pass_gate else "fail",
                       "numerical_qualification": numerical,
                       "projected_campaign_seconds": raw_projection,
                       "conservative_projected_seconds": conservative,
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(path, status)
        return status
    except (ResourceStop, RuntimeError) as exc:
        status.update({"status": "fail", "reason": type(exc).__name__ + ": " + str(exc),
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(path, status)
        raise


def check_benchmark(config: Config, result_dir: Path, provenance: dict) -> dict:
    path = result_dir / "benchmark.json"
    if not path.is_file():
        raise RuntimeError("Source-matched TEACH-0003 benchmark is required")
    report = json.loads(path.read_text())
    measured = report.get("measured")
    measured_valid = isinstance(measured, dict) and set(measured) == set(ARMS)
    if measured_valid:
        measured_valid = all(
            row.get("parameters") == EXPECTED_PARAMETERS[arm]
            and row.get("timed_steps") == config.benchmark_steps - config.benchmark_warmup_steps
            and row.get("warmup_steps") == config.benchmark_warmup_steps
            and isinstance(row.get("timed_step_seconds"), list)
            and len(row["timed_step_seconds"]) == row["timed_steps"]
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in row["timed_step_seconds"])
            and type(row.get("median_timed_step_seconds")) in (int, float)
            and math.isfinite(row["median_timed_step_seconds"])
            and row["median_timed_step_seconds"] > 0
            and math.isclose(row["median_timed_step_seconds"],
                             statistics.median(row["timed_step_seconds"]),
                             rel_tol=1e-12, abs_tol=1e-12)
            for arm, row in measured.items() if isinstance(row, dict))
        measured_valid = measured_valid and all(isinstance(row, dict)
                                               for row in measured.values())
    numerical = report.get("numerical_qualification", {})
    numerical_valid = (numerical.get("finite_loss_and_gradients") is True
                       and type(numerical.get("max_wrapped_native_logit_error")) in (int, float)
                       and math.isfinite(numerical["max_wrapped_native_logit_error"])
                       and 0 <= numerical["max_wrapped_native_logit_error"] <= .002)
    projection = (2 * config.steps_per_arm * sum(row["median_timed_step_seconds"]
                  for row in measured.values())) if measured_valid else float("inf")
    recorded_peak = report.get("peak_sampled_mps_allocated_bytes")
    recorded_elapsed = report.get("elapsed_seconds")
    if (report.get("status") != "pass" or not measured_valid or not numerical_valid
            or report.get("experiment") != "TEACH-0003" or report.get("mode") != "benchmark"
            or report.get("config") != json.loads(json.dumps(asdict(config)))
            or report.get("source_worktree_status") != []
            or report.get("source_git_head") != provenance["source_git_head"]
            or report.get("source_sha256") != provenance["source_sha256"]
            or report.get("config_sha256") != provenance["config_sha256"]
            or type(report.get("projected_campaign_seconds")) not in (int, float)
            or not math.isclose(report["projected_campaign_seconds"], projection,
                                rel_tol=1e-12, abs_tol=1e-9)
            or type(report.get("conservative_projected_seconds")) not in (int, float)
            or not math.isfinite(report["conservative_projected_seconds"])
            or not math.isclose(report["conservative_projected_seconds"], 1.5 * projection,
                                rel_tol=1e-12, abs_tol=1e-9)
            or report["conservative_projected_seconds"] > config.max_seconds
            or type(recorded_peak) is not int or not 0 <= recorded_peak <= config.max_mps_bytes
            or type(recorded_elapsed) not in (int, float)
            or not 0 <= recorded_elapsed <= config.benchmark_max_seconds):
        raise RuntimeError("TEACH-0003 benchmark did not qualify this frozen source")
    return report


def pointer_answers(episode: Episode) -> tuple[int | None, int | None]:
    t = episode.tokens
    f_row = (0 if t[12] == t[2] else 1) if episode.task in ("first_hop", "composed") else None
    if episode.task == "direct":
        key = t[12]
    elif episode.task == "composed":
        key = t[3 if f_row == 0 else 5]
    else:
        key = None
    g_row = (0 if key == t[7] else 1) if key is not None else None
    return f_row, g_row


@torch.no_grad()
def evaluate(net: RowHeadTeacher, suite: dict[str, list[Episode]], device: str,
             rep: int, arm: str, result_dir: Path) -> tuple[dict, str]:
    net.eval()
    scores = {}
    archived = {}
    for name, episodes in suite.items():
        rows = []
        for offset in range(0, len(episodes), 128):
            batch = episodes[offset:offset + 128]
            ids = torch.tensor([ep.tokens for ep in batch], dtype=torch.long, device=device)
            answer_logits, f_logits, g_logits = net(ids)
            for index, ep in enumerate(batch):
                options = KEYS if ep.task == "first_hop" else OBJECTS
                pred = options[int(answer_logits[index, options[0]:options[-1] + 1].argmax().item())]
                f_answer, g_answer = pointer_answers(ep)
                f_family, g_family = family_signatures(ep)
                t = ep.tokens
                candidates = (t[3], t[5]) if ep.task == "first_hop" else (t[8], t[10])
                rows.append({"index": offset + index, "tokens": t, "task": ep.task,
                             "answer": ep.answer, "prediction": pred,
                             "f_family": f_family, "g_family": g_family,
                             "candidate_member": pred in candidates,
                             "f_row_answer": f_answer,
                             "f_row_prediction": int(f_logits[index].argmax().item()),
                             "g_row_answer": g_answer,
                             "g_row_prediction": int(g_logits[index].argmax().item())})
        correct = sum(row["answer"] == row["prediction"] for row in rows)
        summary = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                   "wilson_95": wilson_95(correct, len(rows)),
                   "candidate_member": sum(row["candidate_member"] for row in rows)}
        for side in ("f", "g"):
            applicable = [row for row in rows if row[f"{side}_row_answer"] is not None]
            if applicable:
                match = sum(row[f"{side}_row_answer"] == row[f"{side}_row_prediction"]
                            for row in applicable)
                summary[f"{side}_row_correct"] = match
                summary[f"{side}_row_total"] = len(applicable)
                summary[f"{side}_row_accuracy"] = match / len(applicable)
        if name in ("first_hop_pairs_holdout", "direct_pairs_holdout", "factorial"):
            width = 4 if name == "factorial" else 2
            groups = [rows[i:i + width] for i in range(0, len(rows), width)]
            exact = sum(all(row["answer"] == row["prediction"] for row in group)
                        for group in groups)
            summary.update({"exact_groups": exact, "total_groups": len(groups),
                            "group_accuracy": exact / len(groups),
                            "group_wilson_95": wilson_95(exact, len(groups)),
                            "prediction_changes": sum(group[0]["prediction"] != group[1]["prediction"]
                                                      for group in groups)})
            if width == 2:
                side = "f" if name.startswith("first_hop") else "g"
                row_exact = sum(all(row[f"{side}_row_answer"] == row[f"{side}_row_prediction"]
                                    for row in group) for group in groups)
                summary[f"{side}_row_exact_pairs"] = row_exact
                summary[f"{side}_row_pair_accuracy"] = row_exact / len(groups)
            else:
                summary["f_swap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 1), (2, 3)))
                summary["g_remap_prediction_changes"] = sum(
                    group[a]["prediction"] != group[b]["prediction"]
                    for group in groups for a, b in ((0, 2), (1, 3)))
        scores[name] = summary
        archived[name] = rows
    payload = json.dumps(archived, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    path = result_dir / f"predictions-rep{rep}-{arm}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    tmp.replace(path)
    return scores, sha_file(path)


def decision(scores: dict) -> dict:
    dense_clauses, align_clauses = {}, {}
    for rep in ("0", "1"):
        small, large, aligned, random_aux = (scores[rep][arm] for arm in
                                            ("dense_small", "dense_large", "align_true", "align_random"))

        def behavior(s: dict) -> dict:
            return {
                "first_hop_holdout_at_least_0.90": s["first_hop_holdout"]["accuracy"] >= .90,
                "first_hop_pair_at_least_0.80": s["first_hop_pairs_holdout"]["group_accuracy"] >= .80,
                "direct_holdout_at_least_0.90": s["direct_holdout"]["accuracy"] >= .90,
                "direct_pair_at_least_0.80": s["direct_pairs_holdout"]["group_accuracy"] >= .80,
                "both_held_composed_at_least_0.85": s["composed_holdout_holdout"]["accuracy"] >= .85,
                "crossed_cells_at_least_0.85": all(s[name]["accuracy"] >= .85 for name in
                                                   ("composed_holdout_train", "composed_train_holdout")),
                "factorial_at_least_0.65": s["factorial"]["group_accuracy"] >= .65,
                "copy_at_least_0.95": s["copy"]["accuracy"] >= .95,
            }

        dense_clauses[rep] = behavior(large)
        dense_clauses[rep].update({
            "composed_gain_over_small_at_least_0.20": (
                large["composed_holdout_holdout"]["accuracy"]
                - small["composed_holdout_holdout"]["accuracy"] >= .20),
            "factorial_gain_over_small_at_least_0.20": (
                large["factorial"]["group_accuracy"]
                - small["factorial"]["group_accuracy"] >= .20),
        })
        align_clauses[rep] = behavior(aligned)
        align_clauses[rep].update({
            "f_pointer_holdout_at_least_0.90": aligned["first_hop_holdout"]["f_row_accuracy"] >= .90,
            "g_pointer_holdout_at_least_0.90": aligned["direct_holdout"]["g_row_accuracy"] >= .90,
            "f_pointer_pair_at_least_0.80": (
                aligned["first_hop_pairs_holdout"]["f_row_pair_accuracy"] >= .80),
            "g_pointer_pair_at_least_0.80": (
                aligned["direct_pairs_holdout"]["g_row_pair_accuracy"] >= .80),
            "composed_f_g_pointers_at_least_0.90": all(
                aligned["composed_holdout_holdout"][f"{side}_row_accuracy"] >= .90
                for side in ("f", "g")),
        })
        for control_name, control in (("large", large), ("random", random_aux)):
            align_clauses[rep][f"composed_gain_over_{control_name}_at_least_0.15"] = (
                aligned["composed_holdout_holdout"]["accuracy"]
                - control["composed_holdout_holdout"]["accuracy"] >= .15)
            align_clauses[rep][f"factorial_gain_over_{control_name}_at_least_0.15"] = (
                aligned["factorial"]["group_accuracy"]
                - control["factorial"]["group_accuracy"] >= .15)
    scale = all(all(clauses.values()) for clauses in dense_clauses.values())
    alignment = all(all(clauses.values()) for clauses in align_clauses.values())
    return {"verdict": "qualified" if scale or alignment else "not_qualified",
            "dense_scale_qualified": scale, "alignment_qualified": alignment,
            "dense_clauses_by_seed": dense_clauses,
            "alignment_clauses_by_seed": align_clauses}


def run(config: Config, output_dir: Path, result_dir: Path) -> dict:
    config.validate()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for TEACH-0003 scientific run")
    provenance = source_provenance(config)
    benchmark_report = check_benchmark(config, result_dir, provenance)
    torch.mps.set_per_process_memory_fraction(.40)
    start = time.monotonic()
    resource = {"peak_sampled_mps_allocated_bytes": 0}
    status = {"experiment": "TEACH-0003", "mode": "scientific", "status": "running",
              "device": "mps", "torch_version": torch.__version__,
              "platform": platform.platform(), "machine": platform.machine(),
              "config": asdict(config), "benchmark_sha256": sha_file(result_dir / "benchmark.json"),
              "benchmark_conservative_projected_seconds": benchmark_report[
                  "conservative_projected_seconds"], **provenance, **resource}
    status_path = result_dir / "status.json"
    write_json(status_path, status)
    try:
        status["numerical_qualification"] = numerical_check(config, "mps")
        resource_check(config, start, resource, benchmark=False)
        suite = evaluation_suite(config.eval_seed, config.eval_size)
        if sum(len(cells) for cells in suite.values()) != EXPECTED_EVALUATION_ITEMS_PER_ARM:
            raise AssertionError("TEACH-0003 evaluation item count changed")
        if any(symbolic_oracle(ep.tokens) != ep.answer for cells in suite.values() for ep in cells):
            raise AssertionError("TEACH-0003 symbolic evaluation oracle failed")
        suite_payload = {name: [(ep.tokens, ep.answer) for ep in cells]
                         for name, cells in suite.items()}
        status["evaluation_suite_sha256"] = hashlib.sha256(
            json.dumps(suite_payload, sort_keys=True).encode()).hexdigest()
        records, scores = {}, {}
        for rep in range(2):
            key = str(rep)
            records[key], scores[key] = {}, {}
            for arm in ARMS:
                net, optimizer = new_model(config, arm, rep, "mps")
                net.train()
                history, losses = [], []
                for step in range(config.steps_per_arm):
                    resource_check(config, start, resource, benchmark=False)
                    loss = one_update(net, optimizer, config, arm, rep, step, "mps")
                    resource_check(config, start, resource, benchmark=False)
                    losses.append(loss)
                    if (step + 1) % 250 == 0:
                        recent = sum(losses[-100:]) / 100
                        history.append({"step": step + 1, "last_100_loss": recent})
                        status.update({"progress": {"replicate": rep, "arm": arm,
                                                    "step": step + 1, "latest_loss": loss,
                                                    "last_100_loss": recent,
                                                    "elapsed_seconds": time.monotonic() - start,
                                                    **resource}, **resource})
                        write_json(status_path, status)
                checkpoint = output_dir / f"rep{rep}-{arm}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({"model": {name: tensor.detach().cpu()
                                      for name, tensor in net.state_dict().items()},
                            "config": asdict(config), "replicate": rep, "arm": arm}, temporary)
                temporary.replace(checkpoint)
                loss_path = result_dir / f"losses-rep{rep}-{arm}.json.gz"
                loss_path.parent.mkdir(parents=True, exist_ok=True)
                loss_tmp = loss_path.with_suffix(loss_path.suffix + ".tmp")
                loss_tmp.write_bytes(gzip.compress(
                    json.dumps(losses, separators=(",", ":"), allow_nan=False).encode(),
                    compresslevel=9, mtime=0))
                loss_tmp.replace(loss_path)
                resource_check(config, start, resource, benchmark=False)
                score, prediction_sha = evaluate(net, suite, "mps", rep, arm, result_dir)
                resource_check(config, start, resource, benchmark=False)
                records[key][arm] = {"steps": config.steps_per_arm,
                                     "parameters": net.parameter_count,
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
        if sum(len(row) for row in records.values()) != EXPECTED_CAMPAIGN_ARMS:
            raise AssertionError("TEACH-0003 ten-arm campaign incomplete")
        status.update({"status": "complete", "decision": verdict, "scores": scores,
                       "arms": records, "elapsed_seconds": time.monotonic() - start,
                       **resource})
        write_json(result_dir / "report.json", status)
        write_json(status_path, {"experiment": "TEACH-0003", "status": "complete",
                                 "decision": verdict, "report": "report.json",
                                 "source_git_head": provenance["source_git_head"], **resource})
        return status
    except (ResourceStop, RuntimeError, AssertionError) as exc:
        status.update({"status": "stopped", "decision": {"verdict": "incomplete"},
                       "reason": type(exc).__name__ + ": " + str(exc),
                       "elapsed_seconds": time.monotonic() - start, **resource})
        write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "run"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/TEACH-0003"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/TEACH-0003"))
    args = parser.parse_args()
    config = Config()
    result = (benchmark(config, args.result_dir) if args.mode == "benchmark"
              else run(config, args.output_dir, args.result_dir))
    print(json.dumps({"status": result["status"],
                      "decision": result.get("decision", {}).get("verdict"),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
