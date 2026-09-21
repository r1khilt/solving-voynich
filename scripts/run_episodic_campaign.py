"""Execute the registered synthetic campaign with scoped workers and finite caps."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from run_parallel_campaign import CHILD_ENV, run_campaign, source_state
from voynich.episodic_campaign import (
    prepare, train, evaluate, explicit_comparison, load_model, neural_joint, kl_bits, verify_frozen_runs,
)
from voynich.episodic_causal import audit
from voynich.episodic_data import make_task, sample_task, oracle_joint
from voynich.episodic_models import build_model
from voynich.runtime import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def causal_pools(seed):
    arrays, groups, tasks_by_split = [], [], []
    for split, count in enumerate((8, 4, 8)):
        rows, keys, tasks = [], [], []
        for i in range(count):
            task = make_task("pair_parity", seed + split * 10000 + i)
            tasks.append(task)
            rows.append(sample_task(task, 24, 64, seed + 100000 + split * 10000 + i))
            keys.extend([task["task_id"]] * 24)
        arrays.append(np.concatenate(rows))
        groups.append(np.array(keys))
        tasks_by_split.append(tasks)
    return arrays, groups, tasks_by_split


def causal(config, out, device="cpu"):
    """Separate fresh-key audit; no real manuscript and no hidden-label fitting."""
    out = Path(out)
    verify_frozen_runs(config, out)
    root = out / "causal"
    root.mkdir(exist_ok=True)
    arrays, groups, tasks = causal_pools(config["data_seed"] + 50000000)
    write_json(root / "task_manifest.json", {"tasks": tasks, "prefix_length": 64,
               "contexts_per_task": 24, "source": source_state(ROOT)})
    reports = []
    started = time.monotonic()
    for condition in ("gru_fresh", "signed_delta_fresh"):
        for seed in config["model_seeds"]:
            model, spec = load_model(out / "runs" / f"{condition}-s{seed}", device)
            # Teacher qualification is diagnosed only; it cannot tune a basis.
            q = neural_joint(model, spec, arrays[2], 3, device)
            p = np.concatenate([np.stack([oracle_joint(task, prefix, 3)
                                for prefix in arrays[2][groups[2] == task["task_id"]]]) for task in tasks[2]])
            teacher_kl = float(kl_bits(p, q).mean())
            for rank in (4, 16):
                if time.monotonic() - started > 3600:
                    raise TimeoutError("Registered one-hour causal cap")
                name = f"{condition}-s{seed}-r{rank}"
                if (root / f"{name}.json").exists():
                    raise FileExistsError("Preserve existing causal confirmation; no adaptive rerun")
                torch.manual_seed(seed + 800000)
                untrained = build_model(spec).to(device).eval()
                report, artifacts = audit(model, *arrays, train_groups=groups[0],
                    validation_groups=groups[1], test_groups=groups[2], alphabet=4,
                    rank=rank, steps=120, seed=seed + 700000, horizon=3, batch_size=32,
                    untrained_model=untrained)
                report.update(run=name, spec=asdict(spec), teacher_oracle_joint_kl_bits=teacher_kl,
                              teacher_qualification=teacher_kl <= .30,
                              teacher_qualification_threshold_joint_kl_bits=.30)
                torch.save(artifacts, root / f"{name}.pt")
                report["artifact_sha256"] = digest(root / f"{name}.pt")
                write_json(root / f"{name}.json", report)
                reports.append({"run": name, "teacher_kl_bits": teacher_kl,
                                "teacher_qualified": teacher_kl <= .30,
                                "elapsed_seconds": report["elapsed_seconds"]})
                print(json.dumps(reports[-1]), flush=True)
    write_json(root / "summary.json", {"audits": reports, "elapsed_seconds": time.monotonic() - started,
               "no_manuscript_claim": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "analysis", "causal"))
    parser.add_argument("--config", default="configs/exp0012.json")
    parser.add_argument("--out", default="outputs/EXP-0012")
    parser.add_argument("--track", choices=("gpu", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if args.action == "worker":
        if args.track == "gpu":
            torch.set_num_threads(2)
            torch.mps.set_per_process_memory_fraction(.70)
            train(config, out, "mps", exclude="signed_delta_fresh")
        else:
            train(config, out, "cpu", only="signed_delta_fresh")
        return
    if args.action == "causal":
        causal(config, out)
        return
    if args.action == "analysis":
        torch.mps.set_per_process_memory_fraction(.70)
        evaluate(config, out, "mps")
        explicit_comparison(config, out)
        causal(config, out, "cpu")
        return
    if source_state(ROOT)["git_dirty"]:
        raise ValueError("Freeze and publish source before campaign execution")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Registered GPU unavailable")
    started = time.monotonic()
    initial = source_state(ROOT)
    prepare(config, out)
    CHILD_ENV.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.70", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.60",
                     OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                     VECLIB_MAXIMUM_THREADS="4", PYTHONPATH=str(ROOT / "src"))
    plan = {"campaign": "EXP-0012", "max_seconds": 14400, "rss_ceiling_gib": 40,
            "jobs": [{"name": track, "max_seconds": 14340,
                      "command": [sys.executable, str(Path(__file__).resolve()), "worker", "--config",
                                  args.config, "--out", args.out, "--track", track]} for track in ("gpu", "cpu")]}
    run_campaign(plan, out / "supervisor", root=ROOT)
    status = json.loads((out / "supervisor/status.json").read_text())
    if status["state"] != "completed" or any(x.get("returncode") != 0 for x in status["jobs"].values()):
        raise RuntimeError("A training worker failed; do not open confirmation")
    if source_state(ROOT) != initial:
        raise RuntimeError("Scientific source changed during campaign")
    analysis_plan = {"campaign": "EXP-0012-0013-analysis", "max_seconds": 7200, "rss_ceiling_gib": 40,
                     "jobs": [{"name": "analysis", "max_seconds": 7140,
                               "command": [sys.executable, str(Path(__file__).resolve()), "analysis",
                                           "--config", args.config, "--out", args.out]}]}
    analysis_status = run_campaign(analysis_plan, out / "analysis-supervisor", root=ROOT)
    if analysis_status["state"] != "completed":
        raise RuntimeError("Registered analysis phase failed or stopped")
    if source_state(ROOT) != initial:
        raise RuntimeError("Scientific source changed during analysis")
    write_json(out / "complete.json", {"elapsed_seconds": time.monotonic() - started,
               "source": source_state(ROOT), "no_paid_api": True, "no_manuscript_inputs": True})


if __name__ == "__main__":
    main()
