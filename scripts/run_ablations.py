"""Launch explicitly selected, bounded local ablations; dry-run unless --execute."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["small", "reference", "mtp", "qk_norm", "gated_attention"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--data", default="data/processed/zl3b")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.configs)) != len(args.configs):
        parser.error("Need positive steps and unique configs/seeds")
    commands = []
    for name in args.configs:
        if name not in {"small", "reference", "mtp", "qk_norm", "gated_attention", "attention_only", "smoke"}:
            parser.error(f"Unknown config: {name}")
        for seed in args.seeds:
            path = Path(args.output_root) / f"{name}-seed{seed}"
            commands.append([sys.executable, "-m", "voynich.train", "--config", f"configs/{name}.json",
                             "--seed", str(seed), "--steps", str(args.steps), "--data", args.data,
                             "--run-dir", str(path), "--device", args.device])
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": len(commands),
            "maximum_optimizer_steps": len(commands) * args.steps, "commands": commands,
            "test_evaluation": False, "paid_api_calls": False,
            "note": "Matched steps/batch/context approximate equal token exposure; early stopping can differ. Compare manifests."}
    print(json.dumps(plan, indent=2))
    if args.execute:
        root = Path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.json"
        if plan_path.exists():
            raise ValueError("Choose a fresh output root; plans are never overwritten")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
