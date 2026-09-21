"""Bounded hardware timing and analysis-path parity on artificial integer inputs."""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer
from voynich.runtime import environment, write_json


def benchmark(steps=20):
    torch.set_num_threads(4)
    torch.manual_seed(829)
    cfg = ModelConfig(vocab_size=112, dropout=0.1)
    reference = VoynichTransformer(cfg)
    ids = torch.randint(1, 112, (8, 256))
    targets = torch.randint(1, 112, (8, 256))
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {"environment": environment(), "steps": steps, "data": "random integer inputs, no manuscript",
              "measurements": {}}
    for device in devices:
        model = VoynichTransformer(cfg).to(device)
        model.load_state_dict(reference.state_dict())
        x, y = ids.to(device), targets.to(device)
        def synchronize():
            if device == "mps":
                torch.mps.synchronize()
        model.eval()
        with torch.no_grad():
            fast = model(x).logits
            explicit = model(x, cache_names=["blocks.0.attn.result"]).logits
            delta = float((fast - explicit).abs().max().cpu())
            if not torch.allclose(fast, explicit, atol=2e-5, rtol=2e-5):
                raise ValueError(f"Analysis parity failed on {device}: {delta}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
        model.train()
        times = []
        for step in range(steps + 5):
            synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).logits.flatten(0, 1), y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            if step >= 5:
                times.append(time.perf_counter() - started)
        report["measurements"][device] = {
            "mean_seconds_per_step": sum(times) / len(times), "analysis_max_abs_difference": delta,
            "examples_per_batch": 8, "context": 256, "warmup_steps": 5,
        }
    report["selected_device"] = min(report["measurements"],
                                     key=lambda device: report["measurements"][device]["mean_seconds_per_step"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="results/EXP-0002/hardware.json")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Hardware probe must use between 1 and 100 measured steps")
    result = benchmark(args.steps)
    write_json(args.output, result)
    print(json.dumps(result["measurements"], indent=2))
    print("Selected:", result["selected_device"])
