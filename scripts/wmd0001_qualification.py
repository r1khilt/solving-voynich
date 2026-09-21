"""Frozen WMD-0001 validation, with non-invasive candidate audit sidecars.

Run after training; never opens a manuscript or the final synthetic holdout.
The wrapper observes inference outputs and returns them unchanged to scoring.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from voynich.communication import evaluation
from voynich.communication.denoiser import ConditionalDenoiser
from voynich.communication.schema import save_json
from voynich.communication.training import load_checkpoint, source_manifest
from voynich.runtime import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    trained, layout, checkpoint = load_checkpoint(args.checkpoint, resolve_device(args.device))
    sources = source_manifest()
    if sources != checkpoint["source_sha256"]:
        raise ValueError("Qualification must use the exact training source bytes")
    provenance = {
        "source_sha256": sources,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint["step"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_budget": 20000,
        "compiler_beam": 32,
        "device": args.device,
    }
    original_infer = evaluation.infer
    results = {}
    for condition in ("trained", "untrained"):
        for anchors in (0, 2):
            if condition == "untrained":
                torch.manual_seed(checkpoint["training_config"]["seed"])
                model = ConditionalDenoiser(trained.config).to(args.device)
            else:
                model = trained
            audit = []

            def observed_infer(*positional, **kwargs):
                result = original_infer(*positional, **kwargs)
                audit.append({
                    key: value for key, value in result.items() if key != "candidates"
                } | {"candidates": [
                    {key: value for key, value in candidate.items() if key != "hypothesis"}
                    for candidate in result["candidates"]
                ]})
                return result

            evaluation.infer = observed_infer
            started = time.monotonic()
            try:
                result = evaluation.evaluate_joint(
                    model, layout, seed=41021, count=64, anchor_count=anchors,
                    candidates=4, steps=12,
                )
            finally:
                evaluation.infer = original_infer
            result.update(provenance, model_condition=condition,
                          elapsed_seconds=time.monotonic() - started)
            name = f"{condition}-a{anchors}"
            results[name] = result
            save_json(destination / f"{name}.json", result)
            save_json(destination / f"{name}-audit.json", {
                "schema_version": 1, "model_condition": condition,
                "anchor_count": anchors, "input_sha256": result["input_sha256"],
                **provenance, "inferences": audit,
            })
            print(json.dumps({"condition": condition, "anchors": anchors,
                              "seconds": result["elapsed_seconds"],
                              "metrics": result["metrics"]["neural"]}), flush=True)
    for anchors in (0, 2):
        left, right = (results[f"{name}-a{anchors}"] for name in ("trained", "untrained"))
        assert left["input_sha256"] == right["input_sha256"]
        for baseline in ("random_key", "all_null_except_anchors"):
            assert left["metrics"][baseline] == right["metrics"][baseline]
    assert source_manifest() == sources, "Source changed during qualification"
    save_json(destination / "manifest.json", {
        "schema_version": 1, **provenance,
        "selection": "rank zero fixed before gold scoring",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(destination.glob("*.json"))},
    })


if __name__ == "__main__":
    main()
