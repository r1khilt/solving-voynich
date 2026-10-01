"""Frozen random-weight full-size neural prefix benchmark; no cipher inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.fixed_transition_provider import FixedTransitionProvider
from voynich.recurrent_latin_source import ALPHABET, RecurrentSource, score_records
from voynich.recurrent_shared_prefix import decode_shared_prefix
from voynich.recurrent_unit_beam import RecurrentProvider

EXP = "SHARED-PREFIX-SYSTEMS-001"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
ARMS, SEEDS = ("variable", "fixed8-host"), (71831, 71839)
SHAPES = ((32, 1, 32), (128, 64, 128), (224, 1024, 128))
PATHS = ["src/voynich/recurrent_shared_prefix.py", "src/voynich/fixed_transition_provider.py",
         "src/voynich/recurrent_latin_source.py", "src/voynich/recurrent_unit_beam.py",
         "scripts/benchmark_shared_prefix_systems001.py", "scripts/audit_shared_prefix_systems001.py",
         "scripts/run_blind_channel_dev001.py", "scripts/run_blind_channel_dev004.py",
         "scripts/run_latin_source_model001.py", "tests/test_fixed_transition_provider.py",
         "tests/test_shared_prefix_systems001.py", "tests/test_recurrent_shared_prefix.py",
         "docs/experiments/SHARED-PREFIX-SYSTEMS-001.md", "uv.lock"]


def workloads():
    rng = np.random.default_rng(71821)
    pool = tuple("".join(row) for length in (1, 2) for row in itertools.product("ABCDEF", repeat=length))
    result = []
    for letters, count, beam in SHAPES:
        keys = []
        while len(keys) < count:
            key = tuple(rng.choice(pool, size=len(ALPHABET)).tolist())
            if key not in keys:
                keys.append(key)
        plain = ["".join(rng.choice(list(ALPHABET), size=letters)) for _ in range(2)]
        table = dict(zip(ALPHABET, keys[0], strict=True))
        records = ["".join(table[c] for c in text) for text in plain]
        result.append({"letters": letters, "key_count": count, "beam": beam, "keys": keys,
                       "plain": plain, "records": records, "log_weights": [-math.log(count)]*count})
    return result


def literal_checks(model, item, reading):
    text = reading["plaintexts"]
    if text is None or len(text) != 2:
        raise ValueError("Artificial generated support lost")
    compatible = []
    for i, key in enumerate(item["keys"]):
        table = dict(zip(ALPHABET, key, strict=True))
        if ["".join(table[c] for c in t) for t in text] == item["records"]:
            compatible.append(i)
    if tuple(compatible) != reading["compatible_key_indices"] or not compatible:
        raise ValueError("Shared key literal replay failed")
    reference = copy.deepcopy(model).to("cpu").double().eval()
    value = score_records(reference, list(text), "cpu", batch=2, length=max(map(len, text)))
    score = -value["bits"]*math.log(2)+sum(map(len, text))*math.log1p(-1/225)+2*math.log(1/225)
    score += math.log(len(compatible))-math.log(item["key_count"])
    delta = abs(score-reading["joint_log_probability"])
    if delta > .002:
        raise ValueError("Independent float64 full-history score mismatch")
    return {"full_sequence_cpu_float64_score": score, "maximum_delta": delta,
            "compatible_keys_reencoded": len(item["keys"]), "returned_source_letters": sum(map(len, text))}


def check_memory(row):
    if row["driver_bytes"] > 2*1024**3 or row["peak_rss_bytes"] > 4*1024**3:
        raise MemoryError("Sampled 2GiB driver / 4GiB host guard")


class MeasuredRecurrentProvider(RecurrentProvider):
    """Keep instrumentation local to one provider without callback self-cycles."""
    def __init__(self, model, device, after_batch):
        super().__init__(model, device)
        self.after_batch, self.calls, self.real_rows = after_batch, 0, 0

    def advance(self, tokens, states):
        probabilities, following = super().advance(tokens, states)
        self.calls += 1
        self.real_rows += len(tokens)
        self.after_batch({"calls": self.calls, "real_rows": self.real_rows,
                          "padded_rows": 0, "shape": [len(tokens), 1]})
        return probabilities, following


def run_arm(arm, freeze):
    require_frozen(freeze, PATHS)
    if arm not in ARMS or not torch.backends.mps.is_available():
        raise ValueError("Registered arm and MPS required")
    save_new(OUT/f"{arm}-started.json", {"freeze": freeze, "start_unix": time.time(),
             "torch": str(torch.__version__), "numpy": np.__version__, "device": "mps", "cpu_threads": 2})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    items = workloads()
    input_spec = save_new(BULK/f"{arm}-inputs.json.gz", {"seed": 71821, "items": items}, compressed=True)
    trace, completed = [], []
    BULK.mkdir(parents=True, exist_ok=True)
    path = BULK/f"{arm}-trace.jsonl"
    with path.open("x") as stream:
        def sample(info):
            torch.mps.synchronize()
            row = {**info, "driver_bytes": torch.mps.driver_allocated_memory(),
                   "tensor_bytes": torch.mps.current_allocated_memory(), **resource_report(wall, cpu)}
            trace.append(row)
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            check_memory(row)
        try:
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = RecurrentSource().to("mps").eval()
                weights = hashlib.sha256(b"".join(t.cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
                sample({"seed": seed, "stage": "loaded"})
                for index, item in enumerate(items):
                    def after_batch(info):
                        sample({"seed": seed, "item": index, "stage": "transition", **info})
                    provider = (FixedTransitionProvider(model, "mps", batch=8,
                                after_batch=after_batch)
                                if arm == "fixed8-host" else MeasuredRecurrentProvider(model, "mps", after_batch))
                    began, cpu_began = time.monotonic(), time.process_time()
                    reading = decode_shared_prefix(provider, item["keys"], item["records"], 1/225,
                                log_weights=item["log_weights"], beam_width=item["beam"], max_expanded=100_000,
                                max_source_prefixes=100_000, max_channel_cells=10_000_000)
                    elapsed, cpu_used = time.monotonic()-began, time.process_time()-cpu_began
                    checks = literal_checks(model, item, reading)
                    sample({"seed": seed, "item": index, "stage": "verified"})
                    saved = save_new(BULK/f"{arm}-{seed}-{index}.json.gz", {"reading": reading, "checks": checks}, compressed=True)
                    row = {"seed": seed, "item": index, "parameters": sum(p.numel() for p in model.parameters()),
                           "initial_weights_sha256": weights, "reading": saved, "checks": checks,
                           "key_count": item["key_count"], "beam": item["beam"], "source_letters_generated": item["letters"]*2,
                           "observed_glyphs": sum(map(len, item["records"])), "wall_seconds": elapsed,
                           "cpu_seconds": cpu_used, "counters": {k: v for k, v in reading.items() if k not in
                           ("plaintexts", "compatible_key_indices", "joint_log_probability", "discarded_completion_upper_bound")}}
                    completed.append(row)
                    print(json.dumps({"arm": arm, "seed": seed, "item": index, "seconds": elapsed,
                                      "expanded": reading["expanded_prefixes"], "float64_delta": checks["maximum_delta"]}), flush=True)
                    del provider
                    torch.mps.empty_cache()
                del model
                torch.mps.empty_cache()
            save_new(OUT/f"{arm}.json", {"status": "PASS", "arm": arm, "freeze": freeze, "inputs": input_spec,
                     "completed": completed, "trace": artifact(path), "sampled_driver_peak": max(r["driver_bytes"] for r in trace),
                     "sampled_tensor_peak": max(r["tensor_bytes"] for r in trace), "resources": resource_report(wall, cpu)})
        except Exception as error:
            signal.alarm(0)
            save_new(OUT/f"{arm}-failure.json", {"status": "FAIL", "arm": arm, "freeze": freeze, "inputs": input_spec,
                     "type": type(error).__name__, "error": str(error), "completed": completed,
                     "trace": artifact(path), "last_sample": trace[-1] if trace else None, "resources": resource_report(wall, cpu)})
            raise
        finally:
            signal.alarm(0)


def campaign(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/"campaign-started.json", {"freeze": freeze, "start_unix": time.time(), "arms": ARMS})
    BULK.mkdir(parents=True, exist_ok=True)
    wall = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    processes = []
    for arm in ARMS:
        began = time.monotonic()
        with (BULK/f"{arm}.log").open("x") as log:
            try:
                result = subprocess.run([sys.executable, "-u", __file__, "arm", "--arm", arm, "--freeze", freeze],
                       cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, timeout=615, check=False)
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
            except OSError as error:
                code, timed_out = None, False
                log.write(str(error)+"\n")
        row = {"arm": arm, "returncode": code, "outer_timeout": timed_out,
               "outer_wall_seconds": time.monotonic()-began, "log": artifact(BULK/f"{arm}.log")}
        processes.append(row)
        print(json.dumps(row), flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save_new(OUT/"campaign.json", {"freeze": freeze, "processes": processes, "wall_seconds": time.monotonic()-wall,
             "all_children_cpu_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
             "paid_spend_usd": 0, "only_artificial_inputs_and_untrained_models": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("campaign", "arm"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    if args.mode == "campaign":
        campaign(args.freeze)
    else:
        run_arm(args.arm, args.freeze)
