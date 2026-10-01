"""Explicit separately frozen serialization correction; retain original FAIL."""
import argparse
import json
from contextlib import contextmanager

from scripts import benchmark_shared_prefix_systems001 as engine
from scripts import audit_shared_prefix_systems001 as auditor
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import save_new
from scripts.run_latin_source_model001 import ROOT, artifact

EXP = "SHARED-PREFIX-SYSTEMS-002"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
FAIL = ROOT/"results/SHARED-PREFIX-SYSTEMS-001"
PATHS = sorted(set(engine.PATHS+[
    "scripts/benchmark_shared_prefix_systems002.py", "tests/test_shared_prefix_systems002.py",
    "docs/experiments/SHARED-PREFIX-SYSTEMS-002.md",
    "results/SHARED-PREFIX-SYSTEMS-001/campaign.json", "results/SHARED-PREFIX-SYSTEMS-001/audit.json",
    "results/SHARED-PREFIX-SYSTEMS-001/variable-failure.json",
    "results/SHARED-PREFIX-SYSTEMS-001/fixed8-host-failure.json",
]))


@contextmanager
def configured():
    replacements = [(engine, "EXP", EXP), (engine, "OUT", OUT), (engine, "BULK", BULK),
                    (engine, "PATHS", PATHS), (engine, "__file__", str(ROOT/"scripts/benchmark_shared_prefix_systems002.py")),
                    (auditor, "OUT", OUT), (auditor, "PATHS", PATHS)]
    saved = [(module, key, getattr(module, key)) for module, key, _ in replacements]
    try:
        for module, key, value in replacements:
            setattr(module, key, value)
        yield
    finally:
        for module, key, value in saved:
            setattr(module, key, value)


def previous_failure():
    campaign = json.loads((FAIL/"campaign.json").read_text())
    audit = json.loads((FAIL/"audit.json").read_text())
    if (audit["status"] != "PASS" or audit["campaign"] != artifact(FAIL/"campaign.json")
            or [p["arm"] for p in campaign["processes"]] != list(engine.ARMS)
            or any(p["returncode"] != 1 for p in campaign["processes"])):
        raise ValueError("Entire original terminal failed campaign required")
    bindings = {"campaign": artifact(FAIL/"campaign.json"), "audit": artifact(FAIL/"audit.json")}
    for arm in engine.ARMS:
        path = FAIL/f"{arm}-failure.json"
        failure = json.loads(path.read_text())
        if (failure["type"] != "TypeError" or "JSON serializable" not in failure["error"]
                or failure["completed"] or audit["arms"][arm]["result"] != artifact(path)
                or audit["arms"][arm]["status"] != "retained_failure"):
            raise ValueError("Original serialization failure identity differs")
        bindings[arm] = artifact(path)
    return bindings


def run(mode, freeze=None, arm=None):
    if mode not in ("campaign", "arm", "audit"):
        raise ValueError("Unregistered stage")
    if mode == "audit":
        binding = json.loads((OUT/"execution-binding.json").read_text())
        freeze = binding["freeze"]
        require_frozen(freeze, PATHS)
        if binding["previous_failure"] != previous_failure() or binding["wrapper"] != artifact(ROOT/"scripts/benchmark_shared_prefix_systems002.py"):
            raise ValueError("Correction execution binding differs")
        with configured():
            auditor.audit()
        return
    require_frozen(freeze, PATHS)
    previous = previous_failure()
    if mode == "campaign":
        save_new(OUT/"execution-binding.json", {"freeze": freeze, "previous_failure": previous,
                 "wrapper": artifact(ROOT/"scripts/benchmark_shared_prefix_systems002.py"),
                 "only_algorithm_change": "native Python probability scalars at full-history cache insertion",
                 "unchanged_workloads_seeds_models_scores_caps_and_timers": True,
                 "engineering_correction_after_exposure_not_fresh_scientific_qualification": True})
        with configured():
            engine.campaign(freeze)
    else:
        with configured():
            engine.run_arm(arm, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("campaign", "arm", "audit"))
    parser.add_argument("--arm", choices=engine.ARMS)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    run(args.mode, args.freeze, args.arm)
