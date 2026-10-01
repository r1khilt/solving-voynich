"""Read-only publication/version/work accounting; no new model inference."""
import json
import math
import time

from scripts.benchmark_key_proposal_systems001 import (ARMS, DATA_SEED, MODEL_SEED, OUT, PARAMETERS,
    PATHS, SAMPLE_SEED, STEPS)
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact


def account():
    wall, cpu = time.monotonic(), time.process_time()
    bound = {}

    def verify(spec):
        if artifact(ROOT/spec["path"]) != spec or (spec["path"] in bound and bound[spec["path"]] != spec):
            raise ValueError("Publication artifact binding differs")
        bound[spec["path"]] = spec

    def read(name):
        path = OUT/name
        verify(artifact(path))
        return json.loads(path.read_text())

    campaign, audit = read("campaign.json"), read("audit.json")
    freeze = campaign["freeze"]
    require_frozen(freeze, PATHS)
    started = read("campaign-started.json")
    audit_started = read("audit-started.json")
    if (started["freeze"] != freeze or audit_started["freeze"] != freeze
            or started["arms"] != list(ARMS) or [p["arm"] for p in campaign["processes"]] != list(ARMS)
            or audit["status"] != "PASS" or audit["campaign"] != artifact(OUT/"campaign.json")
            or set(audit["arms"]) != set(ARMS)):
        raise ValueError("Complete source-frozen publication inventory differs")
    verify(audit["auditor"])
    rows = {}
    for process in campaign["processes"]:
        arm = process["arm"]
        verify(process["log"])
        value = read(arm+".json")
        header = read(arm+"-started.json")
        checked = audit["arms"][arm]
        if (process["returncode"] != 0 or process["outer_timeout"] or value["status"] != "PASS"
                or checked["status"] != "PASS" or checked["result"] != artifact(OUT/(arm+".json"))
                or value["freeze"] != freeze or header["freeze"] != freeze
                or (header["model_seed"], header["data_seed"], header["sample_seed"], header["cpu_threads"])
                   != (MODEL_SEED, DATA_SEED, SAMPLE_SEED, 2)
                or header["cpu_mha_fastpath"] is not False
                or (header["torch"], header["numpy"]) != ("2.14.0", "2.5.3")
                or value["resources"]["python"] != "3.12.13"
                or value["random_optimizer_updates"] != STEPS or checked["optimizer_updates"] != STEPS
                or value["language_training_or_cipher_panel_files_opened"] is not False
                or value["parameters"] != PARAMETERS[arm.split("-b")[0]]):
            raise ValueError("Complete arm/version/random work differs")
        for field in ("inputs", "initial", "final", "trace"):
            verify(value[field])
        for spec in checked["retained_partial_artifacts"]:
            verify(spec)
        timed = [s["wall_seconds"] for s in value["optimizer_steps"] if s["timed"]]
        mean = math.fsum(timed)/len(timed)
        if len(timed) != 12 or abs(mean-value["mean_timed_step_seconds"]) > 1e-12:
            raise ValueError("Timing arithmetic differs")
        rows[arm] = {"parameters": value["parameters"], "batch": value["batch"], "step_seconds": mean,
                     "episodes_per_second": value["batch"]/mean,
                     "driver_gib": value["sampled_driver_peak"]/1024**3,
                     "host_gib": value["resources"]["peak_rss_bytes"]/1024**3,
                     "proposal_seconds": value["proposal_wall_seconds"],
                     "full_logq_max_delta": checked["replay"]["full_proposal_delta"],
                     "initial_weights_sha256": value["initial_weights_sha256"]}
    for scale in ("small", "large"):
        if rows[scale+"-b1"]["initial_weights_sha256"] != rows[scale+"-b4"]["initial_weights_sha256"]:
            raise ValueError("Paired initialization differs")
    if audit["random_optimizer_updates"] != 60 or campaign["paid_spend_usd"] != 0:
        raise ValueError("Total work/spend differs")
    save_new(OUT/"publication-accounting.json", {"status": "PASS", "freeze": freeze, "arms": rows,
        "bindings": sorted(bound.values(), key=lambda s: s["path"]),
        "checkpoint_bytes": sum(s["bytes"] for s in bound.values() if s["path"].endswith(".pt")),
        "random_optimizer_updates": 60, "language_training_updates": 0, "new_model_inference": False,
        "independent_agent_review": False, "accounting_code": artifact(ROOT/"scripts/account_key_proposal_systems001.py"),
        "resources": resource_report(wall, cpu)})


if __name__ == "__main__":
    account()
