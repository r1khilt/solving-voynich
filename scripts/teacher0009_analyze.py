#!/usr/bin/env python3
"""Independent compact-artifact audit for TEACH-0009."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from voynich.workspace.teacher5_intervene import build_groups, sha_file, stable_json
from voynich.workspace.teacher7_dense_mechanism import score
from voynich.workspace.teacher9_qkv import Config, QKV_CELLS


ROOT = Path(__file__).resolve().parents[1]


def select(screens):
    rows = []
    for mask in range(1, 64):
        key = str(mask)
        rows.append((mask, mask.bit_count(),
                     min(row[key]["donor_groups"]["accuracy"] for row in screens.values()),
                     min(row[key]["donor_items"]["accuracy"] for row in screens.values()),
                     min(row[key]["cross_g_non_injection"]["accuracy"]
                         for row in screens.values())))
    qualified = [row for row in rows if row[2] >= .60 and row[4] >= .90]
    if qualified:
        chosen = sorted(qualified, key=lambda row: (row[1], -row[2], -row[3], row[0]))[0]
        flag = True
    else:
        chosen = sorted(rows, key=lambda row: (-row[2], -row[3], row[1], row[0]))[0]
        flag = False
    names = ("f0", "f1", "g0", "g1", "marker", "query")
    return {"mask": chosen[0], "slots": [slot for slot in range(6)
                                            if chosen[0] & (1 << slot)],
            "slot_names": [name for slot, name in enumerate(names)
                           if chosen[0] & (1 << slot)],
            "size": chosen[1], "discovery_qualified": flag,
            "minimum_group_accuracy": chosen[2], "minimum_item_accuracy": chosen[3],
            "minimum_cross_g_non_injection": chosen[4]}


def sufficient(scores, name):
    row = scores[name]
    return (row["donor_items"]["accuracy"] >= .75
            and row["donor_groups"]["accuracy"] >= .60
            and row["cross_g_non_injection"]["accuracy"] >= .90)


def decision(selection, metrics):
    clauses = {}
    for rep, row in metrics.items():
        scores = row["scores"]
        clauses[rep] = {
            "clean_base_at_least_0.95": scores["clean_base"]["base_items"]["accuracy"] >= .95,
            "clean_donor_at_least_0.95": scores["clean_donor"]["donor_items"]["accuracy"] >= .95,
            "f_positive_at_least_0.95": scores["f_positive"]["donor_items"]["accuracy"] >= .95,
            "ddd_sufficient": sufficient(scores, "qkv_ddd"),
            "all_sources_sufficient": sufficient(scores, "all_sources"),
            "selected_sources_sufficient": sufficient(scores, "selected_sources"),
            "random_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["random"]["donor_items"]["accuracy"] >= .40),
            "mismatch_advantage_at_least_0.40": (
                scores["selected_sources"]["donor_items"]["accuracy"]
                - scores["mismatch"]["donor_items"]["accuracy"] >= .40),
            "numerical_errors_below_1e-6": max(row["numerical_errors"].values()) < 1e-6,
        }
    supported = selection["discovery_qualified"] and all(
        all(row.values()) for row in clauses.values())
    qk = all(sufficient(row["scores"], "qkv_ddb") for row in metrics.values())
    value = all(sufficient(row["scores"], "qkv_bbd") for row in metrics.values())
    full = all(sufficient(row["scores"], "qkv_ddd") for row in metrics.values())
    if qk and value:
        label = "qk_and_v_each_sufficient"
    elif qk:
        label = "qk_only_sufficient"
    elif value:
        label = "v_only_sufficient"
    elif full:
        label = "qk_v_conjunctive"
    else:
        label = "unresolved"
    return {"source_edge": "supported" if supported else "not_supported",
            "qkv_label": label, "clauses_by_seed": clauses}


def verify_sources(report):
    for path, expected in report["source_sha256"].items():
        raw = subprocess.run(["git", "show", f"{report['source_git_head']}:{path}"],
                             cwd=ROOT, capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AssertionError(f"Source mismatch: {path}")


def audit(result_dir=ROOT / "results" / "TEACH-0009",
          checkpoint_dir=ROOT / "outputs" / "TEACH-0004"):
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    config = Config(**report["config"])
    config.validate()
    if report.get("experiment") != "TEACH-0009" or report.get("status") != "complete":
        raise AssertionError("Incomplete report")
    if report.get("selected_heads") != [1, 3]:
        raise AssertionError("Selected heads changed")
    verify_sources(report)
    groups = build_groups(config.suite_seed,
                          config.discovery_groups + config.confirmation_groups)
    if hashlib.sha256(stable_json(groups).encode()).hexdigest() != report["suite_sha256"]:
        raise AssertionError("Suite mismatch")
    if select(report["discovery_screen"]) != report["source_selection"]:
        raise AssertionError("Source selection mismatch")
    confirmation = groups[config.discovery_groups:]
    conditions = tuple(f"qkv_{cell.lower()}" for cell in QKV_CELLS) + (
        "selected_sources", "all_sources", "both_f", "non_f", "queried_f", "other_f",
        "random", "mismatch", "clean_base", "clean_donor", "f_positive")
    rescored = {}
    for rep in ("0", "1"):
        checkpoint = checkpoint_dir / f"rep{rep}-dense_row.pt"
        if sha_file(checkpoint) != report["artifacts"][rep]["checkpoint_sha256"]:
            raise AssertionError("Checkpoint mismatch")
        path = result_dir / f"rows-rep{rep}.json.gz"
        if sha_file(path) != report["artifacts"][rep]["rows_sha256"]:
            raise AssertionError("Rows mismatch")
        rows = json.loads(gzip.decompress(path.read_bytes()))
        if len(rows) != config.confirmation_groups * config.g_tables:
            raise AssertionError("Row count mismatch")
        for index, row in enumerate(rows):
            group, g_index = divmod(index, config.g_tables)
            if row["group"] != group or row["g_index"] != g_index or \
                    row["base_answer"] != confirmation[group]["base_answers"][g_index] or \
                    row["donor_answer"] != confirmation[group]["donor_answers"][g_index]:
                raise AssertionError("Row mismatch")
        base = torch.tensor([row["base_answer"] for row in rows])
        donor = torch.tensor([row["donor_answer"] for row in rows])
        rescored[rep] = {}
        for condition in conditions:
            pred = torch.tensor([row[f"{condition}_prediction"] for row in rows])
            rescored[rep][condition] = score(pred, donor, base,
                                             config.confirmation_groups, config.g_tables)
        if stable_json(rescored[rep]) != stable_json(report["metrics"][rep]["scores"]):
            raise AssertionError("Score mismatch")
        if max(report["metrics"][rep]["numerical_errors"].values()) >= 1e-6:
            raise AssertionError("Numerical mismatch")
        for state in ("base", "donor_g0_repeated"):
            for head in ("1", "3"):
                mass = report["metrics"][rep]["attention_mass"][state][head]
                if abs(sum(mass.values()) - 1) > 1e-5:
                    raise AssertionError("Attention mass mismatch")
    verdict = decision(report["source_selection"], report["metrics"])
    if verdict != report["decision"]:
        raise AssertionError("Decision mismatch")
    payload = {"experiment": "TEACH-0009", "audit": "pass",
               "source_git_head": report["source_git_head"],
               "source_selection": report["source_selection"],
               "decision": verdict, "primary_scores": rescored,
               "suite_sha256": report["suite_sha256"],
               "report_sha256": sha_file(report_path)}
    (result_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    result = audit()
    print(json.dumps({"audit": result["audit"],
                      "source_selection": result["source_selection"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
