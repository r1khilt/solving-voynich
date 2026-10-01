"""Alternate all-case integer decision/comparison accounting; no new scores."""
import json
import signal
import time

from scripts.global_key_read001_common import NAMES, OUT, ROOT, fit_status
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources


def main():
    path = OUT/"evaluation.json"
    value = json.loads(path.read_text())
    panel, _, admitted, _ = fit_status(value["freeze"])
    original_path = ROOT/"results/BLIND-CHANNEL-CONFIRM-002/evaluation.json"
    original = json.loads(original_path.read_text())
    original_audit = json.loads(original_path.with_name("evaluation-audit.json").read_text())
    if (set(value["cases"]) != set(NAMES) or value["original_evaluation"] != artifact(original_path)
            or original_audit["evaluation"] != artifact(original_path) or original_audit["status"] != "PASS"
            or value["prediction_campaign"] != artifact(OUT/"campaign.json")
            or value["prediction_audit"] != artifact(OUT/"audit.json")
            or value["original_fresh_fail_unchanged"] is not True
            or value["exposed_development_not_fresh_qualification"] is not True
            or value["structured_null_or_historical_decipherment_qualified"] is not False):
        raise ValueError("Original/current immutable evaluation identities differ")
    limit_resources(300, 250)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        positives, negatives, comparison_rows = [], [], []
        for name in NAMES:
            row, old, spec = value["cases"][name], original["cases"][name], panel["cases"][name]
            if (row["positive"] != spec["positive"] or row["pair_index"] != spec["pair_index"]
                    or row["positive"] != old["positive"] or row["pair_index"] != old["pair_index"]
                    or not row["fit_complete"] or not row["prediction_complete"] or admitted[name] is None):
                raise ValueError("Case identities/completion differ")
            if not row["positive"]:
                negatives.append(row)
                continue
            for arm in value["totals"]:
                metric = row["arms"][arm]
                if (not metric["available"] or metric["gold_characters"] != 448
                        or len(metric["record_edits"]) != 2
                        or metric["edits"] != sum(metric["record_edits"])
                        or any(type(n) is not int or n < 0 for n in metric["record_edits"])
                        or metric["exact_records"] != sum(n == 0 for n in metric["record_edits"])):
                    raise ValueError("Per-record metric arithmetic differs")
            positives.append((name, row))
            before, after = old["arms"]["mixture"], row["arms"]["mixture"]
            comparison_rows.append({"case": name, "old_edits": before["edits"], "new_edits": after["edits"],
                "edits_reduced": before["edits"]-after["edits"], "old_exact_records": before["exact_records"],
                "new_exact_records": after["exact_records"]})
        if len(positives) != 16 or len(negatives) != 16:
            raise ValueError("Full panel denominator changed")
        for arm, actual in value["totals"].items():
            if actual != {k: sum(r["arms"][arm][k] for _, r in positives) for k in actual}:
                raise ValueError("All-arm totals differ")
            conditional = value["completed_case_metrics_conditional"][arm]
            if conditional != {"cases": 16, "edits": actual["edits"], "gold_characters": 7168}:
                raise ValueError("Completed metrics silently alter denominator")
        errors = [r["arms"]["mixture"]["edits"] for _, r in positives]
        oracle = [r["arms"]["oracle_large"]["edits"] for _, r in positives]
        recovery = {
            "all_32_fits_and_predictions_complete_audited": True,
            "all_32_positive_records_supported": all(r["arms"]["mixture"]["supported_records"] == 2 for _, r in positives),
            "macro_cer_at_most_002": 100*sum(errors) <= 2*7168,
            "every_key_cer_at_most_005": all(100*e <= 5*448 for e in errors),
            "every_key_oracle_excess_at_most_002": all(100*(e-o) <= 2*448 for e, o in zip(errors, oracle, strict=True)),
            "every_key_strictly_beats_frequency_order3": all(r["arms"]["mixture"]["edits"] < r["arms"]["frequency_order3"]["edits"] for _, r in positives)}
        screen = {
            "all_32_fits_and_predictions_complete_audited": True,
            "all_32_evidences_available": all(r["evidence"] is not None for r in value["cases"].values()),
            "all_16_positives_beat_frozen_iid": all(r["evidence"]["gain_bits_vs_iid"] > 0 for _, r in positives),
            "no_shuffle_beats_frozen_iid": all(r["evidence"]["gain_bits_vs_iid"] <= 0 for r in negatives)}
        if (recovery != value["recovery_clauses"] or screen != value["screen_clauses"]
                or value["recovery_gate"] != ("PASS" if all(recovery.values()) else "FAIL")
                or value["iid_screen_gate"] != ("PASS" if all(screen.values()) else "FAIL")):
            raise ValueError("Integer decision clauses differ")
        comparison = value["comparison_to_original_fresh_run"]
        expected = {"cases": comparison_rows, "gold_characters": 7168,
            "old_primary_edits": sum(r["old_edits"] for r in comparison_rows),
            "new_primary_edits": sum(r["new_edits"] for r in comparison_rows),
            "edits_reduced": sum(r["edits_reduced"] for r in comparison_rows),
            "cases_improved": sum(r["edits_reduced"] > 0 for r in comparison_rows),
            "cases_tied": sum(r["edits_reduced"] == 0 for r in comparison_rows),
            "cases_regressed": sum(r["edits_reduced"] < 0 for r in comparison_rows),
            "original_recovery_gate": original["recovery_gate"], "original_iid_screen_gate": original["iid_screen_gate"]}
        if comparison != expected or value["independent_record_edit_checks"] != 192 or value["independent_oracle_records"] != 32:
            raise ValueError("Full comparison/edit/oracle accounting differs")
        outside = [name for name, r in positives if r["truth_diagnosis"]["supporting_keys"] == 0]
        remaining = [(name, r) for name, r in positives if name not in outside]
        save_new(OUT/"evaluation-audit.json", {"status": "PASS", "evaluation": artifact(path),
            "code": artifact(ROOT/"scripts/account_global_key_read001_evaluation.py"),
            "all_32_cases_and_all_arm_totals_and_integer_gates_replayed": True,
            "comparison": expected, "true_tuple_outside_bank_cases": outside,
            "outside_bank_edits": sum(value["cases"][n]["arms"]["mixture"]["edits"] for n in outside),
            "remaining_cases": len(remaining), "remaining_edits": sum(r["arms"]["mixture"]["edits"] for _, r in remaining),
            "remaining_gold_characters": sum(r["arms"]["mixture"]["gold_characters"] for _, r in remaining),
            "remaining_per_key_edit_count_minus_oracle": {n: r["arms"]["mixture"]["edits"]-r["arms"]["oracle_large"]["edits"] for n, r in remaining},
            "equal_reading_strings_not_checked": True, "new_fitting_decoding_or_model_scores_computed": False,
            "independent_agent_review": False, "resources": resource_report(wall, cpu)})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
