"""Independent TEACH-0014 candidate-edge and symbolic-solver audit.

This reads visible episode streams and archived gate logits only. It imports no
generator, model, trainer or neural prediction. Candidate indices are the
adjacent ordinary-symbol occurrences in their visible order. No matching is
needed because each candidate has an unambiguous occurrence pair.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from scripts.teacher0014_suite_audit import EDGE, SYMBOL_START, audit_manifest


RAW_ARMS = {
    "latent_rows_answer", "latent_rows_edge_aux", "latent_rows_causal",
    "latent_rows_wrong_causal", "latent_rows_one_read",
    "latent_rows_mean_address", "latent_rows_recurrent4",
    "latent_rows_diffuse", "raw_null",
}
THRESHOLD_LOGIT = 0.0


def _fraction(correct: int, total: int) -> dict:
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0}


def _symbolic_answer(rows: list[tuple[int, int]], query: int,
                     hops: int) -> int | None:
    mapping = {}
    for left, right in rows:
        if left in mapping:
            return None
        mapping[left] = right
    value = query
    for _ in range(hops):
        if value not in mapping:
            return None
        value = mapping[value]
    return value


def _new_counts() -> dict:
    return {"true_positive_rows": 0, "predicted_rows": 0,
            "true_signal_rows": 0, "found_signal_rows": 0,
            "exact_tables": 0, "items": 0,
            "symbolic_two_hop_correct": 0, "symbolic_two_hop_items": 0,
            "factorial_exact_groups": 0, "factorial_groups": 0}


def _rates(counts: dict) -> dict:
    return {
        "all_row_precision": _fraction(counts["true_positive_rows"],
                                       counts["predicted_rows"]),
        "signal_row_recall": _fraction(counts["found_signal_rows"],
                                       counts["true_signal_rows"]),
        "complete_table_exact": _fraction(counts["exact_tables"],
                                          counts["items"]),
        "symbolic_two_hop_items": _fraction(
            counts["symbolic_two_hop_correct"],
            counts["symbolic_two_hop_items"]),
        "symbolic_factorial_groups": _fraction(
            counts["factorial_exact_groups"], counts["factorial_groups"]),
    }


def audit_parser(manifest: dict, archive: dict) -> dict:
    """Score the preregistered zero-logit candidate threshold on every item."""
    suite = audit_manifest(manifest)
    if (archive.get("experiment") != "TEACH-0014" or
            archive.get("manifest_sha256") != suite["manifest_sha256"] or
            type(archive.get("threshold_logit")) not in (int, float) or
            archive["threshold_logit"] != THRESHOLD_LOGIT):
        raise ValueError("Parser archive identity or threshold mismatch")
    runs = archive.get("runs")
    if not isinstance(runs, dict) or set(runs) != RAW_ARMS:
        raise ValueError("Parser arm set incomplete")
    scored = {}
    for arm, replicates in runs.items():
        if not isinstance(replicates, dict) or set(replicates) != {"0", "1"}:
            raise ValueError(f"{arm}: parser replicate set incomplete")
        scored[arm] = {}
        for rep, panels in replicates.items():
            if not isinstance(panels, dict) or set(panels) != set(manifest["panels"]):
                raise ValueError(f"{arm}/{rep}: parser panel set incomplete")
            counts = {"marked": _new_counts(), "marker_free": _new_counts(),
                      "all": _new_counts()}
            for name, episodes in manifest["panels"].items():
                gate_rows = panels[name]
                if not isinstance(gate_rows, list) or len(gate_rows) != len(episodes):
                    raise ValueError(f"{arm}/{rep}/{name}: parser item coverage")
                factorial_hits = []
                for episode, gate_row in zip(episodes, gate_rows, strict=True):
                    if (not isinstance(gate_row, dict) or
                            set(gate_row) != {"render_id", "logits"} or
                            gate_row["render_id"] != episode["render_id"]):
                        raise ValueError(f"{arm}/{rep}/{name}: parser item identity")
                    ordinary = [value for value in episode["tokens"][1:-3]
                                if value >= SYMBOL_START]
                    logits = gate_row["logits"]
                    if (not isinstance(logits, list) or
                            len(logits) != len(ordinary) - 1 or
                            any(type(value) not in (int, float) or
                                not math.isfinite(value) for value in logits)):
                        raise ValueError(f"{arm}/{rep}/{name}: invalid gate logits")
                    selected = {index for index, value in enumerate(logits)
                                if value > THRESHOLD_LOGIT}
                    true = set(range(0, len(logits), 2))
                    selected_rows = [
                        (ordinary[index], ordinary[index + 1])
                        for index in sorted(selected)]
                    signals = {(left, right)
                               for path in episode["signal_paths"]
                               for left, right in zip(path, path[1:])}
                    found_signals = sum(
                        index in true and (ordinary[index], ordinary[index + 1])
                        in signals for index in selected)
                    symbolic = _symbolic_answer(
                        selected_rows, episode["query"], episode["hops"])
                    hit = symbolic == episode["answer"]
                    if name == "factorial":
                        factorial_hits.append(hit)
                    marked = EDGE in episode["tokens"][1:-3]
                    for group in ("all", "marked" if marked else "marker_free"):
                        bucket = counts[group]
                        bucket["true_positive_rows"] += len(selected & true)
                        bucket["predicted_rows"] += len(selected)
                        bucket["true_signal_rows"] += len(signals)
                        bucket["found_signal_rows"] += found_signals
                        bucket["exact_tables"] += selected == true
                        bucket["items"] += 1
                        if episode["task"] == "composed" and episode["hops"] == 2:
                            bucket["symbolic_two_hop_correct"] += hit
                            bucket["symbolic_two_hop_items"] += 1
                if name == "factorial":
                    if len(factorial_hits) % 4:
                        raise ValueError("Incomplete factorial group")
                    counts["all"]["factorial_exact_groups"] += sum(
                        all(factorial_hits[index:index + 4])
                        for index in range(0, len(factorial_hits), 4))
                    counts["all"]["factorial_groups"] += len(factorial_hits) // 4
            rates = {key: _rates(value) for key, value in counts.items()}
            marked = rates["marked"]
            free = rates["marker_free"]
            overall = rates["all"]
            qualified = (
                marked["all_row_precision"]["accuracy"] >= .95 and
                marked["signal_row_recall"]["accuracy"] >= .95 and
                free["all_row_precision"]["accuracy"] >= .90 and
                free["signal_row_recall"]["accuracy"] >= .90 and
                marked["complete_table_exact"]["accuracy"] >= .85 and
                free["complete_table_exact"]["accuracy"] >= .70 and
                overall["symbolic_two_hop_items"]["accuracy"] >= .90 and
                overall["symbolic_factorial_groups"]["accuracy"] >= .85)
            scored[arm][rep] = {"counts": counts, "rates": rates,
                                "parser_qualified": qualified}
    return {"audit": "pass", "scope": "synthetic_candidate_edges_and_symbolic_solver",
            "manifest_sha256": suite["manifest_sha256"],
            "threshold_logit": THRESHOLD_LOGIT, "runs": scored,
            "answer_only_parser_qualified_both_seeds": all(
                scored["latent_rows_answer"][rep]["parser_qualified"]
                for rep in ("0", "1"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    archive = json.loads(gzip.decompress(args.archive.read_bytes()))
    raw = json.dumps(audit_parser(manifest, archive), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
