"""Independent TEACH-0014 answer-behavior scorer; no model/trainer imports.

This is a design-phase scorer, not the complete checkpoint/resource/causal
outcome auditor. A final registration must freeze its exact source and seeds.
"""

import argparse
import json
from pathlib import Path

from scripts.teacher0014_suite_audit import audit_manifest


ARMS = {
    "oracle_rows_workspace", "latent_rows_answer", "latent_rows_edge_aux",
    "latent_rows_causal", "latent_rows_wrong_causal", "latent_rows_one_read",
    "latent_rows_mean_address", "raw_dense_matched",
    "latent_rows_recurrent4", "latent_rows_diffuse", "raw_null",
}
GROUPED = {
    "first_hop_query_groups", "direct_query_groups", "factorial",
    "order_groups", "boundary_groups", "distractor_groups",
}
CROSSED = (
    "composed_train_train", "composed_confirm_train",
    "composed_train_confirm", "composed_confirm_confirm",
)


def _fraction(correct: list[bool]) -> dict[str, int | float]:
    return {"correct": sum(correct), "total": len(correct),
            "accuracy": sum(correct) / len(correct)}


def _score_panel(name: str, truth: list[int], predicted: list[int]
                 ) -> dict[str, dict[str, int | float]]:
    hits = [a == b for a, b in zip(truth, predicted, strict=True)]
    result = {"items": _fraction(hits)}
    if name in GROUPED:
        if len(hits) % 4:
            raise ValueError(f"{name}: incomplete four-item groups")
        result["groups"] = _fraction([
            all(hits[index:index + 4]) for index in range(0, len(hits), 4)])
    if name == "boundary_groups":
        result["marker_free"] = _fraction(hits[3::4])
    return result


def _pass_twohop(row: dict) -> bool:
    panels = row["panels"]

    def item(name: str) -> float:
        return panels[name]["items"]["accuracy"]

    def group(name: str) -> float:
        return panels[name]["groups"]["accuracy"]
    return (all(item(name) >= .90 for name in CROSSED)
            and item("first_hop_confirm") >= .95
            and item("direct_confirm") >= .95
            and item("copy_confirm") >= .98
            and group("first_hop_query_groups") >= .85
            and group("direct_query_groups") >= .85
            and group("factorial") >= .75
            and group("order_groups") >= .80
            and group("distractor_groups") >= .80
            and group("boundary_groups") >= .70
            and panels["boundary_groups"]["marker_free"]["accuracy"] >= .80
            and item("long_ood") >= .80)


def _pass_extrapolation(row: dict) -> bool:
    panels = row["panels"]
    return (panels["hop_3"]["items"]["accuracy"] >= .75
            and panels["hop_4"]["items"]["accuracy"] >= .60
            and panels["hop_4_long_ood"]["items"]["accuracy"] >= .50)


def _margin(raw: dict, control: dict, panel: str, measure: str) -> float:
    return (raw["panels"][panel][measure]["accuracy"]
            - control["panels"][panel][measure]["accuracy"])


def _raw_advantage(raw: dict, control: dict) -> bool:
    return all(
        _margin(raw, control, panel, measure) >= .15
        for panel, measure in (
            ("factorial", "groups"), ("boundary_groups", "marker_free"),
            ("long_ood", "items")))


def audit_behavior(manifest: dict, artifact: dict) -> dict:
    """Fail closed on incomplete/reordered predictions, then apply fixed gates."""
    suite = audit_manifest(manifest)
    if artifact.get("experiment") != "TEACH-0014" or (
            artifact.get("manifest_sha256") != suite["manifest_sha256"]):
        raise ValueError("Prediction artifact/suite identity mismatch")
    runs = artifact.get("runs")
    if not isinstance(runs, dict) or set(runs) != ARMS:
        raise ValueError("Prediction arm set incomplete or extra")
    scores = {}
    for arm, replicates in runs.items():
        if not isinstance(replicates, dict) or set(replicates) != {"0", "1"}:
            raise ValueError(f"{arm}: both fixed replicate indices required")
        scores[arm] = {}
        for replicate, payload in replicates.items():
            predictions = payload.get("panels")
            if not isinstance(predictions, dict) or set(predictions) != set(
                    manifest["panels"]):
                raise ValueError(f"{arm}/{replicate}: panel set incomplete or extra")
            panel_scores = {}
            for name, episodes in manifest["panels"].items():
                rows = predictions[name]
                if not isinstance(rows, list) or len(rows) != len(episodes):
                    raise ValueError(f"{arm}/{replicate}/{name}: prediction count")
                predicted = []
                for source, prediction in zip(episodes, rows, strict=True):
                    if (not isinstance(prediction, dict) or
                            set(prediction) != {"render_id", "prediction"} or
                            prediction["render_id"] != source["render_id"] or
                            type(prediction["prediction"]) is not int or
                            not 16 <= prediction["prediction"] < 2064):
                        raise ValueError(
                            f"{arm}/{replicate}/{name}: invalid identity or answer")
                    predicted.append(prediction["prediction"])
                panel_scores[name] = _score_panel(
                    name, [episode["answer"] for episode in episodes], predicted)
            scores[arm][replicate] = {"panels": panel_scores,
                                      "twohop_absolute": _pass_twohop(
                                          {"panels": panel_scores}),
                                      "extrapolation_absolute": _pass_extrapolation(
                                          {"panels": panel_scores})}
    null_valid = all(
        scores["raw_null"][rep]["panels"]["composed_confirm_confirm"]
        ["items"]["accuracy"] <= .35 and
        scores["raw_null"][rep]["panels"]["factorial"]["groups"]["accuracy"]
        <= .10 for rep in ("0", "1"))
    oracle = all(scores["oracle_rows_workspace"][rep]["twohop_absolute"]
                 for rep in ("0", "1"))
    raw = all(scores["latent_rows_answer"][rep]["twohop_absolute"]
              and all(_raw_advantage(scores["latent_rows_answer"][rep],
                                     scores[arm][rep]) for arm in (
                  "raw_dense_matched", "latent_rows_one_read",
                  "latent_rows_mean_address")) for rep in ("0", "1"))
    edge_aux_benefit = all(
        scores["latent_rows_edge_aux"][rep]["twohop_absolute"] and
        _margin(scores["latent_rows_edge_aux"][rep],
                scores["latent_rows_answer"][rep], "factorial", "groups") >= .15 and
        _margin(scores["latent_rows_edge_aux"][rep],
                scores["latent_rows_answer"][rep], "boundary_groups",
                "marker_free") >= .15 for rep in ("0", "1"))
    causal_benefit = all(
        _margin(scores["latent_rows_causal"][rep],
                scores["latent_rows_wrong_causal"][rep], "factorial",
                "groups") >= .15 and
        _margin(scores["latent_rows_causal"][rep],
                scores["latent_rows_wrong_causal"][rep], "boundary_groups",
                "marker_free") >= .15 for rep in ("0", "1"))
    diffusion_benefit = all(
        _margin(scores["latent_rows_diffuse"][rep],
                scores["latent_rows_recurrent4"][rep], "factorial",
                "groups") >= .15 and
        _margin(scores["latent_rows_diffuse"][rep],
                scores["latent_rows_recurrent4"][rep], "boundary_groups",
                "marker_free") >= .15 for rep in ("0", "1"))
    if not null_valid:
        primary = "INVALID_NULL_LEAKAGE"
    elif not oracle:
        primary = "EXECUTOR_OPTIMIZATION_INCOMPLETE"
    elif raw:
        primary = "ANSWER_ONLY_RAW_BEHAVIOR_QUALIFIED"
    elif edge_aux_benefit:
        primary = "OBJECTIVE_BOTTLENECK_CANDIDATE"
    else:
        primary = "RAW_BEHAVIOR_NOT_QUALIFIED"
    return {
        "audit": "pass", "scope": "answer_behavior_only",
        "manifest_sha256": suite["manifest_sha256"], "scores": scores,
        "decisions": {
            "primary": primary, "null_valid": null_valid,
            "oracle_twohop": oracle, "raw_twohop_and_advantage": raw,
            "oracle_extrapolation": all(
                scores["oracle_rows_workspace"][rep]["extrapolation_absolute"]
                for rep in ("0", "1")),
            "raw_extrapolation": all(
                scores["latent_rows_answer"][rep]["extrapolation_absolute"]
                for rep in ("0", "1")),
            "edge_aux_benefit": edge_aux_benefit,
            "causal_benefit": causal_benefit,
            "diffusion_behavior_margin": diffusion_benefit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_behavior(json.loads(args.manifest.read_text()),
                            json.loads(args.predictions.read_text()))
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
