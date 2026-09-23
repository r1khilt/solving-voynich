#!/usr/bin/env python3
"""Independent no-model audit of a frozen TEACH-0013 counterfactual suite."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAD, BOS, EDGE, GAP, COMPOSE, DIRECT, COPY, FIRST_HOP, ANSWER = range(9)
SYMBOL_START = 16
TASK_MARKERS = {"composed": COMPOSE, "direct": DIRECT, "copy": COPY,
                "first_hop": FIRST_HOP}
TEACH12_NAMESPACE = "TEACH-0012-v1"
TEACH13_NAMESPACE = "TEACH-0013-v1"
MULTI_VARIANTS = {
    "base", "donor", "marker_free_base", "marker_free_donor",
    "reordered_base", "reordered_donor", "g_content_base", "g_content_donor",
    "binding_base", "binding_donor", "g_binding_base", "g_binding_donor",
    "format_donor", "distractor_donor",
}
SINGLE_VARIANTS = {
    "first_hop_base", "first_hop_donor", "direct_base", "direct_donor", "copy_control",
}


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def stable_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(stable_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def partition(kind, rows):
    left, right = sorted(row[0] for row in rows), sorted(row[1] for row in rows)
    bucket = int(digest([TEACH12_NAMESPACE, kind, left, right])[:16], 16) % 10
    return "train" if bucket < 8 else "confirm" if bucket == 9 else "development"


def oracle(rows, task, query):
    mapping = {}
    for left, right in rows:
        need(left not in mapping, "Duplicate row left side")
        mapping[left] = right
    if task == "copy":
        return query
    need(query in mapping, "Query absent from graph")
    first = mapping[query]
    if task in ("first_hop", "direct"):
        return first
    need(task == "composed" and first in mapping, "Malformed composed program")
    return mapping[first]


def parse_serialized_rows(tokens, expected_rows):
    tokens = tuple(tokens)
    expected_rows = [list(row) for row in expected_rows]
    need(tokens[0] == BOS and tokens[-1] == ANSWER, "Bad BOS/ANSWER suffix")
    cursor, body_end, observed = 1, len(tokens) - 3, []
    while cursor < body_end:
        while cursor < body_end and tokens[cursor] == GAP:
            cursor += 1
        if cursor >= body_end:
            break
        if tokens[cursor] == EDGE:
            need(cursor + 2 < body_end, "Truncated prefix row")
            left, right = tokens[cursor + 1], tokens[cursor + 2]
            cursor += 3
        elif cursor + 1 < body_end and tokens[cursor + 1] == EDGE:
            need(cursor + 2 < body_end, "Truncated infix row")
            left, right = tokens[cursor], tokens[cursor + 2]
            cursor += 3
        else:
            need(cursor + 1 < body_end, "Truncated bare/suffix row")
            left, right = tokens[cursor], tokens[cursor + 1]
            cursor += 2
            if cursor < body_end and tokens[cursor] == EDGE:
                cursor += 1
        need(left >= SYMBOL_START and right >= SYMBOL_START, "Reserved token used as endpoint")
        observed.append([left, right])
    need(cursor == body_end, "Row parser did not consume body")
    need(observed == expected_rows, "Token rows disagree with serialized_rows")


def audit_episode(episode):
    required = {
        "tokens", "f_rows", "g_rows", "distractor_rows", "serialized_rows", "answer",
        "task", "query", "f_partition", "g_partition", "logical_id", "render_id",
        "distractor_chains", "marker_dropout",
    }
    need(set(episode) == required, "Episode fields changed")
    tokens = tuple(episode["tokens"])
    need(len(tokens) <= 128 and PAD not in tokens, "Bad unpadded episode length")
    need(episode["task"] in TASK_MARKERS, "Unknown task")
    need(tokens[-3:] == (TASK_MARKERS[episode["task"]], episode["query"], ANSWER),
         "Task/query suffix mismatch")
    parse_serialized_rows(tokens, episode["serialized_rows"])
    rows = [list(row) for row in (episode["f_rows"] + episode["g_rows"]
                                  + episode["distractor_rows"])]
    need(sorted(map(list, episode["serialized_rows"])) == sorted(rows),
         "Serialized row multiset mismatch")
    need(oracle(rows, episode["task"], episode["query"]) == episode["answer"],
         "Independent oracle mismatch")
    need(partition("F", episode["f_rows"]) == episode["f_partition"]
         and partition("G", episode["g_rows"]) == episode["g_partition"],
         "Family partition mismatch")
    logical = {"f": episode["f_rows"], "g": episode["g_rows"],
               "d": episode["distractor_rows"], "task": episode["task"],
               "query": episode["query"]}
    need(digest(logical) == episode["logical_id"], "Logical ID mismatch")
    need(digest({"logical": episode["logical_id"], "tokens": tokens}) == episode["render_id"],
         "Render ID mismatch")
    need(episode["distractor_chains"] == len(episode["distractor_rows"]) // 2,
         "Distractor count mismatch")


def skeleton(episode):
    return [token if token < SYMBOL_START else -1 for token in episode["tokens"]]


def rows_tuple(rows):
    return tuple(tuple(row) for row in rows)


def gap_signature(episode):
    """Recover the number of physical GAP tokens after each serialized row."""
    tokens, cursor, body_end, gaps = tuple(episode["tokens"]), 1, len(episode["tokens"]) - 3, []
    for _ in episode["serialized_rows"]:
        if tokens[cursor] == EDGE or (cursor + 1 < body_end and tokens[cursor + 1] == EDGE):
            cursor += 3
        else:
            cursor += 2
            if cursor < body_end and tokens[cursor] == EDGE:
                cursor += 1
        count = 0
        while cursor < body_end and tokens[cursor] == GAP:
            count += 1
            cursor += 1
        gaps.append(count)
    need(cursor == body_end, "Could not recover gap signature")
    return tuple(gaps)


def marker_signature(episode):
    """Recover prefix/infix/suffix/none for every physical row slot."""
    tokens, cursor, body_end, result = tuple(episode["tokens"]), 1, len(episode["tokens"]) - 3, []
    for _ in episode["serialized_rows"]:
        if tokens[cursor] == EDGE:
            result.append("prefix")
            cursor += 3
        elif cursor + 1 < body_end and tokens[cursor + 1] == EDGE:
            result.append("infix")
            cursor += 3
        else:
            cursor += 2
            if cursor < body_end and tokens[cursor] == EDGE:
                result.append("suffix")
                cursor += 1
            else:
                result.append("none")
        while cursor < body_end and tokens[cursor] == GAP:
            cursor += 1
    need(cursor == body_end, "Could not recover marker signature")
    return tuple(result)


def audit_layout(layout, episode):
    need(set(layout) == {"roles", "labels", "row_indices"}, "Layout fields changed")
    length = len(episode["tokens"])
    need(len(layout["roles"]) == len(layout["labels"]) == len(layout["row_indices"]) == length,
         "Semantic layout length mismatch")
    need(len(set(layout["labels"])) == length, "Semantic labels are not exhaustive/unique")
    need(layout["roles"][0] == layout["labels"][0] == "bos", "BOS semantic label mismatch")
    need(tuple(layout["roles"][-3:]) == tuple(layout["labels"][-3:])
         == ("task", "query", "answer"),
         "Suffix semantic labels mismatch")
    for index, row_index in enumerate(layout["row_indices"]):
        if row_index >= 0:
            need(row_index < len(episode["serialized_rows"]), "Semantic row index out of range")
            need(episode["tokens"][index] != GAP, "Gap assigned to a logical row")


def _variants(group, name):
    value = group[name]
    return value if name in MULTI_VARIANTS else [value]


def audit_group(group, split):
    expected_fields = ({
        "group_id", "split", "semantic_layouts", "key_base", "key_donor",
        "base_answers", "recipient_answers", "fixed_donor_answer", "render_seed",
    } | MULTI_VARIANTS | SINGLE_VARIANTS)
    need(set(group) == expected_fields, "Counterfactual group fields changed")
    need(group["split"] == split, "Group stored in wrong split")
    for name in MULTI_VARIANTS | SINGLE_VARIANTS:
        episodes, layouts = _variants(group, name), group["semantic_layouts"][name]
        need(len(episodes) == len(layouts), f"Layout count mismatch: {name}")
        need(len(episodes) == (3 if name in MULTI_VARIANTS else 1),
             f"Variant arity mismatch: {name}")
        for episode, layout in zip(episodes, layouts, strict=True):
            audit_episode(episode)
            audit_layout(layout, episode)
            need(episode["f_partition"] == episode["g_partition"] == "confirm",
                 "Mechanism episode is not confirm/confirm")

    base, donor = group["base"], group["donor"]
    need(len({tuple(skeleton(ep)) for ep in base + donor}) == 1,
         "Primary counterfactual skeletons diverged")
    need(tuple(episode["answer"] for episode in base) == tuple(group["base_answers"]),
         "Base answer list mismatch")
    need(tuple(episode["answer"] for episode in donor) == tuple(group["recipient_answers"]),
         "Recipient answer list mismatch")
    need(group["fixed_donor_answer"] == donor[0]["answer"], "Fixed donor answer mismatch")
    need(len(set(group["base_answers"])) == len(set(group["recipient_answers"])) == 3,
         "Recipient answers are not distinguishable")
    f0, f1 = base[0]["f_rows"], donor[0]["f_rows"]
    need([row[0] for row in f0] == [row[0] for row in f1],
         "F content donor changed row identities")
    need(sum(left != right for left, right in zip(f0, f1, strict=True)) == 1,
         "F content donor does not change exactly one row")
    changed_index = next(index for index, (left, right) in enumerate(zip(
        f0, f1, strict=True)) if left != right)
    need(f0[changed_index][0] == base[0]["query"]
         and f0[changed_index][1] == group["key_base"]
         and f1[changed_index][1] == group["key_donor"]
         and group["key_donor"] in {row[1] for row in f0 if row[0] != base[0]["query"]},
         "F content donor is not the registered queried-row replacement")
    need(group["key_base"] != group["key_donor"], "F keys are identical")
    need(all(base[index]["g_rows"] == donor[index]["g_rows"] for index in range(3)),
         "F donor changed a recipient G table")
    need(base[0]["distractor_rows"] == donor[0]["distractor_rows"],
         "F donor changed distractors")

    for recipient, (binding_base, binding_donor, g_content_base, g_content_donor,
                    g_binding_base, g_binding_donor) in enumerate(zip(
            group["binding_base"], group["binding_donor"],
            group["g_content_base"], group["g_content_donor"],
            group["g_binding_base"], group["g_binding_donor"], strict=True)):
        need(rows_tuple(binding_base["f_rows"]) == rows_tuple(f0)
             and rows_tuple(binding_base["g_rows"]) == rows_tuple(base[recipient]["g_rows"]),
             "F binding base differs from primary base")
        need([row[0] for row in binding_donor["f_rows"]] == [row[0] for row in f0]
             and sorted(row[1] for row in binding_donor["f_rows"])
             == sorted(row[1] for row in f0),
             "F binding donor changed endpoint inventory")
        need(sum(left != right for left, right in zip(
            f0, binding_donor["f_rows"], strict=True)) == 2,
             "F binding donor is not a two-row swap")

        need(rows_tuple(g_content_base["f_rows"]) == rows_tuple(f0)
             and [row[0] for row in g_content_base["g_rows"]]
             == [row[0] for row in g_content_donor["g_rows"]],
             "G content donor changed F or G row identities")
        g_content_changed = [(left, right) for left, right in zip(
            g_content_base["g_rows"], g_content_donor["g_rows"], strict=True)
            if left != right]
        need(len(g_content_changed) == 1
             and g_content_changed[0][0][0] == group["key_base"]
             and g_content_changed[0][1][1]
             not in {row[1] for row in g_content_base["g_rows"]},
             "G content donor is not one fresh matched-row replacement")

        need(rows_tuple(g_binding_base["g_rows"]) == rows_tuple(g_content_base["g_rows"])
             and [row[0] for row in g_binding_donor["g_rows"]]
             == [row[0] for row in g_binding_base["g_rows"]]
             and sorted(row[1] for row in g_binding_donor["g_rows"])
             == sorted(row[1] for row in g_binding_base["g_rows"]),
             "G binding donor changed endpoint inventory")
        need(sum(left != right for left, right in zip(
            g_binding_base["g_rows"], g_binding_donor["g_rows"], strict=True)) == 2,
             "G binding donor is not a two-row swap")

    for original, marker_free, reordered in zip(
            base, group["marker_free_base"], group["reordered_base"], strict=True):
        need(rows_tuple(original["serialized_rows"]) == rows_tuple(marker_free["serialized_rows"])
             and gap_signature(original) == gap_signature(marker_free)
             and all(style == "none" for style in marker_signature(marker_free)),
             "Marker-free factor changed row order or gaps")
        expected_order = rows_tuple(original["serialized_rows"])[1:] \
            + rows_tuple(original["serialized_rows"])[:1]
        need(rows_tuple(reordered["serialized_rows"]) == expected_order
             and skeleton(original) == skeleton(reordered)
             and gap_signature(original) == gap_signature(reordered),
             "Order factor changed physical slot structure")
    for original, marker_free, reordered in zip(
            donor, group["marker_free_donor"], group["reordered_donor"], strict=True):
        need(rows_tuple(original["serialized_rows"]) == rows_tuple(marker_free["serialized_rows"])
             and gap_signature(original) == gap_signature(marker_free)
             and all(style == "none" for style in marker_signature(marker_free)),
             "Donor marker-free factor changed row order or gaps")
        expected_order = rows_tuple(original["serialized_rows"])[1:] \
            + rows_tuple(original["serialized_rows"])[:1]
        need(rows_tuple(reordered["serialized_rows"]) == expected_order
             and skeleton(original) == skeleton(reordered)
             and gap_signature(original) == gap_signature(reordered),
             "Donor order factor changed physical slot structure")

    rotation = {"prefix": "infix", "infix": "suffix", "suffix": "prefix"}
    for original, format_donor, nuisance in zip(
            donor, group["format_donor"], group["distractor_donor"], strict=True):
        need(rows_tuple(format_donor["serialized_rows"])
             == rows_tuple(original["serialized_rows"])
             and gap_signature(format_donor) == gap_signature(original)
             and marker_signature(format_donor)
             == tuple(rotation[style] for style in marker_signature(original)),
             "Format factor changed content, order, gaps, or used wrong rotation")
        need(rows_tuple(nuisance["f_rows"]) == rows_tuple(original["f_rows"])
             and rows_tuple(nuisance["g_rows"]) == rows_tuple(original["g_rows"])
             and rows_tuple(nuisance["distractor_rows"])
             != rows_tuple(original["distractor_rows"])
             and skeleton(nuisance) == skeleton(original),
             "Distractor factor is not isolated")

    for left_layout, right_layout, left_episode, right_episode in zip(
            group["semantic_layouts"]["base"], group["semantic_layouts"]["donor"],
            base, donor, strict=True):
        common = set(left_layout["labels"]) & set(right_layout["labels"])
        for label in common:
            if (label.startswith("other_f.") or label.startswith("other_g.")) \
                    and label.endswith(".left"):
                left_position = left_layout["labels"].index(label)
                right_position = right_layout["labels"].index(label)
                need(left_episode["tokens"][left_position]
                     == right_episode["tokens"][right_position],
                     "Canonical semantic label changed logical row identity")
    distractors = base[0]["distractor_rows"]
    g_outputs = {right for _, right in base[0]["g_rows"]}
    need(distractors[1][1] in g_outputs and distractors[3][1] in g_outputs,
         "Registered false paths do not terminate at signal objects")

    logical = {"f0": f0, "f1": f1,
               "f_binding": group["binding_donor"][0]["f_rows"],
               "g": [episode["g_rows"] for episode in base],
               "g_content": [episode["g_rows"] for episode in group["g_content_donor"]],
               "g_binding": [episode["g_rows"] for episode in group["g_binding_donor"]],
               "d": base[0]["distractor_rows"],
               "nuisance_d": group["distractor_donor"][0]["distractor_rows"],
               "query": base[0]["query"],
               "key_base": group["key_base"], "key_donor": group["key_donor"]}
    identifier = digest([TEACH13_NAMESPACE, logical])
    need(identifier == group["group_id"], "TEACH-0013 group ID mismatch")
    expected_split = "discovery" if int(identifier[:16], 16) % 10 < 5 else "confirmation"
    need(expected_split == split, "TEACH-0013 split hash mismatch")


def audit_payload(payload):
    need(payload.get("experiment") == "TEACH-0013"
         and payload.get("namespace") == TEACH13_NAMESPACE, "Wrong suite identity")
    need(set(payload.get("splits", {})) == {"discovery", "confirmation"},
         "Missing suite split")
    group_ids = []
    for split, groups in payload["splits"].items():
        for group in groups:
            audit_group(group, split)
            group_ids.append(group["group_id"])
    need(len(group_ids) == len(set(group_ids)), "Duplicate logical groups")
    stats = payload.get("generation_stats")
    need(isinstance(stats, dict) and set(stats) == {
        "attempts", "candidate_rejected", "duplicate_id", "split_full",
        "accepted_discovery", "accepted_confirmation"},
         "Missing or malformed generator rejection counts")
    need(all(type(value) is int and value >= 0 for value in stats.values())
         and stats["accepted_discovery"] == len(payload["splits"]["discovery"])
         and stats["accepted_confirmation"] == len(payload["splits"]["confirmation"])
         and stats["attempts"] == stats["candidate_rejected"] + stats["duplicate_id"]
         + stats["split_full"] + stats["accepted_discovery"]
         + stats["accepted_confirmation"],
         "Generator rejection counts do not reconcile")
    labels = sorted({label for group in payload["splits"]["discovery"]
                     for layouts in group["semantic_layouts"].values()
                     for layout in layouts for label in layout["labels"]})
    need(labels == payload["semantic_label_order"], "Semantic label order mismatch")
    return {"groups": len(group_ids),
            "split_counts": {name: len(groups) for name, groups in payload["splits"].items()},
            "unique_semantic_labels": len(labels)}


def audit(suite_path: Path, manifest_path: Path, output_path: Path):
    manifest = json.loads(manifest_path.read_text())
    need(file_digest(suite_path) == manifest["suite_gzip_sha256"], "Suite gzip hash mismatch")
    raw = gzip.decompress(suite_path.read_bytes())
    need(hashlib.sha256(raw).hexdigest() == manifest["suite_uncompressed_sha256"],
         "Suite content hash mismatch")
    payload = json.loads(raw)
    summary = audit_payload(payload)
    need(summary["split_counts"] == manifest["split_counts"], "Manifest split counts mismatch")
    need(payload["generation_stats"] == manifest["generation_stats"],
         "Manifest generator counts mismatch")
    report = {"audit": "pass", "experiment": "TEACH-0013",
              "suite_gzip_sha256": manifest["suite_gzip_sha256"], **summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "outputs/TEACH-0013/suite.json.gz")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "results/TEACH-0013/suite-manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/TEACH-0013/suite-audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.suite, args.manifest, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
