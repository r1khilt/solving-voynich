"""Independent, no-model structural audit for TEACH-0014 suite manifests.

This module deliberately imports no TEACH-0014 generator, model or trainer.
It audits recorded episodes and grouped counterfactuals, not neural inference.
"""

import argparse
import hashlib
import json
from pathlib import Path


PAD, BOS, EDGE, GAP, COMPOSE, DIRECT, COPY, FIRST_HOP, ANSWER, HOP3, HOP4 = range(11)
SYMBOL_START = 16
VOCAB_SIZE = SYMBOL_START + 2048
CONTEXT_LENGTH = 192
SPLIT_NAMESPACE = "TEACH-0014-latent-edge-v1"
GROUPED = {
    "first_hop_query_groups", "direct_query_groups", "factorial",
    "order_groups", "boundary_groups", "distractor_groups",
}
SINGLE = {
    "composed_train_train", "composed_confirm_train",
    "composed_train_confirm", "composed_confirm_confirm",
    "first_hop_confirm", "direct_confirm", "copy_confirm",
    "long_ood", "alias_terminal", "alias_inner",
    "hop_3", "hop_4", "hop_4_long_ood",
}
EXPECTED_PANELS = GROUPED | SINGLE
MARKERS = {
    COPY: ("copy", 0),
    FIRST_HOP: ("first_hop", 1),
    DIRECT: ("direct", 1),
    COMPOSE: ("composed", 2),
    HOP3: ("composed", 3),
    HOP4: ("composed", 4),
}


def _digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _family_partition(kind: str, rows: list[list[int]]) -> str:
    token = _digest([SPLIT_NAMESPACE, kind, sorted(rows)])
    bucket = int(token[:16], 16) % 10
    return "train" if bucket < 8 else "development" if bucket == 8 else "confirm"


def _path_rows(paths: list[list[int]]) -> list[list[int]]:
    return [[path[index], path[index + 1]]
            for path in paths for index in range(len(path) - 1)]


def _follow(rows: list[list[int]], query: int, hops: int) -> int:
    mapping = {}
    for left, right in rows:
        if left in mapping:
            raise ValueError("Rows are not a function")
        mapping[left] = right
    value = query
    for _ in range(hops):
        if value not in mapping:
            raise ValueError("Missing requested outgoing edge")
        value = mapping[value]
    return value


def _audit_episode(record: dict) -> dict:
    tokens = record["tokens"]
    if (not 4 <= len(tokens) <= CONTEXT_LENGTH or tokens[0] != BOS or
            tokens[-1] != ANSWER or tokens[-3] not in MARKERS):
        raise ValueError("Malformed visible episode envelope")
    if any(not isinstance(token, int) or not 0 <= token < VOCAB_SIZE
           for token in tokens):
        raise ValueError("Visible token outside vocabulary")
    task, hops = MARKERS[tokens[-3]]
    query = tokens[-2]
    if not SYMBOL_START <= query < VOCAB_SIZE:
        raise ValueError("Invalid visible query")
    if record["task"] != task or record["hops"] != hops or record["query"] != query:
        raise ValueError("Task, hop or query metadata disagrees with visible tokens")
    body = tokens[1:-3]
    if any(token not in (EDGE, GAP) and token < SYMBOL_START for token in body):
        raise ValueError("Unregistered body token")
    positions = [index + 1 for index, token in enumerate(body)
                 if token >= SYMBOL_START]
    if not positions or len(positions) % 2:
        raise ValueError("Incomplete visible pair grammar")
    ordered = [[tokens[positions[index]], tokens[positions[index + 1]]]
               for index in range(0, len(positions), 2)]
    row_positions = [[positions[index], positions[index + 1]]
                     for index in range(0, len(positions), 2)]
    if record["serialized_rows"] != ordered or record["row_positions"] != row_positions:
        raise ValueError("Visible row/position metadata mismatch")
    signals = record["signal_paths"]
    distractors = record["distractor_paths"]
    if len(signals) != 4 or len({len(path) for path in signals}) != 1:
        raise ValueError("Expected four equal-length signal paths")
    signal_hops = len(signals[0]) - 1
    if not 1 <= signal_hops <= 4 or any(len(path) != 3 for path in distractors):
        raise ValueError("Malformed graph path lengths")
    if task == "composed" and hops != signal_hops:
        raise ValueError("Composed hop count does not match graph")
    signal_symbols = [symbol for path in signals for symbol in path]
    if len(set(signal_symbols)) != len(signal_symbols):
        raise ValueError("Signal paths overlap")
    prefix = [symbol for path in distractors for symbol in path[:2]]
    terminals = [path[-1] for path in distractors]
    if (len(set(prefix)) != len(prefix) or set(prefix) & set(signal_symbols) or
            len(set(terminals)) != len(terminals) or set(terminals) & set(prefix)):
        raise ValueError("Malformed distractor endpoints")
    shared = [terminal for terminal in terminals if terminal in set(signal_symbols)]
    alias_mode = record["alias_mode"]
    if alias_mode == "none":
        if shared:
            raise ValueError("Unregistered endpoint alias")
    elif alias_mode == "terminal":
        if len(shared) != 1 or shared[0] not in {path[-1] for path in signals}:
            raise ValueError("Incorrect terminal alias")
    elif alias_mode == "inner":
        if len(shared) != 1 or shared[0] not in {
                symbol for path in signals for symbol in path[1:-1]}:
            raise ValueError("Incorrect inner alias")
    else:
        raise ValueError("Unknown alias mode")
    signal_rows = _path_rows(signals)
    distractor_rows = _path_rows(distractors)
    logical_rows = signal_rows + distractor_rows
    if len(logical_rows) > 32 or sorted(logical_rows) != sorted(ordered):
        raise ValueError("Visible and logical edges disagree")
    answer = _follow(ordered, query, hops)
    if record["answer"] != answer:
        raise ValueError("Visible oracle answer mismatch")
    stages = [
        _family_partition(f"stage-{index}", [
            [path[index], path[index + 1]] for path in signals])
        for index in range(signal_hops)
    ]
    graph_partition = _family_partition("graph", signal_rows)
    graph_id = _digest({"signal": signals, "distractors": distractors})
    logical_id = _digest({"graph": graph_id, "task": task, "query": query})
    render_id = _digest({"logical": logical_id, "tokens": tokens})
    if (record["stage_partitions"] != stages or
            record["graph_partition"] != graph_partition or
            record["graph_id"] != graph_id or
            record["logical_id"] != logical_id or
            record["render_id"] != render_id):
        raise ValueError("Split or identity hash mismatch")
    if not 0 <= record["marker_dropout"] <= 1:
        raise ValueError("Invalid marker dropout metadata")
    return {
        "signal_paths": signals,
        "distractors": len(distractors),
        "stages": stages,
        "graph_partition": graph_partition,
        "graph_id": graph_id,
        "answer": answer,
        "query": query,
        "task": task,
        "hops": hops,
        "marker_dropout": record["marker_dropout"],
        "alias_mode": alias_mode,
        "stage_rows": tuple(tuple(tuple((path[index], path[index + 1]))
                                      for path in signals)
                            for index in range(signal_hops)),
        "length": len(tokens),
        "rows": len(ordered),
    }


def _check_groups(name: str, rows: list[dict], size: int) -> None:
    if name not in GROUPED:
        return
    for group_index in range(size):
        group = rows[4 * group_index:4 * group_index + 4]
        if len(group) != 4:
            raise ValueError(f"{name}: incomplete quartet")
        if name in ("first_hop_query_groups", "direct_query_groups"):
            if len({item["graph_id"] for item in group}) != 1:
                raise ValueError(f"{name}: query quartet does not share a graph")
            endpoint = 0 if name == "first_hop_query_groups" else 1
            expected_queries = {path[endpoint] for path in group[0]["signal_paths"]}
            if {item["query"] for item in group} != expected_queries:
                raise ValueError(f"{name}: query quartet incomplete")
        elif name == "factorial":
            if len({item["answer"] for item in group}) != 4 or (
                    len({item["query"] for item in group}) != 1):
                raise ValueError("Factorial answers/query invalid")
            f_signatures = {item["stage_rows"][0] for item in group}
            g_signatures = {item["stage_rows"][1] for item in group}
            crossed = {(item["stage_rows"][0], item["stage_rows"][1])
                       for item in group}
            if (len(f_signatures) != 2 or len(g_signatures) != 2 or
                    len(crossed) != 4):
                raise ValueError("Factorial F-by-G crossing incomplete")
        elif name in ("order_groups", "boundary_groups"):
            if len({item["graph_id"] for item in group}) != 1 or (
                    len({item["answer"] for item in group}) != 1):
                raise ValueError(f"{name}: graph/answer changed")
            if name == "boundary_groups" and [
                    item["marker_dropout"] for item in group] != [0.0, .25, .5, 1.0]:
                raise ValueError("Boundary dropout grid incomplete")
        elif name == "distractor_groups":
            if [item["distractors"] for item in group] != [0, 1, 4, 8]:
                raise ValueError("Distractor count grid incomplete")
            if (len({item["stage_rows"] for item in group}) != 1 or
                    len({item["answer"] for item in group}) != 1):
                raise ValueError("Distractor group changed signal computation")


def audit_manifest(manifest: dict) -> dict:
    if manifest.get("experiment") != "TEACH-0014" or (
            manifest.get("split_namespace") != SPLIT_NAMESPACE):
        raise ValueError("Experiment or split namespace mismatch")
    size = manifest.get("group_count")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Invalid group count")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid suite seed")
    panels = manifest.get("panels")
    if not isinstance(panels, dict) or set(panels) != EXPECTED_PANELS:
        raise ValueError("Suite panel set differs from registration")
    all_rows = []
    summary = {}
    for name, items in panels.items():
        expected = size * (4 if name in GROUPED else 1)
        if not isinstance(items, list) or len(items) != expected:
            raise ValueError(f"{name}: wrong item count")
        rows = [_audit_episode(item) for item in items]
        for row in rows:
            if name.startswith("composed_"):
                _, f_part, g_part = name.split("_", 2)
                if row["task"] != "composed" or row["hops"] != 2 or (
                        row["stages"] != [f_part, g_part]):
                    raise ValueError(f"{name}: family cell mismatch")
            elif name.endswith("_confirm") and name in (
                    "first_hop_confirm", "direct_confirm", "copy_confirm"):
                task = name.removesuffix("_confirm")
                if row["task"] != task or row["stages"] != ["confirm", "confirm"]:
                    raise ValueError(f"{name}: task/family mismatch")
            elif name in GROUPED or name in (
                    "long_ood", "alias_terminal", "alias_inner"):
                if row["stages"] != ["confirm", "confirm"]:
                    raise ValueError(f"{name}: grouped family mismatch")
            if name.startswith("alias_") and row["alias_mode"] != name.removeprefix(
                    "alias_"):
                raise ValueError(f"{name}: alias mode mismatch")
            if name == "long_ood" and row["distractors"] != 8:
                raise ValueError("Long two-hop panel lacks eight distractors")
            if name in ("hop_3", "hop_4", "hop_4_long_ood"):
                expected_hops = 3 if name == "hop_3" else 4
                if row["hops"] != expected_hops or (
                        row["graph_partition"] != "confirm"):
                    raise ValueError(f"{name}: hop/family mismatch")
                if name == "hop_4_long_ood" and (
                        row["distractors"] != 8 or row["rows"] != 32):
                    raise ValueError("Four-hop long panel incomplete")
        _check_groups(name, rows, size)
        all_rows.extend(rows)
        summary[name] = {
            "items": len(rows),
            "min_tokens": min(row["length"] for row in rows),
            "max_tokens": max(row["length"] for row in rows),
            "max_rows": max(row["rows"] for row in rows),
        }
    return {
        "audit": "pass",
        "experiment": "TEACH-0014",
        "panels": summary,
        "items": len(all_rows),
        "max_tokens": max(row["length"] for row in all_rows),
        "max_rows": max(row["rows"] for row in all_rows),
        "manifest_sha256": _digest(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = audit_manifest(manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
