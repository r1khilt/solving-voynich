"""Separate full arithmetic replay; extraction/generator are explicitly shared.

Recounts training distributions and evaluates the transported odds directly,
without the production fit/distributions/gain/cross_fit/permutation functions.
This is same-author alternate arithmetic, not independent paleography.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import math
import random
import resource
import time

from voynich import paragraph_carry as pc
from scripts import run_paragraph_carry001 as controller

ROOT = controller.ROOT


def alternate_model(rows):
    vocabulary = tuple(sorted({word[0] for row in rows for word in row.words})) + (pc.UNKNOWN,)
    first_global, first_context, global_middle = defaultdict(list), defaultdict(list), []
    context_middle, edge = defaultdict(list), defaultdict(list)
    for row in rows:
        kind = "paragraph" if row.start else "continuation"
        first_global[kind].append(row.words[0][0])
        first_context[(row.context, kind)].append(row.words[0][0])
        for pos in range(1, len(row.words)):
            target, terminal = row.words[pos][0], row.words[pos - 1][-1]
            global_middle.append(target)
            context_middle[row.context].append(target)
            edge[(row.context, terminal)].append(target)
    return vocabulary, first_global, first_context, global_middle, context_middle, edge


def alternate_gain(model, event):
    vocabulary, starts, contextual_starts, middle, contextual_middle, edges = model
    context, kind, last = tuple(event["context"]), event["kind"], event["last"]
    global_mid_counts = Counter(middle)
    mid_counts = Counter(contextual_middle[context])
    edge_counts = Counter(edges[(context, last)])
    global_start_counts = Counter(starts[kind])
    start_counts = Counter(contextual_starts[(context, kind)])
    transported = {}
    ratios = {}
    for symbol in vocabulary:
        global_mid = (global_mid_counts[symbol] + 1) / (len(middle) + len(vocabulary))
        mid = (mid_counts[symbol] + 50 * global_mid) / (len(contextual_middle[context]) + 50)
        conditional = (edge_counts[symbol] + 20 * mid) / (len(edges[(context, last)]) + 20)
        global_start = (global_start_counts[symbol] + 1) / (len(starts[kind]) + len(vocabulary))
        baseline = (start_counts[symbol] + 50 * global_start) / (len(contextual_starts[(context, kind)]) + 50)
        ratios[symbol] = conditional / mid
        transported[symbol] = baseline * ratios[symbol]
    target = event["target"] if event["target"] in vocabulary else pc.UNKNOWN
    return math.log2(ratios[target]) - math.log2(math.fsum(transported.values()))


def independent_junctions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.page].append(row)
    events = []
    for page, lines in sorted(groups.items()):
        lines.sort(key=lambda row: row.number)
        for left, right in zip(lines, lines[1:]):
            if (right.number != left.number + 1 or left.leaf != right.leaf
                    or right.locator not in ("+", "*")
                    or left.context != right.context or left.end != right.start):
                continue
            events.append({"leaf": left.leaf, "page": page, "left": left.number, "right": right.number,
                           "context": right.context, "kind": "paragraph" if right.start else "continuation",
                           "last": left.words[-1][-1], "target": right.words[0][0],
                           "stratum": (min(right.depth, 3), min(len(left.words) // 4, 3),
                                       min(len(right.words) // 4, 3))})
    return events


def close(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for left, right in zip(a, b):
            close(left, right)
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 2e-11
    else:
        assert a == b


def replay_cell(rows, cell, saved_ledger, start_wall, start_cpu):
    assignment = pc.folds(rows)  # Shared immutable hash assignment, not selection.
    models = {fold: alternate_model([row for row in rows if assignment[row.leaf] != fold])
              for fold in range(pc.N_FOLDS)}
    assert cell["manifest"]["leaf_fold"] == assignment and cell["manifest"]["lines"] == len(rows)
    assert len(cell["manifest"]["models"]) == pc.N_FOLDS
    for f, manifest in enumerate(cell["manifest"]["models"]):
        training = [row for row in rows if assignment[row.leaf] != f]
        assert manifest["fold"] == f
        assert manifest["training_leaves"] == sorted({row.leaf for row in training})
        assert manifest["test_leaves"] == sorted(leaf for leaf, fold in assignment.items() if fold == f)
        assert manifest["training_lines"] == len(training)
        assert manifest["alphabet"] == list(models[f][0])
        # Reconstruct the production count-key representation independently.
        counts = defaultdict(Counter)
        for row in training:
            kind = "paragraph" if row.start else "continuation"
            counts[("start", kind)][row.words[0][0]] += 1
            counts[("start", kind, row.context)][row.words[0][0]] += 1
            for i in range(1, len(row.words)):
                target = row.words[i][0]
                counts[("mid",)][target] += 1
                counts[("mid", row.context)][target] += 1
                counts[("edge", row.context, row.words[i - 1][-1])][target] += 1
        digest = pc.json_sha(sorted((repr(key), sorted(value.items())) for key, value in counts.items()))
        assert manifest["count_digest"] == digest
    events = independent_junctions(rows)
    replayed = [{**event, "fold": assignment[event["leaf"]],
                 "gain": alternate_gain(models[assignment[event["leaf"]]], event)} for event in events]
    replayed.sort(key=lambda row: (row["page"], row["left"]))
    close(replayed, saved_ledger)
    assert pc.json_sha(saved_ledger) == cell["ledger_sha"]
    # Every production aggregate/interval is replayed over independently scored
    # events. Bootstrap/order/gate helpers are shared and disclosed below.
    close(pc.summarize(replayed, cell["bootstrap_seed"], n_boot=pc.N_BOOT), cell["summary"])
    groups = defaultdict(list)
    for index, event in enumerate(events):
        groups[(event["leaf"], tuple(event["context"]), event["kind"], tuple(event["stratum"]))].append(index)
    rng = random.Random(cell["permutation_seed"])
    contrasts = []
    # Cache independent alternate probabilities: the permutation only changes
    # terminals within a finite held-out stratum, never training counts.
    cache = {}
    for index in range(pc.N_PERM):
        controller.guard(start_wall, start_cpu)
        permuted = [dict(event) for event in events]
        for indices in groups.values():
            states = [events[i]["last"] for i in indices]
            rng.shuffle(states)
            for i, terminal in zip(indices, states):
                permuted[i]["last"] = terminal
        totals, counts = Counter(), Counter()
        for event in permuted:
            f = assignment[event["leaf"]]
            key = f, tuple(event["context"]), event["kind"], event["last"], event["target"]
            if key not in cache:
                cache[key] = alternate_gain(models[f], event)
            totals[event["kind"]] += cache[key]
            counts[event["kind"]] += 1
        contrasts.append(totals["continuation"] / counts["continuation"] - totals["paragraph"] / counts["paragraph"]
                         if counts["continuation"] and counts["paragraph"] else None)
    close(contrasts, cell["null"]["contrasts"])
    # Rank p uses audited saved doubles, preserving the original exact tie rule.
    real = cell["summary"]["difference_bits"]
    finite = [x for x in cell["null"]["contrasts"] if x is not None]
    p = (1 + sum(x >= real for x in finite)) / (1 + len(finite)) if real is not None and finite else None
    assert p == cell["null"]["one_sided_p"]
    assert cell["null"]["events"] == len(events)
    assert cell["null"]["permutable_events"] == sum(len(indices) for indices in groups.values() if len(indices) > 1)
    assert cell["null"]["movable_events"] == sum(len(indices) for indices in groups.values()
        if len({events[i]["last"] for i in indices}) > 1)
    for manifest in cell["manifest"]["models"]:
        assert not set(manifest["training_leaves"]) & set(manifest["test_leaves"])
    return len(saved_ledger), len(contrasts)


def _audit_work():
    out = ROOT / "results" / pc.EXPERIMENT
    if (out / "audit.json").exists() or (out / "audit-failure.json").exists():
        raise FileExistsError("Audit already attempted")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    result_path = out / "result.json"
    result = json.loads(result_path.read_text())
    assert result["complete"] and not (out / "failure.json").exists()
    started = json.loads((out / "started.json").read_text())
    assert started["freeze"] == result["freeze"] and started["inputs"] == result["inputs"]
    assert started["limits"] == {"wall_s": 600, "cpu_s": 550, "process_rss_bytes": 1024 ** 3,
                                 "private_bytes": 16 * 1024 ** 2, "workers": 1, "paid_usd": 0}
    assert 0 <= result["resource"]["wall_s"] <= 600 and 0 <= result["resource"]["cpu_s"] <= 550
    assert result["resource"]["process_peak_rss_bytes"] <= 1024 ** 3 and result["resource"]["paid_usd"] == 0
    assert controller.fingerprint() == result["inputs"]
    controller.require_frozen(result["freeze"], result["inputs"])
    archive = ROOT / result["private_archive"]["path"]
    assert result["private_archive"]["bytes"] == archive.stat().st_size <= 16 * 1024 ** 2
    assert pc.sha(archive) == result["private_archive"]["sha256"]
    with gzip.open(archive, "rt") as stream:
        ledgers = json.load(stream)
    views, extraction = controller.input_views()
    assert extraction == result["extraction"]
    allocation = [(f"control:{family}:{seed}:{mode}", pc.controls(family, seed, mode))
                  for family, seed in pc.CONTROL_ALLOCATION for mode in pc.MODES]
    allocation.extend(views.items())
    assert len(allocation) == len(result["cells"]) and set(ledgers) == {name for name, _ in allocation}
    scored, permutations = 0, 0
    try:
        for index, ((name, rows), cell) in enumerate(zip(allocation, result["cells"])):
            assert cell["name"] == name
            assert cell["input_rows_sha"] == pc.json_sha([asdict(row) for row in rows])
            assert cell["bootstrap_seed"] == 97701 + index and cell["permutation_seed"] == 97801 + index
            a, b = replay_cell(rows, cell, ledgers[name], start_wall, start_cpu)
            scored += a
            permutations += b
            print(f"audit {index + 1}/{len(allocation)} completed", flush=True)
        assert result["decisions"] == controller.decisions(result["cells"])
        assert controller.fingerprint() == result["inputs"]
        controller.require_frozen(result["freeze"], result["inputs"])
        controller.guard(start_wall, start_cpu)
        receipt = {"experiment": pc.EXPERIMENT, "pass": True, "result_sha256": pc.sha(result_path),
                   "events_recomputed": scored, "permutations_recomputed": permutations,
                   "scope": "Independent training recount, transported odds, junction reconstruction, full event/null scores; shared parser/generator/fold/bootstrap/gate helpers; same author, not expert",
                   "resource": {"wall_s": time.monotonic() - start_wall,
                                "cpu_s": time.process_time() - start_cpu,
                                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                "paid_usd": 0}}
        controller.write_exclusive(out / "audit.json", receipt)
        return receipt
    except BaseException as exc:
        controller.write_exclusive(out / "audit-failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


def audit():
    out = ROOT / "results" / pc.EXPERIMENT
    if any((out / name).exists() for name in ("audit-started.json", "audit.json", "audit-failure.json")):
        raise FileExistsError("Audit already attempted")
    controller.write_exclusive(out / "audit-started.json", {"experiment": pc.EXPERIMENT,
                               "started_utc": datetime.now(timezone.utc).isoformat(),
                               "limits": {"wall_s": 600, "cpu_s": 550,
                                          "process_rss_bytes": 1024 ** 3, "paid_usd": 0}})
    controller.hard_limits()
    try:
        return _audit_work()
    except BaseException as exc:
        if not (out / "audit-failure.json").exists():
            controller.write_exclusive(out / "audit-failure.json",
                                       {"error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        controller.clear_limits()


if __name__ == "__main__":
    audit()
