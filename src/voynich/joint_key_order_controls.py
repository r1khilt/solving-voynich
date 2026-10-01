"""Count/name-preserving order perturbations with disclosed gold construction.

Gold source segmentation is used only to construct controls. Model inputs remain
unsegmented ciphertext; no source string, prefix mask or unit boundary is passed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from voynich.joint_key_proposal import unit_pool
from voynich.joint_key_training import pack_episodes
from voynich.recurrent_latin_source import encode


def cipher_digest(records):
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def controlled_episode(episode, texts, *, condition, seed):
    if condition not in {"unit_shuffle", "glyph_shuffle"} or type(seed) is not int:
        raise ValueError("Declared condition and integer seed required")
    records, key, metadata = episode
    pool = unit_pool(6)
    if len(records) != 2 or len(key) != 23 or len(metadata["windows"]) != 2:
        raise ValueError("Two records and complete dictionary required")
    rng = np.random.default_rng(seed)
    changed, details = [], []
    for record, window in zip(records, metadata["windows"], strict=True):
        source = encode(texts[window["segment"]])[window["start"]:window["start"]+window["length"]]
        if len(source) != window["length"] or not len(source):
            raise ValueError("Gold construction crosses source boundary")
        units = [pool[key[int(c)]] for c in source]
        flattened = tuple(c for unit in units for c in unit)
        if flattened != tuple(record):
            raise ValueError("Source segmentation does not re-encode original cipher")
        all_symbols, seen, stop, prefix_glyphs = set(record), set(), 0, 0
        for stop, unit in enumerate(units, 1):
            seen.update(unit)
            prefix_glyphs += len(unit)
            if seen == all_symbols:
                break
        suffix_units = units[stop:]
        if condition == "unit_shuffle":
            order = rng.permutation(len(suffix_units)).tolist()
            suffix = tuple(c for i in order for c in suffix_units[i])
        else:
            suffix = tuple(int(c) for c in rng.permutation(record[prefix_glyphs:]))
        new = tuple(record[:prefix_glyphs])+suffix
        if (Counter(new) != Counter(record) or len(new) != len(record)
                or tuple(dict.fromkeys(new)) != tuple(dict.fromkeys(record))
                or new[:prefix_glyphs] != tuple(record[:prefix_glyphs])):
            raise ValueError("Order control changed counts, names or frozen prefix")
        changed.append(new)
        details.append({"prefix_source_units": stop, "prefix_cipher_glyphs": prefix_glyphs,
            "record_glyphs": len(record), "mutable_source_units": len(suffix_units),
            "mutable_glyphs": len(record)-prefix_glyphs,
            "changed_glyph_positions": sum(a != b for a, b in zip(new, record, strict=True))})
    control_metadata = {"used_rows": metadata["used_rows"], "counterfactual": True,
        "original_episode_metadata": metadata,
        "control_cipher_ids_sha256": cipher_digest(changed)}
    return (tuple(changed), key, control_metadata), {
        "condition": condition, "seed": seed, "original_cipher_ids_sha256": cipher_digest(records),
        "controlled_cipher_ids_sha256": cipher_digest(changed), "records": details,
        "gold_segmentation_only_for_control_construction": True,
        "key_and_used_rows_unchanged": True,
    }


@torch.inference_mode()
def score_episode(model, episode, *, saved_original=None):
    model.eval()
    records, target = pack_episodes([episode])
    if next(model.parameters()).device.type != "cpu" or next(model.parameters()).dtype != torch.float64:
        raise ValueError("CPUfloat64 diagnostic only")
    memory, padding = model.encode_records(records)
    logits = model.decode_partial(memory, padding, target[:, :-1])
    row_logq = F.log_softmax(logits, -1).gather(-1, target[..., None]).squeeze(-1)[0]
    proposed, logq = model.propose(memory, padding, greedy=True)
    guess = proposed[0, 0]
    used = episode[2]["used_rows"]
    result = {"joint_logq": float(row_logq.sum()), "row_logq": row_logq.tolist(),
        "greedy_key": guess.tolist(), "greedy_logq": float(logq[0, 0]),
        "correct_rows": int((guess == target[0]).sum()),
        "used_correct_rows": int((guess[used] == target[0, used]).sum()),
        "used_rows": len(used), "whole_key_exact": bool(torch.equal(guess, target[0]))}
    if not all(math.isfinite(v) for v in [*result["row_logq"], result["greedy_logq"]]):
        raise ValueError("Nonfinite diagnostic probabilities")
    if saved_original is not None:
        old = torch.tensor([saved_original["key"]], dtype=torch.long)
        old_logits = model.decode_partial(memory, padding, old[:, :-1])
        old_lp = F.log_softmax(old_logits, -1).gather(-1, old[..., None]).sum()
        deficits = old_logits.max(-1).values-old_logits.gather(-1, old[..., None]).squeeze(-1)
        numeric = {"true_logq_delta": abs(result["joint_logq"]-saved_original["joint_logq"]),
            "saved_greedy_logq_delta": abs(float(old_lp)-saved_original["greedy_logq"]),
            "saved_greedy_max_cpu_deficit": float(deficits.max()),
            "cpu_vs_mps_greedy_changed_rows": int((guess != old[0]).sum())}
        if (max(numeric["true_logq_delta"], numeric["saved_greedy_logq_delta"]) > .002
                or numeric["saved_greedy_max_cpu_deficit"] > 1e-4):
            raise ValueError("Original checkpoint CPUdouble replay gate failed")
        result["numeric"] = numeric
    return result


def summarize(rows, *, seed=72353, bootstraps=2048):
    by = {(r["checkpoint"], r["episode"], r["condition"]): r for r in rows}
    if len(by) != len(rows):
        raise ValueError("Duplicate diagnostic record")
    checkpoints = sorted({r["checkpoint"] for r in rows})
    cases = sorted({r["episode"] for r in rows})
    if not cases or not checkpoints or len(by) != len(cases)*len(checkpoints)*3:
        raise ValueError("Complete checkpoint/case/condition grid required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(cases), size=(bootstraps, len(cases)))
    results = {}
    for step in checkpoints:
        own = [by[step, i, "original"] for i in cases]
        comparisons = {}
        for condition in ("unit_shuffle", "glyph_shuffle"):
            control = [by[step, i, condition] for i in cases]
            delta = np.asarray([(a["joint_logq"]-b["joint_logq"])/23
                               for a, b in zip(own, control, strict=True)])
            comparisons[condition] = {"mean_control_minus_original_nats_per_row": float(delta.mean()),
                "descriptive_episode_bootstrap_95_interval": np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
                "episodes_control_density_lower": int((delta > 0).sum()),
                "paired_delta_nats_per_row": delta.tolist(),
                "original_used_correct_rows": sum(r["used_correct_rows"] for r in own),
                "control_used_correct_rows": sum(r["used_correct_rows"] for r in control),
                "used_rows": sum(r["used_rows"] for r in own),
                "control_greedy_key_changed_episodes": sum(a["greedy_key"] != b["greedy_key"] for a, b in zip(own, control, strict=True)),
                "control_whole_keys_exact": sum(r["whole_key_exact"] for r in control)}
        results[str(step)] = {"original_nats_per_row": -math.fsum(r["joint_logq"] for r in own)/(23*len(cases)),
            "original_whole_keys_exact": sum(r["whole_key_exact"] for r in own), "comparisons": comparisons}
    first, last = str(checkpoints[0]), str(checkpoints[-1])
    unit = results[last]["comparisons"]["unit_shuffle"]
    grown = unit["mean_control_minus_original_nats_per_row"]-results[first]["comparisons"]["unit_shuffle"]["mean_control_minus_original_nats_per_row"]
    supported = (len(checkpoints) == 2 and unit["mean_control_minus_original_nats_per_row"] >= .01
                 and unit["descriptive_episode_bootstrap_95_interval"][0] > 0 and grown > 0)
    return {"episodes": len(cases), "checkpoints": results, "bootstrap_seed": seed,
        "bootstraps": bootstraps, "selected_minus_initial_unit_order_effect": grown,
        "exploratory_unit_order_diagnostic": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "not_a_recovery_or_circuit_qualification": True}
