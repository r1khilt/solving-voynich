"""Independent, explicitly gated replay of selected developmental channels.

No production inference, decoding, distance or coding helpers are imported.
Importing this module performs no file I/O. The command refuses artifact reads
unless --after-evaluation is supplied. This is an audit of selected frozen
models, not a replay of all structural-search candidates or a recovery claim.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-001"
TOLERANCE = 1e-7


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _distribution(values, label):
    numbers = list(values)
    if (not numbers or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1 for value in numbers)
            or abs(math.fsum(numbers) - 1) > 1e-12):
        raise ValueError(f"Invalid normalized distribution: {label}")


def _parse(source: dict, channel: dict):
    """Validate public JSON independently and return explicit transition rows."""
    alphabet, states, glyphs = source["alphabet"], channel["states"], channel["glyph_alphabet"]
    for values, single, label in ((alphabet, True, "source"), (states, False, "states"), (glyphs, True, "glyphs")):
        if (not values or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value or (single and len(value) != 1) for value in values)):
            raise ValueError(f"Invalid {label} alphabet")
    if source.get("schema_version") != 1 or channel.get("schema_version") != 1:
        raise ValueError("Unsupported model schema")
    order = source["order"]
    if type(order) is not int or order not in (0, 1):
        raise ValueError("Only source order zero or one is supported")
    contexts = [""] if order == 0 else ["", *alphabet]
    if set(source["probabilities"]) != set(contexts):
        raise ValueError("Wrong source contexts")
    for context in contexts:
        row = source["probabilities"][context]
        if set(row) != set(alphabet):
            raise ValueError("Wrong source row support")
        _distribution(row.values(), "source")
    if set(channel["initial"]) != set(states):
        raise ValueError("Wrong initial-state support")
    _distribution(channel["initial"].values(), "initial")
    rho = channel["stop_probability"]
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("Invalid stopping law")
    maximum = channel["max_emission_length"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid emission-length bound")
    rows = {}
    for row in channel["rows"]:
        key = row["state"], row["letter"]
        if key in rows or key[0] not in states or key[1] not in alphabet:
            raise ValueError("Wrong or duplicate channel row")
        emissions = row["emissions"]
        _distribution([item["probability"] for item in emissions], "emissions")
        for item in emissions:
            if (item["next_state"] not in states or not isinstance(item["glyphs"], str)
                    or not 1 <= len(item["glyphs"]) <= maximum or set(item["glyphs"]) - set(glyphs)):
                raise ValueError("Invalid channel emission")
        rows[key] = emissions
    if set(rows) != {(state, letter) for state in states for letter in alphabet}:
        raise ValueError("Channel rows do not cover the complete declared support")
    return alphabet, states, rows


def _logsum(values):
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def infer_record(source: dict, channel: dict, record: str) -> dict:
    """Independent incoming-edge log-sum and maximum-joint-path dynamic programs.

    A node is (source context, state) at a consumed glyph offset. Incoming
    masses are collected and summed together. The separate max table retains
    the first traversed representative of an exact tie, using declared model
    orders. This is joint Viterbi, not a most-probable-plaintext claim.
    """
    alphabet, states, rows = _parse(source, channel)
    if not isinstance(record, str):
        raise ValueError("Observed records must be strings")
    unsupported = {"log_likelihood": None, "joint_log_probability": None,
                   "plaintext": None, "prediction_supported": False}
    if set(record) - set(channel["glyph_alphabet"]):
        return unsupported
    incoming = [{} for _ in range(len(record) + 1)]
    maximum = [{} for _ in range(len(record) + 1)]
    back = {}
    for state in states:
        probability = channel["initial"][state]
        if probability:
            node = "", state
            incoming[0][node] = [math.log(probability)]
            maximum[0][node] = math.log(probability)
    continuation = math.log1p(-channel["stop_probability"])
    for offset in range(len(record)):
        for node, contributions in incoming[offset].items():
            context, state = node
            forward = _logsum(contributions)
            for letter in alphabet:
                source_probability = source["probabilities"][context][letter]
                if not source_probability:
                    continue
                for item in rows[state, letter]:
                    probability, unit = item["probability"], item["glyphs"]
                    if not probability or not record.startswith(unit, offset):
                        continue
                    end = offset + len(unit)
                    target = (letter if source["order"] else ""), item["next_state"]
                    weight = continuation + math.log(source_probability) + math.log(probability)
                    incoming[end].setdefault(target, []).append(forward + weight)
                    candidate = maximum[offset][node] + weight
                    if candidate > maximum[end].get(target, -math.inf):
                        maximum[end][target] = candidate
                        back[end, target] = offset, node, letter
    if not incoming[-1]:
        return unsupported
    final_stop = math.log(channel["stop_probability"])
    likelihood = _logsum([_logsum(values) for values in incoming[-1].values()]) + final_stop
    last = max(maximum[-1], key=maximum[-1].__getitem__)
    joint = maximum[-1][last] + final_stop
    letters, offset = [], len(record)
    while offset:
        offset, last, letter = back[offset, last]
        letters.append(letter)
    return {"log_likelihood": likelihood, "joint_log_probability": joint,
            "plaintext": "".join(reversed(letters)), "prediction_supported": True}


def edit_distance(first: str, second: str) -> int:
    """Unit-cost insertion/deletion/substitution distance, with linear memory."""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for col, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate_channel(source: dict, channel: dict | None, records, gold_records=None) -> dict:
    """Return strict-JSON likelihood, literal joint-decoding and edit metrics.

    An impossible observation has null likelihood/plaintext and contributes
    zero decoded characters. Against gold it counts as deleting all gold
    characters. Null cases have no gold-derived metrics, including exactness.
    """
    if isinstance(records, (str, bytes)) or isinstance(gold_records, (str, bytes)):
        raise ValueError("Supply sequences of separate records, not a bare string")
    records = list(records)
    if not records or any(not isinstance(record, str) for record in records):
        raise ValueError("Need a nonempty list of observed records")
    if gold_records is not None:
        gold_records = list(gold_records)
        if len(gold_records) != len(records) or any(not isinstance(record, str) for record in gold_records):
            raise ValueError("Gold records must correspond one-to-one with observations")
    details = []
    for index, record in enumerate(records):
        inferred = (infer_record(source, channel, record) if channel is not None else
                    {"log_likelihood": None, "joint_log_probability": None,
                     "plaintext": None, "prediction_supported": False})
        gold = None if gold_records is None else gold_records[index]
        inferred.update(edits=None if gold is None else edit_distance(inferred["plaintext"] or "", gold),
                        gold_characters=None if gold is None else len(gold))
        details.append(inferred)
    supported = all(item["prediction_supported"] for item in details)
    return {"log_likelihood": math.fsum(item["log_likelihood"] for item in details) if supported else None,
            "edits": None if gold_records is None else sum(item["edits"] for item in details),
            "gold_characters": None if gold_records is None else sum(map(len, gold_records)),
            "decoded_characters": sum(len(item["plaintext"] or "") for item in details),
            "exact_records": None if gold_records is None else sum(
                item["prediction_supported"] and item["plaintext"] == gold
                for item, gold in zip(details, gold_records, strict=True)),
            "records": details}


def model_bits(channel: dict, context: dict, source_index: int = 0) -> int:
    """Count the specified finite code fields without importing its encoder."""
    denominator, states = context["denominator"], channel["states"]
    if (type(denominator) is not int or denominator < 1 or not 0 <= source_index < context["source_count"]
            or len(states) > context["max_states"] or channel["glyph_alphabet"] != context["glyph_alphabet"]
            or channel["stop_probability"] != context["stop_probability"]):
        raise ValueError("Channel is outside the shared coding context")

    def width(cardinality):
        if cardinality < 1:
            raise ValueError("Empty coding cardinality")
        return (cardinality - 1).bit_length()

    def grid(values, positive):
        counts = [round(value * denominator) for value in values]
        if (sum(counts) != denominator or any(count < int(positive) for count in counts)
                or any(abs(value * denominator - count) > 1e-9 for value, count in zip(values, counts, strict=True))):
            raise ValueError("Channel probabilities are not the declared coded weights")

    grid([channel["initial"][state] for state in states], False)
    bits = width(context["source_count"]) + width(context["max_states"])
    bits += width(math.comb(denominator + len(states) - 1, len(states) - 1))
    for row in channel["rows"]:
        alternatives = row["emissions"]
        if (not 1 <= len(alternatives) <= min(context["max_alternatives"], denominator)
                or len({(item["next_state"], item["glyphs"]) for item in alternatives}) != len(alternatives)):
            raise ValueError("Invalid coded emission support")
        grid([item["probability"] for item in alternatives], True)
        bits += width(context["max_alternatives"])
        for item in alternatives:
            if not 1 <= len(item["glyphs"]) <= context["max_emission_length"]:
                raise ValueError("Emission length outside coding context")
            bits += width(len(states)) + width(context["max_emission_length"])
            bits += len(item["glyphs"]) * width(len(context["glyph_alphabet"]))
        bits += width(math.comb(denominator - 1, len(alternatives) - 1))
    return bits


def compare_known(expected, actual, path="root", tolerance=TOLERANCE) -> float:
    """Require every audit-known field; permit additional production baselines."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) - set(actual):
            raise AssertionError(f"Missing fields at {path}")
        return max((compare_known(value, actual[key], f"{path}.{key}", tolerance)
                    for key, value in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"List shape differs at {path}")
        return max((compare_known(left, right, f"{path}[{index}]", tolerance)
                    for index, (left, right) in enumerate(zip(expected, actual, strict=True))), default=0.)
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise AssertionError(f"Invalid numeric value at {path}")
        delta = abs(expected - actual)
        if delta > tolerance:
            raise AssertionError(f"Numeric mismatch at {path}: {delta}")
        return delta
    if type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"Exact mismatch at {path}")
    return 0.


def baseline_from_fit(records, glyphs) -> dict:
    """Reconstruct the registered iid baseline weights, stop grid and code."""
    counts = Counter("".join(records))
    if not records or set(counts) - set(glyphs) or not glyphs or len(glyphs) > 256:
        raise ValueError("Invalid iid baseline fitting data")
    total = sum(counts.values()) + len(glyphs) / 2
    targets = [256 * (counts[glyph] + .5) / total for glyph in glyphs]
    weights = [max(1, math.floor(target)) for target in targets]
    while sum(weights) != 256:
        if sum(weights) < 256:
            order = sorted(range(len(weights)), key=lambda i: (weights[i] - targets[i], i))
            weights[order[0]] += 1
        else:
            order = sorted((i for i, weight in enumerate(weights) if weight > 1),
                           key=lambda i: (targets[i] - weights[i], i))
            weights[order[0]] -= 1
    rho_count = max(1, min(4095, round(4096 * len(records) / (len(records) + sum(counts.values())))))
    remainder, rank = 256 - len(weights), 0
    for index, weight in enumerate(weights[:-1]):
        slots, value = len(weights) - index - 1, weight - 1
        rank += math.comb(remainder + slots, slots) - math.comb(remainder - value + slots, slots)
        remainder -= value
    cardinality = math.comb(255, len(weights) - 1)
    width = (cardinality - 1).bit_length()
    probability_code = format(rank, f"0{width}b") if width else ""
    code = "1" + format(rho_count - 1, "012b") + probability_code
    return {"family": "iid_glyph_geometric", "glyph_alphabet": list(glyphs),
            "denominator": 256, "counts": weights, "stop_denominator": 4096,
            "stop_count": rho_count, "model_code": code, "model_bits": len(code)}


def baseline_score(baseline, records) -> float:
    rho = baseline["stop_count"] / baseline["stop_denominator"]
    counts = Counter("".join(records))
    probabilities = dict(zip(baseline["glyph_alphabet"], baseline["counts"], strict=True))
    return (len(records) * math.log(rho) + sum(counts.values()) * math.log1p(-rho)
            + math.fsum(count * math.log(probabilities[glyph] / baseline["denominator"])
                        for glyph, count in counts.items()))


def _read(root: Path, relative: str, expected_sha: str | None = None, expected_bytes: int | None = None):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact paths must be repository-relative")
    raw = (root / path).read_bytes()
    if expected_sha is not None and digest(raw) != expected_sha:
        raise ValueError(f"Artifact hash mismatch: {relative}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError(f"Artifact byte count mismatch: {relative}")
    return raw


def audit(root: Path, *, after_evaluation: bool, manifest_path="data/manifests/blind_channel_dev001.json",
          evaluation_path=f"results/{EXPERIMENT}/evaluation.json") -> dict:
    """Validate artifact links and replay selected models; gate precedes all I/O."""
    if not after_evaluation:
        raise PermissionError("Answer access requires --after-evaluation after the evaluator has run")
    evaluation_raw = _read(root, evaluation_path)
    compact = json.loads(evaluation_raw)
    prediction_spec = compact["full_predictions"]
    prediction_raw = _read(root, prediction_spec["path"], prediction_spec["sha256"], prediction_spec.get("bytes"))
    evaluation = json.loads(prediction_raw)
    # Compare all common fields between the compact report and hashed full
    # predictions, omitting only the deliberately removed record details.
    projection = json.loads(prediction_raw)
    for case in projection["cases"].values():
        for split in case["splits"].values():
            for arm in ("learned", "oracle"):
                if arm in split:
                    split[arm].pop("records")
    projection_delta = compare_known(projection, compact)
    manifest_raw = _read(root, manifest_path, evaluation["manifest_sha256"])
    manifest = json.loads(manifest_raw)
    source_spec = manifest["source"]
    source_raw = _read(root, source_spec["path"], source_spec["sha256"], source_spec.get("bytes"))
    if evaluation["source_sha256"] != digest(source_raw):
        raise ValueError("Evaluation source hash differs")
    source = json.loads(source_raw)["source_model"]
    if set(evaluation["cases"]) != set(manifest["cases"]):
        raise ValueError("Evaluation case inventory differs")
    maximum_delta, artifacts, records_checked = projection_delta, {}, 0
    source_freezes = set()
    for name, specification in manifest["cases"].items():
        case = evaluation["cases"][name]
        freeze_path = f"results/{EXPERIMENT}/{name}_freeze.json"
        freeze_raw = _read(root, freeze_path, case["freeze_sha256"])
        frozen = json.loads(freeze_raw)
        if frozen["case_id"] != name or frozen["source_sha256"] != digest(source_raw):
            raise ValueError("Frozen model case/source mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", frozen["source_freeze"]):
            raise ValueError("Malformed source-freeze commit identifier")
        source_freezes.add(frozen["source_freeze"])
        output = frozen["full_output"]
        output_raw = _read(root, output["path"], output["sha256"], output.get("bytes"))
        full_output = json.loads(gzip.decompress(output_raw))
        if full_output["channel"] != frozen["channel"] or full_output["score"] != frozen["score"]:
            raise ValueError("Frozen channel/score differs from hashed full output")
        if full_output["config"] != frozen["config"]:
            raise ValueError("Frozen search configuration differs from full output")
        inputs = {}
        for split in ("fit", "transfer", "answer"):
            spec = specification["artifacts"][split]
            raw = _read(root, spec["path"], spec["sha256"], spec.get("bytes"))
            inputs[split] = json.loads(raw)
            artifacts[spec["path"]] = digest(raw)
        if frozen["input_sha256"] != specification["artifacts"]["fit"]["sha256"]:
            raise ValueError("Frozen fitting-input hash differs")
        answer = inputs["answer"]
        if answer["case_id"] != name or answer["positive"] is not specification["positive"]:
            raise ValueError("Answer case/positive status differs")
        expected_baseline = baseline_from_fit(inputs["fit"]["records"], inputs["fit"]["context"]["glyph_alphabet"])
        compare_known(expected_baseline, frozen["baseline"], name + ".baseline")
        expected_case = {"positive": answer["positive"], "family": specification["family"], "splits": {},
                         "model_bits_with_selector": None if frozen["score"] is None else frozen["score"]["model_bits"] + 1,
                         "baseline_model_bits": expected_baseline["model_bits"]}
        for split in ("fit", "transfer"):
            observed = inputs[split]
            if observed["case_id"] != name or observed["split"] != split:
                raise ValueError("Observed case/split identity differs")
            gold = answer["plaintext"][split] if answer["positive"] else None
            learned = evaluate_channel(source, frozen["channel"], observed["records"], gold)
            expected_case["splits"][split] = {
                "learned": learned, "baseline_log_likelihood": baseline_score(expected_baseline, observed["records"]),
                "glyph_characters": sum(map(len, observed["records"]))}
            records_checked += len(observed["records"])
            if answer["positive"]:
                expected_case["splits"][split]["oracle"] = evaluate_channel(
                    source, answer["gold_channel"], observed["records"], gold)
                records_checked += len(observed["records"])
            if split == "fit" and frozen["channel"] is not None:
                bits = model_bits(frozen["channel"], observed["context"], full_output.get("source_index", 0))
                if learned["log_likelihood"] is None:
                    raise ValueError("Frozen fitted model does not support its fitting records")
                data_bits = -learned["log_likelihood"] / math.log(2)
                expected_score = {"model_bits": bits, "data_bits": data_bits,
                                  "total_bits": bits + data_bits, "log_likelihood": learned["log_likelihood"]}
                maximum_delta = max(maximum_delta, compare_known(expected_score, frozen["score"], name + ".score"))
            elif frozen["channel"] is None and frozen["score"] is not None:
                raise ValueError("Missing frozen channel has a nonempty score")
        fit, transfer = expected_case["splits"]["fit"], expected_case["splits"]["transfer"]
        fit_ll, transfer_ll = fit["learned"]["log_likelihood"], transfer["learned"]["log_likelihood"]
        if fit_ll is None or transfer_ll is None:
            saving = gain = None
        else:
            saving = (expected_case["baseline_model_bits"] - fit["baseline_log_likelihood"] / math.log(2)
                      - expected_case["model_bits_with_selector"] + fit_ll / math.log(2))
            gain = (transfer_ll - transfer["baseline_log_likelihood"]) / (math.log(2) * transfer["glyph_characters"])
        expected_case.update(fit_saving_vs_iid_bits=saving, transfer_gain_vs_iid_bits_per_glyph=gain,
                             diagnostic_flag=saving is not None and saving >= 32 and gain >= .05)
        maximum_delta = max(maximum_delta, compare_known(expected_case, case, name))
        artifacts[freeze_path], artifacts[output["path"]] = digest(freeze_raw), digest(output_raw)
    if len(source_freezes) != 1:
        raise ValueError("Cases do not share one frozen source revision")
    positives = [case for case in evaluation["cases"].values() if case["positive"]]
    negatives = [case for case in evaluation["cases"].values() if not case["positive"]]
    summary = {"positives_cer_at_most_10pct": sum(
        case["splits"]["transfer"]["learned"]["edits"] <= .1 * case["splits"]["transfer"]["learned"]["gold_characters"]
        for case in positives), "positives_flagged": sum(case["diagnostic_flag"] for case in positives),
        "nulls_flagged": sum(case["diagnostic_flag"] for case in negatives)}
    compare_known(summary, compact["diagnostic_summary"], "diagnostic_summary")
    return {"experiment": EXPERIMENT, "status": "pass", "scope": "selected_frozen_models_only",
            "cases_checked": len(manifest["cases"]), "record_evaluations_checked": records_checked,
            "maximum_numeric_delta": maximum_delta, "absolute_tolerance": TOLERANCE,
            "source_freeze": next(iter(source_freezes)), "manifest_sha256": digest(manifest_raw),
            "source_sha256": digest(source_raw), "evaluation_sha256": digest(evaluation_raw),
            "full_predictions_sha256": digest(prediction_raw),
            "auditor_source_sha256": digest(Path(__file__).read_bytes()), "artifacts": artifacts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-evaluation", action="store_true")
    parser.add_argument("--manifest", default="data/manifests/blind_channel_dev001.json")
    parser.add_argument("--evaluation", default=f"results/{EXPERIMENT}/evaluation.json")
    parser.add_argument("--output", default=f"results/{EXPERIMENT}/audit.json")
    arguments = parser.parse_args(argv)
    if not arguments.after_evaluation:
        parser.error("Refusing answer access: supply --after-evaluation only after frozen-model evaluation")
    result = audit(ROOT, after_evaluation=True, manifest_path=arguments.manifest,
                   evaluation_path=arguments.evaluation)
    destination = ROOT / arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
