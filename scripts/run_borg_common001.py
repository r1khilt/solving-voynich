"""Input preparation and one bounded common-name historical candidate seal.

No published Borg key, corrected plaintext or translation is opened here.
Historical success is deliberately absent until a separate post-seal answer
acquisition/visual-key collation is registered and completed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

import numpy as np

from scripts.audit_borg_glyph_resources import atlas_labels
from scripts.borg_control_parser import PINNED_SHA256, parse_bytes
from voynich.borg_common import (COMMON, LATIN, Objective, decode, digest,
                                encode_records, fit, select_body, train_source)


ROOT = Path(__file__).resolve().parents[1]
EXP = "BORG-COMMON-001"
OUT = ROOT / "results" / EXP
BULK = ROOT / "outputs" / EXP
PREP = OUT / "prepare.json"
RAW = "data/raw/borg/transcription-0001r-0204v.txt"
BRIDGE = "results/BORG-GLYPH-BRIDGE-005/visual_ledger.json"
ATLAS = "data/manifests/borg_glyph_resource_inventory.json"
CORPUS = "results/LATIN-SOURCE-001/corpus.json"
SOURCE_MANIFEST = "data/manifests/naibbe003_data.json"
PATHS = ("src/voynich/borg_common.py", "scripts/run_borg_common001.py",
    "scripts/audit_borg_common001.py", "tests/test_borg_common_audit001.py",
    "tests/test_borg_common001.py", "docs/experiments/BORG-COMMON-001.md",
    "docs/research/borg-common-pilot-and-three-day-priorities-2026-10-10.md",
    "scripts/borg_control_parser.py", "scripts/audit_borg_glyph_resources.py",
    BRIDGE, ATLAS, CORPUS, SOURCE_MANIFEST,
    "results/NAIBBE-003/source_selection.json",
    "results/BORG-GLYPH-BRIDGE-005/audit.json")
RESTARTS, SWEEPS = 12, 8
CPU_CAP, WALL_CAP, RSS_CAP = 900, 1020, 1024**3
BULK_CAP = 16 * 1024**2
CONTROL_FIT, CONTROL_TRANSFER = 6000, 4000
CONTROL_AUTHORS = ("phi0588", "phi1212")
KEY_SEEDS, SEARCH_SEEDS = (96701, 96709), (96721, 96729)
HIST_SEEDS = (96741, 96749, 96757)


def save_new(path, value, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    if path.is_relative_to(BULK) and len(raw) > BULK_CAP:
        raise MemoryError("Private artifact exceeds16MiB")
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": digest(raw)}


def load_bound(entry, compressed=False):
    path = (ROOT / entry["path"]).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Artifact escapes repository")
    raw = path.read_bytes()
    if digest(raw) != entry["sha256"] or len(raw) != entry["bytes"]:
        raise ValueError("Bound artifact changed")
    return json.loads(gzip.decompress(raw) if compressed else raw)


def source_inputs():
    manifest = json.loads((ROOT / SOURCE_MANIFEST).read_text())
    answer = {}
    for language in ("latin", "english"):
        entry = manifest["lms"][language]
        raw = (ROOT / entry["path"]).read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError("Source file changed")
        text = raw.decode().strip()
        if len(text) != 317326 or set(text) - set(LATIN):
            raise ValueError("Wrong source alphabet/length")
        answer[language] = {"entry": {**entry, "bytes": len(raw)}, "text": text}
    return answer


def control_records(rows, letters):
    """First 10k allowed letters; censor y/z with hard context resets.

    Existing canonical Latin archives fold v->u and retain k.  This declared
    control conversion maps k->c only.  It does not reverse that v/u folding.
    Cuts at64 letters are artificial test lineation, never manuscript evidence.
    """
    accepted = set(letters)
    records, current, count = [], [], 0
    for row in rows:
        for char in row["text"].replace("k", "c"):
            if char not in accepted:
                if current:
                    records.append("".join(current))
                    current = []
                continue
            current.append(char)
            count += 1
            if len(current) == 64:
                records.append("".join(current))
                current = []
            if count == CONTROL_FIT + CONTROL_TRANSFER:
                if current:
                    records.append("".join(current))
                return records
        if current:
            records.append("".join(current))
            current = []
    raise ValueError("Insufficient control source")


def split_records(records, cut):
    fit_rows, transfer_rows, seen = [], [], 0
    for value in records:
        left = max(0, min(len(value), cut - seen))
        if left:
            fit_rows.append(value[:left])
        if left < len(value):
            transfer_rows.append(value[left:])
        seen += len(value)
    return fit_rows, transfer_rows


def prepare_payload():
    blob = (ROOT / RAW).read_bytes()
    parsed = parse_bytes(blob, expected_sha256=PINNED_SHA256)
    atlas = json.loads((ROOT / ATLAS).read_text())
    singles = {value for value in atlas_labels(ROOT, atlas) if len(value) == 1}
    if singles != set(COMMON):
        raise ValueError("Common-name hypothesis differs from image atlas")
    ledger = json.loads((ROOT / BRIDGE).read_text())
    excluded_lines = sorted({row["target"]["source_line"] for row in ledger["observations"]
        if row["shape_status"] != "local_atlas_resemblance"
        or row["context_status"] != "main_symbol_run"})
    audit = json.loads((ROOT / "results/BORG-GLYPH-BRIDGE-005/audit.json").read_text())
    if not audit["status"].startswith("PASS") or audit["ledger_sha256"] != digest((ROOT / BRIDGE).read_bytes()):
        raise ValueError("Historical preparation ledger lacks its accounting closure")
    historical = {"fit": select_body(blob, parsed, leaves=range(20, 61), excluded_lines=frozenset(excluded_lines)),
                  "transfer": select_body(blob, parsed, leaves=range(120, 161), excluded_lines=frozenset(excluded_lines))}
    if (not historical["fit"] or not historical["transfer"]
            or ({row["leaf"] for row in historical["fit"]}
                & {row["leaf"] for row in historical["transfer"]})):
        raise ValueError("Empty/overlapping historical splits")
    counts = {split: {"records": len(rows), "characters": sum(len(row["symbols"]) for row in rows),
        "leaves": sorted({row["leaf"] for row in rows}),
        "types": dict(sorted(Counter("".join(row["symbols"] for row in rows)).items()))}
              for split, rows in historical.items()}
    if any(row["characters"] < 4000 or row["characters"] > 40000 for row in counts.values()):
        raise ValueError("Historical fixed panel outside4k–40k character bounds")
    sources = source_inputs()
    corpus = json.loads((ROOT / CORPUS).read_text())
    controls, gold, control_bindings = {}, {}, {}
    letters = "".join(letter for letter in LATIN if letter not in "yz")
    # Exact64-letter overlap with the Caesar source is checked before fitting;
    # no replacement is selected if the predetermined source fails this check.
    source_windows = {sources["latin"]["text"][i:i+64]
                      for i in range(len(sources["latin"]["text"]) - 63)}
    for index, author in enumerate(CONTROL_AUTHORS):
        entry = corpus["authors"][author]["artifact"]
        payload = load_bound(entry, compressed=True)
        if payload["author"] != author or payload["alphabet"] != corpus["alphabet"]:
            raise ValueError("Control source identity differs")
        rows = control_records(payload["records"], letters)
        if any(value[i:i+64] in source_windows for value in rows for i in range(len(value)-63)):
            raise ValueError("Control64-letter source overlap")
        plain_fit, plain_transfer = split_records(rows, CONTROL_FIT)
        rng = np.random.Generator(np.random.PCG64(KEY_SEEDS[index]))
        completed_key = rng.permutation(len(letters))
        inverse = {letters[int(letter)]: COMMON[symbol]
                   for symbol, letter in enumerate(completed_key)}
        crypt_fit = ["".join(inverse[letter] for letter in row) for row in plain_fit]
        crypt_transfer = ["".join(inverse[letter] for letter in row) for row in plain_transfer]
        name = f"control-{index}"
        controls[name] = {"fit": crypt_fit, "transfer": crypt_transfer,
                          "search_seed": SEARCH_SEEDS[index], "kind": "positive"}
        full_key = [LATIN.index(letters[int(letter)]) for letter in completed_key]
        full_key += [LATIN.index(letter) for letter in "yz"]
        gold[name] = {"plain_transfer": plain_transfer, "key": full_key}
        shuffled_fit = ["".join(rng.permutation(list(row))) for row in crypt_fit]
        shuffled_transfer = ["".join(rng.permutation(list(row))) for row in crypt_transfer]
        controls[f"shuffle-{index}"] = {"fit": shuffled_fit, "transfer": shuffled_transfer,
                    "search_seed": SEARCH_SEEDS[index] + 100, "kind": "paired_row_shuffle"}
        control_bindings[author] = entry
    if set(controls) != {"control-0", "control-1", "shuffle-0", "shuffle-1"}:
        raise ValueError("Incomplete control allocation")
    return {"experiment": EXP, "historical": historical, "controls": controls}, gold, {
        "historical_counts": counts, "excluded_lines": excluded_lines,
        "raw_source_sha256": PINNED_SHA256,
        "source_files": {name: row["entry"] for name, row in sources.items()},
        "control_source_files": control_bindings,
        "control_known_letters": letters, "new_key_seeds": list(KEY_SEEDS),
        "source_overlap64": 0,
        "historical_name_semantics": "hypothesis_not_global_glyph_verification",
        "historical_heldout_answer": "not_acquired_not_opened"}


def prepare():
    if PREP.exists() or (BULK / "inputs.json.gz").exists() or (BULK / "control_gold.json.gz").exists():
        raise FileExistsError("Preserve original prepared inputs")
    inputs, gold, metadata = prepare_payload()
    input_artifact = save_new(BULK / "inputs.json.gz", inputs, compressed=True)
    gold_artifact = save_new(BULK / "control_gold.json.gz", gold, compressed=True)
    return save_new(PREP, {"experiment": EXP, "status": "PREPARED_NO_MODEL_SCORES",
        "inputs": input_artifact, "control_gold": gold_artifact, **metadata,
        "preparation_source_hashes": {name: digest((ROOT/name).read_bytes()) for name in PATHS}})


def run(freeze):
    wall, cpu = time.monotonic(), time.process_time()
    if (any((OUT / name).exists() for name in ("started.json", "sealed_predictions.json",
            "result.json", "failure.json", "audit-started.json", "audit.json", "audit-failure.json"))
            or (BULK / "transfers.json.gz").exists()):
        raise FileExistsError("One original producer only")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != freeze:
        raise ValueError("HEAD differs from declared freeze")
    freeze_paths = (*PATHS, str(PREP.relative_to(ROOT)))
    for name in freeze_paths:
        if subprocess.check_output(["git", "show", f"{freeze}:{name}"], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError(f"Unfrozen dependency: {name}")
    hashes = {name: digest((ROOT/name).read_bytes()) for name in freeze_paths}
    preparation = json.loads(PREP.read_text())
    if preparation.get("preparation_source_hashes") != {name: hashes[name] for name in PATHS}:
        raise ValueError("Preparation source fingerprint differs from freeze")
    inputs = load_bound(preparation["inputs"], compressed=True)
    sources = source_inputs()
    for language in sources:
        if sources[language]["entry"] != preparation["source_files"][language]:
            raise ValueError("Prepared source binding changed")
    save_new(OUT / "started.json", {"experiment": EXP, "freeze": freeze,
        "source_hashes": hashes, "started_unix": time.time(), "paid_cost_usd": 0})
    old_alarm = signal.getsignal(signal.SIGALRM)
    old_cpu_signal = signal.getsignal(signal.SIGXCPU)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Wall limit")))
    signal.signal(signal.SIGXCPU, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU limit")))
    signal.alarm(WALL_CAP)
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (min(CPU_CAP, hard) if hard > 0 else CPU_CAP, hard))

    def guard():
        if time.process_time() - cpu >= CPU_CAP or time.monotonic() - wall >= WALL_CAP:
            raise TimeoutError("Resource limit")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_CAP:
            raise MemoryError("RSS limit")

    try:
        if (set(inputs["controls"]) != {"control-0", "control-1", "shuffle-0", "shuffle-1"}
                or {name for name, row in inputs["controls"].items() if row["kind"] == "positive"}
                != {"control-0", "control-1"}):
            raise ValueError("Incomplete control allocation")
        tables = {language: train_source(row["text"]) for language, row in sources.items()}
        runs = {}
        for name, row in inputs["controls"].items():
            runs[name] = (row["fit"], "latin", row["search_seed"])
        historical_fit = [row["symbols"] for row in inputs["historical"]["fit"]]
        shuffle_rng = np.random.Generator(np.random.PCG64(96767))
        historical_shuffle = ["".join(shuffle_rng.permutation(list(row))) for row in historical_fit]
        runs.update({"borg-latin": (historical_fit, "latin", HIST_SEEDS[0]),
                     "borg-english": (historical_fit, "english", HIST_SEEDS[1]),
                     "borg-shuffle-latin": (historical_shuffle, "latin", HIST_SEEDS[2])})
        if len(runs) != 7:
            raise ValueError("Incomplete fit allocation")
        fits = {}
        for name, (records, language, seed) in runs.items():
            guard()
            objective = Objective(encode_records(records, COMMON), tables[language])
            result = fit(objective, seed=seed, restarts=RESTARTS, sweeps=SWEEPS, guard=guard)
            result.update(language=language, fit_characters=objective.characters,
                          fit_records=len(records), input_sha256=digest(json.dumps(records).encode()))
            fits[name] = result
            print(json.dumps({"case": name, "status": "fit_complete", "cpu_seconds": time.process_time()-cpu}), flush=True)
        # This durable seal precedes every control answer read and every transfer
        # decoding/score.  An external Borg answer phase is a separate future run.
        seal = save_new(OUT / "sealed_predictions.json", {"experiment": EXP, "freeze": freeze,
            "source_hashes": hashes, "preparation_sha256": digest(PREP.read_bytes()),
            "fits": fits, "historical_answer_opened": False,
            "seal_unix": time.time()})
        gold = load_bound(preparation["control_gold"], compressed=True)
        if set(gold) != {"control-0", "control-1"}:
            raise ValueError("Incomplete control answers")
        transfers, metrics = {}, {}
        for name, row in inputs["controls"].items():
            key = fits[name]["key"]
            decoded = decode(row["transfer"], key)
            obj = Objective(encode_records(row["transfer"], COMMON), tables["latin"])
            transfers[name] = {"decoded": decoded, "score": float(obj.scores(np.asarray(key)[None])[0]),
                               "characters": obj.characters}
            if row["kind"] == "positive":
                truth = gold[name]
                errors = sum(a != b for real, pred in zip(truth["plain_transfer"], decoded, strict=True)
                             for a, b in zip(real, pred, strict=True))
                counts = Counter("".join(row["transfer"]))
                used = [COMMON.index(symbol) for symbol in counts]
                matches = sum(key[index] == truth["key"][index] for index in used)
                metrics[name] = {"errors": errors, "characters": obj.characters,
                    "used_rows": len(used), "correct_used_rows": matches,
                    "pass": errors * 100 <= obj.characters and matches * 100 >= 95 * len(used)}
        historical_transfer = [row["symbols"] for row in inputs["historical"]["transfer"]]
        for name, language in (("borg-latin", "latin"), ("borg-english", "english")):
            key = fits[name]["key"]
            obj = Objective(encode_records(historical_transfer, COMMON), tables[language])
            transfers[name] = {"decoded": decode(historical_transfer, key),
                "score": float(obj.scores(np.asarray(key)[None])[0]), "characters": obj.characters}
        if set(transfers) != {"control-0", "control-1", "shuffle-0", "shuffle-1", "borg-latin", "borg-english"}:
            raise ValueError("Incomplete transfer allocation")
        private = save_new(BULK / "transfers.json.gz", transfers, compressed=True)
        guard()
        for name, expected in hashes.items():
            if digest((ROOT/name).read_bytes()) != expected:
                raise ValueError("Frozen input changed during original run")
        result = {"experiment": EXP, "freeze": freeze, "seal": seal, "private_transfers": private,
            "control_metrics": metrics, "control_competence": "PASS"
            if set(metrics) == {"control-0", "control-1"} and all(row["pass"] for row in metrics.values()) else "FAIL",
            "historical_recovery": "UNASSESSED_NO_EXTERNAL_ANSWER",
            "historical_answer_opened": False,
            "transfer_scores": {name: {key:value for key,value in row.items() if key != "decoded"}
                                for name,row in transfers.items()},
            "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "paid_cost_usd": 0, "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "thread_environment": {k:os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}}
        save_new(OUT / "result.json", result)
        print(json.dumps({k:result[k] for k in ("control_competence", "historical_recovery", "cpu_seconds", "wall_seconds")}), flush=True)
    except BaseException as error:
        save_new(OUT / "failure.json", {"experiment": EXP, "freeze": freeze,
            "exception": type(error).__name__, "message": str(error),
            "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall,
            "no_retry_or_extension": True})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGXCPU, old_cpu_signal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(), sort_keys=True))
    else:
        if not args.freeze:
            raise ValueError("Original producer requires explicit published freeze")
        run(args.freeze)


if __name__ == "__main__":
    main()
