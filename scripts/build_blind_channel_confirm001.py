"""Evaluator-only fresh-key construction after pipeline and corpus freezes.

Public seeds provide reproducibility, not cryptographic answer isolation.
The fitting runner never imports this module or reads its answer artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
from dataclasses import asdict
from pathlib import Path

from scripts.build_blind_channel_dev001 import encode_records, make_channel, shuffled_records
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from voynich.finite_state_channel_fit import CodingContext

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-CONFIRM-001"
MANIFEST = "data/manifests/blind_channel_confirm001.json"
CORPUS = "data/manifests/blind_channel_confirmation_corpora.json"
KEY_COUNT = 8


def construct(payloads: dict, alphabet: tuple[str, ...], windows) -> dict:
    """Pure construction; no performance-based filtering or redraws."""
    result, seen_keys = {}, set()
    for index in range(KEY_COUNT):
        key_seed = 60129 + 104729 * index
        channel = make_channel(alphabet, "B", key_seed)
        signature = tuple(channel.rows["s0", letter][0].glyphs for letter in alphabet)
        if signature in seen_keys:
            raise ValueError("Duplicate generated key; stop, do not redraw")
        seen_keys.add(signature)
        selected = {role: windows(payloads[author], role, index)
                    for role, author in (("fit", "sallust"), ("transfer", "tacitus"))}
        plain = {role: [row["text"] for row in rows] for role, rows in selected.items()}
        if (len(plain["fit"]) != 4 or len(plain["transfer"]) != 2
                or any(len(text) != 224 or set(text) - set(alphabet)
                       for rows in plain.values() for text in rows)):
            raise ValueError("Fresh text schedule or alphabet differs")
        encoded = {role: encode_records(rows, channel) for role, rows in plain.items()}
        context = asdict(CodingContext(alphabet, channel.glyph_alphabet, 32, 2, 2, 3,
                                      stop_probability=1 / 225))
        for is_null in (False, True):
            name = f"B-key{index + 1}" + ("-shuffle" if is_null else "")
            rng = random.Random(key_seed + 3_000_000)
            records = {role: shuffled_records(rows, rng) if is_null else rows
                       for role, rows in encoded.items()}
            result[name] = {"positive": not is_null, "context": context,
                            "records": records, "search_seed": 63101 + 257 * (2 * index + int(is_null)),
                            "answer": {"positive": not is_null,
                                "gold_channel": None if is_null else channel.to_dict(),
                                "plaintext": None if is_null else plain,
                                "source_windows": {role: [{k: v for k, v in row.items() if k != "text"}
                                                         for row in rows] for role, rows in selected.items()},
                                "key_seed": key_seed,
                                "null_seed": key_seed + 3_000_000 if is_null else None}}
    return result


def main():
    from scripts.prepare_blind_channel_confirmation_corpora import fixed_windows
    from scripts.run_blind_channel_confirm001 import SOURCE_PATHS, write_new

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-freeze", required=True)
    parser.add_argument("--corpus-freeze", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    require_frozen(args.pipeline_freeze, SOURCE_PATHS)
    require_frozen(args.corpus_freeze, [CORPUS])
    if (ROOT / MANIFEST).exists() or (ROOT / f"data/processed/{EXPERIMENT}").exists():
        raise FileExistsError("Fresh panel already attempted; no selective regeneration")
    corpus = json.loads((ROOT / CORPUS).read_text())
    if (corpus.get("status") != "prepared" or set(corpus["sources"]) != {"sallust", "tacitus"}
            or corpus["sources"]["sallust"].get("role") != "F"
            or corpus["sources"]["tacitus"].get("role") != "T"):
        raise ValueError("Corpus preparation/role/overlap screen is not qualified")
    write_new(ROOT / f"data/processed/{EXPERIMENT}/construction_started.json",
              {"pipeline_freeze": args.pipeline_freeze, "corpus_freeze": args.corpus_freeze})
    payloads = {name: checked_artifact({"path": corpus["sources"][name]["derived_path"],
                                       "sha256": corpus["sources"][name]["derived_sha256"]})
                for name in ("sallust", "tacitus")}
    alphabet = tuple(corpus["alphabet"])
    if "".join(alphabet) != "abcdefghiklmnopqrstuxyz":
        raise ValueError("Registered normalized alphabet differs")
    cases = {}
    for name, case in construct(payloads, alphabet, fixed_windows).items():
        artifacts = {role: write_new(ROOT / f"data/processed/{EXPERIMENT}/{name}/{role}.json",
                                     {"case_id": name, "split": role, "records": records,
                                      "context": case["context"]})
                     for role, records in case["records"].items()}
        artifacts["answer"] = write_new(ROOT / f"data/processed/{EXPERIMENT}/{name}/answer.json", case["answer"])
        cases[name] = {"positive": case["positive"], "family": "B", "search_seed": case["search_seed"],
                       "artifacts": artifacts}
    write_new(ROOT / MANIFEST, {"experiment": EXPERIMENT, "status": "fresh_conditional_B_qualification",
        "pipeline_freeze": args.pipeline_freeze, "corpus_freeze": args.corpus_freeze,
        "corpus_manifest_sha256": digest((ROOT / CORPUS).read_bytes()), "cases": cases,
        "assistance": ["Latin and normalized alphabet", "one deterministic channel state", "one/two-glyph units",
                       "ABCDEF inventory", "record boundaries", "mean source-length prior224"],
        "answer_isolation": "Procedural, not cryptographic; builder performs no fitting or score-based selection",
        "final_author_roles": {"fit": "sallust", "transfer": "tacitus"}})
    print(json.dumps({"experiment": EXPERIMENT, "cases": len(cases), "key_count": KEY_COUNT}))


if __name__ == "__main__":
    main()
