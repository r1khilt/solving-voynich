"""Replay EXP-0030 data and compare stored targets with pre-null cipher streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import voynich.latent_recovery as lr
from voynich.runtime import digest, write_json


EXPECTED_RAW_SHA256 = {
    "english": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
    "latin": "84ac8411841a4d8f5f4a49b6a2cd1f466917c6a5af72916d5e0b2b1ecb2f659c",
    "finnish": "23e57fdc123b738f49e31a68ceef2b74b77a83baa0359ffd820f682905283631",
}
SAVED_KEYS = ("world", "language", "text", "mask", "ciphered", "filler_family", "filler_rate")


def alignment_checks(source: str, noisy: str, mask: list[int], tags: list[str], target: str) -> dict:
    return {
        "lengths_match": len(noisy) == len(mask) == len(tags),
        "full_inverse_matches_source": "".join(ch for ch, keep in zip(noisy, mask) if keep) == source,
        "window_target_is_source_prefix": source.startswith(target.rstrip(" ")),
    }


def audit(root: Path, experiment_id: str = "EXP-0030") -> dict:
    slug = experiment_id.lower().replace("-", "")
    manifest = json.loads((root / "data" / "manifests" / f"{slug}_data.json").read_text())
    if manifest["experiment"] != experiment_id:
        raise ValueError("manifest experiment mismatch")
    raw = root / "data" / "raw" / "latent_corpora"
    raw_sha = {lang: digest(raw / f"{lang}.txt") for lang in EXPECTED_RAW_SHA256}
    if raw_sha != EXPECTED_RAW_SHA256:
        raise ValueError("raw corpus checksum mismatch")
    texts = {
        lang: lr.clean_plaintext((raw / f"{lang}.txt").read_text(encoding="utf-8", errors="replace"))
        for lang in EXPECTED_RAW_SHA256
    }

    captured: list[tuple[str, str, list[int], list[str]]] = []
    historical = lr.insert_nulls

    def capture(source, rng, rate, alphabet, filler_families=None):
        noisy, mask, tags = historical(source, rng, rate, alphabet, filler_families)
        captured.append((source, noisy, mask, tags))
        return noisy, mask, tags

    lr.insert_nulls = capture
    try:
        train, validation = lr.generate_dataset(
            {lang: texts[lang] for lang in ("english", "latin")},
            manifest["n_train"],
            manifest["n_val"],
            manifest["data_seed"],
            manifest["filler_rate"],
            tuple(manifest["filler_families"]),
        )
        holdout = lr.generate_finnish_holdout(
            texts["finnish"],
            manifest["n_holdout"],
            manifest["finnish_seed"],
            manifest["filler_rate"],
            tuple(manifest["filler_families"]),
        )
    finally:
        lr.insert_nulls = historical

    split_rows = {"train": train, "validation": validation, "finnish_holdout": holdout}
    cursor = 0
    splits = {}
    all_ok = True
    for name, rows in split_rows.items():
        path = root / "data" / "processed" / slug / f"{name}.jsonl"
        saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        digest_ok = digest(path) == manifest["derived_sha256"][name]
        exact_rows = len(saved) == len(rows) and all(
            all(row.get(key) == stored.get(key) for key in SAVED_KEYS)
            for row, stored in zip(rows, saved)
        )
        world_c = [row for row in rows if row["world"] == lr.WORLD_C]
        traces = captured[cursor : cursor + len(world_c)]
        cursor += len(world_c)
        counts = {
            family: {
                "rows": 0,
                "length_failures": 0,
                "full_inverse_failures": 0,
                "window_target_failures": 0,
            }
            for family in manifest["filler_families"]
        }
        for row, (source, noisy, mask, tags) in zip(world_c, traces):
            family = row["filler_family"]
            check = alignment_checks(source, noisy, mask, tags, row["ciphered"])
            item = counts[family]
            item["rows"] += 1
            item["length_failures"] += not check["lengths_match"]
            item["full_inverse_failures"] += not check["full_inverse_matches_source"]
            item["window_target_failures"] += not check["window_target_is_source_prefix"]
        split_ok = (
            digest_ok
            and exact_rows
            and len(traces) == len(world_c)
            and all(all(v == 0 for k, v in item.items() if k != "rows") for item in counts.values())
        )
        all_ok = all_ok and split_ok
        splits[name] = {
            "n_saved": len(saved),
            "n_replayed": len(rows),
            "jsonl_sha256": digest(path),
            "digest_matches_manifest": digest_ok,
            "all_saved_fields_match_replay": exact_rows,
            "world_c": counts,
            "passed": split_ok,
        }
    all_ok = all_ok and cursor == len(captured)
    if splits["finnish_holdout"]["world_c"]["copy_mutate"]["rows"] < 60:
        all_ok = False
    return {
        "experiment": experiment_id,
        "passed": all_ok,
        "raw_sha256": raw_sha,
        "manifest_sha256": digest(root / "data" / "manifests" / f"{slug}_data.json"),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "captured_world_c": len(captured),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--experiment-id", default="EXP-0030")
    args = parser.parse_args()
    report = audit(args.root, args.experiment_id)
    out = args.root / "results" / args.experiment_id / "source_oracle_audit.json"
    write_json(out, report)
    print(json.dumps({"passed": report["passed"], "splits": report["splits"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
