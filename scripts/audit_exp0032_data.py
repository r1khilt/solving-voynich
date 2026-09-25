"""Separate all-row source replay and true-channel audit for EXP-0032."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

import voynich.latent_recovery as channel
from voynich.runtime import write_json


FAMILIES = ("random_char", "periodic", "copy_mutate")
SEEDS = {"train": 320032, "validation": 320033, "polish_holdout": 320034, "iid_control": 320035}
COUNTS = {"train": 12000, "validation": 1200, "polish_holdout": 600, "iid_control": 600}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_clean_polish(raw: str) -> str:
    start = raw.index("=Żabusia.=")
    end = raw.rfind("UWAGI DO WYDANIA ELEKTRONICZNEGO")
    body = raw[start:end].translate(str.maketrans({"ł": "l", "Ł": "L"}))
    body = unicodedata.normalize("NFKD", body)
    body = "".join(ch for ch in body if unicodedata.category(ch) != "Mn")
    return channel.clean_plaintext(body)


def capture_sample(piece: str, rng: np.random.Generator, world: int, alphabet: str) -> tuple[dict, str | None]:
    seen: list[str] = []
    original = channel.insert_nulls

    def tapped(ciphered: str, *args: object, **kwargs: object) -> tuple[str, list[int], list[str]]:
        seen.append(ciphered)
        return original(ciphered, *args, **kwargs)

    channel.insert_nulls = tapped
    try:
        row = channel.make_sample(piece, rng, world, 0.3, alphabet, FAMILIES)
    finally:
        channel.insert_nulls = original
    if world == channel.WORLD_C and len(seen) != 1:
        raise AssertionError("WORLD_C did not expose exactly one pre-null source")
    if world != channel.WORLD_C and seen:
        raise AssertionError("non-C world invoked null insertion")
    return row, seen[0] if seen else None


def check_record(saved: dict, fresh: dict, true_source: str | None) -> bool:
    if saved != fresh:
        raise AssertionError("stored row does not match independent seeded replay")
    text, mask = saved["text"], saved["mask"]
    if len(text) != 128 or len(mask) != 128:
        raise AssertionError("window text/mask lengths differ")
    if saved["world"] == channel.WORLD_C:
        if true_source is None:
            raise AssertionError("missing pre-null source")
        kept = "".join(ch for ch, m in zip(text, mask, strict=True) if m)
        if kept != saved["ciphered"] or not true_source.startswith(kept.rstrip(" ")):
            raise AssertionError("true pre-null cipher is not the keep-mask source")
        return True
    return False


def audit(root: Path) -> dict:
    manifest_path = root / "data/manifests/exp0032_data.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "EXP-0032" or manifest["counts"] != COUNTS or manifest["seeds"] != SEEDS:
        raise AssertionError("data registration drift")
    raw_dir = root / "data/raw/latent_corpora"
    source_paths = {
        "english": raw_dir / "english.clean.txt",
        "latin": raw_dir / "latin.clean.txt",
        "polish_34635": raw_dir / "polish_34635.txt",
    }
    for name, path in source_paths.items():
        if digest(path) != manifest["source_sha256"][name]:
            raise AssertionError(f"source hash mismatch: {name}")
    polish = independent_clean_polish(source_paths["polish_34635"].read_text(encoding="utf-8-sig"))
    if hashlib.sha256(polish.encode()).hexdigest() != manifest["polish_clean_sha256"]:
        raise AssertionError("Polish preprocessing replay mismatch")
    english = source_paths["english"].read_text()
    latin = source_paths["latin"].read_text()
    split = manifest["train_val_source_split_char"]
    source = {"english": english, "latin": latin}
    stores = {}
    for name in COUNTS:
        path = root / f"data/processed/exp0032/{name}.jsonl"
        if digest(path) != manifest["derived_sha256"][name]:
            raise AssertionError(f"derived hash mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != COUNTS[name]:
            raise AssertionError(f"derived count mismatch: {name}")
        stores[name] = rows
    true_source_rows = 0
    family_counts = {}
    for name in ("train", "validation"):
        rng = np.random.default_rng(SEEDS[name])
        parts = {language: (text[:split[language]] if name == "train" else text[split[language] + 128 :]) for language, text in source.items()}
        for saved in stores[name]:
            world = int(rng.choice(4, p=[0.10, 0.15, 0.60, 0.15]))
            language = str(rng.choice(tuple(parts)))
            piece = channel.chunks_from_text(parts[language], rng, 1)[0]
            alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=36, replace=False))
            fresh, true_source = capture_sample(piece, rng, world, alphabet)
            fresh["language"] = language
            true_source_rows += int(check_record(saved, fresh, true_source))
        family_counts[name] = {f: sum(row["filler_family"] == f for row in stores[name]) for f in FAMILIES}
    flat = re.sub(r"\s+", " ", polish).strip()
    rng = np.random.default_rng(SEEDS["polish_holdout"])
    blocks = rng.choice(len(flat) // 128, size=COUNTS["polish_holdout"], replace=False)
    if len(set(map(int, blocks))) != len(blocks):
        raise AssertionError("Polish source blocks overlap")
    for saved, block_index in zip(stores["polish_holdout"], blocks, strict=True):
        piece = flat[int(block_index) * 128 : (int(block_index) + 1) * 128]
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        fresh, true_source = capture_sample(piece, rng, channel.WORLD_C, alphabet)
        fresh["language"] = "polish_34635"
        fresh["source_block_index"] = int(block_index)
        true_source_rows += int(check_record(saved, fresh, true_source))
    family_counts["polish_holdout"] = {f: sum(row["filler_family"] == f for row in stores["polish_holdout"]) for f in FAMILIES}
    for name, counts in family_counts.items():
        if counts != manifest["family_counts"][name]:
            raise AssertionError("filler-family counts do not match manifest")
    rng = np.random.default_rng(SEEDS["iid_control"])
    for saved in stores["iid_control"]:
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        chars = rng.choice(list(alphabet), size=128, replace=True)
        mask = np.zeros(128, dtype=np.uint8)
        mask[rng.choice(128, size=90, replace=False)] = 1
        text = "".join(chars)
        fresh = {
            "world": channel.WORLD_C,
            "text": text,
            "mask": mask.astype(int).tolist(),
            "ciphered": "".join(ch for ch, m in zip(text, mask, strict=True) if m),
            "plaintext": "",
            "alphabet": alphabet,
            "filler_family": "iid_indistinguishable",
            "filler_rate": 0.3,
            "language": "iid_uniform_control",
        }
        if saved != fresh:
            raise AssertionError("iid control replay mismatch")
    if true_source_rows < 8000:
        raise AssertionError("too few true-source WORLD_C rows")
    return {
        "experiment": "EXP-0032",
        "passed": True,
        "manifest_sha256": digest(manifest_path),
        "auditor_sha256": digest(Path(__file__)),
        "rows_replayed": sum(COUNTS.values()),
        "true_source_rows_checked": true_source_rows,
        "polish_nonoverlap_blocks": len(blocks),
        "family_counts": family_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0032/data_audit.json"
    if out.exists():
        raise FileExistsError("data audit already exists; refusing overwrite")
    result = audit(args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
