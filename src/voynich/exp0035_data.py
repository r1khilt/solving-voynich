"""Frozen training posterior targets and fresh Portuguese data for EXP-0035."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

import voynich.latent_recovery as channel
from voynich.exp0033_alignment import alignment_counts
from voynich.exp0034_weighted_oracle import weighted_posterior
from voynich.runtime import write_json


EXPERIMENT = "EXP-0035"
TRAIN_SHA = "378276e449cd25934c62e770a104eadb0275d7e1e9813baa6c45b6fe453f7240"
VALIDATION_SHA = "aa7404ebd06d206ed8798fdb2f0f931167fa0834f862fb3a7a8fe2383f2d125a"
PORTUGUESE_RAW_SHA = "0fc3dbf384544d81d87e5a731e67b7976ac3a57acb378f0f354d12a3b52bd0c7"
PORTUGUESE_CLEAN_SHA = "6c5a4271c2ce38b3bfb289085fd32ec635fd07f6a8bb23de13c18d3be1b0dddf"
START = "*** START OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***"
FAMILIES = ("random_char", "periodic", "copy_mutate")
PORTUGUESE_SEED = 350035
IID_SEED = 350036
N_HOLDOUT = 600
N_IID = 600


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_portuguese(raw: str) -> str:
    start = raw.index(START) + len(START)
    end = raw.index(END)
    if end <= start or end < len(raw) * 0.8:
        raise ValueError("Portuguese Gutenberg boundaries changed")
    body = "".join(ch for ch in unicodedata.normalize("NFKD", raw[start:end]) if not unicodedata.combining(ch))
    return channel.clean_plaintext(body)


def portuguese_rows(clean: str) -> list[dict]:
    flat = re.sub(r"\s+", " ", clean).strip()
    blocks = len(flat) // 128
    if blocks < 2 * N_HOLDOUT:
        raise ValueError("Portuguese source too short for nonoverlap blocks")
    rng = np.random.default_rng(PORTUGUESE_SEED)
    chosen = rng.choice(blocks, size=N_HOLDOUT, replace=False)
    rows = []
    for block in chosen:
        piece = flat[int(block) * 128:(int(block) + 1) * 128]
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        row = channel.make_sample(piece, rng, channel.WORLD_C, 0.3, alphabet, FAMILIES)
        row.update(language="portuguese_55752", source_block_index=int(block))
        rows.append(row)
    if len({row["source_block_index"] for row in rows}) != N_HOLDOUT:
        raise AssertionError("Portuguese source blocks not unique")
    if sum(row["filler_family"] == "copy_mutate" for row in rows) < 150:
        raise AssertionError("too few fresh Portuguese copy rows")
    return rows


def iid_rows() -> list[dict]:
    rng = np.random.default_rng(IID_SEED)
    rows = []
    for _ in range(N_IID):
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        text = "".join(rng.choice(list(alphabet), size=128, replace=True))
        mask = np.zeros(128, dtype=np.uint8)
        mask[rng.choice(128, size=90, replace=False)] = 1
        keep = mask.astype(int).tolist()
        rows.append({"world": channel.WORLD_C, "text": text, "mask": keep,
                     "ciphered": "".join(ch for ch, bit in zip(text, keep, strict=True) if bit),
                     "plaintext": "", "alphabet": alphabet, "filler_family": "iid_indistinguishable",
                     "filler_rate": 0.3, "language": "iid_uniform_control_exp0035"})
    return rows


def training_targets(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    previous = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    path = root / "data/processed/exp0032/train.jsonl"
    if digest(path) != TRAIN_SHA or previous["derived_sha256"]["train"] != TRAIN_SHA:
        raise ValueError("EXP-0032 training data drift")
    english_path = root / "data/raw/latent_corpora/english.clean.txt"
    latin_path = root / "data/raw/latent_corpora/latin.clean.txt"
    if digest(english_path) != previous["source_sha256"]["english"] or digest(latin_path) != previous["source_sha256"]["latin"]:
        raise ValueError("training source drift")
    sources = {"english": english_path.read_text(), "latin": latin_path.read_text()}
    parts = {name: text[:previous["train_val_source_split_char"][name]] for name, text in sources.items()}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 12000:
        raise ValueError("training count drift")
    hard = np.asarray([row["mask"] for row in rows], dtype=np.float32)
    uniform = hard.copy()
    weighted = hard.copy()
    rng = np.random.default_rng(320032)
    copy_count = 0
    for index, saved in enumerate(rows):
        world = int(rng.choice(4, p=[0.10, 0.15, 0.60, 0.15]))
        language = str(rng.choice(tuple(parts)))
        piece = channel.chunks_from_text(parts[language], rng, 1)[0]
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=36, replace=False))
        source_seen: list[str] = []
        original = channel.insert_nulls

        def capture(source: str, *args: object, **kwargs: object) -> tuple[str, list[int], list[str]]:
            source_seen.append(source)
            return original(source, *args, **kwargs)

        channel.insert_nulls = capture
        try:
            fresh = channel.make_sample(piece, rng, world, 0.3, alphabet, FAMILIES)
        finally:
            channel.insert_nulls = original
        fresh["language"] = language
        if fresh != saved:
            raise AssertionError(f"training source replay mismatch at row {index}")
        if world == channel.WORLD_C and saved["filler_family"] == "copy_mutate":
            if len(source_seen) != 1:
                raise AssertionError("copy source missing from replay")
            total, counts = alignment_counts(saved["text"], saved["ciphered"])
            if total < 1:
                raise AssertionError("gold copy alignment absent")
            uniform[index] = np.asarray([count / total for count in counts], dtype=np.float32)
            posterior = weighted_posterior(saved["text"], source_seen[0], alphabet,
                                           len(saved["ciphered"]), saved["mask"])
            weighted[index] = np.asarray(posterior["posterior_keep_probabilities"], dtype=np.float32)
            if abs(float(uniform[index].sum()) - len(saved["ciphered"])) > 1e-4 or abs(float(weighted[index].sum()) - len(saved["ciphered"])) > 1e-4:
                raise AssertionError("soft target retained count mismatch")
            copy_count += 1
    if copy_count != previous["family_counts"]["train"]["copy_mutate"]:
        raise AssertionError("copy row count drift")
    if not np.all((uniform >= 0) & (uniform <= 1)) or not np.all((weighted >= 0) & (weighted <= 1)):
        raise AssertionError("soft target outside probability range")
    meta = {"training_rows_replayed": len(rows), "copy_soft_rows": copy_count,
            "mean_copy_weighted_abs_gold_difference": float(np.mean(np.abs(weighted - hard)[np.asarray([r["filler_family"] == "copy_mutate" and r["world"] == channel.WORLD_C for r in rows])])),
            "mean_copy_uniform_abs_gold_difference": float(np.mean(np.abs(uniform - hard)[np.asarray([r["filler_family"] == "copy_mutate" and r["world"] == channel.WORLD_C for r in rows])]))}
    return uniform, weighted, meta


def dump_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest(path)


def prepare(root: Path) -> dict:
    manifest_path = root / "data/manifests/exp0035_data.json"
    out_dir = root / "data/processed/exp0035"
    if manifest_path.exists() or out_dir.exists():
        raise FileExistsError("EXP-0035 output already exists; refusing overwrite")
    previous = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    if digest(root / "data/processed/exp0032/validation.jsonl") != VALIDATION_SHA:
        raise ValueError("validation source drift")
    if digest(Path(channel.__file__)) != previous["generator_sha256"]:
        raise ValueError("corrected channel source changed")
    raw_path = root / "data/raw/latent_corpora/portuguese_55752.txt"
    if digest(raw_path) != PORTUGUESE_RAW_SHA:
        raise ValueError("Portuguese raw source changed")
    clean = clean_portuguese(raw_path.read_text(encoding="utf-8-sig"))
    if hashlib.sha256(clean.encode()).hexdigest() != PORTUGUESE_CLEAN_SHA:
        raise ValueError("Portuguese preprocessing drift")
    uniform, weighted, soft_meta = training_targets(root)
    holdout = portuguese_rows(clean)
    iid = iid_rows()
    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir / "train_soft_targets.npz", uniform=uniform, weighted=weighted)
    hashes = {"train_soft_targets": digest(out_dir / "train_soft_targets.npz"),
              "portuguese_holdout": dump_jsonl(out_dir / "portuguese_holdout.jsonl", holdout),
              "iid_control": dump_jsonl(out_dir / "iid_control.jsonl", iid)}
    manifest = {"experiment": EXPERIMENT,
                "source_url": "https://www.gutenberg.org/ebooks/55752", "source_title": "Dom Casmurro",
                "source_author": "Machado de Assis", "source_language": "Portuguese",
                "source_rights": "Project Gutenberg official page states public domain in USA; raw text remains Git-ignored",
                "portuguese_raw_sha256": PORTUGUESE_RAW_SHA, "portuguese_clean_sha256": PORTUGUESE_CLEAN_SHA,
                "portuguese_clean_characters": len(clean),
                "preprocessing": "Slice exact Gutenberg START/END; Unicode NFKD strip combining marks; clean_plaintext; disjoint 128-char blocks",
                "prior_train_sha256": TRAIN_SHA, "prior_validation_sha256": VALIDATION_SHA,
                "prior_data_manifest_sha256": digest(root / "data/manifests/exp0032_data.json"),
                "generator_sha256": digest(Path(channel.__file__)), "builder_sha256": digest(Path(__file__)),
                "seeds": {"training_replay": 320032, "portuguese_holdout": PORTUGUESE_SEED, "iid_control": IID_SEED},
                "counts": {"train": 12000, "portuguese_holdout": len(holdout), "iid_control": len(iid)},
                "family_counts": {f: sum(row["filler_family"] == f for row in holdout) for f in FAMILIES},
                "soft_targets": soft_meta, "derived_sha256": hashes}
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = prepare(args.root)
    print(json.dumps({k: result[k] for k in ("counts", "family_counts", "soft_targets", "derived_sha256")}, indent=2))


if __name__ == "__main__":
    main()
