"""Independent full-row soft-target and fresh-source audit for EXP-0035."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import unicodedata
from pathlib import Path

import numpy as np

import voynich.latent_recovery as channel
from scripts.audit_exp0032_data import capture_sample
from scripts.audit_exp0033_alignment import recursive_alignment
from scripts.audit_exp0034_weighted_oracle import explicit_posterior
from voynich.runtime import write_json


FAMILIES = ("random_char", "periodic", "copy_mutate")
MAX_SECONDS = 300


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_clean_portuguese(raw: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***"
    text = raw.split(start_marker, 1)[1].split(end_marker, 1)[0]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return channel.clean_plaintext(text)


def check_soft_targets(root: Path, manifest: dict) -> dict:
    previous = json.loads((root / "data/manifests/exp0032_data.json").read_text())
    if manifest["prior_data_manifest_sha256"] != digest(root / "data/manifests/exp0032_data.json"):
        raise AssertionError("prior source manifest drift")
    training_path = root / "data/processed/exp0032/train.jsonl"
    if digest(training_path) != manifest["prior_train_sha256"]:
        raise AssertionError("training rows changed")
    rows = [json.loads(line) for line in training_path.read_text().splitlines()]
    soft_path = root / "data/processed/exp0035/train_soft_targets.npz"
    if digest(soft_path) != manifest["derived_sha256"]["train_soft_targets"]:
        raise AssertionError("soft target file checksum drift")
    with np.load(soft_path) as saved:
        uniform = saved["uniform"]
        weighted = saved["weighted"]
    if uniform.shape != (12000, 128) or weighted.shape != (12000, 128) or uniform.dtype != np.float32 or weighted.dtype != np.float32:
        raise AssertionError("soft target shape/dtype drift")
    english_path = root / "data/raw/latent_corpora/english.clean.txt"
    latin_path = root / "data/raw/latent_corpora/latin.clean.txt"
    if digest(english_path) != previous["source_sha256"]["english"] or digest(latin_path) != previous["source_sha256"]["latin"]:
        raise AssertionError("original training source drift")
    corpora = {"english": english_path.read_text(), "latin": latin_path.read_text()}
    pieces = {name: text[:previous["train_val_source_split_char"][name]] for name, text in corpora.items()}
    rng = np.random.default_rng(320032)
    copy_checked = 0
    maximum_weighted_error = 0.0
    maximum_uniform_error = 0.0
    for index, row in enumerate(rows):
        world = int(rng.choice(4, p=[0.10, 0.15, 0.60, 0.15]))
        language = str(rng.choice(tuple(pieces)))
        piece = channel.chunks_from_text(pieces[language], rng, 1)[0]
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=36, replace=False))
        fresh, source = capture_sample(piece, rng, world, alphabet)
        fresh["language"] = language
        if fresh != row:
            raise AssertionError(f"independent train replay differs at {index}")
        if world == channel.WORLD_C and row["filler_family"] == "copy_mutate":
            if source is None:
                raise AssertionError("missing copy full source")
            total, keep = recursive_alignment(row["text"], row["ciphered"])
            expected_uniform = np.asarray([count / total for count in keep])
            expected_weighted = np.asarray(explicit_posterior(row["text"], source, alphabet,
                                                               len(row["ciphered"]), row["mask"])["keep"])
            uniform_error = float(np.max(np.abs(expected_uniform - uniform[index])))
            weighted_error = float(np.max(np.abs(expected_weighted - weighted[index])))
            maximum_uniform_error = max(maximum_uniform_error, uniform_error)
            maximum_weighted_error = max(maximum_weighted_error, weighted_error)
            if uniform_error > 1e-6 or weighted_error > 1e-6:
                raise AssertionError(f"copy target mismatch at {index}: {uniform_error}, {weighted_error}")
            if abs(float(uniform[index].sum()) - len(row["ciphered"])) > 1e-4 or abs(float(weighted[index].sum()) - len(row["ciphered"])) > 1e-4:
                raise AssertionError("copy soft-target sum mismatch")
            copy_checked += 1
        elif not np.array_equal(uniform[index], np.asarray(row["mask"], dtype=np.float32)) or not np.array_equal(weighted[index], np.asarray(row["mask"], dtype=np.float32)):
            raise AssertionError(f"non-copy hard target altered at {index}")
    if copy_checked != manifest["soft_targets"]["copy_soft_rows"]:
        raise AssertionError("copy soft-row count drift")
    return {"training_rows_replayed": len(rows), "copy_soft_rows_recomputed": copy_checked,
            "max_weighted_probability_error": maximum_weighted_error,
            "max_uniform_probability_error": maximum_uniform_error}


def check_fresh_controls(root: Path, manifest: dict) -> dict:
    raw_path = root / "data/raw/latent_corpora/portuguese_55752.txt"
    if digest(raw_path) != manifest["portuguese_raw_sha256"]:
        raise AssertionError("Portuguese raw text drift")
    clean = independent_clean_portuguese(raw_path.read_text(encoding="utf-8-sig"))
    if hashlib.sha256(clean.encode()).hexdigest() != manifest["portuguese_clean_sha256"]:
        raise AssertionError("independent Portuguese clean text drift")
    flat = re.sub(r"\s+", " ", clean).strip()
    holdout_path = root / "data/processed/exp0035/portuguese_holdout.jsonl"
    iid_path = root / "data/processed/exp0035/iid_control.jsonl"
    if digest(holdout_path) != manifest["derived_sha256"]["portuguese_holdout"] or digest(iid_path) != manifest["derived_sha256"]["iid_control"]:
        raise AssertionError("fresh derived-file hash drift")
    holdout = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    iid = [json.loads(line) for line in iid_path.read_text().splitlines()]
    if len(holdout) != 600 or len(iid) != 600:
        raise AssertionError("fresh split count drift")
    rng = np.random.default_rng(350035)
    blocks = rng.choice(len(flat) // 128, 600, replace=False)
    if len(set(map(int, blocks))) != 600:
        raise AssertionError("Portuguese blocks overlap")
    for saved, block in zip(holdout, blocks, strict=True):
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        fresh, source = capture_sample(flat[int(block) * 128:(int(block) + 1) * 128], rng,
                                       channel.WORLD_C, alphabet)
        if source is None:
            raise AssertionError("missing Portuguese pre-null source")
        fresh.update(language="portuguese_55752", source_block_index=int(block))
        if saved != fresh:
            raise AssertionError("Portuguese row replay mismatch")
        if "".join(ch for ch, bit in zip(saved["text"], saved["mask"], strict=True) if bit) != saved["ciphered"]:
            raise AssertionError("Portuguese gold source mismatch")
    if {family: sum(r["filler_family"] == family for r in holdout) for family in FAMILIES} != manifest["family_counts"]:
        raise AssertionError("Portuguese family counts drift")
    rng = np.random.default_rng(350036)
    for saved in iid:
        alphabet = "".join(rng.choice(list(channel.CIPHER_POOL), size=40, replace=False))
        text = "".join(rng.choice(list(alphabet), size=128, replace=True))
        mask = np.zeros(128, dtype=np.uint8)
        mask[rng.choice(128, size=90, replace=False)] = 1
        kept = mask.astype(int).tolist()
        fresh = {"world": channel.WORLD_C, "text": text, "mask": kept,
                 "ciphered": "".join(ch for ch, bit in zip(text, kept, strict=True) if bit),
                 "plaintext": "", "alphabet": alphabet, "filler_family": "iid_indistinguishable",
                 "filler_rate": 0.3, "language": "iid_uniform_control_exp0035"}
        if saved != fresh:
            raise AssertionError("fresh iid control replay mismatch")
    return {"portuguese_rows_replayed": len(holdout), "iid_rows_replayed": len(iid),
            "nonoverlap_source_blocks": len(blocks), "family_counts": manifest["family_counts"]}


def audit(root: Path) -> dict:
    manifest_path = root / "data/manifests/exp0035_data.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "EXP-0035" or manifest["builder_sha256"] != digest(root / "src/voynich/exp0035_data.py"):
        raise AssertionError("builder/manifest identity drift")
    if manifest["generator_sha256"] != digest(Path(channel.__file__)):
        raise AssertionError("channel changed")
    target = check_soft_targets(root, manifest)
    fresh = check_fresh_controls(root, manifest)
    return {"experiment": "EXP-0035", "passed": True, "manifest_sha256": digest(manifest_path),
            "auditor_sha256": digest(Path(__file__)), **target, **fresh}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.root / "results/EXP-0035/data_audit.json"
    if out.exists():
        raise FileExistsError("EXP-0035 data audit already exists")

    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError("EXP-0035 data audit exceeded 300 seconds")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_SECONDS)
    try:
        result = audit(args.root)
    finally:
        signal.alarm(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
