"""Frozen synthetic data preparation for EXP-0032; no model scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from voynich.latent_recovery import (
    CIPHER_POOL,
    PRIMARY_FILLER_RATE,
    SEQ_LEN,
    WORLD_C,
    chunks_from_text,
    clean_plaintext,
    make_sample,
)
from voynich.runtime import write_json


EXPERIMENT = "EXP-0032"
FAMILIES = ("random_char", "periodic", "copy_mutate")
RAW_SHA = {
    "english": "4beea3799a297c19b2a0781056e851873558c1317c67d482d2982e3940e018e7",
    "latin": "298bdb17fb07ace8b861102975e0d961dc054bb57a0397e37d91dc40e6254a1f",
    "polish_34635": "d5521a54c38616e4b2255af8fe8deb6e95f4428e721d140973c672a482b28b5c",
}
POLISH_CLEAN_SHA = "f04f01cfa8e1cb92435b22f637011432eab84ce3e94fe9c22180ec5c2026964f"
N_TRAIN = 12000
N_VAL = 1200
N_POLISH = 600
N_IID = 600
TRAIN_SEED = 320032
VAL_SEED = 320033
POLISH_SEED = 320034
IID_SEED = 320035
WORLD_PROBS = np.array([0.10, 0.15, 0.60, 0.15])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_polish_raw(raw: str) -> str:
    start = raw.index("=Żabusia.=")
    end = raw.rfind("UWAGI DO WYDANIA ELEKTRONICZNEGO")
    if end <= start or end < int(len(raw) * 0.8):
        raise ValueError("Polish text boundary drift")
    body = raw[start:end].replace("ł", "l").replace("Ł", "L")
    body = "".join(ch for ch in unicodedata.normalize("NFKD", body) if not unicodedata.combining(ch))
    return clean_plaintext(body)


def generate_mixed(corpora: dict[str, str], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    languages = tuple(corpora)
    rows = []
    for _ in range(n):
        world = int(rng.choice(4, p=WORLD_PROBS))
        language = str(rng.choice(languages))
        piece = chunks_from_text(corpora[language], rng, 1)[0]
        alphabet = "".join(rng.choice(list(CIPHER_POOL), size=36, replace=False))
        row = make_sample(piece, rng, world, PRIMARY_FILLER_RATE, alphabet, FAMILIES)
        row["language"] = language
        rows.append(row)
    return rows


def generate_polish(clean: str) -> list[dict]:
    flat = re.sub(r"\s+", " ", clean).strip()
    blocks = len(flat) // SEQ_LEN
    if blocks < 2 * N_POLISH:
        raise ValueError("Polish source too short for disjoint 128-character blocks")
    rng = np.random.default_rng(POLISH_SEED)
    chosen = rng.choice(blocks, size=N_POLISH, replace=False)
    rows = []
    for block_index in chosen:
        piece = flat[int(block_index) * SEQ_LEN : (int(block_index) + 1) * SEQ_LEN]
        alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
        row = make_sample(piece, rng, WORLD_C, PRIMARY_FILLER_RATE, alphabet, FAMILIES)
        row["language"] = "polish_34635"
        row["source_block_index"] = int(block_index)
        rows.append(row)
    return rows


def generate_iid() -> list[dict]:
    rng = np.random.default_rng(IID_SEED)
    rows = []
    for _ in range(N_IID):
        alphabet = "".join(rng.choice(list(CIPHER_POOL), size=40, replace=False))
        chars = rng.choice(list(alphabet), size=SEQ_LEN, replace=True)
        keep = np.zeros(SEQ_LEN, dtype=np.uint8)
        keep[rng.choice(SEQ_LEN, size=round((1 - PRIMARY_FILLER_RATE) * SEQ_LEN), replace=False)] = 1
        text = "".join(chars)
        rows.append({
            "world": WORLD_C,
            "text": text,
            "mask": keep.astype(int).tolist(),
            "ciphered": "".join(ch for ch, m in zip(text, keep, strict=True) if m),
            "plaintext": "",
            "alphabet": alphabet,
            "filler_family": "iid_indistinguishable",
            "filler_rate": PRIMARY_FILLER_RATE,
            "language": "iid_uniform_control",
        })
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest(path)


def prepare(root: Path) -> dict:
    raw_dir = root / "data/raw/latent_corpora"
    paths = {
        "english": raw_dir / "english.clean.txt",
        "latin": raw_dir / "latin.clean.txt",
        "polish_34635": raw_dir / "polish_34635.txt",
    }
    for key, path in paths.items():
        if digest(path) != RAW_SHA[key]:
            raise ValueError(f"input checksum changed: {key}")
    clean_polish = clean_polish_raw(paths["polish_34635"].read_text(encoding="utf-8-sig"))
    if hashlib.sha256(clean_polish.encode()).hexdigest() != POLISH_CLEAN_SHA:
        raise ValueError("Polish clean-text checksum changed")
    clean_path = raw_dir / "polish_34635.clean.txt"
    if clean_path.exists() and digest(clean_path) != POLISH_CLEAN_SHA:
        raise ValueError("saved Polish clean-text checksum changed")
    english, latin = paths["english"].read_text(), paths["latin"].read_text()
    split = {"english": int(len(english) * 0.8), "latin": int(len(latin) * 0.8)}
    train_sources = {"english": english[:split["english"]], "latin": latin[:split["latin"]]}
    val_sources = {"english": english[split["english"] + SEQ_LEN :], "latin": latin[split["latin"] + SEQ_LEN :]}
    rows = {
        "train": generate_mixed(train_sources, N_TRAIN, TRAIN_SEED),
        "validation": generate_mixed(val_sources, N_VAL, VAL_SEED),
        "polish_holdout": generate_polish(clean_polish),
        "iid_control": generate_iid(),
    }
    out_dir = root / "data/processed/exp0032"
    manifest_path = root / "data/manifests/exp0032_data.json"
    if out_dir.exists() or manifest_path.exists():
        raise FileExistsError("EXP-0032 data or manifest already exists; refusing overwrite")
    out_dir.mkdir(parents=True)
    derived = {name: dump_jsonl(out_dir / f"{name}.jsonl", group) for name, group in rows.items()}
    manifest = {
        "experiment": EXPERIMENT,
        "source_url": "https://www.gutenberg.org/ebooks/34635",
        "source_title": "Menazerya ludzka",
        "source_author": "Gabriela Zapolska",
        "source_language": "Polish",
        "source_status": "Project Gutenberg page states public domain in the USA; raw text stays ignored",
        "source_sha256": RAW_SHA,
        "polish_clean_sha256": POLISH_CLEAN_SHA,
        "polish_clean_characters": len(clean_polish),
        "preprocessing": "Slice raw from first =Żabusia.= to last UWAGI DO WYDANIA ELEKTRONICZNEGO; ł/Ł→l/L; Unicode NFKD strip combining marks; clean_plaintext; 128-char disjoint source blocks",
        "train_val_source_split_char": split,
        "train_val_gap_chars": SEQ_LEN,
        "family_counts": {name: {f: sum(row["filler_family"] == f for row in group) for f in FAMILIES} for name, group in rows.items()},
        "counts": {name: len(group) for name, group in rows.items()},
        "seeds": {"train": TRAIN_SEED, "validation": VAL_SEED, "polish_holdout": POLISH_SEED, "iid_control": IID_SEED},
        "filler_rate": PRIMARY_FILLER_RATE,
        "derived_sha256": derived,
        "generator_sha256": digest(root / "src/voynich/latent_recovery.py"),
        "builder_sha256": digest(Path(__file__)),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = prepare(args.root)
    print(json.dumps({"experiment": result["experiment"], "counts": result["counts"], "derived_sha256": result["derived_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
