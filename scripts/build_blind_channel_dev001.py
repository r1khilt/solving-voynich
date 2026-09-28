"""Source-only calibration and exposed developmental data generation.

Never imported by the ciphertext-only runner. Public deterministic seeds make
this separation procedural, not cryptographic secrecy. No final authors.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from voynich.finite_state_channel import Channel, Emission, SourceModel
from voynich.finite_state_channel_fit import CodingContext


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-001"
LENGTH = 224
RHO = 1 / 225
TAUS = (.25, 1., 4., 16., 64., 256.)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, data: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def segments(payload: dict) -> list[str]:
    starts = [0, *payload["body_boundaries"][:-1]]
    return [payload["text"][a:b] for a, b in zip(starts, payload["body_boundaries"], strict=True)]


def estimate_source(records: list[str], alphabet: tuple[str, ...], order: int, tau: float) -> SourceModel:
    if not records or not all(records) or any(set(row) - set(alphabet) for row in records):
        raise ValueError("Need nonempty in-alphabet source records")
    counts = Counter("".join(records))
    total = sum(counts.values()) + .5 * len(alphabet)
    base = {char: (counts[char] + .5) / total for char in alphabet}
    probabilities = {"": base}
    if order == 1:
        pairs = Counter((left, right) for record in records for left, right in zip(record, record[1:]))
        for left in alphabet:
            total = sum(pairs[left, right] for right in alphabet)
            probabilities[left] = {right: (pairs[left, right] + tau * base[right]) / (total + tau)
                                   for right in alphabet}
    return SourceModel(alphabet, order, probabilities)


def source_bits(source: SourceModel, records: list[str]) -> float:
    score = 0.
    for record in records:
        context = ""
        for char in record:
            score -= math.log2(source.probabilities[context][char])
            context = char if source.order else ""
    return score


def encode_records(records: list[str], channel: Channel) -> list[str]:
    if len(channel.states) != 1 or any(len(row) != 1 for row in channel.rows.values()):
        raise ValueError("Development generator expects one-state deterministic channels")
    state = channel.states[0]
    return ["".join(channel.rows[state, char][0].glyphs for char in record) for record in records]


def make_channel(alphabet: tuple[str, ...], family: str, seed: int) -> Channel:
    rng = random.Random(seed)
    if family == "A":
        glyphs = tuple(chr(65 + i) for i in range(len(alphabet)))
        units = list(glyphs)
    elif family == "B":
        glyphs = tuple("ABCDEF")
        pairs = ["".join(pair) for pair in itertools.product(glyphs, repeat=2)]
        units = [*glyphs, *rng.sample(pairs, len(alphabet) - len(glyphs))]
    else:
        raise ValueError("Only exposed A/B development families permitted")
    rng.shuffle(units)
    return Channel(("s0",), glyphs, {"s0": 1.},
                   {("s0", char): (Emission("s0", unit, 1.),)
                    for char, unit in zip(alphabet, units, strict=True)}, RHO, 2)


def shuffled_records(records: list[str], rng: random.Random) -> list[str]:
    result = []
    for record in records:
        chars = list(record)
        rng.shuffle(chars)
        result.append("".join(chars))
    return result


def main() -> None:
    corpus_path = ROOT / "data/manifests/blind_channel_development_corpora.json"
    corpus = json.loads(corpus_path.read_text())
    payloads = {}
    for name, spec in corpus["sources"].items():
        raw = (ROOT / spec["derived_path"]).read_bytes()
        if digest(raw) != spec["derived_sha256"]:
            raise ValueError("Derived corpus hash changed")
        payloads[name] = json.loads(raw)
    alphabet = tuple(corpus["alphabet"])
    prior, validation = segments(payloads["caesar"]), segments(payloads["virgil"])
    candidates = []
    for order, tau in [(0, 1.), *((1, tau) for tau in TAUS)]:
        model = estimate_source(prior, alphabet, order, tau)
        candidates.append({"order": order, "tau": tau,
                           "validation_bits_per_character": source_bits(model, validation) / 50_000})
    chosen = min(candidates, key=lambda row: row["validation_bits_per_character"])
    source = estimate_source(prior + validation, alphabet, chosen["order"], chosen["tau"])
    source_artifact = save(ROOT / f"results/{EXPERIMENT}/source_selection.json", {
        "experiment": EXPERIMENT, "status": "source_only_selection", "candidates": candidates,
        "selected": chosen, "source_model": source.to_dict(),
        "corpus_manifest_sha256": digest(corpus_path.read_bytes()), "fixed_stop_probability": RHO,
        "fitting_authors": ["caesar", "virgil"], "characters_per_author": 50_000})
    cases = {}
    for index, (family, key_index) in enumerate(itertools.product(("A", "B"), (1, 2))):
        case_id = f"{family}-key{key_index}"
        seed = 47001 + index * 101
        channel = make_channel(alphabet, family, seed)
        base = index * 4000
        offsets = {"fit": [base + n * 256 for n in range(4)],
                   "transfer": [base + 2048 + n * 256 for n in range(2)]}
        plain = {split: [payloads["cicero"]["text"][start:start + LENGTH] for start in starts]
                 for split, starts in offsets.items()}
        assert all(len(record) == LENGTH for records in plain.values() for record in records)
        encoded = {split: encode_records(records, channel) for split, records in plain.items()}
        context = CodingContext(alphabet, channel.glyph_alphabet, 32, 2, 2, 3,
                                stop_probability=RHO)
        for is_null in (False, True):
            name = case_id + ("-shuffle" if is_null else "")
            rng = random.Random(seed + 1_000_000)
            records = {split: shuffled_records(rows, rng) if is_null else rows
                       for split, rows in encoded.items()}
            artifacts = {}
            for split, rows in records.items():
                artifacts[split] = save(ROOT / f"data/processed/{EXPERIMENT}/{name}/{split}.json", {
                    "case_id": name, "split": split, "records": rows, "context": asdict(context)})
            answer = {"case_id": name, "positive": not is_null,
                      "gold_channel": None if is_null else channel.to_dict(),
                      "plaintext": None if is_null else plain, "source_offsets": offsets,
                      "key_seed": seed, "null_seed": seed + 1_000_000 if is_null else None}
            artifacts["answer"] = save(ROOT / f"data/processed/{EXPERIMENT}/{name}/answer.json", answer)
            cases[name] = {"family": family, "positive": not is_null, "artifacts": artifacts,
                           "search_seed": 48101 + index * 101 + int(is_null),
                           "glyph_alphabet": list(channel.glyph_alphabet)}
    save(ROOT / "data/manifests/blind_channel_dev001.json", {
        "experiment": EXPERIMENT, "status": "exposed_development_not_confirmation",
        "source": source_artifact, "cases": cases, "raw_source_authors": ["caesar", "virgil", "cicero"],
        "source_code_sha256": digest(Path(__file__).read_bytes()),
        "oracle_assistance": ["candidate Latin source/alphabet", "complete glyph inventory", "record boundaries",
                              "max states2, emission length2, alternatives3", "mean source length224"],
        "answer_isolation": "procedural only; deterministic generator seeds are public",
        "final_authors_accessed": False})
    print(json.dumps({"source_selection": chosen, "generated_cases": list(cases),
                      "fit_plaintext_characters_per_positive": 896, "transfer_per_positive": 448}))


if __name__ == "__main__":
    main()
