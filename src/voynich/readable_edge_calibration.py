"""EXP-0041: known-readable controls for the EXP-0039 boundary assay."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import time
import unicodedata

from .boundary_cross_transcription import (
    GC_SHA,
    SPLIT_SHA,
    ZL_TRAIN_SHA,
    ZL_VALID_SHA,
    average,
    fit,
    load_gc,
    load_zl,
    score,
    sha,
    shuffle,
)


EXPERIMENT = "EXP-0041"
EXP39_SHA = "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329"
N_SHUFFLES = 200
SOURCES = {
    "english": ("01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
                "alice was beginning to get very tired", 236),
    "finnish": ("23e57fdc123b738f49e31a68ceef2b74b77a83baa0359ffd820f682905283631",
                "mieleni minun tekevi", 348),
    "italian_historical": ("4669dcc00ee61ceffe92d871e61ea430cec87b35cbab24f19a4c0b1c7da521b2",
                           "nel mezzo del cammin di nostra vita", 350),
    "spanish": ("534f41d59f7142163fa0964076ac6351845c006ea6433778be637fac6d5b04d7",
                "en un lugar de la mancha", 7189),
}


def shuffle_seed(view_index: int) -> int:
    # The bijective positive control must replay the *same* permutations as English.
    return 410400 if view_index == 4 else 410400 + 1000 * view_index


def fold_words(raw: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", raw)
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z]+", folded.lower())


def source_window(root: Path, language: str, size: int) -> tuple[list[tuple[str, ...]], str]:
    expected, phrase, position = SOURCES[language]
    path = root / "data/raw/multilang_corpora" / f"{language}.raw.txt"
    if sha(path) != expected:
        raise ValueError(f"{language} source checksum drift")
    words = fold_words(path.read_text(encoding="utf-8"))
    if words[position:position + len(phrase.split())] != phrase.split():
        raise ValueError(f"{language} anchor drift")
    selected = words[position:position + size]
    if len(selected) != size or any(re.fullmatch(r"[a-z]+", word) is None for word in selected):
        raise ValueError(f"{language} selected words invalid")
    digest = hashlib.sha256(" ".join(selected).encode("ascii")).hexdigest()
    return [tuple(word) for word in selected], digest


def templates(root: Path) -> tuple[list[dict], list[dict]]:
    paths = (root / "data/raw/v101/GC2a-n.txt", root / "data/manifests/zl3b_split.json",
             root / "data/processed/zl3b/train.jsonl",
             root / "data/processed/zl3b/validation.jsonl")
    if [sha(path) for path in paths] != [GC_SHA, SPLIT_SHA, ZL_TRAIN_SHA, ZL_VALID_SHA]:
        raise ValueError("EXP-0039 manuscript input drift")
    split = json.loads(paths[1].read_text())["leaf_assignments"]
    _, _, common_ids = load_gc(paths[0], split)
    train, _ = load_zl(paths[2], common_ids)
    valid, _ = load_zl(paths[3], common_ids)
    if sum(len(g["words"]) for g in train) != 19071 or sum(len(g["words"]) for g in valid) != 2685:
        raise ValueError("EXP-0039 skeleton drift")
    return train, valid


def bijection(words: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    shuffled = letters[:]
    random.Random(410100).shuffle(shuffled)
    forward = dict(zip(letters, shuffled))
    backward = dict(zip(shuffled, letters))
    transformed = [tuple(forward[ch] for ch in word) for word in words]
    if [tuple(backward[ch] for ch in word) for word in transformed] != words:
        raise AssertionError("Bijection did not decode")
    return transformed


def homophones(words: list[tuple[str, ...]], seed: int) -> list[tuple[str, ...]]:
    transformed = []
    for index, word in enumerate(words):
        generator = random.Random(seed * 1_000_003 + index)
        transformed.append(tuple(f"{ch}{generator.randrange(2)}" for ch in word))
    if [tuple(ch[0] for ch in word) for word in transformed] != words:
        raise AssertionError("Homophonic channel did not decode")
    return transformed


def transpositions(words: list[tuple[str, ...]], seed: int) -> list[tuple[str, ...]]:
    transformed = []
    decoded = []
    for index, word in enumerate(words):
        order = list(range(len(word)))
        random.Random(seed * 1_000_003 + index).shuffle(order)
        ciphertext = tuple(word[original] for original in order)
        original = [""] * len(word)
        for character, offset in zip(ciphertext, order):
            original[offset] = character
        transformed.append(ciphertext)
        decoded.append(tuple(original))
    if decoded != words:
        raise AssertionError("Transposition channel did not decode")
    return transformed


def wrap(words: list[tuple[str, ...]], skeleton: list[dict]) -> list[dict]:
    result = []
    cursor = 0
    for group in skeleton:
        end = cursor + len(group["words"])
        result.append({"leaf": group["leaf"], "locus": group["locus"],
                       "context": group["context"], "words": words[cursor:end]})
        cursor = end
    if cursor != len(words):
        raise ValueError("Source/template length mismatch")
    return result


def lexical(train: list[tuple[str, ...]], valid: list[tuple[str, ...]]) -> dict:
    known = set(train)
    counts = Counter(valid)
    return {"train_types": len(known), "validation_types": len(counts),
            "validation_novel_token_share": sum(word not in known for word in valid) / len(valid),
            "validation_singleton_type_share": sum(n == 1 for n in counts.values()) / len(counts)}


def evaluate_view(train_words: list[tuple[str, ...]], valid_words: list[tuple[str, ...]],
                  train_skeleton: list[dict], valid_skeleton: list[dict], seed: int) -> dict:
    train = wrap(train_words, train_skeleton)
    valid = wrap(valid_words, valid_skeleton)
    model = fit(train)
    actual = score(valid, model)
    real_last = average(actual, "last")
    shuffled = [average(score(shuffle(valid, seed + rep), model), "last")
                for rep in range(N_SHUFFLES)]
    null_mean = sum(shuffled) / N_SHUFFLES
    return {"train_groups": len(train), "validation_groups": len(valid),
            "train_pairs": sum(len(g["words"]) - 1 for g in train),
            "validation_pairs": sum(len(g["words"]) - 1 for g in valid),
            "alphabet_size_train_plus_unk": model["n_symbols"],
            "last_gain_bits_per_pair": real_last,
            "last_minus_shuffle_bits_per_pair": real_last - null_mean,
            "last_minus_first_bits_per_pair": real_last - average(actual, "first"),
            "identity_over_last_bits_per_pair": average(actual, "identity"),
            "null_mean_bits_per_pair": null_mean,
            "null_scores_bits_per_pair": shuffled,
            "lexical": lexical(train_words, valid_words)}


def run(root: Path) -> dict:
    started = time.monotonic()
    train_skeleton, valid_skeleton = templates(root)
    n_train = sum(len(g["words"]) for g in train_skeleton)
    n_valid = sum(len(g["words"]) for g in valid_skeleton)
    if sha(root / "results/EXP-0039/results.json") != EXP39_SHA:
        raise ValueError("EXP-0039 reference drift")
    reference = json.loads((root / "results/EXP-0039/results.json").read_text())[
        "views"]["zl_basic_matched"]
    windows = {}
    views = {}
    source_windows = {}
    for index, language in enumerate(SOURCES):
        words, digest = source_window(root, language, n_train + n_valid)
        source_windows[language] = digest
        windows[language] = words
        views[language] = evaluate_view(words[:n_train], words[n_train:],
                                        train_skeleton, valid_skeleton, shuffle_seed(index))
    english = windows["english"]
    encrypted = bijection(english)
    views["english_bijection"] = evaluate_view(encrypted[:n_train], encrypted[n_train:],
                                                train_skeleton, valid_skeleton, shuffle_seed(4))
    for index, seed in enumerate(range(410200, 410208), start=5):
        encrypted = homophones(english, seed)
        views[f"english_homophonic_{seed}"] = evaluate_view(
            encrypted[:n_train], encrypted[n_train:], train_skeleton, valid_skeleton,
            shuffle_seed(index))
    for index, seed in enumerate(range(410300, 410308), start=13):
        encrypted = transpositions(english, seed)
        views[f"english_transposition_{seed}"] = evaluate_view(
            encrypted[:n_train], encrypted[n_train:], train_skeleton, valid_skeleton,
            shuffle_seed(index))
    for key in ("last_gain_bits_per_pair", "last_minus_shuffle_bits_per_pair",
                "last_minus_first_bits_per_pair", "identity_over_last_bits_per_pair"):
        if abs(views["english"][key] - views["english_bijection"][key]) > 1e-10:
            raise AssertionError(f"Substitution invariance failed for {key}")
    if time.monotonic() - started > 180:
        raise TimeoutError("EXP-0041 runner CPU cap exceeded")
    return {"experiment": EXPERIMENT, "status": "complete",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "wall_seconds": time.monotonic() - started,
            "input_sha256": {"gc": GC_SHA, "split": SPLIT_SHA, "zl_train": ZL_TRAIN_SHA,
                             "zl_validation": ZL_VALID_SHA, "exp0039_result": EXP39_SHA,
                             **{f"{name}_raw": values[0] for name, values in SOURCES.items()}},
            "source_window_sha256": source_windows,
            "manuscript_reference": {
                "last_gain_bits_per_pair": reference["real_mean"]["last"],
                "last_minus_shuffle_bits_per_pair": reference["contrasts"]["last_minus_null"],
                "last_minus_first_bits_per_pair": reference["contrasts"]["last_minus_first"]},
            "invariance_control": "pass", "views": views,
            "limits": "One source window per work; artificial line/metadata wrapping; exposed manuscript reference; no language or channel identification."}


def main() -> None:
    root = Path.cwd()
    report = run(root)
    path = root / "results/EXP-0041/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"experiment": EXPERIMENT, "wall_seconds": report["wall_seconds"],
                      "views": {name: round(row["last_minus_shuffle_bits_per_pair"], 5)
                                for name, row in report["views"].items()}}, indent=2))


if __name__ == "__main__":
    main()
