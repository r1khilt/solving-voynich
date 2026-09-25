"""Independent source/channel/score replay for EXP-0041."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import unicodedata
import re

from scripts.audit_exp0039 import aggregate, gc_source, model_fit, score, scramble, zl_source


ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    "gc": "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f",
    "split": "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e",
    "zl_train": "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4",
    "zl_validation": "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3",
    "exp0039_result": "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329",
    "english_raw": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
    "finnish_raw": "23e57fdc123b738f49e31a68ceef2b74b77a83baa0359ffd820f682905283631",
    "italian_historical_raw": "4669dcc00ee61ceffe92d871e61ea430cec87b35cbab24f19a4c0b1c7da521b2",
    "spanish_raw": "534f41d59f7142163fa0964076ac6351845c006ea6433778be637fac6d5b04d7",
}
ANCHORS = {
    "english": (236, "alice was beginning to get very tired"),
    "finnish": (348, "mieleni minun tekevi"),
    "italian_historical": (350, "nel mezzo del cammin di nostra vita"),
    "spanish": (7189, "en un lugar de la mancha"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float) -> None:
    if not math.isclose(a, b, abs_tol=1e-10, rel_tol=0):
        raise AssertionError((a, b))


def shuffle_seed(view_index: int) -> int:
    if view_index == 4:
        return 410400
    return 410400 + 1000 * view_index


def chunks(words: list[tuple[str, ...]], skeleton: list[tuple]) -> list[tuple]:
    rows = []
    cursor = 0
    for leaf, locus, context, source in skeleton:
        stop = cursor + len(source)
        rows.append((leaf, locus, context, words[cursor:stop]))
        cursor = stop
    assert cursor == len(words)
    return rows


def load_sources(total: int) -> tuple[dict, dict]:
    streams, hashes = {}, {}
    for name, (start, phrase) in ANCHORS.items():
        path = ROOT / "data/raw/multilang_corpora" / f"{name}.raw.txt"
        assert digest(path) == HASHES[f"{name}_raw"]
        text = unicodedata.normalize("NFKD", path.read_text(encoding="utf-8"))
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        words = re.findall(r"[a-z]+", text.casefold())
        assert words[start:start + len(phrase.split())] == phrase.split()
        selected = words[start:start + total]
        assert len(selected) == total and all(word.isascii() and word.isalpha() for word in selected)
        hashes[name] = hashlib.sha256(" ".join(selected).encode("ascii")).hexdigest()
        streams[name] = [tuple(word) for word in selected]
    return streams, hashes


def channels(plain: list[tuple[str, ...]]) -> dict[str, list[tuple[str, ...]]]:
    output = {}
    letters = list("abcdefghijklmnopqrstuvwxyz")
    shuffled = letters.copy()
    random.Random(410100).shuffle(shuffled)
    table = dict(zip(letters, shuffled))
    output["english_bijection"] = [tuple(table[ch] for ch in word) for word in plain]
    inverse = dict(zip(shuffled, letters))
    assert [tuple(inverse[ch] for ch in word) for word in output["english_bijection"]] == plain
    for seed in range(410200, 410208):
        stream = []
        for index, word in enumerate(plain):
            rng = random.Random(index + seed * 1_000_003)
            stream.append(tuple(ch + str(rng.choice((0, 1))) for ch in word))
        assert [tuple(symbol[:1] for symbol in word) for word in stream] == plain
        output[f"english_homophonic_{seed}"] = stream
    for seed in range(410300, 410308):
        stream = []
        for index, word in enumerate(plain):
            permutation = list(range(len(word)))
            random.Random(index + seed * 1_000_003).shuffle(permutation)
            cipher = tuple(word[j] for j in permutation)
            inverse_word = [None] * len(word)
            for j, orig in enumerate(permutation):
                inverse_word[orig] = cipher[j]
            assert tuple(inverse_word) == word
            stream.append(cipher)
        output[f"english_transposition_{seed}"] = stream
    return output


def replay(name: str, words: list[tuple[str, ...]], fit_template: list[tuple],
           hold_template: list[tuple], ordinal: int, saved: dict) -> None:
    n_fit = sum(len(row[3]) for row in fit_template)
    source = chunks(words[:n_fit], fit_template)
    held = chunks(words[n_fit:], hold_template)
    model = model_fit(source)
    actual = score(held, model)
    last = aggregate(actual, 0)
    null = [aggregate(score(scramble(held, shuffle_seed(ordinal) + rep), model), 0)
            for rep in range(200)]
    assert len(null) == len(saved["null_scores_bits_per_pair"])
    for observed, expected in zip(null, saved["null_scores_bits_per_pair"]):
        close(observed, expected)
    null_mean = sum(null) / len(null)
    checks = {"last_gain_bits_per_pair": last,
              "last_minus_shuffle_bits_per_pair": last - null_mean,
              "last_minus_first_bits_per_pair": last - aggregate(actual, 1),
              "identity_over_last_bits_per_pair": aggregate(actual, 4),
              "null_mean_bits_per_pair": null_mean}
    for key, value in checks.items():
        close(value, saved[key])
    assert len(model[0]) + 1 == saved["alphabet_size_train_plus_unk"]
    assert len(source) == saved["train_groups"] and len(held) == saved["validation_groups"]
    assert sum(len(row[3]) - 1 for row in source) == saved["train_pairs"]
    assert sum(len(row[3]) - 1 for row in held) == saved["validation_pairs"]
    train_words = words[:n_fit]
    held_words = words[n_fit:]
    vocab = set(train_words)
    counts = Counter(held_words)
    lexical = saved["lexical"]
    assert len(vocab) == lexical["train_types"] and len(counts) == lexical["validation_types"]
    close(sum(word not in vocab for word in held_words) / len(held_words),
          lexical["validation_novel_token_share"])
    close(sum(n == 1 for n in counts.values()) / len(counts),
          lexical["validation_singleton_type_share"])
    print(f"{name}: independent source/channel/score replay pass")


def main() -> None:
    started = time.monotonic()
    paths = {
        "gc": ROOT / "data/raw/v101/GC2a-n.txt",
        "split": ROOT / "data/manifests/zl3b_split.json",
        "zl_train": ROOT / "data/processed/zl3b/train.jsonl",
        "zl_validation": ROOT / "data/processed/zl3b/validation.jsonl",
        "exp0039_result": ROOT / "results/EXP-0039/results.json",
    }
    for name, path in paths.items():
        assert digest(path) == HASHES[name]
    saved = json.loads((ROOT / "results/EXP-0041/results.json").read_text())
    assert saved["input_sha256"] == HASHES
    assert saved["source_sha256"] == digest(ROOT / "src/voynich/readable_edge_calibration.py")
    reference = json.loads(paths["exp0039_result"].read_text())["views"]["zl_basic_matched"]
    close(saved["manuscript_reference"]["last_gain_bits_per_pair"], reference["real_mean"]["last"])
    close(saved["manuscript_reference"]["last_minus_shuffle_bits_per_pair"],
          reference["contrasts"]["last_minus_null"])
    split = json.loads(paths["split"].read_text())["leaf_assignments"]
    _, _, matched = gc_source(paths["gc"], split)
    fit_template, _ = zl_source(paths["zl_train"], matched)
    hold_template, _ = zl_source(paths["zl_validation"], matched)
    n_fit = sum(len(row[3]) for row in fit_template)
    n_hold = sum(len(row[3]) for row in hold_template)
    assert (n_fit, n_hold) == (19071, 2685)
    plain, source_hashes = load_sources(n_fit + n_hold)
    assert source_hashes == saved["source_window_sha256"]
    all_streams = {**plain, **channels(plain["english"])}
    assert list(all_streams) == list(saved["views"])
    assert len(all_streams) == 21
    for ordinal, (name, stream) in enumerate(all_streams.items()):
        replay(name, stream, fit_template, hold_template, ordinal, saved["views"][name])
    for key in ("last_gain_bits_per_pair", "last_minus_shuffle_bits_per_pair",
                "last_minus_first_bits_per_pair", "identity_over_last_bits_per_pair"):
        close(saved["views"]["english"][key], saved["views"]["english_bijection"][key])
    if time.monotonic() - started > 180:
        raise TimeoutError("EXP-0041 audit wall cap exceeded")
    audit = {"experiment": "EXP-0041", "audit": "pass", "views": len(all_streams),
             "total_shuffles": len(all_streams) * 200,
             "wall_seconds": time.monotonic() - started,
             "result_sha256": digest(ROOT / "results/EXP-0041/results.json")}
    target = ROOT / "results/EXP-0041/audit.json"
    target.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
