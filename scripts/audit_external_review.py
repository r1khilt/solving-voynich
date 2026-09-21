"""Reproduce descriptive audit checks using training/validation data only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from voynich.latent_recovery import recon_accuracy, recon_edit_similarity, rank_bucket_sequence
from voynich.runtime import PageWindows


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(data):
    result = {"scope": "descriptive software/data checks; no fitting or manuscript test access",
              "splits": {}, "input_sha256": {}, "source_sha256": {}}
    for split in ("train", "validation"):
        pages = PageWindows(data, split, 2048)
        texts = [row["text"] for row in pages.pages]
        c_count = sum(text.count("c") for text in texts)
        ch_count = sum(text.count("ch") for text in texts)
        lengths = [window.length for window in pages.windows]
        result["splits"][split] = {
            "pages": len(texts), "normalized_text_codepoints": sum(map(len, texts)),
            "encoded_tokens_with_page_bos_eos": sum(map(len, pages.sequences.values())),
            "pages_below_2048_codepoints": sum(len(t) < 2048 for t in texts),
            "c_occurrences": c_count, "ch_occurrences": ch_count,
            "fraction_c_followed_by_h": ch_count / c_count if c_count else None,
            "fixed_windows_256": len(PageWindows(data, split, 256).windows),
            "fixed_windows_2048": len(lengths),
            "padded_slot_fraction_2048": 1 - sum(lengths)/(2048*len(lengths)),
            "blocks_with_50_to_75_percent_padding": sum(.50 <= 1-n/2048 <= .75 for n in lengths),
        }
        result["input_sha256"][f"{split}.jsonl"] = digest(data/f"{split}.jsonl")
    result["input_sha256"]["tokenizer.json"] = digest(data/"tokenizer.json")
    text = "abcdefghij"
    mask = np.ones(len(text), dtype=int)
    mask[2] = 0
    result["one_deletion_fixture"] = {
        "target": text, "selected": "abdefghij",
        "positional_score": recon_accuracy(text, text, mask),
        "aligned_similarity": recon_edit_similarity(text, text, mask),
        "exact_recovery": False,
    }
    text, kept = "aaaabbbc", [3, 4, 5, 6, 7]
    encoded = rank_bucket_sequence(text)
    result["rank_remapping_fixture"] = {
        "original_text": text, "selected_text": "abbbc",
        "original_encoding_then_deletion": [encoded[i] for i in kept],
        "legacy_rebuilt_encoding_after_deletion": rank_bucket_sequence("abbbc"),
    }
    for path in (Path(__file__), ROOT/'src/voynich/runtime.py', ROOT/'src/voynich/tokenizer.py',
                 ROOT/'src/voynich/latent_recovery.py'):
        result["source_sha256"][str(path.relative_to(ROOT))] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
