"""Read-only manuscript adapters preserving observations and frozen split boundaries."""

from pathlib import Path
import hashlib
import json

from ..runtime import corpus_identity
from ..tokenizer import ALTERNATIVE, UNREADABLE, UNKNOWN_SPAN, UNCERTAIN_SPACE, RARE_BASE
from .schema import Observation, save_json


def export_manuscript(data_dir, output_dir, *, split="validation", window=128, preparation_manifest=None):
    """Export literal transcription units. This neither scores nor translates a manuscript.

    Vocabulary is a public transcription inventory from the train-fitted tokenizer.
    Unknown codepoints are refused, never silently mapped or removed. Editorial and
    uncertain tokens remain marked in the manifest; no word boundary is assumed.
    Final-test export is deliberately unavailable in this implementation.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Only train/validation are available; the manuscript final holdout is sealed")
    if type(window) is not int or window < 1:
        raise ValueError("Invalid observation window")
    directory = Path(data_dir)
    identity = corpus_identity(directory, preparation_manifest)
    pages = [
        json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines() if line.strip()
    ]
    training = [
        json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line.strip()
    ]
    if any(p["split"] != "train" for p in training) or any(p["split"] != split for p in pages):
        raise ValueError("Split labels do not match requested corpus")
    symbols = sorted({c for p in training for c in p["text"]})
    unknown = sorted({c for p in pages for c in p["text"]} - set(symbols))
    if unknown:
        raise ValueError(
            f"Unseen transcription units require an explicit vocabulary revision: {[ord(c) for c in unknown]}"
        )
    if len(symbols) < 6:
        raise ValueError("Insufficient transcription alphabet")
    token_to_id = {c: i + 2 for i, c in enumerate(symbols)}
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new")
    output.mkdir(parents=True, exist_ok=True)
    observations = []
    page_ids = set()
    for page in pages:
        page_id = page["page_id"]
        if page_id in page_ids:
            raise ValueError("Duplicate manuscript page")
        page_ids.add(page_id)
        text = page["text"]
        for start in range(0, len(text), window):
            span = text[start : start + window]
            if not span:
                continue
            obs = Observation(
                f"{page_id}:{start}",
                tuple(token_to_id[c] for c in span),
                len(symbols),
                source_group=f"manuscript-leaf:{page.get('leaf_id', page_id)}",
                split=split,
            )
            observations.append(obs)
    path = output / "observations.jsonl"
    path.write_text("".join(json.dumps(o.to_dict(), ensure_ascii=False) + "\n" for o in observations))
    manifest = {
        "schema_version": 1,
        "corpus_sha256": identity,
        "split": split,
        "window": window,
        "observations": len(observations),
        "pages": len(page_ids),
        "alphabet_size": len(symbols),
        "symbol_inventory": symbols,
        "uncertain_codepoints": [
            ord(c) for c in symbols if c in {ALTERNATIVE, UNREADABLE, UNKNOWN_SPAN, UNCERTAIN_SPACE}
        ],
        "rare_glyph_codepoints": [ord(c) for c in symbols if RARE_BASE + 128 <= ord(c) <= RARE_BASE + 255],
        "observations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "interpretation": "Literal transcription inventory, including layout/uncertainty units; not linguistic units",
        "claim": "Read-only format export; no scientific scoring or decipherment",
    }
    save_json(output / "manifest.json", manifest)
    return manifest
