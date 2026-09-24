"""Fresh outcome-blind groups for cross-distractor rank-shift qualification."""

from dataclasses import asdict
import json
import random

from .teacher15_tasks import TransferGroup, _sample_group


EXPERIMENT = "TEACH-0017"
NAMESPACE = "TEACH-0017-cross-distractor-v1"
SEEDS = {"discovery": 87111, "confirmation": 87121}


def generate_split(split: str, count: int = 128) -> list[TransferGroup]:
    if split not in SEEDS or count <= 0:
        raise ValueError("Invalid TEACH-0017 split/count")
    rng = random.Random(SEEDS[split])
    seen: set[str] = set()
    return [_sample_group(rng, split, seen) for _ in range(count)]


def split_manifest(groups: list[TransferGroup], split: str) -> dict:
    if not groups or split not in SEEDS or any(
            group.split != split for group in groups):
        raise ValueError("Invalid TEACH-0017 group manifest")
    return json.loads(json.dumps({
        "experiment": EXPERIMENT, "namespace": NAMESPACE,
        "generator": "TEACH-0015-key-transfer-v1",
        "split": split, "seed": SEEDS[split],
        "group_count": len(groups),
        "groups": [asdict(group) for group in groups],
    }))
