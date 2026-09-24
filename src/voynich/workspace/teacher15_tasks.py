"""Prospective fresh three-recipient key-transfer groups for TEACH-0015.

No model or TEACH-0014 final result is read here. Every signal-stage family
is in the frozen TEACH-0014 confirm partition, disjoint from train families.
"""

from dataclasses import asdict, dataclass
import json
import random

from .teacher14_tasks import (
    SYMBOL_START, VOCAB_SIZE, Episode, RenderSpec, _paths, digest,
    family_partition, make_episode,
)


NAMESPACE = "TEACH-0015-key-transfer-v1"
DISCOVERY_SEED = 85111
CONFIRMATION_SEED = 85121
MARKED = RenderSpec(0.0, 2, ("prefix", "infix", "suffix"))
MARKER_FREE = RenderSpec(1.0, 2, ("prefix", "infix", "suffix"))


@dataclass(frozen=True)
class Cell:
    f: int
    g: int
    distractor: int
    marked: bool
    order: int
    task: str
    episode: Episode


@dataclass(frozen=True)
class TransferGroup:
    group_id: str
    split: str
    key0: int
    key1: int
    recipient_outputs: tuple[tuple[int, int], ...]
    donor_answer: int
    cells: tuple[Cell, ...]


def _stage(rows: tuple[tuple[int, int], ...], index: int) -> str:
    return family_partition(f"stage-{index}", rows)


def _surface_seed(group_id: str, distractor: int, marked: bool,
                  order: int) -> int:
    return int(digest([group_id, distractor, marked, order])[:16], 16)


def _sample_group(rng: random.Random, wanted_split: str,
                  seen: set[str]) -> TransferGroup:
    if wanted_split not in ("discovery", "confirmation"):
        raise ValueError("Unknown TEACH-0015 split")
    for _ in range(1_000_000):
        f0, d0 = _paths(rng, 2, 4, "none")
        f1_list = list(f0)
        f1_list[0] = (f0[1][0], *f0[0][1:])
        f1_list[1] = (f0[0][0], *f0[1][1:])
        f1 = tuple(f1_list)
        first0 = tuple((path[0], path[1]) for path in f0)
        first1 = tuple((path[0], path[1]) for path in f1)
        if _stage(first0, 0) != "confirm" or _stage(first1, 0) != "confirm":
            continue
        g_tables = [tuple(path[2] for path in f0)]
        second0 = tuple((path[1], path[2]) for path in f0)
        if _stage(second0, 1) != "confirm":
            continue
        used = {symbol for path in f0 + d0 for symbol in path}
        for _table in range(2):
            for _attempt in range(1000):
                available = tuple(symbol for symbol in range(
                    SYMBOL_START, VOCAB_SIZE) if symbol not in used)
                values = tuple(rng.sample(available, 4))
                second = tuple((f0[index][1], values[index])
                               for index in range(4))
                if _stage(second, 1) == "confirm":
                    g_tables.append(values)
                    used.update(values)
                    break
            else:
                raise RuntimeError("Cannot sample confirm G remapping")
        available = tuple(symbol for symbol in range(
            SYMBOL_START, VOCAB_SIZE) if symbol not in used)
        values = rng.sample(available, 12)
        d1 = tuple(tuple(values[index:index + 3])
                   for index in range(0, len(values), 3))
        specification = {"f0": first0, "f1": first1,
                         "g": g_tables, "d": (d0, d1)}
        group_id = digest(specification)
        split = ("discovery" if int(digest([NAMESPACE, group_id])[:16], 16) % 2 == 0
                 else "confirmation")
        if split != wanted_split or group_id in seen:
            continue
        seen.add(group_id)
        query = f0[0][0]
        key0, key1 = f0[0][1], f1[1][1]
        outputs = tuple((table[0], table[1]) for table in g_tables)
        if len(set(value for pair in outputs for value in pair)) != 6:
            raise RuntimeError("Recipient targets are not six-way distinct")
        cells = []
        for d_index, distractors in enumerate((d0, d1)):
            for marked, spec in ((True, MARKED), (False, MARKER_FREE)):
                for order in (0, 1):
                    seed = _surface_seed(group_id, d_index, marked, order)
                    for f_index, base in enumerate((f0, f1)):
                        for g_index, table in enumerate(g_tables):
                            paths = tuple((path[0], path[1], table[index])
                                          for index, path in enumerate(base))
                            episode = make_episode(
                                paths, distractors, task="composed",
                                query=query, rng=random.Random(seed), spec=spec)
                            if episode.stage_partitions != ("confirm", "confirm"):
                                raise RuntimeError("Counterfactual family partition drift")
                            cells.append(Cell(f_index, g_index, d_index,
                                              marked, order, "composed", episode))
        for f_index, base in enumerate((f0, f1)):
            for g_index, table in enumerate(g_tables):
                paths = tuple((path[0], path[1], table[index])
                              for index, path in enumerate(base))
                for task, task_query in (
                        ("first_hop", query),
                        ("direct", key0 if f_index == 0 else key1),
                        ("copy", query)):
                    episode = make_episode(
                        paths, d0, task=task, query=task_query,
                        rng=random.Random(_surface_seed(group_id, 0, True, 0)),
                        spec=MARKED)
                    cells.append(Cell(f_index, g_index, 0, True, 0,
                                      task, episode))
        return TransferGroup(group_id, split, key0, key1,
                             outputs, outputs[0][1], tuple(cells))
    raise RuntimeError("Could not sample TEACH-0015 group")


def generate_split(split: str, count: int = 128) -> list[TransferGroup]:
    if split not in ("discovery", "confirmation") or count <= 0:
        raise ValueError("Invalid TEACH-0015 split/count")
    rng = random.Random(DISCOVERY_SEED if split == "discovery"
                        else CONFIRMATION_SEED)
    seen: set[str] = set()
    return [_sample_group(rng, split, seen) for _ in range(count)]


def split_manifest(groups: list[TransferGroup], split: str) -> dict:
    if not groups or any(group.split != split for group in groups):
        raise ValueError("Wrong or empty TEACH-0015 split")
    return json.loads(json.dumps({
        "experiment": "TEACH-0015", "namespace": NAMESPACE,
        "split": split,
        "seed": DISCOVERY_SEED if split == "discovery"
        else CONFIRMATION_SEED,
        "group_count": len(groups),
        "groups": [asdict(group) for group in groups]}))
