"""Streaming, split-auditable two-table episodes for TEACH-0001.

The split hash is over *unordered table symbol sets*. Reversing assignments
therefore cannot move a factorial counterpart into a different partition.
"""

from dataclasses import dataclass
import hashlib
import random


PAD, BOS, FIRST, SECOND, COMPOSE, DIRECT, COPY, ANSWER = range(8)
NAMES = tuple(range(8, 16))
KEYS = tuple(range(16, 24))
OBJECTS = tuple(range(24, 32))
VOCAB_SIZE = 32
SEQUENCE_LENGTH = 14


def _bucket(kind: str, left: tuple[int, ...], right: tuple[int, ...]) -> int:
    material = bytes([ord(kind), *sorted(left), 255, *sorted(right)])
    return int.from_bytes(hashlib.blake2s(material, digest_size=8).digest(), "big") % 10


def _partition(bucket: int, requested: str) -> bool:
    if requested == "train":
        return bucket < 8
    if requested == "holdout":
        return bucket >= 8
    raise ValueError(f"Unknown partition: {requested}")


def table_partitions(names: tuple[int, int], keys: tuple[int, int],
                     objects: tuple[int, int]) -> tuple[str, str]:
    f = "train" if _bucket("F", names, keys) < 8 else "holdout"
    g = "train" if _bucket("G", keys, objects) < 8 else "holdout"
    return f, g


@dataclass(frozen=True)
class Episode:
    tokens: tuple[int, ...]
    answer: int
    task: str
    f_partition: str
    g_partition: str


def family_signatures(episode: Episode) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Canonical F/G family keys, independent of assignment and row order."""
    tokens = episode.tokens
    return ((tuple(sorted((tokens[2], tokens[4]))),
             tuple(sorted((tokens[3], tokens[5])))),
            (tuple(sorted((tokens[7], tokens[9]))),
             tuple(sorted((tokens[8], tokens[10])))))


def make_episode(names: tuple[int, int], assigned_keys: tuple[int, int],
                 table_keys: tuple[int, int], objects: tuple[int, int],
                 task: str, query: int) -> Episode:
    if len(set(names)) != 2 or any(x not in NAMES for x in names):
        raise ValueError("Names must be distinct name tokens")
    if set(assigned_keys) != set(table_keys) or len(set(table_keys)) != 2:
        raise ValueError("Both tables must use the same two distinct key tokens")
    if any(x not in KEYS for x in table_keys):
        raise ValueError("Keys must be key tokens")
    if len(set(objects)) != 2 or any(x not in OBJECTS for x in objects):
        raise ValueError("Objects must be distinct object tokens")
    if task == "composed":
        if query not in names:
            raise ValueError("Composed query must name one first-table row")
        key = assigned_keys[names.index(query)]
        answer = objects[table_keys.index(key)]
        marker = COMPOSE
    elif task == "direct":
        if query not in table_keys:
            raise ValueError("Direct query must name one second-table key")
        answer = objects[table_keys.index(query)]
        marker = DIRECT
    elif task == "copy":
        if query not in OBJECTS:
            raise ValueError("Copy query must be an object token")
        answer = query
        marker = COPY
    else:
        raise ValueError(f"Unknown task: {task}")
    tokens = (
        BOS, FIRST, names[0], assigned_keys[0], names[1], assigned_keys[1],
        SECOND, table_keys[0], objects[0], table_keys[1], objects[1],
        marker, query, ANSWER,
    )
    f_part, g_part = table_partitions(names, table_keys, objects)
    return Episode(tokens, answer, task, f_part, g_part)


def symbolic_oracle(tokens: tuple[int, ...]) -> int:
    """Parse the frozen token grammar independently of Episode.answer."""
    if len(tokens) != SEQUENCE_LENGTH or tokens[0:2] != (BOS, FIRST):
        raise ValueError("Malformed episode prefix")
    if tokens[6] != SECOND or tokens[13] != ANSWER:
        raise ValueError("Malformed episode markers")
    names = (tokens[2], tokens[4])
    assigned = (tokens[3], tokens[5])
    keys = (tokens[7], tokens[9])
    objects = (tokens[8], tokens[10])
    marker, query = tokens[11], tokens[12]
    if len(set(names)) != 2 or any(token not in NAMES for token in names):
        raise ValueError("Malformed first-table names")
    if len(set(keys)) != 2 or set(assigned) != set(keys) or any(token not in KEYS for token in keys):
        raise ValueError("Malformed first/second-table keys")
    if len(set(objects)) != 2 or any(token not in OBJECTS for token in objects):
        raise ValueError("Malformed second-table objects")
    first = dict(zip(names, assigned, strict=True))
    second = dict(zip(keys, objects, strict=True))
    if marker == COMPOSE:
        return second[first[query]]
    if marker == DIRECT:
        return second[query]
    if marker == COPY and query in OBJECTS:
        return query
    raise ValueError("Unknown query marker or invalid query")


def sample_episode(rng: random.Random, *, f_partition: str, g_partition: str,
                   task: str) -> Episode:
    for _ in range(10_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        objects = tuple(rng.sample(OBJECTS, 2))
        if table_partitions(names, keys, objects) != (f_partition, g_partition):
            continue
        assigned = keys if rng.randrange(2) == 0 else keys[::-1]
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        if task == "composed":
            query = rng.choice(names)
        elif task == "direct":
            query = rng.choice(keys)
        elif task == "copy":
            query = rng.choice(OBJECTS)
        else:
            raise ValueError(f"Unknown task: {task}")
        return make_episode(names, assigned, table_keys, objects, task, query)
    raise RuntimeError("Could not sample an episode from requested partition")


def sample_factorial(rng: random.Random, *, partition: str = "holdout") -> tuple[Episode, ...]:
    """Return F0/G0, F1/G0, F0/G1, F1/G1 with four distinct answers."""
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        out = tuple(rng.sample(OBJECTS, 4))
        if not _partition(_bucket("F", names, keys), partition):
            continue
        if not all(_partition(_bucket("G", keys, pair), partition)
                   for pair in (out[:2], out[2:])):
            continue
        query = rng.choice(names)
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        f0 = keys if rng.randrange(2) == 0 else keys[::-1]
        f1 = f0[::-1]
        return tuple(
            make_episode(names, assigned, table_keys, output, "composed", query)
            for output in (out[:2], out[2:])
            for assigned in (f0, f1)
        )
    raise RuntimeError("Could not sample a factorial bundle")


def training_batch(seed: int, batch_size: int, *, null_labels: bool = False
                   ) -> tuple[list[Episode], list[int]]:
    """Stateless per-step stream: no finite training table is ever materialized."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = random.Random(seed)
    label_rng = random.Random(seed ^ 0x5EEDBEEF)
    episodes = []
    labels = []
    for _ in range(batch_size):
        roll = rng.random()
        task = "composed" if roll < 0.70 else "direct" if roll < 0.90 else "copy"
        episode = sample_episode(rng, f_partition="train", g_partition="train", task=task)
        episodes.append(episode)
        label = label_rng.choice(OBJECTS) if null_labels and task == "composed" else episode.answer
        labels.append(label)
    return episodes, labels


def evaluation_suite(seed: int, size: int = 256) -> dict[str, list[Episode]]:
    if size <= 0:
        raise ValueError("size must be positive")
    rng = random.Random(seed)
    suite = {}
    for f_part, g_part in (("train", "train"), ("holdout", "train"),
                           ("train", "holdout"), ("holdout", "holdout")):
        label = f"composed_{f_part}_{g_part}"
        suite[label] = [sample_episode(rng, f_partition=f_part, g_partition=g_part,
                                       task="composed") for _ in range(size)]
    for task in ("direct", "copy"):
        suite[task] = [sample_episode(rng, f_partition="holdout", g_partition="holdout",
                                      task=task) for _ in range(size)]
    suite["factorial"] = [ep for _ in range(size // 2) for ep in sample_factorial(rng)]
    return suite
