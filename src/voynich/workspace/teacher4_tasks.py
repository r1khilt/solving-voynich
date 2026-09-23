"""Fresh-family two-table episodes for TEACH-0004 architecture contrasts."""

from dataclasses import dataclass
import hashlib
import random


PAD, BOS, FIRST, SECOND, COMPOSE, DIRECT, COPY, ANSWER, FIRST_HOP = range(9)
NAMES = tuple(range(9, 21))
KEYS = tuple(range(21, 33))
OBJECTS = tuple(range(33, 45))
VOCAB_SIZE = 45
SEQUENCE_LENGTH = 14
SPLIT_NAMESPACE = b"TEACH-0004-v1/"


@dataclass(frozen=True)
class Episode:
    tokens: tuple[int, ...]
    answer: int
    task: str
    f_partition: str
    g_partition: str


def _partition(kind: bytes, left: tuple[int, int], right: tuple[int, int]) -> str:
    value = SPLIT_NAMESPACE + kind + bytes([*sorted(left), 255, *sorted(right)])
    bucket = int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "holdout"


def table_partitions(names: tuple[int, int], keys: tuple[int, int],
                     outputs: tuple[int, int]) -> tuple[str, str]:
    return _partition(b"F/", names, keys), _partition(b"G/", keys, outputs)


def family_signatures(episode: Episode) -> tuple[tuple, tuple]:
    t = episode.tokens
    return ((tuple(sorted((t[2], t[4]))), tuple(sorted((t[3], t[5])))),
            (tuple(sorted((t[7], t[9]))), tuple(sorted((t[8], t[10])))))


def make_episode(names: tuple[int, int], assigned: tuple[int, int],
                 table_keys: tuple[int, int], outputs: tuple[int, int],
                 task: str, query: int) -> Episode:
    if len(set(names)) != 2 or any(n not in NAMES for n in names):
        raise ValueError("Distinct name tokens required")
    if len(set(table_keys)) != 2 or set(assigned) != set(table_keys) or any(
            k not in KEYS for k in table_keys):
        raise ValueError("Both tables must share two distinct key tokens")
    if len(set(outputs)) != 2 or any(o not in OBJECTS for o in outputs):
        raise ValueError("Distinct output tokens required")
    if task == "first_hop" and query in names:
        marker, answer = FIRST_HOP, assigned[names.index(query)]
    elif task == "composed" and query in names:
        marker = COMPOSE
        answer = outputs[table_keys.index(assigned[names.index(query)])]
    elif task == "direct" and query in table_keys:
        marker, answer = DIRECT, outputs[table_keys.index(query)]
    elif task == "copy" and query in OBJECTS:
        marker, answer = COPY, query
    else:
        raise ValueError("Task/query mismatch")
    tokens = (BOS, FIRST, names[0], assigned[0], names[1], assigned[1],
              SECOND, table_keys[0], outputs[0], table_keys[1], outputs[1],
              marker, query, ANSWER)
    f_part, g_part = table_partitions(names, table_keys, outputs)
    return Episode(tokens, answer, task, f_part, g_part)


def symbolic_oracle(tokens: tuple[int, ...]) -> int:
    """Read tokens and calculate the answer without calling make_episode."""
    if len(tokens) != SEQUENCE_LENGTH or tokens[:2] != (BOS, FIRST) or \
            tokens[6] != SECOND or tokens[13] != ANSWER:
        raise ValueError("Malformed grammar")
    names, assigned = (tokens[2], tokens[4]), (tokens[3], tokens[5])
    keys, outputs = (tokens[7], tokens[9]), (tokens[8], tokens[10])
    if len(set(names)) != 2 or any(n not in NAMES for n in names) or \
            len(set(keys)) != 2 or set(assigned) != set(keys) or any(k not in KEYS for k in keys) or \
            len(set(outputs)) != 2 or any(o not in OBJECTS for o in outputs):
        raise ValueError("Malformed table")
    marker, query = tokens[11], tokens[12]
    if marker == FIRST_HOP and query in names:
        return assigned[names.index(query)]
    if marker == COMPOSE and query in names:
        return outputs[keys.index(assigned[names.index(query)])]
    if marker == DIRECT and query in keys:
        return outputs[keys.index(query)]
    if marker == COPY and query in OBJECTS:
        return query
    raise ValueError("Malformed query")


def sample_episode(rng: random.Random, *, f_partition: str, g_partition: str,
                   task: str) -> Episode:
    if f_partition not in ("train", "holdout") or g_partition not in ("train", "holdout"):
        raise ValueError("Unknown partition")
    if task not in ("first_hop", "direct", "composed", "copy"):
        raise ValueError("Unknown task")
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        outputs = tuple(rng.sample(OBJECTS, 2))
        if table_partitions(names, keys, outputs) != (f_partition, g_partition):
            continue
        assigned = keys if rng.randrange(2) == 0 else keys[::-1]
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        if task in ("first_hop", "composed"):
            query = rng.choice(names)
        elif task == "direct":
            query = rng.choice(keys)
        else:
            query = rng.choice(OBJECTS)
        return make_episode(names, assigned, table_keys, outputs, task, query)
    raise RuntimeError("Could not sample requested partition")


def sample_factorial(rng: random.Random) -> tuple[Episode, Episode, Episode, Episode]:
    """Two first-table assignments × two second-table output assignments."""
    for _ in range(100_000):
        names = tuple(rng.sample(NAMES, 2))
        keys = tuple(rng.sample(KEYS, 2))
        four_outputs = tuple(rng.sample(OBJECTS, 4))
        if _partition(b"F/", names, keys) != "holdout" or any(
                _partition(b"G/", keys, pair) != "holdout"
                for pair in (four_outputs[:2], four_outputs[2:])):
            continue
        query = rng.choice(names)
        table_keys = keys if rng.randrange(2) == 0 else keys[::-1]
        f0 = keys if rng.randrange(2) == 0 else keys[::-1]
        f1 = f0[::-1]
        return tuple(
            make_episode(names, assigned, table_keys, outputs, "composed", query)
            for outputs in (four_outputs[:2], four_outputs[2:])
            for assigned in (f0, f1)
        )
    raise RuntimeError("Could not sample held-out factorial quartet")


def schedule(arm: str, step: int) -> tuple[tuple[str, float], ...]:
    if not 0 <= step < 5000:
        raise ValueError("Step outside frozen schedule")
    if arm == "baseline":
        return (("composed", .70), ("direct", .20), ("copy", .10))
    if arm in ("curriculum", "null"):
        if step < 1500:
            return (("first_hop", .45), ("direct", .45), ("copy", .10))
        if step < 3000:
            return (("first_hop", .30), ("direct", .30), ("composed", .30), ("copy", .10))
        return (("first_hop", .10), ("direct", .20), ("composed", .60), ("copy", .10))
    raise ValueError("Unknown arm")


def training_batch(seed: int, batch_size: int, *, arm: str, step: int
                   ) -> tuple[list[Episode], list[int]]:
    if batch_size <= 0:
        raise ValueError("Positive batch size required")
    weights = schedule(arm, step)
    rng = random.Random(seed)
    null_rng = random.Random(seed ^ 0x52C0FFEE)
    episodes, targets = [], []
    for _ in range(batch_size):
        roll = rng.random()
        cumulative = 0.0
        task = weights[-1][0]
        for candidate, weight in weights:
            cumulative += weight
            if roll < cumulative:
                task = candidate
                break
        ep = sample_episode(rng, f_partition="train", g_partition="train", task=task)
        target = null_rng.choice(OBJECTS) if arm == "null" and task == "composed" else ep.answer
        episodes.append(ep)
        targets.append(target)
    return episodes, targets


def evaluation_suite(seed: int, size: int = 256) -> dict[str, list[Episode]]:
    if size <= 0 or size % 2:
        raise ValueError("Positive even size required")
    rng = random.Random(seed)
    suite = {}
    for f, g in (("train", "train"), ("holdout", "train"),
                 ("train", "holdout"), ("holdout", "holdout")):
        suite[f"composed_{f}_{g}"] = [
            sample_episode(rng, f_partition=f, g_partition=g, task="composed")
            for _ in range(size)
        ]
    for f in ("train", "holdout"):
        suite[f"first_hop_{f}"] = [
            sample_episode(rng, f_partition=f, g_partition="train", task="first_hop")
            for _ in range(size)
        ]
    first_pairs = []
    for _ in range(size // 2):
        ep = sample_episode(rng, f_partition="holdout", g_partition="train", task="first_hop")
        t = ep.tokens
        names = (t[2], t[4])
        other = names[1] if t[12] == names[0] else names[0]
        first_pairs.extend((ep, make_episode(names, (t[3], t[5]), (t[7], t[9]),
                                             (t[8], t[10]), "first_hop", other)))
    suite["first_hop_pairs_holdout"] = first_pairs
    for g in ("train", "holdout"):
        suite[f"direct_{g}"] = [
            sample_episode(rng, f_partition="train", g_partition=g, task="direct")
            for _ in range(size)
        ]
    direct_pairs = []
    for _ in range(size // 2):
        ep = sample_episode(rng, f_partition="train", g_partition="holdout", task="direct")
        t = ep.tokens
        keys = (t[7], t[9])
        other = keys[1] if t[12] == keys[0] else keys[0]
        direct_pairs.extend((ep, make_episode((t[2], t[4]), (t[3], t[5]), keys,
                                              (t[8], t[10]), "direct", other)))
    suite["direct_pairs_holdout"] = direct_pairs
    suite["copy"] = [sample_episode(rng, f_partition="holdout", g_partition="holdout",
                                     task="copy") for _ in range(size)]
    suite["factorial"] = [ep for _ in range(size // 2) for ep in sample_factorial(rng)]
    return suite
