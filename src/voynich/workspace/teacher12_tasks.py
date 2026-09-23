"""Raw serialized graph programs for TEACH-0012; generation only, no training on import."""

from dataclasses import dataclass, replace
import hashlib
import json
import random


PAD, BOS, EDGE, GAP, COMPOSE, DIRECT, COPY, FIRST_HOP, ANSWER = range(9)
SYMBOL_START = 16
SYMBOL_COUNT = 2048
SYMBOLS = tuple(range(SYMBOL_START, SYMBOL_START + SYMBOL_COUNT))
VOCAB_SIZE = SYMBOL_START + SYMBOL_COUNT
CONTEXT_LENGTH = 128
SPLIT_NAMESPACE = "TEACH-0012-v1"
TASK_MARKERS = {
    "composed": COMPOSE,
    "direct": DIRECT,
    "copy": COPY,
    "first_hop": FIRST_HOP,
}


@dataclass(frozen=True)
class RenderSpec:
    marker_dropout: float
    max_gaps: int
    styles: tuple[str, ...]

    def validate(self) -> None:
        if not 0 <= self.marker_dropout <= 1:
            raise ValueError("marker_dropout outside [0,1]")
        if not 0 <= self.max_gaps <= 3:
            raise ValueError("max_gaps outside [0,3]")
        if not self.styles or any(x not in ("prefix", "infix", "suffix")
                                  for x in self.styles):
            raise ValueError("Unknown or empty row style set")


@dataclass(frozen=True)
class Episode:
    tokens: tuple[int, ...]
    f_rows: tuple[tuple[int, int], ...]
    g_rows: tuple[tuple[int, int], ...]
    distractor_rows: tuple[tuple[int, int], ...]
    serialized_rows: tuple[tuple[int, int], ...]
    answer: int
    task: str
    query: int
    f_partition: str
    g_partition: str
    logical_id: str
    render_id: str
    distractor_chains: int
    marker_dropout: float

    @property
    def rows(self) -> tuple[tuple[int, int], ...]:
        return self.f_rows + self.g_rows + self.distractor_rows


def _digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _partition(kind: str, left: tuple[int, ...], right: tuple[int, ...]) -> str:
    digest = _digest([SPLIT_NAMESPACE, kind, sorted(left), sorted(right)])
    bucket = int(digest[:16], 16) % 10
    return "train" if bucket < 8 else "confirm" if bucket == 9 else "development"


def table_partitions(f_rows: tuple[tuple[int, int], ...],
                     g_rows: tuple[tuple[int, int], ...]) -> tuple[str, str]:
    return (_partition("F", tuple(a for a, _ in f_rows), tuple(b for _, b in f_rows)),
            _partition("G", tuple(a for a, _ in g_rows), tuple(b for _, b in g_rows)))


def oracle_from_rows(rows: tuple[tuple[int, int], ...], task: str, query: int) -> int:
    mapping = {}
    for left, right in rows:
        if left in mapping:
            raise ValueError("Rows are not a function")
        mapping[left] = right
    if task == "copy":
        return query
    if query not in mapping:
        raise ValueError("Query has no outgoing row")
    first = mapping[query]
    if task in ("first_hop", "direct"):
        return first
    if task == "composed" and first in mapping:
        return mapping[first]
    raise ValueError("Malformed composed graph")


def symbolic_oracle(episode: Episode) -> int:
    return oracle_from_rows(episode.rows, episode.task, episode.query)


def _serialize(rows: tuple[tuple[int, int], ...], task: str, query: int,
               rng: random.Random, spec: RenderSpec) -> tuple[tuple[int, ...],
                                                               tuple[tuple[int, int], ...]]:
    spec.validate()
    ordered = list(rows)
    rng.shuffle(ordered)
    body = [BOS]
    for index, (left, right) in enumerate(ordered):
        marker = rng.random() >= spec.marker_dropout
        style = rng.choice(spec.styles)
        if not marker:
            body.extend((left, right))
        elif style == "prefix":
            body.extend((EDGE, left, right))
        elif style == "infix":
            body.extend((left, EDGE, right))
        else:
            body.extend((left, right, EDGE))
        if index + 1 < len(ordered):
            body.extend([GAP] * rng.randint(0, spec.max_gaps))
    body.extend((TASK_MARKERS[task], query, ANSWER))
    if len(body) > CONTEXT_LENGTH:
        raise ValueError(f"Serialized program length {len(body)} exceeds context")
    return tuple(body), tuple(ordered)


def _make_episode(f_rows: tuple[tuple[int, int], ...],
                  g_rows: tuple[tuple[int, int], ...],
                  distractor_rows: tuple[tuple[int, int], ...],
                  task: str, query: int, rng: random.Random, spec: RenderSpec) -> Episode:
    if len(f_rows) != 4 or len(g_rows) != 4 or len(distractor_rows) % 2:
        raise ValueError("Expected four F/G rows and complete two-row distractor chains")
    rows = f_rows + g_rows + distractor_rows
    left = [a for a, _ in rows]
    if len(set(left)) != len(left):
        raise ValueError("Every row left side must be unique")
    answer = oracle_from_rows(rows, task, query)
    f_partition, g_partition = table_partitions(f_rows, g_rows)
    logical = _digest({"f": f_rows, "g": g_rows, "d": distractor_rows,
                       "task": task, "query": query})
    tokens, serialized = _serialize(rows, task, query, rng, spec)
    render = _digest({"logical": logical, "tokens": tokens})
    return Episode(tokens, f_rows, g_rows, distractor_rows, serialized, answer, task,
                   query, f_partition, g_partition, logical, render,
                   len(distractor_rows) // 2, spec.marker_dropout)


def rerender(episode: Episode, rng: random.Random, spec: RenderSpec, *,
             task: str | None = None, query: int | None = None) -> Episode:
    return _make_episode(episode.f_rows, episode.g_rows, episode.distractor_rows,
                         task or episode.task, episode.query if query is None else query,
                         rng, spec)


def _draw_components(rng: random.Random, distractor_chains: int):
    if not 0 <= distractor_chains <= 6:
        raise ValueError("distractor_chains outside [0,6]")
    count = 12 + 3 * distractor_chains
    symbols = rng.sample(SYMBOLS, count)
    names = tuple(symbols[:4])
    keys = tuple(symbols[4:8])
    objects = tuple(symbols[8:12])
    assigned = list(keys)
    rng.shuffle(assigned)
    table_keys = list(keys)
    rng.shuffle(table_keys)
    object_order = list(objects)
    rng.shuffle(object_order)
    f_rows = tuple(zip(names, assigned, strict=True))
    g_rows = tuple(zip(table_keys, object_order, strict=True))
    distractors = []
    offset = 12
    for _ in range(distractor_chains):
        name, key, obj = symbols[offset:offset + 3]
        offset += 3
        distractors.extend(((name, key), (key, obj)))
    return f_rows, g_rows, tuple(distractors)


def sample_episode(rng: random.Random, *, f_partition: str, g_partition: str,
                   task: str, distractor_chains: int, spec: RenderSpec) -> Episode:
    if f_partition not in ("train", "development", "confirm") or g_partition not in (
            "train", "development", "confirm"):
        raise ValueError("Unknown partition")
    if task not in TASK_MARKERS:
        raise ValueError("Unknown task")
    for _ in range(100_000):
        f_rows, g_rows, distractors = _draw_components(rng, distractor_chains)
        if table_partitions(f_rows, g_rows) != (f_partition, g_partition):
            continue
        if task in ("first_hop", "composed"):
            query = rng.choice(tuple(a for a, _ in f_rows))
        elif task == "direct":
            query = rng.choice(tuple(a for a, _ in g_rows))
        else:
            query = rng.choice(tuple(b for _, b in g_rows))
        return _make_episode(f_rows, g_rows, distractors, task, query, rng, spec)
    raise RuntimeError("Could not sample requested logical partition")


def schedule(step: int) -> tuple[tuple[str, float], ...]:
    if not 0 <= step < 8000:
        raise ValueError("Step outside frozen schedule")
    if step < 2000:
        return (("first_hop", .45), ("direct", .45), ("copy", .10))
    if step < 4000:
        return (("first_hop", .25), ("direct", .25), ("composed", .40), ("copy", .10))
    return (("first_hop", .10), ("direct", .15), ("composed", .70), ("copy", .05))


def difficulty(rng: random.Random, step: int) -> tuple[int, RenderSpec]:
    if step < 2000:
        return rng.randint(0, 1), RenderSpec(0.0, 1, ("prefix",))
    if step < 4000:
        return rng.randint(0, 2), RenderSpec(.10, 1, ("prefix", "infix", "suffix"))
    return rng.randint(1, 4), RenderSpec(.25, 2, ("prefix", "infix", "suffix"))


def training_batch(seed: int, batch_size: int, *, step: int, null_composed: bool = False
                   ) -> tuple[list[Episode], list[int]]:
    if batch_size <= 0:
        raise ValueError("Positive batch size required")
    rng = random.Random(seed)
    null_rng = random.Random(seed ^ 0x1200C0DE)
    weighted = schedule(step)
    episodes, answers = [], []
    for _ in range(batch_size):
        roll, cumulative, task = rng.random(), 0.0, weighted[-1][0]
        for candidate, weight in weighted:
            cumulative += weight
            if roll < cumulative:
                task = candidate
                break
        chains, spec = difficulty(rng, step)
        episode = sample_episode(rng, f_partition="train", g_partition="train",
                                 task=task, distractor_chains=chains, spec=spec)
        answer = episode.answer
        if null_composed and task == "composed":
            answer = null_rng.choice(tuple(b for _, b in episode.g_rows))
        episodes.append(episode)
        answers.append(answer)
    return episodes, answers


def _query_group(rng: random.Random, task: str, size: int) -> list[Episode]:
    items = []
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample_episode(rng, f_partition="confirm", g_partition="confirm",
                              task=task, distractor_chains=4, spec=spec)
        queries = tuple(a for a, _ in base.f_rows) if task == "first_hop" else tuple(
            a for a, _ in base.g_rows)
        for query in queries:
            items.append(rerender(base, random.Random(rng.randrange(2**63)), spec,
                                  task=task, query=query))
    return items


def _factorial(rng: random.Random, size: int) -> list[Episode]:
    items = []
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    while len(items) < 4 * size:
        base = sample_episode(rng, f_partition="confirm", g_partition="confirm",
                              task="composed", distractor_chains=4, spec=spec)
        query = base.f_rows[0][0]
        alternate_name = base.f_rows[1][0]
        f0 = base.f_rows
        assignment = {left: right for left, right in f0}
        assignment[query], assignment[alternate_name] = (
            assignment[alternate_name], assignment[query])
        f1 = tuple((left, assignment[left]) for left, _ in f0)
        new_objects = tuple(rng.sample(sorted(set(SYMBOLS) - set(
            x for row in base.rows for x in row)), 4))
        g1 = tuple((left, right) for (left, _), right in zip(
            base.g_rows, new_objects, strict=True))
        if table_partitions(f1, base.g_rows) != ("confirm", "confirm") or \
                table_partitions(f0, g1) != ("confirm", "confirm"):
            continue
        quartet = []
        for g_rows in (base.g_rows, g1):
            for f_rows in (f0, f1):
                quartet.append(_make_episode(
                    f_rows, g_rows, base.distractor_rows, "composed", query,
                    random.Random(rng.randrange(2**63)), spec))
        if len({ep.answer for ep in quartet}) == 4:
            items.extend(quartet)
    return items


def _order_groups(rng: random.Random, size: int) -> list[Episode]:
    items = []
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample_episode(rng, f_partition="confirm", g_partition="confirm",
                              task="composed", distractor_chains=4, spec=spec)
        for _ in range(4):
            items.append(rerender(base, random.Random(rng.randrange(2**63)), spec))
    return items


def _boundary_groups(rng: random.Random, size: int) -> list[Episode]:
    items = []
    for _ in range(size):
        base = sample_episode(
            rng, f_partition="confirm", g_partition="confirm", task="composed",
            distractor_chains=4, spec=RenderSpec(0.0, 0, ("prefix",)))
        specs = (
            RenderSpec(0.0, 0, ("prefix",)),
            RenderSpec(.25, 2, ("prefix", "infix", "suffix")),
            RenderSpec(.50, 2, ("prefix", "infix", "suffix")),
            RenderSpec(1.0, 2, ("prefix", "infix", "suffix")),
        )
        items.extend(rerender(base, random.Random(rng.randrange(2**63)), spec)
                     for spec in specs)
    return items


def _distractor_groups(rng: random.Random, size: int) -> list[Episode]:
    items = []
    spec = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    for _ in range(size):
        base = sample_episode(rng, f_partition="confirm", g_partition="confirm",
                              task="composed", distractor_chains=0, spec=spec)
        used = {x for row in base.f_rows + base.g_rows for x in row}
        for chains in (0, 1, 2, 4):
            available = sorted(set(SYMBOLS) - used)
            values = rng.sample(available, 3 * chains)
            distractors = []
            for offset in range(0, len(values), 3):
                name, key, obj = values[offset:offset + 3]
                distractors.extend(((name, key), (key, obj)))
            items.append(_make_episode(
                base.f_rows, base.g_rows, tuple(distractors), "composed", base.query,
                random.Random(rng.randrange(2**63)), spec))
    return items


def evaluation_suite(seed: int, size: int = 128) -> dict[str, list[Episode]]:
    if size <= 0:
        raise ValueError("Positive evaluation group count required")
    rng = random.Random(seed)
    hard = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    suite = {}
    for f, g in (("train", "train"), ("confirm", "train"),
                 ("train", "confirm"), ("confirm", "confirm")):
        suite[f"composed_{f}_{g}"] = [
            sample_episode(rng, f_partition=f, g_partition=g, task="composed",
                           distractor_chains=4, spec=hard) for _ in range(size)]
    for task in ("first_hop", "direct", "copy"):
        suite[f"{task}_confirm"] = [
            sample_episode(rng, f_partition="confirm", g_partition="confirm", task=task,
                           distractor_chains=4, spec=hard) for _ in range(size)]
    suite["first_hop_query_groups"] = _query_group(rng, "first_hop", size)
    suite["direct_query_groups"] = _query_group(rng, "direct", size)
    suite["factorial"] = _factorial(rng, size)
    suite["order_groups"] = _order_groups(rng, size)
    suite["distractor_groups"] = _distractor_groups(rng, size)
    suite["boundary_groups"] = _boundary_groups(rng, size)
    suite["long_ood"] = [
        sample_episode(rng, f_partition="confirm", g_partition="confirm", task="composed",
                       distractor_chains=6, spec=hard) for _ in range(size)]
    return suite


def pad_batch(episodes: list[Episode]) -> tuple[list[list[int]], list[int]]:
    lengths = [len(ep.tokens) for ep in episodes]
    width = max(lengths)
    if width > CONTEXT_LENGTH:
        raise ValueError("Batch exceeds context length")
    return [list(ep.tokens) + [PAD] * (width - len(ep.tokens)) for ep in episodes], lengths


def replace_rows(episode: Episode, *, f_rows=None, g_rows=None, distractor_rows=None,
                 rng: random.Random, spec: RenderSpec) -> Episode:
    """Counterfactual helper used by later causal studies."""
    return _make_episode(
        episode.f_rows if f_rows is None else tuple(f_rows),
        episode.g_rows if g_rows is None else tuple(g_rows),
        episode.distractor_rows if distractor_rows is None else tuple(distractor_rows),
        episode.task, episode.query, rng, spec)


def with_tokens(episode: Episode, tokens: tuple[int, ...]) -> Episode:
    """Testing helper; semantic fields stay fixed while a serialization is supplied."""
    return replace(episode, tokens=tokens, render_id=_digest(
        {"logical": episode.logical_id, "tokens": tokens}))
