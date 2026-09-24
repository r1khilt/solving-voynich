"""Fresh episodic graph generator and public-grammar ceiling for TEACH-0014.

This is source under design, not a frozen campaign or a manuscript model.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import random


PAD, BOS, EDGE, GAP, COMPOSE, DIRECT, COPY, FIRST_HOP, ANSWER, HOP3, HOP4 = range(11)
SYMBOL_START = 16
SYMBOL_COUNT = 2048
VOCAB_SIZE = SYMBOL_START + SYMBOL_COUNT
CONTEXT_LENGTH = 192
SPLIT_NAMESPACE = "TEACH-0014-latent-edge-v1"
TASK_MARKERS = {
    "composed": {2: COMPOSE, 3: HOP3, 4: HOP4},
    "first_hop": {1: FIRST_HOP},
    "direct": {1: DIRECT},
    "copy": {0: COPY},
}


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def family_partition(kind: str, rows: tuple[tuple[int, int], ...]) -> str:
    """Partition logical rows before order, marker or distractor rendering."""
    token = digest((SPLIT_NAMESPACE, kind, sorted(rows)))
    bucket = int(token[:16], 16) % 10
    return "train" if bucket < 8 else "development" if bucket == 8 else "confirm"


@dataclass(frozen=True)
class RenderSpec:
    marker_dropout: float = .25
    max_gaps: int = 2
    styles: tuple[str, ...] = ("prefix", "infix", "suffix")

    def validate(self) -> None:
        if not 0.0 <= self.marker_dropout <= 1.0:
            raise ValueError("marker_dropout must be in [0,1]")
        if not 0 <= self.max_gaps <= 3:
            raise ValueError("max_gaps must be in [0,3]")
        if not self.styles or any(style not in ("prefix", "infix", "suffix")
                                  for style in self.styles):
            raise ValueError("Unknown or empty row style set")


@dataclass(frozen=True)
class Episode:
    tokens: tuple[int, ...]
    signal_paths: tuple[tuple[int, ...], ...]
    distractor_paths: tuple[tuple[int, ...], ...]
    serialized_rows: tuple[tuple[int, int], ...]
    row_positions: tuple[tuple[int, int], ...]
    query: int
    answer: int
    task: str
    hops: int
    stage_partitions: tuple[str, ...]
    graph_partition: str
    graph_id: str
    logical_id: str
    render_id: str
    marker_dropout: float
    alias_mode: str

    @property
    def signal_rows(self) -> tuple[tuple[int, int], ...]:
        return tuple((path[index], path[index + 1])
                     for path in self.signal_paths for index in range(len(path) - 1))

    @property
    def distractor_rows(self) -> tuple[tuple[int, int], ...]:
        return tuple((path[index], path[index + 1])
                     for path in self.distractor_paths for index in range(len(path) - 1))

    @property
    def rows(self) -> tuple[tuple[int, int], ...]:
        return self.signal_rows + self.distractor_rows


def rows_oracle(rows: tuple[tuple[int, int], ...], query: int, hops: int) -> int:
    """Follow the unique outgoing edge the requested number of times."""
    if not 0 <= hops <= 4:
        raise ValueError("Only 0-4 hops are registered")
    if not SYMBOL_START <= query < VOCAB_SIZE:
        raise ValueError("Query outside ordinary symbol vocabulary")
    mapping: dict[int, int] = {}
    for left, right in rows:
        if left < SYMBOL_START or right < SYMBOL_START or (
                left >= VOCAB_SIZE or right >= VOCAB_SIZE):
            raise ValueError("Ordinary symbol outside vocabulary")
        if left in mapping:
            raise ValueError("Rows are not a function")
        mapping[left] = right
    value = query
    for _ in range(hops):
        if value not in mapping:
            raise ValueError("Requested path is incomplete")
        value = mapping[value]
    return value


def strip_pair_rows(tokens: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Public deterministic parser for the current two-operand surface grammar.

    It uses only the visible serialized tokens, never Episode row metadata.
    """
    if len(tokens) < 4 or tokens[0] != BOS or tokens[-1] != ANSWER:
        raise ValueError("Malformed episode envelope")
    if tokens[-3] not in (COMPOSE, DIRECT, COPY, FIRST_HOP, HOP3, HOP4):
        raise ValueError("Missing task marker")
    if not SYMBOL_START <= tokens[-2] < VOCAB_SIZE:
        raise ValueError("Malformed query symbol")
    body = tokens[1:-3]
    if any(token not in (EDGE, GAP) and not SYMBOL_START <= token < VOCAB_SIZE
           for token in body):
        raise ValueError("Unknown body token")
    ordinary = tuple(token for token in body if token >= SYMBOL_START)
    if len(ordinary) % 2:
        raise ValueError("Odd number of edge operands")
    return tuple(zip(ordinary[::2], ordinary[1::2], strict=True))


def visible_oracle(tokens: tuple[int, ...]) -> int:
    rows = strip_pair_rows(tokens)
    marker = tokens[-3]
    hops = {COPY: 0, FIRST_HOP: 1, DIRECT: 1, COMPOSE: 2, HOP3: 3, HOP4: 4}[marker]
    return rows_oracle(rows, tokens[-2], hops)


def _paths(rng: random.Random, hops: int, distractors: int,
           alias_mode: str) -> tuple[tuple[tuple[int, ...], ...],
                                     tuple[tuple[int, ...], ...]]:
    if not 1 <= hops <= 4:
        raise ValueError("Signal hop count outside [1,4]")
    if not 0 <= distractors <= 8:
        raise ValueError("Distractor count outside [0,8]")
    if alias_mode not in ("none", "terminal", "inner"):
        raise ValueError("Unknown alias mode")
    if alias_mode != "none" and distractors == 0:
        raise ValueError("Alias requires a distractor chain")
    if alias_mode == "inner" and hops < 2:
        raise ValueError("Inner alias requires at least two signal hops")
    count = 4 * (hops + 1) + 3 * distractors
    symbols = rng.sample(range(SYMBOL_START, VOCAB_SIZE), count)
    signals = tuple(tuple(symbols[i * (hops + 1):(i + 1) * (hops + 1)])
                    for i in range(4))
    offset = 4 * (hops + 1)
    distractor_paths = [tuple(symbols[offset + 3 * i:offset + 3 * (i + 1)])
                        for i in range(distractors)]
    if alias_mode == "terminal":
        chosen = rng.randrange(4)
        path = distractor_paths[0]
        distractor_paths[0] = (path[0], path[1], signals[chosen][-1])
    elif alias_mode == "inner":
        chosen = rng.randrange(4)
        stage = rng.randrange(1, hops)
        path = distractor_paths[0]
        distractor_paths[0] = (path[0], path[1], signals[chosen][stage])
    return signals, tuple(distractor_paths)


def _render(rows: tuple[tuple[int, int], ...], task: str, hops: int, query: int,
            rng: random.Random, spec: RenderSpec
            ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...],
                       tuple[tuple[int, int], ...]]:
    spec.validate()
    if task not in TASK_MARKERS or hops not in TASK_MARKERS[task]:
        raise ValueError("Task and hop count disagree")
    ordered = list(rows)
    rng.shuffle(ordered)
    tokens = [BOS]
    positions = []
    for index, (left, right) in enumerate(ordered):
        marker = rng.random() >= spec.marker_dropout
        style = rng.choice(spec.styles)
        if not marker:
            positions.append((len(tokens), len(tokens) + 1))
            tokens.extend((left, right))
        elif style == "prefix":
            positions.append((len(tokens) + 1, len(tokens) + 2))
            tokens.extend((EDGE, left, right))
        elif style == "infix":
            positions.append((len(tokens), len(tokens) + 2))
            tokens.extend((left, EDGE, right))
        else:
            positions.append((len(tokens), len(tokens) + 1))
            tokens.extend((left, right, EDGE))
        if index + 1 < len(ordered):
            tokens.extend([GAP] * rng.randint(0, spec.max_gaps))
    tokens.extend((TASK_MARKERS[task][hops], query, ANSWER))
    if len(tokens) > CONTEXT_LENGTH:
        raise ValueError(f"Serialized program length {len(tokens)} exceeds context")
    return tuple(tokens), tuple(ordered), tuple(positions)


def make_episode(signal_paths: tuple[tuple[int, ...], ...],
                 distractor_paths: tuple[tuple[int, ...], ...], *,
                 task: str, query: int, rng: random.Random,
                 spec: RenderSpec, alias_mode: str = "none") -> Episode:
    if len(signal_paths) != 4 or len({len(path) for path in signal_paths}) != 1:
        raise ValueError("Expected four equal-length signal paths")
    signal_hops = len(signal_paths[0]) - 1
    if not 1 <= signal_hops <= 4 or any(len(path) != 3 for path in distractor_paths):
        raise ValueError("Malformed signal or distractor path")
    signal_symbols = [symbol for path in signal_paths for symbol in path]
    if len(set(signal_symbols)) != len(signal_symbols):
        raise ValueError("Signal paths must be symbol-disjoint")
    distractor_prefix = [symbol for path in distractor_paths for symbol in path[:2]]
    if len(set(distractor_prefix)) != len(distractor_prefix) or (
            set(distractor_prefix) & set(signal_symbols)):
        raise ValueError("Distractor left and middle symbols must be fresh")
    distractor_terminals = [path[-1] for path in distractor_paths]
    if len(set(distractor_terminals)) != len(distractor_terminals) or (
            set(distractor_terminals) & set(distractor_prefix)):
        raise ValueError("Distractor terminals must be distinct from one another and left sides")
    shared_terminals = [path[-1] for path in distractor_paths
                        if path[-1] in set(signal_symbols)]
    if alias_mode == "none" and shared_terminals:
        raise ValueError("Unregistered signal alias")
    if alias_mode == "terminal" and (
            len(shared_terminals) != 1 or
            shared_terminals[0] not in {path[-1] for path in signal_paths}):
        raise ValueError("Terminal alias must share one signal endpoint")
    if alias_mode == "inner" and (
            len(shared_terminals) != 1 or
            shared_terminals[0] not in {
                symbol for path in signal_paths for symbol in path[1:-1]}):
        raise ValueError("Inner alias must share one signal intermediate")
    if alias_mode not in ("none", "terminal", "inner"):
        raise ValueError("Unknown alias mode")
    if task == "composed":
        hops = signal_hops
    elif task == "copy":
        hops = 0
    else:
        hops = 1
    signal_rows = tuple((path[index], path[index + 1])
                        for path in signal_paths for index in range(signal_hops))
    distractor_rows = tuple((path[index], path[index + 1])
                            for path in distractor_paths for index in range(2))
    rows = signal_rows + distractor_rows
    left = tuple(a for a, _ in rows)
    if len(set(left)) != len(left):
        raise ValueError("Every row left side must be unique")
    answer = rows_oracle(rows, query, hops)
    stages = tuple(
        family_partition(f"stage-{index}", tuple(
            (path[index], path[index + 1]) for path in signal_paths))
        for index in range(signal_hops)
    )
    graph_partition = family_partition("graph", signal_rows)
    graph_id = digest({"signal": signal_paths, "distractors": distractor_paths})
    logical_id = digest({"graph": graph_id, "task": task, "query": query})
    tokens, serialized, positions = _render(rows, task, hops, query, rng, spec)
    if strip_pair_rows(tokens) != serialized or visible_oracle(tokens) != answer:
        raise RuntimeError("Visible grammar and logical oracle disagree")
    render_id = digest({"logical": logical_id, "tokens": tokens})
    return Episode(tokens, signal_paths, distractor_paths, serialized, positions,
                   query, answer, task, hops, stages, graph_partition, graph_id,
                   logical_id, render_id, spec.marker_dropout, alias_mode)


def sample_episode(rng: random.Random, *, signal_hops: int, task: str,
                   distractors: int, spec: RenderSpec,
                   stage_partitions: tuple[str, ...] | None = None,
                   graph_partition: str | None = None,
                   alias_mode: str = "none") -> Episode:
    if stage_partitions is not None and len(stage_partitions) != signal_hops:
        raise ValueError("One requested partition per stage is required")
    allowed = {"train", "development", "confirm"}
    if stage_partitions is not None and any(p not in allowed for p in stage_partitions):
        raise ValueError("Unknown stage partition")
    if graph_partition is not None and graph_partition not in allowed:
        raise ValueError("Unknown graph partition")
    for _ in range(100_000):
        signals, distractor_paths = _paths(rng, signal_hops, distractors, alias_mode)
        if task == "first_hop":
            query = rng.choice(signals)[0]
        elif task == "direct":
            if signal_hops < 2:
                raise ValueError("Direct query requires at least two signal stages")
            query = rng.choice(signals)[1]
        elif task == "copy":
            query = rng.choice(signals)[-1]
        elif task == "composed":
            if signal_hops < 2:
                raise ValueError("Composed task requires at least two signal stages")
            query = rng.choice(signals)[0]
        else:
            raise ValueError("Unknown task")
        try:
            episode = make_episode(signals, distractor_paths, task=task, query=query,
                                   rng=rng, spec=spec, alias_mode=alias_mode)
        except ValueError as exc:
            if "exceeds context" not in str(exc):
                raise
            continue
        if stage_partitions is not None and episode.stage_partitions != stage_partitions:
            continue
        if graph_partition is not None and episode.graph_partition != graph_partition:
            continue
        return episode
    raise RuntimeError("Could not sample requested family partitions")


def rerender(episode: Episode, rng: random.Random, spec: RenderSpec, *,
             task: str | None = None, query: int | None = None) -> Episode:
    return make_episode(
        episode.signal_paths, episode.distractor_paths,
        task=episode.task if task is None else task,
        query=episode.query if query is None else query,
        rng=rng, spec=spec, alias_mode=episode.alias_mode)


def _fresh_distractors(rng: random.Random,
                       signals: tuple[tuple[int, ...], ...],
                       count: int) -> tuple[tuple[int, ...], ...]:
    used = {symbol for path in signals for symbol in path}
    available = tuple(symbol for symbol in range(SYMBOL_START, VOCAB_SIZE)
                      if symbol not in used)
    values = rng.sample(available, 3 * count)
    return tuple(tuple(values[index:index + 3])
                 for index in range(0, len(values), 3))


def _query_groups(rng: random.Random, task: str, size: int,
                  spec: RenderSpec) -> list[Episode]:
    episodes = []
    for _ in range(size):
        base = sample_episode(
            rng, signal_hops=2, task=task, distractors=4, spec=spec,
            stage_partitions=("confirm", "confirm"))
        endpoint = 0 if task == "first_hop" else 1
        for path in base.signal_paths:
            episodes.append(rerender(
                base, random.Random(rng.randrange(2**63)), spec,
                query=path[endpoint]))
    return episodes


def _factorial(rng: random.Random, size: int,
               spec: RenderSpec) -> list[Episode]:
    """F swap and independent G remap; all four table families are confirm."""
    episodes = []
    attempts = 0
    max_attempts = 30_000 * size
    while len(episodes) < 4 * size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not fill TEACH-0014 factorial panel")
        f0, distractors = _paths(rng, 2, 4, "none")
        f1_list = list(f0)
        f1_list[0] = (f0[1][0], *f0[0][1:])
        f1_list[1] = (f0[0][0], *f0[1][1:])
        f1 = tuple(f1_list)
        first0 = tuple((path[0], path[1]) for path in f0)
        first1 = tuple((path[0], path[1]) for path in f1)
        second0 = tuple((path[1], path[2]) for path in f0)
        if any(family_partition(kind, rows) != "confirm" for kind, rows in (
                ("stage-0", first0), ("stage-0", first1),
                ("stage-1", second0))):
            continue
        used = {symbol for path in f0 + distractors for symbol in path}
        available = tuple(symbol for symbol in range(SYMBOL_START, VOCAB_SIZE)
                          if symbol not in used)
        objects = rng.sample(available, 4)
        g0 = tuple((path[0], path[1], objects[index])
                   for index, path in enumerate(f0))
        g1 = tuple((path[0], path[1], objects[index])
                   for index, path in enumerate(f1))
        second1 = tuple((path[1], path[2]) for path in g0)
        if family_partition("stage-1", second1) != "confirm":
            continue
        query = f0[0][0]
        quartet = [
            make_episode(paths, distractors, task="composed", query=query,
                         rng=random.Random(rng.randrange(2**63)), spec=spec)
            for paths in (f0, f1, g0, g1)
        ]
        if len({episode.answer for episode in quartet}) != 4:
            raise RuntimeError("Factorial answers are not four-way distinct")
        if any(ep.stage_partitions != ("confirm", "confirm") for ep in quartet):
            raise RuntimeError("Factorial family partition drift")
        episodes.extend(quartet)
    return episodes


def _rerender_groups(rng: random.Random, size: int,
                     spec: RenderSpec) -> list[Episode]:
    episodes = []
    for _ in range(size):
        base = sample_episode(
            rng, signal_hops=2, task="composed", distractors=4, spec=spec,
            stage_partitions=("confirm", "confirm"))
        episodes.extend(rerender(
            base, random.Random(rng.randrange(2**63)), spec) for _ in range(4))
    return episodes


def _boundary_groups(rng: random.Random, size: int) -> list[Episode]:
    episodes = []
    specs = (
        RenderSpec(0.0, 0, ("prefix",)),
        RenderSpec(.25, 2, ("prefix", "infix", "suffix")),
        RenderSpec(.5, 2, ("prefix", "infix", "suffix")),
        RenderSpec(1.0, 2, ("prefix", "infix", "suffix")),
    )
    for _ in range(size):
        base = sample_episode(
            rng, signal_hops=2, task="composed", distractors=4, spec=specs[0],
            stage_partitions=("confirm", "confirm"))
        episodes.extend(rerender(
            base, random.Random(rng.randrange(2**63)), spec) for spec in specs)
    return episodes


def _distractor_groups(rng: random.Random, size: int,
                       spec: RenderSpec) -> list[Episode]:
    episodes = []
    for _ in range(size):
        base = sample_episode(
            rng, signal_hops=2, task="composed", distractors=0, spec=spec,
            stage_partitions=("confirm", "confirm"))
        for count in (0, 1, 4, 8):
            distractors = _fresh_distractors(rng, base.signal_paths, count)
            episodes.append(make_episode(
                base.signal_paths, distractors, task="composed", query=base.query,
                rng=random.Random(rng.randrange(2**63)), spec=spec))
    return episodes


def evaluation_suite(seed: int, size: int = 128) -> dict[str, list[Episode]]:
    """Fresh grouped behavior suite; no TEACH-0012 family or render IDs are reused."""
    if size <= 0:
        raise ValueError("Positive group count required")
    rng = random.Random(seed)
    hard = RenderSpec(.25, 2, ("prefix", "infix", "suffix"))
    suite: dict[str, list[Episode]] = {}
    for f_partition, g_partition in (
            ("train", "train"), ("confirm", "train"),
            ("train", "confirm"), ("confirm", "confirm")):
        key = f"composed_{f_partition}_{g_partition}"
        suite[key] = [
            sample_episode(
                rng, signal_hops=2, task="composed", distractors=4, spec=hard,
                stage_partitions=(f_partition, g_partition))
            for _ in range(size)
        ]
    for task in ("first_hop", "direct", "copy"):
        suite[f"{task}_confirm"] = [
            sample_episode(
                rng, signal_hops=2, task=task, distractors=4, spec=hard,
                stage_partitions=("confirm", "confirm"))
            for _ in range(size)
        ]
    suite["first_hop_query_groups"] = _query_groups(rng, "first_hop", size, hard)
    suite["direct_query_groups"] = _query_groups(rng, "direct", size, hard)
    suite["factorial"] = _factorial(rng, size, hard)
    suite["order_groups"] = _rerender_groups(rng, size, hard)
    suite["boundary_groups"] = _boundary_groups(rng, size)
    suite["distractor_groups"] = _distractor_groups(rng, size, hard)
    suite["long_ood"] = [
        sample_episode(
            rng, signal_hops=2, task="composed", distractors=8,
            spec=RenderSpec(0.0, 3, ("prefix", "infix", "suffix")),
            stage_partitions=("confirm", "confirm"))
        for _ in range(size)
    ]
    for alias_mode in ("terminal", "inner"):
        suite[f"alias_{alias_mode}"] = [
            sample_episode(
                rng, signal_hops=2, task="composed", distractors=4, spec=hard,
                stage_partitions=("confirm", "confirm"), alias_mode=alias_mode)
            for _ in range(size)
        ]
    for hops in (3, 4):
        suite[f"hop_{hops}"] = [
            sample_episode(
                rng, signal_hops=hops, task="composed", distractors=4, spec=hard,
                graph_partition="confirm")
            for _ in range(size)
        ]
    suite["hop_4_long_ood"] = [
        sample_episode(
            rng, signal_hops=4, task="composed", distractors=8,
            spec=RenderSpec(0.0, 3, ("prefix", "infix", "suffix")),
            graph_partition="confirm")
        for _ in range(size)
    ]
    return suite


def training_batch(seed: int, batch_size: int, *, step: int,
                   null_composed: bool = False
                   ) -> tuple[list[Episode], list[int]]:
    """Fixed-from-step-zero task mix; each arm gets the same logical episodes."""
    if batch_size <= 0 or not 0 <= step < 6000:
        raise ValueError("Invalid TEACH-0014 training batch or step")
    rng = random.Random(seed)
    null_rng = random.Random(seed ^ 0x14C0FFEE)
    episodes: list[Episode] = []
    labels: list[int] = []
    tasks = ("first_hop", "direct", "composed", "copy")
    weights = (.20, .20, .50, .10)
    for _ in range(batch_size):
        task = rng.choices(tasks, weights=weights, k=1)[0]
        distractors = rng.randint(0, 6)
        spec = RenderSpec(
            marker_dropout=rng.choice((0.0, .25, .5, 1.0)),
            max_gaps=rng.randint(0, 2),
            styles=("prefix", "infix", "suffix"))
        episode = sample_episode(
            rng, signal_hops=2, task=task, distractors=distractors,
            spec=spec, stage_partitions=("train", "train"))
        label = episode.answer
        if null_composed and task == "composed":
            label = null_rng.choice(tuple(path[-1] for path in episode.signal_paths))
        episodes.append(episode)
        labels.append(label)
    return episodes, labels


def suite_manifest(suite: dict[str, list[Episode]], *, seed: int,
                   group_count: int) -> dict:
    """Stable, model-free evidence for a later independent suite audit."""
    if group_count <= 0:
        raise ValueError("Positive group count required")
    panels = {name: [asdict(episode) for episode in episodes]
              for name, episodes in suite.items()}
    return {
        "experiment": "TEACH-0014",
        "split_namespace": SPLIT_NAMESPACE,
        "seed": seed,
        "group_count": group_count,
        "panels": json.loads(json.dumps(panels)),
    }
