"""Versioned observations and joint latent configurations, with no oracle side channels."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch


SCHEMA_VERSION = 1
GRAMMARS = ("SOV", "SVO", "VSO", "VOS", "OSV", "OVS")
MORPHOLOGIES = ("suffix", "prefix", "none")
FAMILIES = ("procedure", "taxonomy", "copy")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def object_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Anchor:
    """Explicit known symbol correspondence; its provenance is part of the evidence."""

    observed: int
    canonical: int
    source_group: str

    def __post_init__(self):
        if (
            not integer(self.observed)
            or not integer(self.canonical)
            or min(self.observed, self.canonical) < 2
        ):
            raise ValueError("Anchor symbols must be integers >=2")
        if not isinstance(self.source_group, str) or not self.source_group:
            raise ValueError("Anchors require a named independent source group")


@dataclass(frozen=True)
class Observation:
    document_id: str
    symbols: tuple[int, ...]
    alphabet_size: int
    anchors: tuple[Anchor, ...] = ()
    source_group: str = ""
    split: str = "unassigned"

    def __post_init__(self):
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        if (
            not isinstance(self.document_id, str)
            or not isinstance(self.source_group, str)
            or not self.document_id
            or not self.source_group
        ):
            raise ValueError("Document id and source group are required")
        if not integer(self.alphabet_size) or self.alphabet_size < 2:
            raise ValueError("Invalid alphabet size")
        if not self.symbols or any(
            not integer(x) or not 2 <= x < self.alphabet_size + 2 for x in self.symbols
        ):
            raise ValueError("Observation contains an invalid or reserved symbol")
        if self.split not in {"train", "validation", "test", "unassigned"}:
            raise ValueError("Invalid split")
        mapping = {}
        for anchor in self.anchors:
            if (
                not isinstance(anchor, Anchor)
                or max(anchor.observed, anchor.canonical) >= self.alphabet_size + 2
            ):
                raise ValueError("Anchor is outside the alphabet")
            if anchor.observed in mapping:
                raise ValueError("Duplicate or conflicting anchor")
            if anchor.source_group == self.source_group:
                raise ValueError("An anchor cannot claim independence from its own text source")
            mapping[anchor.observed] = anchor.canonical

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        expected = {
            "schema_version",
            "document_id",
            "symbols",
            "alphabet_size",
            "anchors",
            "source_group",
            "split",
        }
        if set(value) != expected or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported or non-observation fields; gold targets must remain separate")
        return cls(
            **{
                **{k: v for k, v in value.items() if k not in {"schema_version", "anchors"}},
                "anchors": tuple(Anchor(**a) for a in value["anchors"]),
            }
        )

    @property
    def digest(self):
        return object_digest(self.to_dict())


@dataclass(frozen=True)
class JointLayout:
    """Global inverse key, aligned keep/boundary decisions, and linguistic family variables."""

    alphabet_size: int
    max_observation: int
    max_anchors: int = 4

    def __post_init__(self):
        if any(not integer(x) for x in (self.alphabet_size, self.max_observation, self.max_anchors)):
            raise ValueError("Layout sizes must be integers")
        if self.alphabet_size < 6 or self.max_observation < 1 or self.max_anchors < 0:
            raise ValueError("Invalid layout capacity")

    @property
    def latent_length(self):
        return self.alphabet_size + 2 * self.max_observation + 3

    @property
    def condition_length(self):
        return self.max_observation + 1 + 3 * self.max_anchors

    @property
    def vocab_size(self):
        return self.alphabet_size + 4

    @property
    def condition_vocab_size(self):
        return 2 * self.alphabet_size + 6

    @property
    def key_slice(self):
        return slice(0, self.alphabet_size)

    @property
    def keep_slice(self):
        return slice(self.alphabet_size, self.alphabet_size + self.max_observation)

    @property
    def boundary_slice(self):
        return slice(self.alphabet_size + self.max_observation, self.alphabet_size + 2 * self.max_observation)

    @property
    def global_start(self):
        return self.latent_length - 3

    def tensors(self, observations, device="cpu"):
        if not observations:
            raise ValueError("Empty batch")
        batch = len(observations)
        condition = torch.zeros((batch, self.condition_length), dtype=torch.long)
        allowed = torch.zeros((batch, self.latent_length, self.vocab_size), dtype=torch.bool)
        fixed = torch.full((batch, self.latent_length), -1, dtype=torch.long)
        types = torch.zeros((batch, self.latent_length), dtype=torch.long)
        types[:, self.keep_slice] = 1
        types[:, self.boundary_slice] = 2
        types[:, self.global_start :] = torch.tensor([3, 4, 5])
        allowed[:, self.key_slice, 2] = True  # 0/null maps to latent token 2.
        allowed[:, self.key_slice, 4:] = True  # canonical symbols start at 2.
        allowed[:, self.keep_slice, 2:4] = True
        allowed[:, self.boundary_slice, 2:4] = True
        for offset, count in enumerate((len(GRAMMARS), len(MORPHOLOGIES), len(FAMILIES))):
            allowed[:, self.global_start + offset, 2 : 2 + count] = True
        for row, obs in enumerate(observations):
            if obs.alphabet_size != self.alphabet_size or len(obs.symbols) > self.max_observation:
                raise ValueError("Observation exceeds layout; truncation is never implicit")
            if len(obs.anchors) > self.max_anchors:
                raise ValueError("Too many anchors")
            n = len(obs.symbols)
            known_symbols = set(obs.symbols) | {a.observed for a in obs.anchors}
            for key_index in range(self.alphabet_size):
                if key_index + 2 not in known_symbols:
                    fixed[row, key_index] = 0
                    allowed[row, key_index] = False
                    allowed[row, key_index, 0] = True
            condition[row, :n] = torch.tensor(obs.symbols)
            condition[row, self.max_observation] = self.alphabet_size + 2
            for i, anchor in enumerate(obs.anchors):
                start = self.max_observation + 1 + 3 * i
                condition[row, start : start + 3] = torch.tensor(
                    [self.alphabet_size + 3, anchor.observed, self.alphabet_size + 4 + anchor.canonical]
                )
                fixed[row, anchor.observed - 2] = anchor.canonical + 2
            for block in (self.keep_slice, self.boundary_slice):
                padding = slice(block.start + n, block.stop)
                fixed[row, padding] = 0
                allowed[row, padding] = False
                allowed[row, padding, 0] = True
        return {
            "condition": condition.to(device),
            "allowed": allowed.to(device),
            "fixed": fixed.to(device),
            "latent_types": types.to(device),
        }

    def pack(self, inverse_key, keep, boundaries, *, grammar, morphology, family):
        if len(inverse_key) != self.alphabet_size or len(keep) != len(boundaries):
            raise ValueError("Incorrect target shape")
        if not 0 < len(keep) <= self.max_observation:
            raise ValueError("Target length exceeds layout")
        if any(not integer(v) or v == 1 or not 0 <= v < self.alphabet_size + 2 for v in inverse_key):
            raise ValueError("Invalid inverse key")
        if any(type(v) not in (bool, int) or v not in (0, 1) for v in (*keep, *boundaries)):
            raise ValueError("Masks must be binary")
        clean = torch.zeros(self.latent_length, dtype=torch.long)
        clean[self.key_slice] = torch.tensor(inverse_key) + 2
        clean[self.keep_slice.start : self.keep_slice.start + len(keep)] = (
            torch.tensor(keep, dtype=torch.long) + 2
        )
        clean[self.boundary_slice.start : self.boundary_slice.start + len(keep)] = (
            torch.tensor(boundaries, dtype=torch.long) + 2
        )
        clean[self.global_start :] = torch.tensor(
            [GRAMMARS.index(grammar) + 2, MORPHOLOGIES.index(morphology) + 2, FAMILIES.index(family) + 2]
        )
        return clean

    def unpack(self, tokens, observation):
        values = tokens.detach().cpu().tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        if len(values) != self.latent_length or any(not integer(x) for x in values):
            raise ValueError("Malformed latent configuration")
        inputs = self.tensors([observation])
        for i, value in enumerate(values):
            if not 0 <= value < self.vocab_size or not bool(inputs["allowed"][0, i, value]):
                raise ValueError("Candidate is outside the field domain")
            fixed = int(inputs["fixed"][0, i])
            if fixed >= 0 and fixed != value:
                raise ValueError("Candidate violates an immutable anchor or padding")
        n = len(observation.symbols)
        return {
            "inverse_key": tuple(v - 2 if v else 0 for v in values[self.key_slice]),
            "unresolved_key_symbols": tuple(i + 2 for i, v in enumerate(values[self.key_slice]) if v == 0),
            "keep": tuple(bool(v - 2) for v in values[self.keep_slice][:n]),
            "boundaries": tuple(bool(v - 2) for v in values[self.boundary_slice][:n]),
            "grammar": GRAMMARS[values[-3] - 2],
            "morphology": MORPHOLOGIES[values[-2] - 2],
            "family": FAMILIES[values[-1] - 2],
        }


def save_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(destination)


def checked_log_probability(probability):
    if (
        not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Invalid probability")
    return math.log(probability) if probability else -math.inf
