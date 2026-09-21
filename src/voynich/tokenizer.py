"""Train-only EVA codepoint tokenizer, not a claim about manuscript glyph units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ALTERNATIVE = "\ue000"
UNREADABLE = "\ue001"
UNKNOWN_SPAN = "\ue002"
UNCERTAIN_SPACE = "\ue003"
RARE_BASE = 0xE100

SPECIAL_TOKENS = (
    "<pad>", "<unk>", "<bos>", "<eos>", "<space>", "<line>",
    "<alternative>", "<unreadable>", "<unknown_span>", "<uncertain_space>",
)
CHAR_SPECIAL = {
    " ": "<space>", "\n": "<line>", ALTERNATIVE: "<alternative>",
    UNREADABLE: "<unreadable>", UNKNOWN_SPAN: "<unknown_span>",
    UNCERTAIN_SPACE: "<uncertain_space>",
}


class EVATokenizer:
    """Each normalized EVA codepoint is a token; high-ASCII escapes are atomic.

    Fit only on training-page texts. All control IDs have fixed semantics; unseen
    ordinary/rare characters map to <unk>. Uncertainty IDs should be excluded
    from primary text-prediction targets (they remain visible as context).
    """

    def __init__(self, tokens: Iterable[str]):
        self.tokens = list(tokens)
        if self.tokens[:len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError("Tokenizer must start with the fixed special vocabulary")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate vocabulary entries")
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "EVATokenizer":
        chars = set().union(*(set(text) for text in texts))
        return cls([*SPECIAL_TOKENS, *sorted(chars - CHAR_SPECIAL.keys())])

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    pad_id = property(lambda self: self.token_to_id["<pad>"])
    unk_id = property(lambda self: self.token_to_id["<unk>"])
    bos_id = property(lambda self: self.token_to_id["<bos>"])
    eos_id = property(lambda self: self.token_to_id["<eos>"])
    space_id = property(lambda self: self.token_to_id["<space>"])
    line_id = property(lambda self: self.token_to_id["<line>"])

    @property
    def uncertainty_ids(self) -> set[int]:
        return {self.unk_id, *(self.token_to_id[s] for s in SPECIAL_TOKENS[6:])}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(CHAR_SPECIAL.get(c, c), self.unk_id) for c in text]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        inverse = {v: k for k, v in CHAR_SPECIAL.items()}
        output = []
        for index in ids:
            token = self.tokens[index]
            if token in inverse:
                output.append(inverse[token])
            elif token == "<unk>":
                output.append("\ufffd")
            elif token not in SPECIAL_TOKENS or not skip_special:
                output.append(token)
        return "".join(output)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"schema_version": 1, "unit": "normalized_eva_codepoint",
                                         "tokens": self.tokens}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EVATokenizer":
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != 1 or payload.get("unit") != "normalized_eva_codepoint":
            raise ValueError("Unsupported tokenizer format")
        return cls(payload["tokens"])
