"""String-context view for the separate backward decoder; no source-state IDs."""
from collections.abc import Mapping
from functools import lru_cache

import numpy as np


class LiteralCountView(Mapping):
    def __init__(self, model, cache_size=200000):
        self.model = model
        self.get_row = lru_cache(maxsize=cache_size)(self._row)

    def _locate(self, context):
        if not isinstance(context, str) or len(context) >= len(self.model.levels):
            raise KeyError(context)
        code = 0
        for char in context:
            if char not in self.model.alphabet:
                raise KeyError(context)
            code = code * len(self.model.alphabet) + self.model.alphabet.index(char)
        level = self.model.levels[len(context)]
        index = int(np.searchsorted(level['contexts'], np.uint64(code)))
        if index >= len(level['contexts']) or int(level['contexts'][index]) != code:
            raise KeyError(context)
        return level, code

    def _row(self, context):
        level, code = self._locate(context)
        base = len(self.model.alphabet)
        first, end = np.searchsorted(level['joints'], np.array([base * code, base * (code + 1)], dtype=np.uint64))
        return {self.model.alphabet[int(c) % base]: int(n)
                for c, n in zip(level['joints'][first:end], level['frequencies'][first:end])}

    def __getitem__(self, context):
        return self.get_row(context)

    def __contains__(self, context):
        try:
            self._locate(context)
            return True
        except KeyError:
            return False

    def __len__(self):
        return sum(len(level['contexts']) for level in self.model.levels)

    def __iter__(self):
        for depth, level in enumerate(self.model.levels):
            for code in level['contexts']:
                code, text = int(code), []
                for _ in range(depth):
                    code, letter = divmod(code, len(self.model.alphabet))
                    text.append(self.model.alphabet[letter])
                yield ''.join(reversed(text))
