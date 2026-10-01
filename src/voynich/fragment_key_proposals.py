"""Corpus fragments propose coherent blocks; ranking is not a posterior law."""
from __future__ import annotations

import ctypes as C
import hashlib
import itertools
import math
import platform
import subprocess
from pathlib import Path

import numpy as np


class Match(C.Structure):
    _fields_ = [("gram", C.c_int32), ("consumed", C.c_int32), ("key", C.c_int32*23)]


def build_fragment_native(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cpp = Path(__file__).with_name("native_fragment_match.cpp")
    library = directory/("fragment.dylib" if platform.system() == "Darwin" else "fragment.so")
    if library.exists():
        raise FileExistsError("Fresh build required")
    command = ["clang++", "-std=c++17", "-O3", "-fPIC", "-ffp-contract=off",
               "-dynamiclib" if platform.system() == "Darwin" else "-shared", str(cpp), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    version = subprocess.run(["clang++", "--version"], capture_output=True, text=True, check=True, timeout=10)
    return {"library_path": str(library.resolve()), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(), "command": command,
            "compiler": version.stdout, "stdout": result.stdout, "stderr": result.stderr, "abi": 1}


def gram_table(compact, length=12):
    if type(length) is not int or not 1 <= length < len(compact.levels):
        raise ValueError("Retained context depth required")
    codes = compact.levels[length]["contexts"].copy()
    table = np.empty((len(codes), length), dtype=np.int32)
    for position in reversed(range(length)):
        table[:, position] = codes % len(compact.alphabet)
        codes //= len(compact.alphabet)
    counts = compact.levels[length]["totals"].copy()
    if not len(table):
        raise ValueError("Nonempty fragment library required")
    return table, counts


class FragmentMatcher:
    def __init__(self, grams, counts, build, *, rows=23, glyphs=6):
        grams = np.asarray(grams)
        counts = np.asarray(counts)
        if (type(rows) is not int or not 1 <= rows <= 23 or type(glyphs) is not int or not 1 <= glyphs <= 6
                or grams.ndim != 2 or not 1 <= grams.shape[1] <= 12 or not len(grams)
                or len(grams) > 1_000_000 or grams.dtype.kind not in "iu"
                or np.any(grams < 0) or np.any(grams >= rows) or counts.shape != (len(grams),)
                or counts.dtype.kind not in "iu" or np.any(counts <= 0)
                or len({tuple(g) for g in grams.tolist()}) != len(grams)):
            raise ValueError("Unique integer fragments and positive integer counts required")
        self.grams = np.ascontiguousarray(grams, dtype=np.int32)
        self.counts = counts.copy()
        self.rows, self.glyphs = rows, glyphs
        used = np.array([len(set(g)) for g in self.grams.tolist()])
        # Frequency-only and count times integrated iid partial-key prior.
        frequency = np.log(counts.astype(np.float64))
        self.ranks = np.ascontiguousarray([frequency, frequency-used*math.log(glyphs+glyphs**2)])
        library = Path(build["library_path"])
        if (hashlib.sha256(library.read_bytes()).hexdigest() != build["library_sha256"]
                or hashlib.sha256(Path(__file__).with_name("native_fragment_match.cpp").read_bytes()).hexdigest() != build["cpp_sha256"]):
            raise ValueError("Pinned fragment code/library changed")
        self.library = C.CDLL(str(library))
        self.library.fragment_abi.restype = C.c_int
        if self.library.fragment_abi() != 1 or build["abi"] != 1:
            raise ValueError("Fragment ABI changed")
        self.function = self.library.match_fragments
        self.function.argtypes = [C.POINTER(C.c_int32), C.POINTER(C.c_double),
            C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.POINTER(C.c_int32),
            C.c_int32, C.c_int32, C.c_uint64, C.POINTER(Match), C.POINTER(C.c_uint64)]
        self.function.restype = C.c_int
        self.pool = tuple(tuple(p) for n in (1, 2) for p in itertools.product(range(glyphs), repeat=n))

    def match(self, cipher, *, keep=16, node_cap=50_000_000):
        cipher = tuple(cipher)
        if (not cipher or any(type(g) is not int or not 0 <= g < self.glyphs for g in cipher)
                or type(keep) is not int or not 1 <= keep <= 4096
                or type(node_cap) is not int or not 1 <= node_cap < 2**64):
            raise ValueError("Bounded literal ciphertext and caps required")
        # Beyond two glyphs per source symbol can never be consumed.
        observed = np.array(cipher[:2*self.grams.shape[1]], dtype=np.int32)
        output, stats = (Match*(2*keep))(), (C.c_uint64*3)()
        status = self.function(self.grams.ctypes.data_as(C.POINTER(C.c_int32)),
            self.ranks.ctypes.data_as(C.POINTER(C.c_double)), len(self.grams), self.grams.shape[1],
            self.rows, self.glyphs, observed.ctypes.data_as(C.POINTER(C.c_int32)), len(observed),
            keep, node_cap, output, stats)
        if status == 2:
            raise RuntimeError("Fragment node cap hit; no complete ranked output")
        if status:
            raise ValueError(f"Fragment native status {status}")
        matches = [{"gram": int(m.gram), "consumed": int(m.consumed),
                    "partial_key": list(m.key)[:self.rows]} for m in output[:stats[2]]]
        for m in matches:
            emitted = tuple(g for r in self.grams[m["gram"]] for g in self.pool[m["partial_key"][r]])
            if (any(m["partial_key"][r] < 0 for r in set(self.grams[m["gram"]]))
                    or emitted != cipher[:m["consumed"]]):
                raise ValueError("Native fragment does not emit literal prefix")
        return {"nodes": int(stats[0]), "total_matches": int(stats[1]), "matches": matches,
                "all_templates_and_bindings_scanned": True, "output_rank_truncated": stats[1] > stats[2]}


def window_offsets(records, *, length=12, anchors=16):
    if type(length) is not int or length < 1 or type(anchors) is not int or anchors < 1:
        raise ValueError("Positive window parameters required")
    windows = []
    for record, observed in enumerate(records):
        if len(observed) < length:
            continue
        last = len(observed)-length
        central = [i*last//max(1, anchors-1) for i in range(anchors)]
        windows.extend((record, offset) for offset in sorted({c+d for c in central for d in (-1, 0, 1) if 0 <= c+d <= last}))
    return tuple(windows)


def coherent_key(base, partial, pool):
    if len(base) != len(partial) or any(type(v) is not int or not -1 <= v < len(pool) for v in partial):
        raise ValueError("Compatible partial/full key dimensions required")
    return tuple(base[i] if p == -1 else pool[p] for i, p in enumerate(partial))
