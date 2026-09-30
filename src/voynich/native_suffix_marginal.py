"""Explicitly built optional native marginal scorer for an unchanged dense source.

Compilation occurs only when build_native is called. Its output directory must
be new, and its provenance binds compiler, flags, source and binary hashes.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voynich.dense_suffix_adapter import DenseSuffixAdapter

CPP = Path(__file__).with_name('native') / 'suffix_marginal.cpp'


@dataclass(frozen=True)
class MarginalResult:
    log_likelihood: float
    reachable_nodes: int
    edges: int


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_native(directory):
    """Compile once in a new directory; never install globally or download tools."""
    directory = Path(directory).resolve()
    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        raise RuntimeError('C++ compiler unavailable; no automatic installation')
    compiler = str(Path(compiler).resolve())
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=15)
    flags = ['-std=c++17', '-O3', '-fPIC', '-ffp-contract=off',
             '-dynamiclib' if sys.platform == 'darwin' else '-shared']
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / ('suffix_marginal.dylib' if sys.platform == 'darwin' else 'suffix_marginal.so')
    command = [compiler, *flags, str(CPP), '-o', str(library)]
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    (directory / 'compiler.log').write_text(run.stdout)
    run.check_returncode()
    provenance = {'compiler': compiler, 'compiler_version': version, 'flags': flags,
                  'command': command, 'platform': platform.platform(), 'machine': platform.machine(),
                  'source_sha256': digest(CPP), 'library_path': str(library),
                  'library_sha256': digest(library), 'abi': 1}
    (directory / 'build.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    return provenance


def settings(source, units, record, rho, max_nodes, max_edges):
    units = tuple(units)
    if (len(units) != len(source.alphabet) or any(not isinstance(u, str) or not u for u in units)
            or not isinstance(record, str) or len(record) > 1_000_000
            or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
            or any(type(v) is not int or not 1 <= v <= np.iinfo(np.uint64).max for v in (max_nodes, max_edges))):
        raise ValueError('Invalid marginal lattice settings')
    return units


def marginal_python(source, units, record, rho, *, max_nodes=500_000, max_edges=2_000_000):
    """Same forward recursion as decode, without Viterbi/path storage."""
    units = settings(source, units, record, rho, max_nodes, max_edges)
    n = len(record)
    graph = [{} for _ in range(n + 1)]
    graph[0][source.state('')] = 0.
    nodes, edges, transitions = 1, 0, {}
    cont, stop = math.log1p(-rho), math.log(rho)
    for offset in range(n):
        matches = [(i, offset + len(unit)) for i, unit in enumerate(units) if record.startswith(unit, offset)]
        for state, prefix in graph[offset].items():
            for letter, end in matches:
                key = state, letter
                if key not in transitions:
                    transitions[key] = source.step(state, letter), cont + math.log(source.probabilities[state, letter])
                following, weight = transitions[key]
                edges += 1
                if edges > max_edges:
                    raise RuntimeError('Exact lattice edge cap; no pruning')
                value = prefix + weight
                if following not in graph[end]:
                    nodes += 1
                    if nodes > max_nodes:
                        raise RuntimeError('Exact lattice node cap; no pruning')
                    graph[end][following] = value
                else:
                    current = graph[end][following]
                    high, low = max(current, value), min(current, value)
                    graph[end][following] = high + math.log1p(math.exp(low - high))
        graph[offset].clear()
    if not graph[n]:
        return MarginalResult(-math.inf, nodes, edges)
    peak = max(graph[n].values())
    return MarginalResult(peak + math.log(math.fsum(math.exp(v - peak) for v in graph[n].values())) + stop, nodes, edges)


class _Result(ctypes.Structure):
    _fields_ = [('log_likelihood', ctypes.c_double), ('nodes', ctypes.c_uint64), ('edges', ctypes.c_uint64)]


class NativeMarginal:
    def __init__(self, source, build):
        if not isinstance(source, DenseSuffixAdapter):
            raise ValueError('Native marginal scorer requires a validated dense source')
        self.build = dict(build)
        self.alphabet, self.root = tuple(source.alphabet), source.state('')
        if (build['abi'] != 1 or build['source_sha256'] != digest(CPP)
                or build['library_sha256'] != digest(build['library_path'])):
            raise ValueError('Native source/library identity mismatch')
        probabilities, transitions = source.probabilities, source.transitions
        if (probabilities.dtype != np.float64 or transitions.dtype != np.uint32
                or not probabilities.flags.c_contiguous or not transitions.flags.c_contiguous
                or probabilities.ndim != 2 or probabilities.shape != transitions.shape
                or probabilities.shape[1] != len(source.alphabet)
                or not 1 <= probabilities.shape[0] <= np.iinfo(np.uint32).max
                or probabilities.flags.writeable or transitions.flags.writeable):
            raise ValueError('Dense source memory layout or immutability mismatch')
        # Keep immutable array references alive even if a caller replaces source attributes.
        self.probabilities, self.transitions = probabilities, transitions
        self.library = ctypes.CDLL(build['library_path'])
        self.library.suffix_marginal_abi.restype = ctypes.c_int
        if self.library.suffix_marginal_abi() != 1 or ctypes.sizeof(_Result) != 24:
            raise ValueError('Native ABI mismatch')
        double_pointer, uint_pointer = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32)
        self.function = self.library.suffix_marginal
        self.function.argtypes = [double_pointer, uint_pointer, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64), uint_pointer, uint_pointer, ctypes.c_uint32, ctypes.c_uint64,
            ctypes.c_double, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_Result), ctypes.c_char_p, ctypes.c_uint64]
        self.function.restype = ctypes.c_int
        self.probability_pointer = probabilities.ctypes.data_as(double_pointer)
        self.transition_pointer = transitions.ctypes.data_as(uint_pointer)

    def score(self, units, record, rho, *, max_nodes=500_000, max_edges=2_000_000):
        units = settings(self, units, record, rho, max_nodes, max_edges)
        offsets, letters, ends = [0], [], []
        for offset in range(len(record)):
            for letter, unit in enumerate(units):
                if record.startswith(unit, offset):
                    letters.append(letter)
                    ends.append(offset + len(unit))
            offsets.append(len(letters))
        offsets = np.array(offsets, dtype=np.uint64)
        letters, ends = np.array(letters, dtype=np.uint32), np.array(ends, dtype=np.uint32)
        result, error = _Result(), ctypes.create_string_buffer(256)
        code = self.function(self.probability_pointer, self.transition_pointer,
            len(self.probabilities), len(self.alphabet), self.root,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            letters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), ends.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(record), len(letters), rho, max_nodes, max_edges, ctypes.byref(result), error, len(error))
        if code:
            exception = {1: ValueError, 2: RuntimeError, 3: MemoryError, 4: ArithmeticError}.get(code, RuntimeError)
            raise exception(error.value.decode('utf-8', errors='replace'))
        return MarginalResult(result.log_likelihood, result.nodes, result.edges)
