"""Pinned C++ DAG engine, separate from the frozen full-key likelihood kernel."""
from __future__ import annotations

import ctypes as C
import hashlib
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from voynich.source_key_particles import validate_source


class Terminal(C.Structure):
    _fields_ = [("key", C.c_int32*23), ("mass", C.c_double), ("best", C.c_double)]


class Stats(C.Structure):
    _fields_ = [(field, C.c_uint64) for field in
        ("expanded", "generated", "merged", "pruned", "peak_active", "peak_layer",
         "terminals", "returned", "tables", "trace_count")]+[(field, C.c_double) for field in
        ("found", "returned_mass", "lost_upper", "evidence_upper")]+[("complete", C.c_int32), ("stop", C.c_int32)]


def build_source_state_native(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cpp = Path(__file__).with_name("native_source_state.cpp")
    library = directory/("source_state.dylib" if platform.system() == "Darwin" else "source_state.so")
    if library.exists():
        raise FileExistsError("Fresh namespace build required")
    command = ["clang++", "-std=c++17", "-O3", "-fPIC", "-ffp-contract=off",
               "-dynamiclib" if platform.system() == "Darwin" else "-shared", str(cpp), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    version = subprocess.run(["clang++", "--version"], capture_output=True, text=True, check=True, timeout=10)
    return {"abi": 1, "library_path": str(library.resolve()),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(), "command": command,
            "compiler": version.stdout, "stdout": result.stdout, "stderr": result.stderr}


class NativeStateLattice:
    def __init__(self, probabilities, transitions, build):
        validate_source(probabilities, transitions)
        if (probabilities.dtype != np.float64 or transitions.dtype != np.uint32
                or not probabilities.flags.c_contiguous or not transitions.flags.c_contiguous
                or probabilities.shape[1] > 23 or len(probabilities) >= 2**32):
            raise ValueError("Contiguous double/uint32 source and bounded dimensions required")
        self.probabilities, self.transitions = probabilities, transitions
        library = Path(build["library_path"])
        cpp = Path(__file__).with_name("native_source_state.cpp")
        if (hashlib.sha256(library.read_bytes()).hexdigest() != build["library_sha256"]
                or hashlib.sha256(cpp.read_bytes()).hexdigest() != build["cpp_sha256"]):
            raise ValueError("Pinned state code/library changed")
        self.library = C.CDLL(str(library))
        self.library.source_state_abi.restype = C.c_int
        if self.library.source_state_abi() != build["abi"] or build["abi"] != 1:
            raise ValueError("Source state ABI changed")
        self.function = self.library.source_state_search
        self.function.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_uint32), C.c_uint32, C.c_int32,
            C.POINTER(C.c_int32), C.c_int32, C.POINTER(C.c_int32), C.c_int32, C.c_double,
            *([C.c_uint64]*5), *([C.c_int32]*3), *([C.c_uint64]*3), C.c_double,
            C.POINTER(Terminal), C.POINTER(Stats), C.POINTER(C.c_uint64)]
        self.function.restype = C.c_int

    def search(self, cipher, *, glyphs=6, rho=1/225, width=4096,
               max_expanded=1_000_000, max_generated=40_000_000, max_active=500_000,
               max_terminals=4096, merge=True, schedule="balanced", guidance="none",
               guide_prewidth=None, guide_max_tables=1_000_000, guide_cache_entries=1024,
               max_seconds=120.):
        cipher = tuple(map(tuple, cipher))
        guide_prewidth = 4*width if guide_prewidth is None else guide_prewidth
        if (type(glyphs) is not int or not 1 <= glyphs <= 6 or not 1 <= len(cipher) <= 2
                or any(not c or len(c)>4096 or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)
                or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
                or any(type(v) is not int or not 1 <= v <= 10**12 for v in
                       (width, max_expanded, max_generated, max_active, max_terminals, guide_prewidth, guide_max_tables, guide_cache_entries))
                or width > 1_000_000 or max_terminals > 100_000 or guide_cache_entries > 4096
                or type(merge) is not bool or schedule not in ("balanced", "sequential") or guidance not in ("none", "iid")
                or isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600):
            raise ValueError("Bounded literal source-state search configuration required")
        lengths = np.array(list(map(len, cipher)), dtype=np.int32)
        observation = np.array([g for c in cipher for g in c], dtype=np.int32)
        output, stats = (Terminal*max_terminals)(), Stats()
        trace = (C.c_uint64*(8*(int(lengths.sum())+1)))()
        status = self.function(self.probabilities.ctypes.data_as(C.POINTER(C.c_double)),
            self.transitions.ctypes.data_as(C.POINTER(C.c_uint32)), len(self.probabilities), self.probabilities.shape[1],
            observation.ctypes.data_as(C.POINTER(C.c_int32)), len(cipher), lengths.ctypes.data_as(C.POINTER(C.c_int32)),
            glyphs, rho, width, max_expanded, max_generated, max_active, max_terminals,
            int(merge), int(schedule == "balanced"), int(guidance == "iid"),
            guide_max_tables, guide_cache_entries, guide_prewidth, max_seconds, output, C.byref(stats), trace)
        if status == 2:
            raise MemoryError("Native unpruned pending-state allocation cap")
        if status == 3:
            raise RuntimeError("Native iid guide table work cap")
        if status:
            raise RuntimeError(f"Native source-state failure {status}")
        terminals = [{"used_key": tuple(item.key)[:self.probabilities.shape[1]],
            "log_mass": item.mass, "best_leaf_log_mass": item.best} for item in output[:stats.returned]]
        trace_names = ("glyph_layer", "incoming_states", "retained_states", "expanded", "generated",
                       "merged_arrivals", "pruned_states", "pending_states")
        history = [{name: int(trace[8*i+j]) for j, name in enumerate(trace_names)} for i in range(stats.trace_count)]
        for value in (stats.found, stats.returned_mass, stats.lost_upper, stats.evidence_upper):
            if math.isnan(value) or value == math.inf:
                raise ArithmeticError("Invalid returned native mass")
        return {"terminals": terminals, "terminal_states": int(stats.terminals),
            "terminal_output_truncated": stats.returned < stats.terminals,
            "found_log_mass": stats.found, "returned_terminal_log_mass": stats.returned_mass,
            "unresolved_log_mass_upper": stats.lost_upper, "evidence_log_upper": stats.evidence_upper,
            "complete_search": bool(stats.complete), "interval_arithmetic_certificate": False,
            "full_key_posterior_claimed": False,
            "stop_reason": ("frontier_exhausted", "expansion_cap", "generated_preflight_cap", "time_cap")[stats.stop],
            "expanded": int(stats.expanded), "generated": int(stats.generated), "merged_arrivals": int(stats.merged),
            "pruned_states": int(stats.pruned), "maximum_active_states": int(stats.peak_active),
            "maximum_unpruned_layer": int(stats.peak_layer), "guide_tables_built": int(stats.tables), "trace": history,
            "merge": merge, "schedule": schedule, "guidance": guidance, "guide_prewidth": guide_prewidth,
            "scope": "Compiled Markov history sums; finite beam loses mass. No full-key posterior, reading MAP, interval certificate or historical recovery claim."}
