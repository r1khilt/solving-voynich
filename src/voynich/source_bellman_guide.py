"""Proper-edge Bellman backups of a heuristic tail; never new edge weights."""
from __future__ import annotations

import ctypes as C
import hashlib
import itertools
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from voynich.native_source_state import NativeStateLattice
from voynich.source_key_particles import validate_source
from voynich.source_state_lattice import State, selected_record
from voynich.source_prefix_inverse import logsum

BASE_SHA = "51bc370ea6ac4ca903a2a03270ea8dc9bedb83d1419ec6a7870d6ecbb146a8eb"


def actions(state, cipher, p, goto, *, glyphs=6, rho=1/225, schedule="balanced"):
    record = selected_record(state.offsets, tuple(map(len, cipher)), schedule)
    if record is None:
        return []
    pool = tuple(x for n in (1, 2) for x in itertools.product(range(glyphs), repeat=n))
    result = []
    for row in range(p.shape[1]):
        probability = float(p[state.contexts[record], row])
        if not probability:
            continue
        for n in (1, 2):
            offset = state.offsets[record]
            unit = tuple(cipher[record][offset:offset+n])
            if len(unit) != n:
                continue
            code = pool.index(unit)
            if state.key[row] >= 0 and state.key[row] != code:
                continue
            offsets, contexts, key = list(state.offsets), list(state.contexts), list(state.key)
            offsets[record] += n
            contexts[record] = int(goto[contexts[record], row])
            edge = math.log1p(-rho)+math.log(probability)
            if key[row] < 0:
                key[row], edge = code, edge-math.log(len(pool))
            if offsets[record] == len(cipher[record]):
                contexts[record], edge = 0, edge+math.log(rho)
            result.append((State(tuple(offsets), tuple(contexts), tuple(key)), edge))
    return result


def backup(state, depth, cipher, p, goto, tail, **law):
    """No pruning or floor in this mathematical operator; empty sums are zero."""
    if selected_record(state.offsets, tuple(map(len, cipher)), law.get("schedule", "balanced")) is None:
        return 0.
    if depth == 0:
        return tail(state)
    return logsum(edge+backup(child, depth-1, cipher, p, goto, tail, **law)
                  for child, edge in actions(state, cipher, p, goto, **law))


HEADER = r'''
struct LookaheadConfig { int depth=0,layers=0; uint64_t calls=0,actions=0; };
static thread_local LookaheadConfig lookahead;
extern "C" int source_bellman_configure(int depth,int layers) {
  if(depth<0 || depth>2 || layers<0 || layers>8192) return 1;
  lookahead=LookaheadConfig{};lookahead.depth=depth;lookahead.layers=layers;return 0;
}
extern "C" void source_bellman_stats(uint64_t* out) { out[0]=lookahead.calls;out[1]=lookahead.actions; }
struct BackedGuide {
  IidGuide& tail; const double* p; const uint32_t* next;
  const int32_t* cipher;std::array<int32_t,2> lengths;
  int records,rows,glyphs;bool balanced;double rho;
  double value(const State& s,int depth) {
    int record=-1;
    for(int r=0;r<records;++r) if(s.off[r]<lengths[r] && (record<0 ||
      (balanced && uint64_t(s.off[r])*lengths[record]<uint64_t(s.off[record])*lengths[r]))) record=r;
    if(record<0) return 0.;
    if(!depth) return tail(s);
    double total=-INFINITY;int offset=s.off[record],start=record ? lengths[0] : 0;
    for(int row=0;row<rows;++row) {
      double probability=p[size_t(s.ctx[record])*rows+row];if(!probability) continue;
      for(int n=1;n<=2;++n) {
        if(offset+n>lengths[record]) continue;
        int code=n==1 ? cipher[start+offset] : glyphs+glyphs*cipher[start+offset]+cipher[start+offset+1];
        if(s.key[row]>=0 && s.key[row]!=code) continue;
        State child=s;child.off[record]+=n;child.ctx[record]=next[size_t(s.ctx[record])*rows+row];
        double edge=std::log1p(-rho)+std::log(probability);
        if(s.key[row]<0) {child.key[row]=int8_t(code);edge-=std::log(glyphs+glyphs*glyphs);}
        if(child.off[record]==lengths[record]) {child.ctx[record]=0;edge+=std::log(rho);}
        if(++lookahead.actions>64000000ULL) throw std::length_error("lookahead action cap");
        total=add(total,edge+value(child,depth-1));
      }
    }
    return total;
  }
  double operator()(const State& s) {
    if(lookahead.depth && s.off[0]+s.off[1]<lookahead.layers) {
      ++lookahead.calls;return std::max(-2000.,value(s,lookahead.depth));
    }
    return tail(s);
  }
};
extern "C" int source_bellman_value(const double* p,const uint32_t* next,int rows,
    const int32_t* cipher,const int32_t* lengths,int records,int glyphs,double rho,
    const int64_t* input,int depth,int balanced,double* out) {
  try {
    std::array<int32_t,2> n{lengths[0],records==2 ? lengths[1] : 0};
    IidGuide tail(cipher,n,records,rows,glyphs,p,rho,100000,64);
    BackedGuide guide{tail,p,next,cipher,n,records,rows,glyphs,bool(balanced),rho};
    State s;s.off={uint16_t(input[0]),uint16_t(input[1])};s.ctx={uint32_t(input[2]),uint32_t(input[3])};
    for(int r=0;r<23;++r) s.key[r]=int8_t(input[4+r]);
    *out=guide.value(s,depth);return 0;
  } catch(const std::exception&) { return 1; }
}
'''


def instrument(text):
    if hashlib.sha256(text.encode()).hexdigest() != BASE_SHA:
        raise ValueError("Frozen source-state dependency changed")
    replacements = (
        ('extern "C" int source_state_abi()', HEADER+'\nextern "C" int source_state_abi()'),
        ('    uint64_t serial=0,active=1;', '    BackedGuide backed{guide,p,goto_state,cipher,lengths,records,rows,glyphs,bool(balanced),rho};\n    uint64_t serial=0,active=1;'),
        ('candidate.value.mass+guide(candidate.state)', 'candidate.value.mass+backed(candidate.state)'),
    )
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError("Unique source insertion required")
        text = text.replace(before, after)
    return text


def build_bellman(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = directory/"source_bellman.cpp"
    library = directory/("source_bellman.dylib" if platform.system() == "Darwin" else "source_bellman.so")
    if body.exists() or library.exists():
        raise FileExistsError("Fresh compiled namespace required")
    body.write_text(instrument(Path(__file__).with_name("native_source_state.cpp").read_text()))
    command = ["clang++", "-std=c++17", "-O3", "-fPIC", "-ffp-contract=off", "-dynamiclib" if platform.system() == "Darwin" else "-shared", str(body), "-o", str(library)]
    run = subprocess.run(command, text=True, capture_output=True, timeout=120, check=True)
    version = subprocess.run(["clang++", "--version"], text=True, capture_output=True, timeout=10, check=True)
    return {"abi": 1, "cpp_sha256": BASE_SHA, "cpp_sha256_is_frozen_base_dependency": True,
        "generated_cpp_path": str(body.resolve()), "generated_cpp_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library_path": str(library.resolve()), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "command": command, "compiler": version.stdout, "stdout": run.stdout, "stderr": run.stderr}


class NativeBellmanGuide(NativeStateLattice):
    def __init__(self, p, goto, build):
        if (hashlib.sha256(Path(build["generated_cpp_path"]).read_bytes()).hexdigest() != build["generated_cpp_sha256"]
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != build["module_sha256"]):
            raise ValueError("Pinned generated guide/module changed")
        super().__init__(p, goto, build)
        self.configure = self.library.source_bellman_configure
        self.configure.argtypes, self.configure.restype = [C.c_int, C.c_int], C.c_int
        self.stats = self.library.source_bellman_stats
        self.stats.argtypes, self.stats.restype = [C.POINTER(C.c_uint64)], None
        self.value_function = self.library.source_bellman_value
        self.value_function.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_uint32), C.c_int,
            C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.c_int, C.c_int, C.c_double,
            C.POINTER(C.c_int64), C.c_int, C.c_int, C.POINTER(C.c_double)]
        self.value_function.restype = C.c_int

    def search(self, cipher, *, lookahead_depth=1, lookahead_layers=16, **cfg):
        if type(lookahead_depth) is not int or not 0 <= lookahead_depth <= 2 or type(lookahead_layers) is not int or not 0 <= lookahead_layers <= 8192:
            raise ValueError("Bounded depth and early layer cutoff required")
        if self.configure(lookahead_depth, lookahead_layers):
            raise ValueError("Native guide configure failed")
        try:
            result = super().search(cipher, **cfg)
            counts = (C.c_uint64*2)()
            self.stats(counts)
            return {**result, "lookahead_depth": lookahead_depth, "lookahead_layers": lookahead_layers,
                "lookahead_calls": int(counts[0]), "lookahead_actions": int(counts[1]),
                "tail_is_heuristic_with_floor": True}
        finally:
            self.configure(0, 0)

    def value(self, cipher, state, depth, *, glyphs=2, rho=.25, schedule="balanced"):
        # An engineering arithmetic oracle, not a full-size inference API.
        validate_source(self.probabilities, self.transitions)
        cipher = tuple(map(tuple, cipher))
        if (type(depth) is not int or not 0 <= depth <= 8 or not 1 <= len(cipher) <= 2
                or type(glyphs) is not int or not 1 <= glyphs <= 6
                or any(not c or len(c)>8 or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)
                or isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1
                or schedule not in ("balanced", "sequential") or self.probabilities.shape[1] > 3
                or len(state.offsets) != len(cipher) or len(state.contexts) != len(cipher)
                or len(state.key) != self.probabilities.shape[1]
                or any(type(x) is not int or not 0 <= x <= len(c) for x,c in zip(state.offsets,cipher,strict=True))
                or any(type(x) is not int or not 0 <= x < len(self.probabilities) for x in state.contexts)
                or any(type(x) is not int or not -1 <= x < glyphs+glyphs**2 for x in state.key)):
            raise ValueError("Tiny bounded value state required")
        observation = np.array([g for c in cipher for g in c], dtype=np.int32)
        lengths = np.array(list(map(len, cipher)), dtype=np.int32)
        table = np.array(list(state.offsets)+[0]*(2-len(cipher))+list(state.contexts)+[0]*(2-len(cipher))+list(state.key)+[-1]*(23-len(state.key)), dtype=np.int64)
        out = C.c_double()
        self.configure(0, 0)
        status = self.value_function(self.probabilities.ctypes.data_as(C.POINTER(C.c_double)),
            self.transitions.ctypes.data_as(C.POINTER(C.c_uint32)), self.probabilities.shape[1],
            observation.ctypes.data_as(C.POINTER(C.c_int32)), lengths.ctypes.data_as(C.POINTER(C.c_int32)),
            len(cipher), glyphs, rho, table.ctypes.data_as(C.POINTER(C.c_int64)), depth, int(schedule=="balanced"), C.byref(out))
        if status:
            raise RuntimeError("Tiny native value cap/failure")
        return out.value
