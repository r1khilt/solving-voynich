"""Separate observational build of the frozen source-state engine.

Known-answer paths annotate ranks/survival; they NEVER change candidates,
edge weights, ordering, original guide/cache or terminal selection. The
generated C++ and its frozen base are both pinned. No gold reinsertion.
"""
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
from voynich.source_state_lattice import selected_record

BASE_SHA = "51bc370ea6ac4ca903a2a03270ea8dc9bedb83d1419ec6a7870d6ecbb146a8eb"
FIELDS = ("layer", "incoming", "aliases", "path_present", "alive_before", "alive_after",
    "pre_rank", "alias_best_pre_rank", "pre_kept", "guided_candidates", "guide_rank", "final_kept",
    "ghost_pre_better", "ghost_pre_equal", "ghost_guide_better", "ghost_guide_equal",
    "path_expanded", "terminal_group_rank", "terminal_group_returned",
    "terminal_used_mapping_best_rank", "terminal_used_mapping_returned", "observer_tables_built")
FLOATS = ("prefix_log_mass", "state_log_mass", "pre_priority", "guide_priority",
          "ghost_pre_priority", "ghost_guide_priority")


class Diagnostic(C.Structure):
    _fields_ = [(f, C.c_uint64) for f in FIELDS]+[(f, C.c_double) for f in FLOATS]


HEADER = r'''
struct Diagnostic {
  uint64_t layer,incoming,aliases,path_present,alive_before,alive_after,pre_rank,alias_best_pre_rank,
    pre_kept,guided_candidates,guide_rank,final_kept,ghost_pre_better,ghost_pre_equal,
    ghost_guide_better,ghost_guide_equal,path_expanded,terminal_group_rank,terminal_group_returned,
    terminal_used_mapping_best_rank,terminal_used_mapping_returned,observer_tables_built;
  double prefix_log_mass,state_log_mass,pre_priority,guide_priority,ghost_pre_priority,ghost_guide_priority;
};
struct Observer {
  const int64_t* gold=nullptr; const double* prefix=nullptr; Diagnostic* output=nullptr;
  int horizon=0; bool alive=true; uint64_t id=0;
  bool valid(int layer) const { return gold && gold[30*layer+29]; }
  State target(int layer) const {
    State s; const int64_t* t=gold+30*layer;
    s.off={uint16_t(t[0]),uint16_t(t[1])};s.ctx={uint32_t(t[2]),uint32_t(t[3])};
    for(int r=0;r<23;++r) s.key[r]=int8_t(t[4+r]);s.id=id;return s;
  }
  bool same(const State& s,int layer) const {
    State t=target(layer);return s.off==t.off && s.ctx==t.ctx && s.key==t.key;
  }
  Diagnostic* begin(int layer,uint64_t incoming) {
    if(!valid(layer)) return nullptr;
    Diagnostic* d=output+layer;*d=Diagnostic{};d->layer=layer;d->incoming=incoming;
    d->alive_before=alive;d->prefix_log_mass=prefix[layer];
    d->state_log_mass=d->pre_priority=d->guide_priority=NAN;
    d->ghost_pre_priority=d->ghost_guide_priority=NAN;return d;
  }
};
static thread_local Observer observer;
extern "C" int source_prune_configure(const int64_t* gold,const double* prefix,Diagnostic* out,int horizon) {
  observer=Observer{};
  if(!gold && !prefix && !out) return 0;
  if(!gold || !prefix || !out || horizon<1 || horizon>8192) return 1;
  observer.gold=gold;observer.prefix=prefix;observer.output=out;observer.horizon=horizon;return 0;
}
'''

PRE = r'''
      Diagnostic* d=observer.begin(layer,incoming);
      const uint64_t parent_gold_id=observer.id;
      State goal;
      auto is_path=[&](const State& s) { return d && observer.alive && observer.same(s,layer) && (merge || s.id==parent_gold_id); };
      if(d) {
        goal=observer.target(layer);
        d->ghost_pre_priority=d->prefix_log_mass+bound(goal);
        d->ghost_guide_priority=d->prefix_log_mass+(guided ? audit_guide(goal) : bound(goal));
        d->observer_tables_built=audit_guide.built;
        const Candidate* actual=nullptr;
        for(const auto& c:candidates) {
          d->ghost_pre_better+=c.rank>d->ghost_pre_priority;d->ghost_pre_equal+=c.rank==d->ghost_pre_priority;
          if(observer.same(c.state,layer)) {
            ++d->aliases;uint64_t rank=1;
            for(const auto& b:candidates) rank+=better(b,c);
            if(!d->alias_best_pre_rank || rank<d->alias_best_pre_rank) d->alias_best_pre_rank=rank;
          }
          if(is_path(c.state)) actual=&c;
        }
        if(actual) {
          d->path_present=1;d->state_log_mass=actual->value.mass;d->pre_priority=actual->rank;
          d->pre_rank=1;for(const auto& c:candidates) d->pre_rank+=better(c,*actual);
        }
        if(observer.alive && !actual) return 5;
      }
'''

POST_GUIDE = r'''
      if(d) {
        d->guided_candidates=candidates.size();
        const Candidate* actual=nullptr;
        for(const auto& c:candidates) {
          d->ghost_guide_better+=c.rank>d->ghost_guide_priority;d->ghost_guide_equal+=c.rank==d->ghost_guide_priority;
          if(is_path(c.state)) actual=&c;
        }
        d->pre_kept=actual!=nullptr;
        if(actual) {
          d->guide_priority=actual->rank;d->guide_rank=1;
          for(const auto& c:candidates) d->guide_rank+=better(c,*actual);
        }
      }
'''

POST_PRUNE = r'''
      if(d) {
        for(const auto& c:candidates) if(is_path(c.state)) d->final_kept=1;
        observer.alive=observer.alive && d->final_kept;d->alive_after=observer.alive;
      }
'''

CHILD = r'''
            if(d && observer.alive && observer.same(parent,layer) && (merge || parent.id==parent_gold_id) &&
                row==observer.gold[30*layer+27] && n==observer.gold[30*layer+28]) {
              if(!observer.valid(rank) || !observer.same(child,rank)) return 5;
              observer.id=child.id;d->path_expanded=1;
            }
'''

TERMINAL = r'''
        Diagnostic* d=observer.begin(layer,incoming);
        if(d) {
          for(const auto& item:here) if(observer.same(item.first,layer)) {
            ++d->aliases;
            if(observer.alive && (merge || item.first.id==observer.id)) {
              d->path_present=1;d->state_log_mass=item.second.mass;
            }
          }
          if(observer.alive && !d->path_present) return 5;
          d->alive_after=observer.alive;d->observer_tables_built=audit_guide.built;
        }
'''

FINAL = r'''
    if(observer.valid(horizon)) {
      Diagnostic* d=observer.output+horizon;State target=observer.target(horizon);
      for(size_t i=0;i<final.size();++i) {
        if(final[i].first==target.key) {d->terminal_group_rank=i+1;d->terminal_group_returned=i<max_terminals;}
        bool agrees=true;
        for(int r=0;r<rows;++r) if(target.key[r]>=0 && final[i].first[r]!=target.key[r]) agrees=false;
        if(agrees && !d->terminal_used_mapping_best_rank) {
          d->terminal_used_mapping_best_rank=i+1;d->terminal_used_mapping_returned=i<max_terminals;
        }
      }
    }
'''


def instrument(text):
    if hashlib.sha256(text.encode()).hexdigest() != BASE_SHA:
        raise ValueError("Frozen original source-state body changed")
    replacements = (
        ("static double add", HEADER+"\nstatic double add"),
        ("    const auto began=", "    if(observer.gold && observer.horizon!=horizon) return 5;\n    const auto began="),
        ("    uint64_t serial=0,active=1;", "    IidGuide audit_guide(cipher,lengths,records,rows,glyphs,p,rho,10000,64);\n    uint64_t serial=0,active=1;"),
        ("      if(here.empty()) continue;", "      if(here.empty()) { auto d=observer.begin(layer,0);if(d) {if(observer.alive) return 5;d->alive_after=0;}continue; }"),
        ("      if(layer==horizon) {", "      if(layer==horizon) {\n"+TERMINAL),
        ("      auto prune=[&]", PRE+"\n      auto prune=[&]"),
        ("      prune(width);", POST_GUIDE+"\n      prune(width);\n"+POST_PRUNE),
        ("            Layer& target=pending[rank%3];", CHILD+"\n            Layer& target=pending[rank%3];"),
        ("    stats.terminals=final.size();", FINAL+"\n    stats.terminals=final.size();"),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Unique frozen instrumentation anchor required")
        text = text.replace(old, new)
    return text


def build_prune_observer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).with_name("native_source_state.cpp")
    generated = directory/"source_prune.cpp"
    library = directory/("source_prune.dylib" if platform.system() == "Darwin" else "source_prune.so")
    if generated.exists() or library.exists():
        raise FileExistsError("Fresh observer namespace required")
    generated.write_text(instrument(base.read_text()))
    command = ["clang++", "-std=c++17", "-O3", "-fPIC", "-ffp-contract=off",
        "-dynamiclib" if platform.system() == "Darwin" else "-shared", str(generated), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    version = subprocess.run(["clang++", "--version"], capture_output=True, text=True, check=True, timeout=10)
    return {"abi": 1, "library_path": str(library.resolve()), "cpp_sha256": BASE_SHA,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "generated_cpp_path": str(generated.resolve()), "generated_cpp_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "observer_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler": version.stdout, "command": command, "stdout": result.stdout, "stderr": result.stderr,
        "cpp_sha256_is_frozen_base_dependency": True, "observational_instrumentation": True}


def golden_path(cipher, texts, key, probabilities, transitions, *, glyphs=6, rho=1/225, schedule="balanced"):
    validate_source(probabilities, transitions)
    if (type(glyphs) is not int or not 1 <= glyphs <= 6 or isinstance(rho, bool)
            or not math.isfinite(rho) or not 0 < rho < 1):
        raise ValueError("Bounded glyph alphabet and finite geometric parameter required")
    rows = probabilities.shape[1]
    pool = tuple(tuple(u) for n in (1, 2) for u in itertools.product(range(glyphs), repeat=n))
    cipher, texts, key = tuple(map(tuple, cipher)), tuple(map(tuple, texts)), tuple(key)
    if (not 1 <= len(cipher) <= 2 or len(texts) != len(cipher) or rows > 23
            or any(not c or len(c) > 4096 or any(type(g) is not int or not 0 <= g < glyphs for g in c) for c in cipher)
            or len(key) != rows or any(type(k) is not int or not 0 <= k < len(pool) for k in key)
            or any(not t or any(type(r) is not int or not 0 <= r < rows for r in t) for t in texts)
            or tuple(tuple(g for row in text for g in pool[key[row]]) for text in texts) != cipher
            or schedule not in ("balanced", "sequential") or not 0 < rho < 1):
        raise ValueError("Finite literal known-answer path required")
    lengths, horizon = tuple(map(len, cipher)), sum(map(len, cipher))
    targets = np.zeros((horizon+1, 30), dtype=np.int64)
    masses = np.full(horizon+1, np.nan, dtype=np.float64)
    offsets, contexts, indices, partial = [0]*len(cipher), [0]*len(cipher), [0]*len(cipher), [-1]*23
    score = 0.
    while True:
        rank = sum(offsets)
        targets[rank, :2] = list(offsets)+[0]*(2-len(cipher))
        targets[rank, 2:4] = list(contexts)+[0]*(2-len(cipher))
        targets[rank, 4:27], targets[rank, 29], masses[rank] = partial, 1, score
        record = selected_record(offsets, lengths, schedule)
        if record is None:
            targets[rank, 27:29] = -1
            break
        row = texts[record][indices[record]]
        unit = pool[key[row]]
        targets[rank, 27:29] = row, len(unit)
        probability = float(probabilities[contexts[record], row])
        if probability <= 0:
            raise ValueError("Known answer has zero source support")
        edge = math.log1p(-rho)+math.log(probability)
        if partial[row] < 0:
            partial[row], edge = key[row], edge-math.log(len(pool))
        indices[record] += 1
        offsets[record] += len(unit)
        contexts[record] = int(transitions[contexts[record], row])
        if offsets[record] == lengths[record]:
            contexts[record], edge = 0, edge+math.log(rho)
        score += edge
    return targets, masses


class NativePruneObserver(NativeStateLattice):
    def __init__(self, probabilities, transitions, build):
        generated = Path(build["generated_cpp_path"])
        if (hashlib.sha256(generated.read_bytes()).hexdigest() != build["generated_cpp_sha256"]
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != build["observer_module_sha256"]
                or not build["observational_instrumentation"]):
            raise ValueError("Pinned observer/generated source changed")
        super().__init__(probabilities, transitions, build)
        self.configure = self.library.source_prune_configure
        self.configure.argtypes = [C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(Diagnostic), C.c_int]
        self.configure.restype = C.c_int

    def observe(self, cipher, texts, key, **config):
        targets, masses = golden_path(cipher, texts, key, self.probabilities, self.transitions,
            glyphs=config.get("glyphs", 6), rho=config.get("rho", 1/225), schedule=config.get("schedule", "balanced"))
        output = (Diagnostic*len(targets))()
        status = self.configure(targets.ctypes.data_as(C.POINTER(C.c_int64)),
            masses.ctypes.data_as(C.POINTER(C.c_double)), output, len(targets)-1)
        if status:
            raise ValueError("Observer configuration failed")
        try:
            result = super().search(cipher, **config)
            diagnostics = []
            for rank in np.flatnonzero(targets[:, 29]):
                item = output[int(rank)]
                row = {f: int(getattr(item, f)) for f in FIELDS}
                row.update({f: float(getattr(item, f)) if math.isfinite(getattr(item, f)) else None for f in FLOATS})
                row["target_glyph_layer"] = int(rank)
                row["processed"] = bool(item.prefix_log_mass == masses[rank])
                diagnostics.append(row)
            return result, diagnostics
        finally:
            self.configure(None, None, None, 0)
