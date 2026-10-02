"""Unpruned fixed-key posterior paths for a borrowed immutable dense source.

All compatible source paths enter the backward sum. Structural impossibility
is distinct from graph/memory caps; no cap supplies a partial likelihood or
fallback path. Floating arithmetic and finite-resolution draws remain explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _logsum(values):
    values = tuple(values)
    high = max(values,default=-math.inf)
    return high+math.log(math.fsum(math.exp(v-high) for v in values)) if high!=-math.inf else high


@dataclass(frozen=True)
class PosteriorPath:
    text: str
    joint_log_probability: float
    conditional_log_probability: float


class SuffixPosteriorPaths:
    """One complete record and one fixed dictionary; arbitrary dense source order.

    Source arrays are borrowed and retained, not copied. The conservative owned
    envelope covers offset maps, node/edge objects, backward values and cached
    draw choices; borrowed source arrays and caller inputs are excluded from it.
    Whole-process RSS limits are still needed for any actual research driver.
    """

    def __init__(self,source,units,record,rho,*,max_nodes=100_000,max_edges=400_000,
                 max_work_bytes=128*1024**2,max_sample_bytes=64*1024**2):
        alphabet = tuple(source.alphabet)
        p,t = source.probabilities,source.transitions
        if (not 1<=len(alphabet)<=64 or len(set(alphabet))!=len(alphabet)
                or any(not isinstance(a,str) or len(a)!=1 for a in alphabet)
                or isinstance(units,(str,bytes)) or not isinstance(record,str) or len(record)>1_000_000
                or isinstance(rho,bool) or not math.isfinite(rho) or not 0<rho<1
                or any(type(v) is not int or v<1 for v in (max_nodes,max_edges,max_work_bytes,max_sample_bytes))):
            raise ValueError('Bounded fixed-key suffix-posterior settings required')
        units = tuple(units)
        if len(units)!=len(alphabet) or any(not isinstance(u,str) or not u or len(u)>64 for u in units):
            raise ValueError('One nonempty bounded unit per source letter required')
        if (not isinstance(p,np.ndarray) or not isinstance(t,np.ndarray) or p.dtype!=np.float64
                or t.dtype!=np.uint32 or p.ndim!=2 or p.shape!=t.shape or p.shape[1]!=len(alphabet)
                or not len(p) or p.flags.writeable or t.flags.writeable
                or not p.flags.c_contiguous or not t.flags.c_contiguous):
            raise ValueError('Matching immutable dense float64/uint32 source arrays required')
        root = source.state('')
        if type(root) is not int or not 0<=root<len(p):
            raise ValueError('Valid dense source root required')
        self.alphabet,self.units,self.record,self.rho = alphabet,units,record,rho
        self.probabilities,self.transitions,self.root = p,t,root
        self.max_nodes,self.max_edges,self.max_work_bytes = max_nodes,max_edges,max_work_bytes
        self.max_sample_bytes = max_sample_bytes
        self.base_work_bytes = 4096+(len(record)+1)*128+len(record)*8+sum(len(u)*4+128 for u in units)
        self.nodes,self.edges,self.maximum_work_envelope = 0,0,0
        self._reserve(nodes=1)
        levels = [{} for _ in range(len(record)+1)]
        levels[0][root] = 0
        self.outgoing,self.beta,self._choices = [[]],[0.],{}
        stop,cont = math.log(rho),math.log1p(-rho)
        for offset,states in enumerate(levels):
            if offset==len(record):
                continue
            matches = [(a,offset+len(u)) for a,u in enumerate(units) if record.startswith(u,offset)]
            for state,node in states.items():
                row = p[state]
                if (np.any(~np.isfinite(row)) or np.any(row<0) or np.any(row>1)
                        or not math.isclose(math.fsum(map(float,row)),1.,abs_tol=1e-12,rel_tol=0)):
                    raise ValueError('Invalid reachable normalized source row')
                for letter,end in matches:
                    probability = float(row[letter])
                    if not probability:
                        continue
                    following = int(t[state,letter])
                    if not 0<=following<len(p):
                        raise ValueError('Dense transition outside source')
                    fresh = following not in levels[end]
                    self._reserve(nodes=int(fresh),edges=1)
                    if fresh:
                        levels[end][following] = len(self.outgoing)
                        self.outgoing.append([])
                        self.beta.append(0.)
                    self.outgoing[node].append((letter,levels[end][following],cont+math.log(probability)))
        for offset in range(len(record),-1,-1):
            for node in levels[offset].values():
                self.beta[node] = stop if offset==len(record) else _logsum(w+self.beta[nxt] for _,nxt,w in self.outgoing[node])
        self.log_likelihood = self.beta[0]

    def _reserve(self,*,nodes=0,edges=0):
        proposed_nodes,proposed_edges = self.nodes+nodes,self.edges+edges
        owned = self.base_work_bytes+1024*proposed_nodes+512*proposed_edges
        if proposed_nodes>self.max_nodes:
            raise RuntimeError('Posterior lattice node cap; no partial score or path')
        if proposed_edges>self.max_edges:
            raise RuntimeError('Posterior lattice edge cap; no partial score or path')
        if owned>self.max_work_bytes:
            raise MemoryError('Posterior lattice owned-work cap; no partial score or path')
        self.nodes,self.edges = proposed_nodes,proposed_edges
        self.maximum_work_envelope = max(self.maximum_work_envelope,owned)

    def choices(self,node):
        """Conditional edges (letter, next node, probability) for a live node."""
        if type(node) is not int or not 0<=node<self.nodes or self.beta[node]==-math.inf or not self.outgoing[node]:
            raise ValueError('A supported nonterminal posterior node is required')
        if node not in self._choices:
            supported = [(a,nxt,w+self.beta[nxt]) for a,nxt,w in self.outgoing[node] if self.beta[nxt]!=-math.inf]
            high = max(v for _,_,v in supported)
            values = [math.exp(v-high) for _,_,v in supported]
            total = math.fsum(values)
            self._choices[node] = tuple((a,nxt,v/total) for (a,nxt,_),v in zip(supported,values,strict=True))
        return self._choices[node]

    def sample_paths(self,*,draws,seed):
        """Fresh draws with replacement, conditional on this fixed lattice.

        Zero draws or impossible records return no paths. No MAP/top-k/beam
        substitute; finite source zeros are retained. Seed never selects a key.
        """
        if type(draws) is not int or not 0<=draws<=4096 or type(seed) is not int or seed<0:
            raise ValueError('Zero to4096 draws and nonnegative integer seed required')
        if self.log_likelihood==-math.inf:
            return ()
        if draws*(256+4*len(self.record))>self.max_sample_bytes:
            raise MemoryError('Posterior path output cap; no partial sample bank')
        rng,paths = np.random.default_rng(seed),[]
        for _ in range(draws):
            node,text,terms = 0,[],[math.log(self.rho)]
            while self.outgoing[node]:
                choices = self.choices(node)
                cumulative = np.cumsum([v for _,_,v in choices])
                # Keep a rounded float endpoint strictly below the last CDF value.
                threshold = min(rng.random()*cumulative[-1],np.nextafter(cumulative[-1],-math.inf))
                index = int(np.searchsorted(cumulative,threshold,side='right'))
                letter,nxt,_ = choices[index]
                weight = next(w for a,n,w in self.outgoing[node] if a==letter and n==nxt)
                terms.append(weight)
                text.append(self.alphabet[letter])
                node = nxt
            joint = math.fsum(terms)
            paths.append(PosteriorPath(''.join(text),joint,joint-self.log_likelihood))
        return tuple(paths)
