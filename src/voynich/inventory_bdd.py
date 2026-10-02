"""Bounded exact Boolean support and occupancy-conditioned dictionary sampling.

Compile literal one/two-glyph word break. Uniform free-row code assignments
induce dependent presence bits; cardinality counts are weighted by exact onto
assignment counts. No source-language score or recovery guarantee is supplied.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

from voynich.coordinated_dictionary import completion_count,randbelow


class InventoryBudgetExceeded(RuntimeError):
    """An exact computation stopped; no partial graph/count/sampling fallback."""


class InventoryBDD:
    def __init__(self,records,*,glyphs=6,order=None,closed=True,max_nodes=150_000,
                 max_apply_steps=3_000_000,max_apply_entries=300_000,
                 max_polynomial_cells=3_000_000,max_polynomial_work=50_000_000,
                 max_work_bytes=512*1024**2):
        self.records = tuple(map(tuple,records))
        if (type(glyphs) is not int or not 1<=glyphs<=6 or type(closed) is not bool
                or not 1<=len(self.records)<=2 or any(len(r)>4096 or any(
                    type(g) is not int or not 0<=g<glyphs for g in r) for r in self.records)
                or any(type(n) is not int or n<1 for n in (max_nodes,max_apply_steps,
                    max_apply_entries,max_polynomial_cells,max_polynomial_work,max_work_bytes))):
            raise ValueError('Bounded literal inventory-compiler settings required')
        self.units = tuple(u for n in (1,2) for u in itertools.product(range(glyphs),repeat=n))
        self.size = len(self.units)
        self.order = tuple(range(self.size)) if order is None else tuple(order)
        if (len(self.order)!=self.size or any(type(c) is not int for c in self.order)
                or set(self.order)!=set(range(self.size))):
            raise ValueError('Variable order must be a complete code permutation')
        self.rank = {code:i for i,code in enumerate(self.order)}
        self.closed = closed
        self.max_nodes,self.max_apply_steps,self.max_apply_entries = max_nodes,max_apply_steps,max_apply_entries
        self.max_polynomial_cells,self.max_polynomial_work = max_polynomial_cells,max_polynomial_work
        self.max_work_bytes = max_work_bytes
        self.nodes = [(self.size,0,0),(self.size,1,1)]
        self.unique,self.applied = {},{}
        self.apply_steps = self.polynomial_cells = self.polynomial_work = self.polynomial_entries = 0
        self.maximum_owned_envelope = 0
        self._book()
        variables = [self._node(self.rank[c],0,1) for c in range(self.size)]
        roots = []
        for record in self.records:
            suffix = [0]*(len(record)+1)
            suffix[-1] = 1
            for pos in range(len(record)-1,-1,-1):
                value = 0
                for code,unit in enumerate(self.units):
                    visible = min(len(unit),len(record)-pos)
                    end = pos+len(unit)
                    if ((closed and end>len(record)) or unit[:visible]!=record[pos:pos+visible]):
                        continue
                    branch = self.apply('and',variables[code],suffix[min(end,len(record))])
                    value = self.apply('or',value,branch)
                suffix[pos] = value
            roots.append(suffix[0])
        self.record_roots = tuple(roots)
        self.root = roots[0]
        for root in roots[1:]:
            self.root = self.apply('and',self.root,root)

    def _book(self,*,nodes=0,entries=0,cells=0,polynomials=0):
        envelope = (1024**2+512*(len(self.nodes)+nodes)+256*(len(self.applied)+entries)
                    +64*(self.polynomial_cells+cells)+256*(self.polynomial_entries+polynomials))
        if envelope>self.max_work_bytes:
            raise InventoryBudgetExceeded('Inventory owned allocation envelope exhausted')
        self.maximum_owned_envelope = max(self.maximum_owned_envelope,envelope)

    def _node(self,var,low,high):
        if low==high:
            return low
        assert 0<=var<min(self.nodes[low][0],self.nodes[high][0])
        key = var,low,high
        if key not in self.unique:
            if len(self.nodes)>=self.max_nodes:
                raise InventoryBudgetExceeded('Inventory Boolean node cap exhausted')
            self._book(nodes=1)
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def apply(self,operation,a,b):
        if operation not in ('and','or'):
            raise ValueError('Only registered Boolean operations allowed')
        self.apply_steps += 1
        if self.apply_steps>self.max_apply_steps:
            raise InventoryBudgetExceeded('Inventory Boolean apply-work cap exhausted')
        if a>b:
            a,b = b,a
        if a==b:
            return a
        if operation=='and':
            if a==0:
                return 0
            if a==1:
                return b
        else:
            if a==0:
                return b
            if a==1 or b==1:
                return 1
        key = operation,a,b
        if key in self.applied:
            return self.applied[key]
        if len(self.applied)>=self.max_apply_entries:
            raise InventoryBudgetExceeded('Inventory Boolean apply-cache cap exhausted')
        var = min(self.nodes[a][0],self.nodes[b][0])
        al,ah = self.nodes[a][1:] if self.nodes[a][0]==var else (a,a)
        bl,bh = self.nodes[b][1:] if self.nodes[b][0]==var else (b,b)
        low,high = self.apply(operation,al,bl),self.apply(operation,ah,bh)
        value = self._node(var,low,high)
        # Recursive children may fill the cache since the first admission.
        if len(self.applied)>=self.max_apply_entries:
            raise InventoryBudgetExceeded('Inventory Boolean apply-cache cap exhausted')
        self._book(entries=1)
        self.applied[key] = value
        return value

    def accepts(self,codes):
        if any(type(c) is not int or not 0<=c<self.size for c in codes):
            raise ValueError('Literal legal code inventory required')
        present = set(codes)
        node = self.root
        while node>1:
            var,low,high = self.nodes[node]
            node = high if self.order[var] in present else low
        return node==1

    def stats(self):
        return {'nodes':len(self.nodes),'apply_steps':self.apply_steps,'apply_entries':len(self.applied),
            'polynomial_cells':self.polynomial_cells,'polynomial_work':self.polynomial_work,
            'polynomial_entries':self.polynomial_entries,'maximum_owned_envelope':self.maximum_owned_envelope}


class InventoryProfile:
    """One known-row profile. Draw cardinality -> subset -> onto row assignment."""

    def __init__(self,bdd,partial):
        self.bdd = bdd
        self.partial = np.asarray(partial)
        if (self.partial.ndim!=1 or not 1<=len(self.partial)<=23 or self.partial.dtype.kind not in 'iu'
                or np.any(self.partial< -1) or np.any(self.partial>=bdd.size)):
            raise ValueError('One-to23 literal known/unknown source rows required')
        self.partial = self.partial.copy()
        self.partial.setflags(write=False)
        self.free = np.flatnonzero(self.partial<0)
        self.known = set(map(int,self.partial[self.partial>=0]))
        self.forced = sum(1<<bdd.rank[c] for c in self.known)
        self.unforced_suffix = [sum(not bool(self.forced&(1<<j)) for j in range(i,bdd.size))
                                for i in range(bdd.size+1)]
        self.cache = {}
        self.coefficients = self.extend(bdd.root,0)
        self.base = len(self.known)
        self.cardinality_weights = tuple(b*self.onto_count(len(self.free),k,self.base+k)
            for k,b in enumerate(self.coefficients))
        self.total = sum(self.cardinality_weights)
        self.prior_event_probability = Fraction(self.total,bdd.size**len(self.free))

    @staticmethod
    def onto_count(n,missing,allowed):
        if not allowed:
            return int(n==0 and missing==0)
        return completion_count(n,missing,allowed)

    def _work(self,amount):
        self.bdd.polynomial_work += amount
        if self.bdd.polynomial_work>self.bdd.max_polynomial_work:
            raise InventoryBudgetExceeded('Inventory polynomial-work cap exhausted')

    def extend(self,node,start):
        var = self.bdd.nodes[node][0]
        assert start<=var
        skip = self.unforced_suffix[start]-self.unforced_suffix[var]
        value = self.polynomial(node)
        if not skip:
            return value
        self._work(len(value)*(skip+1))
        result = [0]*(len(value)+skip)
        for i,a in enumerate(value):
            for j in range(skip+1):
                result[i+j] += a*math.comb(skip,j)
        return tuple(result)

    def polynomial(self,node):
        if node<=1:
            return (node,)
        if node in self.cache:
            return self.cache[node]
        var,low,high = self.bdd.nodes[node]
        right = self.extend(high,var+1)
        if self.forced&(1<<var):
            result = right
        else:
            left = self.extend(low,var+1)
            self._work(max(len(left),len(right)+1))
            result = tuple((left[i] if i<len(left) else 0)+(right[i-1] if 0<i<=len(right) else 0)
                           for i in range(max(len(left),len(right)+1)))
        # Only retained coefficient tuples are charged; scratch is <=43 cells.
        cells = len(result)
        if self.bdd.polynomial_cells+cells>self.bdd.max_polynomial_cells:
            raise InventoryBudgetExceeded('Inventory polynomial-cell cap exhausted')
        self.bdd._book(cells=cells,polynomials=1)
        self.bdd.polynomial_cells += cells
        self.bdd.polynomial_entries += 1
        self.cache[node] = result
        return result

    def unrank(self,k,rank):
        """Uniform rank in the satisfying unknown-code subsets of size k."""
        if (type(k) is not int or not 0<=k<len(self.coefficients) or type(rank) is not int
                or not 0<=rank<self.coefficients[k]):
            raise ValueError('Valid cardinality/subset rank required')
        chosen,node = set(),self.bdd.root
        for pos,code in enumerate(self.bdd.order):
            var,low,high = self.bdd.nodes[node]
            assert var>=pos
            if code in self.known:
                if var==pos:
                    node = high
                continue
            lo,hi = (low,high) if var==pos else (node,node)
            ways = self.extend(lo,pos+1)
            zero = ways[k] if 0<=k<len(ways) else 0
            if rank<zero:
                node = lo
            else:
                rank -= zero
                chosen.add(code)
                node = hi
                k -= 1
        assert node==1 and k==0 and rank==0
        return chosen

    def sample(self,rng):
        if not self.total:
            raise ValueError('No supporting dictionary completions')
        draw = randbelow(rng,self.total)
        for k,weight in enumerate(self.cardinality_weights):
            if draw<weight:
                break
            draw -= weight
        else:
            raise AssertionError('Cardinality draw lost probability mass')
        required = self.unrank(k,randbelow(rng,self.coefficients[k]))
        allowed = sorted(self.known|required)
        missing = set(required)
        key = self.partial.copy()
        for i,row in enumerate(self.free):
            left = len(self.free)-i-1
            weights = [self.onto_count(left,len(missing)-int(code in missing),len(allowed)) for code in allowed]
            total = sum(weights)
            assert total==self.onto_count(left+1,len(missing),len(allowed))
            draw = randbelow(rng,total)
            for code,weight in zip(allowed,weights,strict=True):
                if draw<weight:
                    key[row] = code
                    missing.discard(code)
                    break
                draw -= weight
            else:
                raise AssertionError('Onto code assignment lost probability mass')
        assert not missing and self.bdd.accepts(list(map(int,key)))
        return key.astype(np.int32)
