"""Exact rolling word-break compilation with bounded protected-root collection."""

from voynich.inventory_bdd import InventoryBDD,InventoryBudgetExceeded


class RollingInventoryBDD(InventoryBDD):
    def __init__(self,records,*,collection_interval=16,collection_nodes=25_000,
                 collection_entries=50_000,max_collection_work=20_000_000,**kwargs):
        self.collections = self.collection_work = self.collected_nodes = 0
        self.created_nodes = self.maximum_live_nodes = self.maximum_apply_entries = 0
        self.sealed = False
        settings = (collection_interval,collection_nodes,collection_entries,max_collection_work)
        if any(type(n) is not int or n<1 for n in settings):
            raise ValueError('Positive bounded collection settings required')
        self.max_collection_work = max_collection_work
        observations = tuple(map(tuple,records))
        # Use frozen manager validation/initialization, without real compilation.
        super().__init__(((),),**kwargs)
        if not 1<=len(observations)<=2:
            raise ValueError('Bounded literal records required')
        # Unit count g+g^2 determines the validated glyph alphabet exactly.
        glyphs = sum(len(u)==1 for u in self.units)
        if any(len(r)>4096 or any(type(g) is not int or not 0<=g<glyphs for g in r)
               for r in observations):
            raise ValueError('Bounded literal records required')
        self.records = observations
        variables = list(range(2,2+self.size))
        roots = []
        for record in observations:
            next_one,next_two = 1,1
            for step,pos in enumerate(range(len(record)-1,-1,-1),1):
                value = 0
                for code,unit in enumerate(self.units):
                    end = pos+len(unit)
                    visible = min(len(unit),len(record)-pos)
                    if ((self.closed and end>len(record)) or unit[:visible]!=record[pos:pos+visible]):
                        continue
                    suffix = 1 if end>=len(record) else next_one if len(unit)==1 else next_two
                    value = self.apply('or',value,self.apply('and',variables[code],suffix))
                next_one,next_two = value,next_one
                if (step%collection_interval==0 or len(self.nodes)>=collection_nodes
                        or len(self.applied)>=collection_entries):
                    mapped = self._collect([*variables,*roots,next_one,next_two])
                    variables,roots = mapped[:self.size],mapped[self.size:-2]
                    next_one,next_two = mapped[-2:]
            roots.append(next_one)
            mapped = self._collect([*variables,*roots])
            variables,roots = mapped[:self.size],mapped[self.size:]
        root = roots[0]
        for other in roots[1:]:
            root = self.apply('and',root,other)
        mapped = self._collect([*variables,*roots,root])
        self.record_roots,self.root = tuple(mapped[self.size:-1]),mapped[-1]
        self.sealed = True

    def _node(self,*args):
        before = len(self.nodes)
        result = super()._node(*args)
        self.created_nodes += len(self.nodes)-before
        self.maximum_live_nodes = max(self.maximum_live_nodes,len(self.nodes))
        return result

    def _book(self,**kwargs):
        super()._book(**kwargs)
        self.maximum_live_nodes = max(self.maximum_live_nodes,len(self.nodes))
        self.maximum_apply_entries = max(self.maximum_apply_entries,len(self.applied)+kwargs.get('entries',0))

    def _charge_collection(self,n):
        self.collection_work += n
        if self.collection_work>self.max_collection_work:
            raise InventoryBudgetExceeded('Inventory collection-work cap exhausted')

    def _collect(self,roots):
        if self.sealed or self.polynomial_entries:
            raise ValueError('Collection only allowed during unprofiled compilation')
        assert all(type(n) is int and 0<=n<len(self.nodes) for n in roots)
        # Extra 512 bytes per old node covers retained-node rebuild, reachability,
        # mapping and stack scratch alongside the old manager. No cache IDs survive.
        self._book(nodes=len(self.nodes))
        self._charge_collection(len(self.applied)+len(self.nodes))
        self.applied = {}
        seen,todo = {0,1},list(roots)
        while todo:
            self._charge_collection(1)
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            if node>1:
                todo.extend(self.nodes[node][1:])
        old,new_nodes = self.nodes,[(self.size,0,0),(self.size,1,1)]
        mapping,new_unique = {0:0,1:1},{}
        for index in range(2,len(old)):
            if index not in seen:
                continue
            self._charge_collection(1)
            var,lo,hi = old[index]
            assert lo<index and hi<index
            node = var,mapping[lo],mapping[hi]
            assert node not in new_unique
            mapping[index] = len(new_nodes)
            new_unique[node] = len(new_nodes)
            new_nodes.append(node)
        self.nodes,self.unique = new_nodes,new_unique
        self.collections += 1
        self.collected_nodes += len(old)-len(self.nodes)
        assert self.created_nodes+2-self.collected_nodes==len(self.nodes)
        return [mapping[n] for n in roots]

    def stats(self):
        return {**super().stats(),'collections':self.collections,'collection_work':self.collection_work,
            'collected_nodes':self.collected_nodes,'created_nodes':self.created_nodes,
            'maximum_live_nodes':self.maximum_live_nodes,'maximum_apply_entries':self.maximum_apply_entries}
