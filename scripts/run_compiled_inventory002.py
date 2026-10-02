"""New namespace, same fixed engineering panel, safely collected compiler."""

import argparse
import json
from contextlib import contextmanager

from scripts import audit_compiled_inventory001 as auditor
from scripts import run_compiled_inventory001 as original
from voynich.rolling_inventory_bdd import RollingInventoryBDD


EXP = 'COMPILED-INVENTORY-002'
ROOT = original.ROOT
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
LIMITS = {**original.LIMITS,'collection_interval':16,'collection_nodes':25_000,
          'collection_entries':50_000,'max_collection_work':20_000_000}
PATHS = sorted(set([*original.PATHS,'src/voynich/rolling_inventory_bdd.py',
    'tests/test_rolling_inventory_bdd.py','scripts/check_rolling_inventory001.py',
    'results/ROLLING-INVENTORY-THEORY-001/result.json','scripts/run_compiled_inventory002.py',
    'scripts/audit_compiled_inventory002.py','tests/test_compiled_inventory002.py',
    'docs/experiments/COMPILED-INVENTORY-002.md','docs/research/rolling-inventory-2026-10-01.md',
    'results/COMPILED-INVENTORY-001/result.json','results/COMPILED-INVENTORY-001/audit.json',
    'results/COMPILED-INVENTORY-001/post-outcome.json']))
ORIGINAL_INPUTS = original.inputs


def inputs(freeze):
    parent,benchmark,queries = ORIGINAL_INPUTS(freeze)
    proof = json.loads((ROOT/'results/ROLLING-INVENTORY-THEORY-001/result.json').read_text())
    assert proof['status']=='PASS_exact_rolling_inventory_laws'
    assert all(original.artifact(ROOT/s['path'])==s for s in proof['inputs'])
    return parent,benchmark,queries


@contextmanager
def namespace():
    # Reuse frozen transport/audit mechanics in one isolated process. Restore all
    # runtime bindings even on failure; original files and namespaces stay intact.
    patches = [(original,'EXP',EXP),(original,'OUT',OUT),(original,'BULK',BULK),
        (original,'PATHS',PATHS),(original,'LIMITS',LIMITS),(original,'inputs',inputs),
        (original,'InventoryBDD',RollingInventoryBDD),(auditor,'OUT',OUT),(auditor,'inputs',inputs)]
    saved = [(module,name,getattr(module,name)) for module,name,_ in patches]
    try:
        for module,name,value in patches:
            setattr(module,name,value)
        yield
    finally:
        for module,name,value in reversed(saved):
            setattr(module,name,value)


def run(freeze):
    with namespace():
        original.run(freeze)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    run(parser.parse_args().freeze)
