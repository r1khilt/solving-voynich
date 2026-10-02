"""Same exact collected compiler, new bounded cumulative-work qualification."""

import argparse
import json
from contextlib import contextmanager

from scripts import run_compiled_inventory002 as previous


EXP = 'COMPILED-INVENTORY-003'
ROOT = previous.ROOT
OUT,BULK = ROOT/'results'/EXP,ROOT/'outputs'/EXP
LIMITS = {**previous.LIMITS,'max_apply_steps':12_000_000}
PATHS = sorted(set([*previous.PATHS,'scripts/run_compiled_inventory003.py',
    'scripts/audit_compiled_inventory003.py','tests/test_compiled_inventory003.py',
    'docs/experiments/COMPILED-INVENTORY-003.md',
    'results/COMPILED-INVENTORY-002/result.json','results/COMPILED-INVENTORY-002/audit.json',
    'results/COMPILED-INVENTORY-002/post-outcome.json','docs/experiments/COMPILED-INVENTORY-002-results.md']))
PREVIOUS_INPUTS = previous.inputs


def inputs(freeze):
    parent,benchmark,queries = PREVIOUS_INPUTS(freeze)
    result = json.loads((ROOT/'results/COMPILED-INVENTORY-002/result.json').read_text())
    audit = json.loads((ROOT/'results/COMPILED-INVENTORY-002/audit.json').read_text())
    assert audit['status']=='PASS_full_compiled_inventory_replay'
    assert audit['result']==previous.original.artifact(ROOT/'results/COMPILED-INVENTORY-002/result.json')
    assert result['source']==parent['source'] and result['source_arrays']==parent['source_arrays']
    assert result['native_build']==benchmark['build'] and result['queries']==queries
    return parent,benchmark,queries


@contextmanager
def namespace():
    patches = [('EXP',EXP),('OUT',OUT),('BULK',BULK),('LIMITS',LIMITS),('PATHS',PATHS),('inputs',inputs)]
    saved = [(name,getattr(previous,name)) for name,_ in patches]
    try:
        for name,value in patches:
            setattr(previous,name,value)
        yield
    finally:
        for name,value in reversed(saved):
            setattr(previous,name,value)


def run(freeze):
    with namespace():
        previous.run(freeze)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    run(parser.parse_args().freeze)
