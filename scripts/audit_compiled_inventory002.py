"""One full collected-compiler panel replay under its independent namespace."""

import json

from scripts import run_compiled_inventory002 as run


def audit():
    assert json.loads((run.OUT/'result.json').read_text())['experiment']==run.EXP
    with run.namespace():
        run.auditor.audit()


if __name__=='__main__':
    audit()
