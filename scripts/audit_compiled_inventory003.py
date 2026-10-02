"""One full new-work-ceiling panel replay, with the original exact audit."""

from scripts import audit_compiled_inventory002 as previous_audit
from scripts import run_compiled_inventory003 as run


def audit():
    with run.namespace():
        previous_audit.audit()


if __name__=='__main__':
    audit()
