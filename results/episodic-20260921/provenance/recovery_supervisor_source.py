"""Resume frozen training after the chat host killed its supervising process.

No scientific source, settings, completed run, or confirmation pool is changed.
Unfinished runs are preserved and replayed from their registered initial seeds.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/Users/rikhil/.codex/worktrees/episodic-rule-recovery/solving-voynich')
OUT = ROOT / 'outputs/EXP-0012'
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from run_parallel_campaign import CHILD_ENV, run_campaign, source_state, write_json


def main():
    os.chdir(ROOT)
    original = json.loads((OUT/'supervisor/manifest.json').read_text())
    source = original['source']
    if source_state(ROOT) != source:
        raise RuntimeError('Frozen source changed; recovery refused')
    launch = datetime.fromisoformat(original['started_utc'])
    elapsed = (datetime.now(timezone.utc)-launch).total_seconds()
    remaining = int(14400-elapsed)
    if remaining < 120:
        raise RuntimeError('Original campaign training deadline exhausted')
    destination = OUT/'interrupted-supervisor'
    if destination.exists() or (OUT/'supervision_recovery.json').exists():
        raise RuntimeError('Recovery must not overwrite earlier evidence')
    (OUT/'supervisor').rename(destination)
    unfinished = OUT/'interrupted-runs'
    unfinished.mkdir()
    restarted = []
    retained = []
    for run in sorted((OUT/'runs').iterdir()):
        if not run.is_dir():
            continue
        if (run/'summary.json').exists():
            retained.append(run.name)
        else:
            restarted.append(run.name)
            run.rename(unfinished/run.name)
    recovery = {
        'reason':'Chat turn interruption killed supervisor; workers became orphans (PPID 1).',
        'original_worker_pids':[99586,99587],
        'original_worker_exit_codes':None,
        'stopped_by_recovery':'SIGTERM to those two PIDs only; exit verified before replay',
        'original_started_utc':original['started_utc'],
        'recovery_started_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_before_recovery_seconds':elapsed,
        'remaining_original_training_budget_seconds':remaining,
        'retained_completed_runs':retained,
        'restarted_incomplete_runs':restarted,
        'policy':'Preserve all prior artifacts; replay incomplete runs from initial seeds without adaptive changes.',
        'confirmation_inspected':False,
        'scientific_source_unchanged':source,
        'recovery_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'resource_limitation':'Supervisor sampling absent between host interruption and worker termination; previous samples retained.',
    }
    write_json(OUT/'supervision_recovery.json',recovery)
    CHILD_ENV.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.70', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.60',
                     OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                     VECLIB_MAXIMUM_THREADS='4', PYTHONPATH=str(ROOT/'src'))
    command=[sys.executable,str(ROOT/'scripts/run_episodic_campaign.py')]
    plan={'campaign':'EXP-0012-interruption-recovery','max_seconds':remaining,'rss_ceiling_gib':40,
          'jobs':[{'name':track,'max_seconds':remaining-60,
                   'command':command+['worker','--config','configs/exp0012.json','--out','outputs/EXP-0012','--track',track]}
                  for track in ('gpu','cpu')]}
    status=run_campaign(plan,OUT/'supervisor',root=ROOT)
    if status['state']!='completed' or source_state(ROOT)!=source:
        raise RuntimeError('Recovery training failed or source drifted; confirmation remains gated')
    analysis={'campaign':'EXP-0012-0013-analysis','max_seconds':7200,'rss_ceiling_gib':40,
              'jobs':[{'name':'analysis','max_seconds':7140,
                       'command':command+['analysis','--config','configs/exp0012.json','--out','outputs/EXP-0012']}]}
    status=run_campaign(analysis,OUT/'analysis-supervisor',root=ROOT)
    if status['state']!='completed' or source_state(ROOT)!=source:
        raise RuntimeError('Analysis failed or source drifted')
    write_json(OUT/'complete.json',{'elapsed_seconds':(datetime.now(timezone.utc)-launch).total_seconds(),
               'source':source,'no_paid_api':True,'no_manuscript_inputs':True,
               'interruption_recovery':'supervision_recovery.json'})


if __name__=='__main__':
    main()
