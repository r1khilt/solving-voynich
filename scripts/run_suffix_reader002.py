"""Fixed source-only calibration, new supplied-key readings, then one evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import signal
import time
from pathlib import Path

import numpy as np

from scripts.audit_blind_channel_dev004 import edit_distance
from scripts.audit_suffix_reader001 import counts_reference
from scripts.audit_suffix_reader002 import infer, reading
from scripts.build_blind_channel_dev001 import make_channel, segments
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_suffix_reader001 import PATHS as OLD_PATHS, corpus_records, gate
from voynich.calibrated_suffix_source import DepthSource, calibrate, extract_features, loss_gradient
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode

ROOT=Path(__file__).resolve().parents[1]
EXP='SUFFIX-READER-002'
DIR=ROOT/'results'/EXP
STARTS=(4.,64.,256.,1024.)
ARMS=('fixed3','fixed12','calibrated12','calibrated3')
OLD_SELECTION='results/SUFFIX-READER-001/source_selection.json'
OLD_PANEL='results/SUFFIX-READER-001/panel_manifest.json'
PATHS=[*OLD_PATHS, 'scripts/run_suffix_reader002.py', 'scripts/audit_suffix_reader002.py',
       'src/voynich/calibrated_suffix_source.py', 'tests/test_calibrated_suffix_source.py',
       'tests/test_suffix_reader002.py', 'docs/experiments/SUFFIX-READER-002.md', OLD_SELECTION, OLD_PANEL]


def source_interval(payload,start,end):
    if not 0<=start<end<=len(payload['text']):
        raise ValueError('Invalid source interval')
    return [payload['text'][max(start,lo):min(end,hi)] for lo,hi in
            zip([0,*payload['body_boundaries'][:-1]],payload['body_boundaries'],strict=True)
            if max(start,lo)<min(end,hi)]


def window_offsets():
    return [[20000+512*i,20256+512*i] for i in range(16)]


def start(stage,freeze,cpu,wall):
    require_frozen(freeze,PATHS)
    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
    def timeout(*_):
        raise TimeoutError('Registered wall cap')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(wall)
    save_new(DIR/f'{stage}_started.json',{'freeze':freeze,'start_unix':time.time()})
    return time.monotonic(),time.process_time()


def prepare(freeze):
    wall,cpu=start('prepare',freeze,1200,1800)
    corpus,payloads=corpus_records(('caesar','virgil'))
    alphabet=tuple(corpus['alphabet'])
    train=segments(payloads['caesar'])
    calibration=source_interval(payloads['virgil'],0,40000)
    selection_text=source_interval(payloads['virgil'],40000,50000)
    counts=collect_counts(train,alphabet,12)
    if counts!=counts_reference(train,12):
        raise ValueError('Independent Caesar counts differ')
    source=SuffixSource(alphabet,12,256.,counts)
    fitting=extract_features(source,calibration)
    selecting=extract_features(source,selection_text)
    candidates=[{'start_index':-1,'step':0,'masses':[256.]*12,
                 'selection_bits_per_character':loss_gradient(np.log([256.]*12),selecting)[0]/math.log(2)}]
    traces={}
    for index,initial in enumerate(STARTS):
        trace=calibrate(fitting,initial)
        traces[str(index)]=trace
        for row in trace:
            if row['step']%100==0:
                candidate={'start_index':index,'initial_mass':initial,'step':row['step'],
                    'masses':np.exp(row['log_masses']).tolist(),
                    'calibration_bits_per_character':row['loss_nats_per_character']/math.log(2),
                    'selection_bits_per_character':loss_gradient(row['log_masses'],selecting)[0]/math.log(2)}
                candidates.append(candidate)
                print(json.dumps(candidate),flush=True)
    chosen=min(candidates,key=lambda r:(r['selection_bits_per_character'],r['start_index'],r['step']))
    check_model=DepthSource(alphabet,chosen['masses'],counts)
    delta=abs(check_model.bits(selection_text)/10000-chosen['selection_bits_per_character'])
    if delta>1e-12:
        raise ValueError('Feature calibration score disagrees with explicit source')
    trace_artifact=save_new(ROOT/'outputs'/EXP/'calibration_trace.json.gz',traces,compressed=True)
    del source,check_model,counts,fitting,selecting
    gc.collect()
    old=json.loads((ROOT/OLD_SELECTION).read_text())
    raw=load_archive(old['source'])
    refit_counts=collect_counts(train+segments(payloads['virgil']),alphabet,12)
    if (raw['alphabet']!=list(alphabet) or raw['order']!=12 or raw['tau']!=256.
            or refit_counts!=raw['counts'] or refit_counts!=counts_reference(train+segments(payloads['virgil']),12)):
        raise ValueError('Shared refit counts differ from fixed baseline')
    calibration_result={'experiment':EXP,'freeze':freeze,'candidates':candidates,'selected':chosen,
        'shared_counts_archive':old['source'],'count_contexts':len(refit_counts),
        'trace':trace_artifact,'feature_source_score_delta':delta,'all_independent_counts_match':True,
        'gradient_fit_chars':40000,'selection_chars':10000,'count_training_chars':50000,
        'refit_characters':100000,'numpy_version':np.__version__}
    save_new(DIR/'calibration.json',calibration_result)
    # Target author is accessed only after calibration and selection are saved.
    _,target=corpus_records(('cicero',))
    payload=target['cicero']
    prior_panel=checked_artifact(json.loads((ROOT/OLD_PANEL).read_text())['panel'])['cases']
    intervals=[(o,o+224) for r in prior_panel.values() for o in r['offsets']]
    old_cases=json.loads((ROOT/'data/manifests/blind_channel_dev001.json').read_text())['cases']
    for case in old_cases.values():
        if case['positive']:
            prior=checked_artifact(case['artifacts']['answer'])
            intervals.extend((o,o+224) for offsets in prior['source_offsets'].values() for o in offsets)
    signatures={tuple(r['units']) for r in prior_panel.values()}
    cases,answers={},{}
    for index,offsets in enumerate(window_offsets()):
        seed=119129+104729*index
        key=make_channel(alphabet,'B',seed)
        units=tuple(key.rows['s0',c][0].glyphs for c in alphabet)
        if units in signatures:
            raise ValueError('Duplicate key; no redraw')
        signatures.add(units)
        plain=[]
        for offset in offsets:
            if any(offset<hi and offset+224>lo for lo,hi in intervals):
                raise ValueError('New window overlaps prior cipher scoring allocation')
            if not any(lo<=offset and offset+224<=hi for lo,hi in
                       zip([0,*payload['body_boundaries'][:-1]],payload['body_boundaries'],strict=True)):
                raise ValueError('Body crossing')
            text=payload['text'][offset:offset+224]
            if len(text)!=224:
                raise ValueError('Short text')
            plain.append(text)
        name=f'B-key{index+1}'
        cases[name]={'units':units,'records':[''.join(units[alphabet.index(c)] for c in p) for p in plain],
                     'offsets':offsets,'seed':seed}
        answers[name]=plain
    panel=save_new(ROOT/'data/processed'/EXP/'panel.json',{'alphabet':alphabet,'cases':cases})
    answer=save_new(ROOT/'data/processed'/EXP/'answers.json',{'cases':answers})
    save_new(DIR/'panel_manifest.json',{'experiment':EXP,'freeze':freeze,'panel':panel,'answers':answer,
        'calibration_sha256':digest((DIR/'calibration.json').read_bytes()),
        'target_sha256':corpus['sources']['cicero']['derived_sha256'],
        'known_keys_supplied':True,'prior_window_overlap':False,'body_contained':True,
        'resources':resource_report(wall,cpu)})


def predict(freeze):
    wall,cpu=start('predict',freeze,1800,2700)
    require_frozen(freeze,[f'results/{EXP}/{s}.json' for s in ('calibration','panel_manifest')])
    meta=json.loads((DIR/'panel_manifest.json').read_text())
    if meta['calibration_sha256']!=digest((DIR/'calibration.json').read_bytes()):
        raise ValueError('Calibration binding differs')
    calibration=json.loads((DIR/'calibration.json').read_text())
    panel=checked_artifact(meta['panel'])
    raw=load_archive(calibration['shared_counts_archive'])
    masses=calibration['selected']['masses']
    models={arm:DepthSource(raw['alphabet'],vector,{h:r for h,r in raw['counts'].items() if len(h)<=len(vector)})
            for arm,vector in zip(ARMS,([256.]*3,[256.]*12,masses,masses[:3]),strict=True)}
    delta=0.
    output={}
    for name,case in panel['cases'].items():
        output[name]={}
        for arm,model in models.items():
            # Counts are already immutable and shared within each model. This
            # lightweight view avoids materializing every row again per record.
            reference={'alphabet':model.alphabet,'masses':model.masses,'counts':model.counts}
            rows=[]
            for observed in case['records']:
                result=decode(model,case['units'],observed,1/225,max_nodes=500_000)
                total,best,nodes=infer(reference,case['units'],observed,1/225)
                if result.plaintext is None or not math.isfinite(total):
                    raise ValueError('Complete known-key support failed')
                delta=max(delta,abs(total-result.log_likelihood),abs(best-result.joint_log_probability),
                          abs(reading(reference,case['units'],observed,result.plaintext,1/225)-best))
                if delta>1e-7 or nodes!=result.reachable_nodes:
                    raise ValueError('Reference inference differs')
                rows.append(result.to_dict())
            output[name][arm]=rows
        artifact=save_new(ROOT/'outputs'/EXP/f'{name}.json.gz',output[name],compressed=True)
        save_new(DIR/f'{name}_prediction.json',{'case':name,'predictions':artifact})
        print(json.dumps({'completed':name}),flush=True)
    artifact=save_new(ROOT/'outputs'/EXP/'predictions.json.gz',{'cases':output},compressed=True)
    save_new(DIR/'prediction_manifest.json',{'experiment':EXP,'freeze':freeze,'predictions':artifact,
        'panel_manifest_sha256':digest((DIR/'panel_manifest.json').read_bytes()),
        'calibration_sha256':digest((DIR/'calibration.json').read_bytes()),
        'reference_records':128,'maximum_score_delta':delta,'resources':resource_report(wall,cpu)})


def evaluate(freeze):
    wall,cpu=start('evaluate',freeze,120,180)
    require_frozen(freeze,[f'results/{EXP}/{s}.json' for s in ('calibration','panel_manifest','prediction_manifest')])
    meta=json.loads((DIR/'panel_manifest.json').read_text())
    predicted=json.loads((DIR/'prediction_manifest.json').read_text())
    if (predicted['panel_manifest_sha256']!=digest((DIR/'panel_manifest.json').read_bytes())
            or predicted['calibration_sha256']!=digest((DIR/'calibration.json').read_bytes())):
        raise ValueError('Evaluation provenance differs')
    panel=checked_artifact(meta['panel'])['cases']
    truth=checked_artifact(meta['answers'])['cases']
    predictions=load_archive(predicted['predictions'])['cases']
    if set(panel)!=set(truth) or set(panel)!=set(predictions) or len(panel)!=16:
        raise ValueError('All cases required')
    rows={}
    for name in panel:
        rows[name]={}
        for arm in ARMS:
            rows[name][arm]=metrics(predictions[name][arm],truth[name])
            if rows[name][arm]['record_edits']!=[edit_distance(r['plaintext'] or '',t)
                for r,t in zip(predictions[name][arm],truth[name],strict=True)]:
                raise ValueError('Independent edit counts differ')
    checks=gate([r['calibrated12']['edits'] for r in rows.values()],[r['fixed12']['edits'] for r in rows.values()])
    totals={arm:sum(r[arm]['edits'] for r in rows.values()) for arm in ARMS}
    result={'experiment':EXP,'freeze':freeze,'status':'prospective_known_key_development_not_blind_recovery',
        'cases':rows,'gates':checks,'reader_gate_pass':all(checks.values()),'total_edits':totals,
        'learned_long_context_beats_learned_short':totals['calibrated12']<totals['calibrated3'],
        'gold_characters':7168,'independent_edit_replays':128,'resources':resource_report(wall,cpu)}
    save_new(DIR/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('gates','reader_gate_pass','total_edits','learned_long_context_beats_learned_short')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','predict','evaluate'))
    parser.add_argument('--freeze',required=True)
    args=parser.parse_args()
    try:
        {'prepare':prepare,'predict':predict,'evaluate':evaluate}[args.stage](args.freeze)
    except Exception as error:
        if not (DIR/f'{args.stage}_failure.json').exists():
            save_new(DIR/f'{args.stage}_failure.json',{'stage':args.stage,'freeze':args.freeze,
                'error_type':type(error).__name__,'error':str(error)})
        raise
